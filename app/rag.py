"""The query pipeline and the dependency container every request resolves through.

`Service` builds the embeddings, the store and the answer chain on first use, so
a missing key costs a 503 rather than an import failure. Below it sit the four
stages of an answer: retrieve, gate, generate, validate citations.
"""

import logging
from typing import Any

from app import chains
from app.config import UNAVAILABLE_MESSAGE, Settings
from app.documents import DocumentRegistry
from app.retries import IngestError, invoke_with_retry

logger = logging.getLogger("knowledge_assistant")

MISSING_KEY_MESSAGE = (
    "GEMINI_API_KEY is not configured. Set it in .env or in the environment and "
    "restart the service."
)


class Service:
    """Everything a request needs, built on first use.

    Each object can be replaced at construction, which is how the offline test
    suite injects hashed embeddings, a stub answer runnable and a temporary store
    without touching a route.
    """

    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry | None = None,
        embeddings: Any | None = None,
        store: Any | None = None,
        answer_chain: Any | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry if registry is not None else DocumentRegistry()
        self.embeddings = embeddings
        self.store = store
        self.answer_chain = answer_chain
        self._store_searchable = store is not None

    def require_gemini(self) -> None:
        """Raise 503 with an actionable message when nothing was injected and no key."""
        if self.embeddings is None and not self.settings.gemini_configured:
            raise IngestError(503, MISSING_KEY_MESSAGE)

    def get_embeddings(self) -> Any:
        self.require_gemini()
        if self.embeddings is None:
            self.embeddings = chains.build_embeddings(self.settings)
        return self.embeddings

    def get_store(self) -> Any:
        """Return the store, rebuilding it if it was opened without an embedder.

        Without an embedder the store can only read and delete metadata. Chroma
        would fall back to its own bundled embedding model for a search, which
        both downloads a model and disagrees with the indexed dimension, so a
        store opened that way is rebuilt as soon as an embedder exists.
        """
        if self.store is not None and self._store_searchable:
            return self.store

        if self.embeddings is not None or self.settings.gemini_configured:
            self.store = chains.build_store(self.settings, self.get_embeddings())
            self._store_searchable = True
        else:
            # Listing, deleting and the health check never embed, so they work
            # without a key.
            self.store = chains.build_store(self.settings, None)
            self._store_searchable = False
        return self.store

    def get_answer_chain(self) -> Any:
        self.require_gemini()
        if self.answer_chain is None:
            self.answer_chain = chains.build_answer_chain(self.settings)
        return self.answer_chain


def retrieve(question: str, service: Service, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve and gate chunks for a question.

    Uses similarity_search_with_score to get raw cosine distances, converts to
    score with score = 1 - distance, then filters with the single threshold at
    RELEVANCE_MAX_DISTANCE. Logs the best distance on one line.

    Returns a list of gated result dicts sorted by ascending distance (best first).
    Each dict holds the chunk text, metadata, distance, and score (1 - distance).
    """
    if top_k is None:
        top_k = service.settings.retrieval_top_k

    # Search always needs an embedder, so a missing key is refused here rather
    # than reaching Chroma.
    service.require_gemini()

    store = service.get_store()
    raw_results: list[tuple] = store.similarity_search_with_score(
        query=question,
        k=top_k,
    )

    # Chroma cosine distance: lower is better. Convert to score where higher is better.
    gated: list[dict[str, Any]] = []
    for doc, distance in raw_results:
        if distance <= service.settings.relevance_max_distance:
            gated.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "distance": distance,
                "score": 1.0 - distance,
            })

    best_distance = raw_results[0][1] if raw_results else None
    logger.info(
        "question=%r best_distance=%.4f gated=%d/%d",
        question[:60],
        best_distance if best_distance is not None else -1.0,
        len(gated),
        len(raw_results),
    )

    return gated


def ask(question: str, service: Service, top_k: int | None = None) -> dict:
    """Retrieve, gate, generate and validate an answer.

    Returns a response dict with answer, answerable, sources and counts.
    """
    gated = retrieve(question, service, top_k)

    if not gated:
        return {
            "answer": UNAVAILABLE_MESSAGE,
            "answerable": False,
            "sources": [],
            "retrieved_count": 0,
            "evidence_count": 0,
            "distinct_document_count": 0,
        }

    context_blocks = _format_context_blocks(gated)
    result = invoke_with_retry(
        service.get_answer_chain(),
        {"context_blocks": context_blocks, "question": question},
        service.settings,
    )

    valid_citations = [c for c in result.citations if 1 <= c <= len(gated)]
    answerable = result.answerable and len(valid_citations) > 0
    answer_text = result.answer.strip()
    if answerable and _is_abstention(answer_text):
        answerable = False
    answer = answer_text if answerable else UNAVAILABLE_MESSAGE

    sources = _build_evidence(gated, valid_citations)

    return {
        "answer": answer,
        "answerable": answerable,
        "sources": sources,
        "retrieved_count": len(gated),
        "evidence_count": len(valid_citations),
        "distinct_document_count": len({s["document_id"] for s in sources}),
    }


def _is_abstention(text: str) -> bool:
    """True when the model opened its answer with the abstention marker."""
    words = text.split(maxsplit=1)
    if not words:
        return True
    return words[0].strip(".,:!\"'").upper() == "UNAVAILABLE"


def _format_context_blocks(gated: list[dict]) -> str:
    """Format gated chunks as numbered context blocks for the prompt."""
    blocks = []
    for i, chunk in enumerate(gated, start=1):
        location = _format_location(chunk["metadata"], len(chunk["text"]))
        header = f"[{i}] DOC: {chunk['metadata']['document_name']} | LOC: {location}"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def _format_location(metadata: dict, text_length: int) -> str:
    """Format a location string from chunk metadata."""
    parts = []
    if "page_number" in metadata:
        parts.append(f"page {metadata['page_number']}")
    parts.append(f"chunk {metadata['chunk_index']}")
    parts.append(f"chars {metadata['start_index']} to {metadata['start_index'] + text_length}")
    return ", ".join(parts)


def _build_evidence(gated: list[dict], valid_citations: list[int]) -> list[dict]:
    """Build evidence entries from valid citations."""
    sources = []
    for idx in valid_citations:
        chunk = gated[idx - 1]
        meta = chunk["metadata"]
        location = {
            "chunk_index": meta["chunk_index"],
            "char_start": meta["start_index"],
            "char_end": meta["start_index"] + len(chunk["text"]),
        }
        if "page_number" in meta:
            location["page_number"] = meta["page_number"]
        sources.append({
            "document_name": meta["document_name"],
            "document_id": meta["document_id"],
            "excerpt": chunk["text"][:500],
            "location": location,
            "score": chunk["score"],
        })
    return sources