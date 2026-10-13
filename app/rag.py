"""Ingest, ask, list, delete, grounding and evidence."""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app import chains
from app.config import MAX_UPLOAD_BYTES, UNAVAILABLE_MESSAGE, Settings

logger = logging.getLogger("knowledge_assistant")

MISSING_KEY_MESSAGE = (
    "GEMINI_API_KEY is not configured. Set it in .env or in the environment and "
    "restart the service."
)


class IngestError(Exception):
    """A domain error that maps to an HTTP status code."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class Service:
    """Everything a request needs, built on first use.

    Each object can be replaced at construction, which is how the offline test
    suite injects hashed embeddings, a stub answer runnable and a temporary store
    without touching a route.
    """

    def __init__(
        self,
        settings: Settings,
        registry: "DocumentRegistry | None" = None,
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
        """Raise 503 when nothing was injected and no key is configured."""
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


class DocumentRegistry:
    """Tracks ingested documents by normalized filename, with replace on re-upload."""

    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, Any]] = {}

    @staticmethod
    def normalize(filename: str) -> str:
        """Return the slug used as the document identifier."""
        stem = Path(filename).stem.lower()
        slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        return slug or "document"

    def upsert(self, document_id: str, filename: str, media_type: str, chunk_count: int) -> dict:
        record = {
            "document_id": document_id,
            "filename": filename,
            "media_type": media_type,
            "chunk_count": chunk_count,
        }
        self._by_id[document_id] = record
        return record

    def get(self, document_id: str) -> dict | None:
        return self._by_id.get(document_id)

    def list(self) -> list[dict]:
        return list(self._by_id.values())

    def remove(self, document_id: str) -> bool:
        return self._by_id.pop(document_id, None) is not None

    def count(self) -> int:
        return len(self._by_id)


def validate_upload(filename: str, size_bytes: int) -> str:
    """Validate extension and size, returning the media type. Raises IngestError."""
    media_type = chains.media_type_for(filename)
    if media_type is None:
        raise IngestError(415, f"Unsupported file type for {filename!r}.")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise IngestError(
            413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
    return media_type


def ingest_document(content: bytes, filename: str, service: Service) -> dict:
    """Validate, load and split one uploaded file, then index it.

    The file is written to a temporary path because LangChain loaders read from
    disk. Returns the registry record for the document.
    """
    media_type = validate_upload(filename, len(content))
    if not content.strip():
        raise IngestError(422, "File contains no extractable text.")
    service.require_gemini()

    document_id = DocumentRegistry.normalize(filename)
    extension = Path(filename).suffix.lower()

    with NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        metadata = {
            "document_id": document_id,
            "document_name": filename,
            "media_type": media_type,
        }
        chunks = chains.load_and_split(tmp_path, extension, service.settings, metadata)
    except Exception as exc:
        raise IngestError(422, "File is corrupted or contains no extractable text.") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not chunks:
        raise IngestError(422, "File contains no extractable text.")

    # Delete the previous version before adding the new one, so re-uploading the
    # same filename replaces it instead of accumulating chunks.
    store = service.get_store()
    store.delete(where={"document_id": document_id})

    vectors = chains.embed_with_retry(
        service.get_embeddings(), [chunk.page_content for chunk in chunks], service.settings
    )
    chains.add_chunks(store, chunks, vectors)

    return service.registry.upsert(document_id, filename, media_type, len(chunks))


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


def documents_in_store(service: Service) -> list[dict]:
    """Derive the document list from the collection metadata.

    The index outlives the process, so it is the source of truth. The registry
    only fills in a name for a document whose chunks carry no readable metadata.
    """
    store = service.get_store()
    records = store._collection.get(include=["metadatas"])

    grouped: dict[str, dict[str, Any]] = {}
    for metadata in records.get("metadatas") or []:
        if not metadata:
            continue
        document_id = metadata.get("document_id")
        if not document_id:
            continue
        entry = grouped.setdefault(
            document_id,
            {
                "document_id": document_id,
                "filename": metadata.get("document_name") or document_id,
                "chunk_count": 0,
            },
        )
        entry["chunk_count"] += 1

    for record in service.registry.list():
        grouped.setdefault(
            record["document_id"],
            {
                "document_id": record["document_id"],
                "filename": record["filename"],
                "chunk_count": record["chunk_count"],
            },
        )

    return sorted(grouped.values(), key=lambda item: item["filename"])


def delete_document(document_id: str, service: Service) -> bool:
    """Delete a document and its chunks. Returns False when the id is unknown."""
    store = service.get_store()
    existing = store._collection.get(where={"document_id": document_id}, limit=1)
    if not existing.get("ids"):
        service.registry.remove(document_id)
        return False

    # Delete forwards keyword arguments to the collection, so the metadata filter
    # removes every chunk of the document in one call.
    store.delete(where={"document_id": document_id})
    service.registry.remove(document_id)
    logger.info("deleted document_id=%s", document_id)
    return True


def list_documents(service: Service) -> list[dict]:
    """Return every ingested document with its chunk count."""
    return documents_in_store(service)


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
    result = _invoke_answer_chain(service, context_blocks, question)

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


def _invoke_answer_chain(service: Service, context_blocks: str, question: str) -> Any:
    """Call the answer runnable, retrying with backoff, then classify the failure.

    A parse or validation failure is a malformed model answer and becomes 502.
    Anything else is the model service being unreachable and becomes 503.
    """
    chain = service.get_answer_chain()
    payload = {"context_blocks": context_blocks, "question": question}
    last_exc: Exception | None = None

    for attempt in range(service.settings.api_max_retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_exc = exc
            if attempt < service.settings.api_max_retries:
                time.sleep(2 ** attempt)

    if _is_malformed_output(last_exc):
        raise IngestError(502, "Model returned malformed output after retry.") from last_exc
    raise chains.ServiceUnavailableError("Model service unavailable after retries.") from last_exc


MALFORMED_OUTPUT_MARKERS = (
    "output parsing",
    "could not parse",
    "failed to parse",
    "could not decode",
    "structured output",
    "json",
    "schema",
)


def _is_malformed_output(exc: Exception | None) -> bool:
    """Tell a bad model answer apart from an unreachable model service.

    Anything unrecognized counts as a service problem, because retrying a service
    problem is useful while reporting one as a bad answer would hide an outage.
    """
    if exc is None:
        return False
    if isinstance(exc, (OutputParserException, ValidationError, json.JSONDecodeError)):
        return True
    if type(exc).__name__ in {"OutputParserException", "ValidationError"}:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in MALFORMED_OUTPUT_MARKERS)


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
