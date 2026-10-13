"""Ingest, ask, list, delete, grounding and evidence."""

import logging
import re
from pathlib import Path
from typing import Any

from app import chains
from app.config import UNAVAILABLE_MESSAGE, Settings

logger = logging.getLogger("knowledge_assistant")

MAX_UPLOAD_MB = 20


class IngestError(Exception):
    """A domain error that maps to an HTTP status code."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


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
    if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        raise IngestError(413, f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
    return media_type


def ingest_document(
    content: bytes,
    filename: str,
    settings: Settings,
    registry: DocumentRegistry,
) -> dict:
    """Validate, load and split one uploaded file, then register it.

    The file is written to a temporary path because LangChain loaders read from
    disk. Returns the registry record for the document.
    """
    media_type = validate_upload(filename, len(content))
    extension = Path(filename).suffix.lower()
    document_id = DocumentRegistry.normalize(filename)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        metadata = {
            "document_id": document_id,
            "document_name": filename,
            "media_type": media_type,
        }
        chunks = chains.load_and_split(tmp_path, extension, settings, metadata)
    except Exception as exc:
        raise IngestError(422, "File is corrupted or contains no extractable text.") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not chunks:
        raise IngestError(422, "File contains no extractable text.")

    # Delete old chunks from the store before adding the new version
    store = chains.build_store(settings, chains.build_embeddings(settings))
    store.delete(where={"document_id": document_id})

    # Embed and add new chunks
    texts = [chunk.page_content for chunk in chunks]
    embeddings = chains.build_embeddings(settings)
    vectors = chains.embed_with_retry(embeddings, texts, settings)
    store.add_texts(texts=texts, metadatas=[c.metadata for c in chunks], ids=[c.metadata["chunk_id"] for c in chunks])

    return registry.upsert(document_id, filename, media_type, len(chunks))


def retrieve(question: str, settings: Settings, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve and gate chunks for a question.

    Uses similarity_search_with_score to get raw cosine distances, converts to
    score with score = 1 - distance, then filters with the single threshold at
    RELEVANCE_MAX_DISTANCE. Logs the best distance on one line.

    Returns a list of gated result dicts sorted by ascending distance (best first).
    Each dict holds the chunk text, metadata, distance, and score (1 - distance).
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    embeddings = chains.build_embeddings(settings)
    store = chains.build_store(settings, embeddings)

    raw_results: list[tuple] = store.similarity_search_with_score(
        query=question,
        k=top_k,
    )

    # Chroma cosine distance: lower is better. Convert to score where higher is better.
    gated: list[dict[str, Any]] = []
    for doc, distance in raw_results:
        if distance <= settings.relevance_max_distance:
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


def delete_document(document_id: str, settings: Settings, registry: DocumentRegistry) -> bool:
    """Delete a document and its chunks from the store and registry. Returns True if found."""
    record = registry.get(document_id)
    if record is None:
        return False

    store = chains.build_store(settings, chains.build_embeddings(settings))
    # langchain-chroma delete() only accepts ids; use the underlying collection for metadata filter
    store._collection.delete(where={"document_id": document_id})
    registry.remove(document_id)
    logger.info("deleted document_id=%s", document_id)
    return True


def list_documents(registry: DocumentRegistry) -> list[dict]:
    """Return all registered documents."""
    return registry.list()


def ask(question: str, settings: Settings, top_k: int | None = None) -> dict:
    """Retrieve, gate, generate and validate an answer.

    Returns a response dict with answer, answerable, sources and counts.
    Raises IngestError(502) when the model returns malformed output after retry.
    """
    gated = retrieve(question, settings, top_k)

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
    chain = chains.build_answer_chain(settings)

    try:
        result = chain.invoke({"context_blocks": context_blocks, "question": question})
    except Exception:
        try:
            result = chain.invoke({"context_blocks": context_blocks, "question": question})
        except Exception as exc:
            raise IngestError(502, "Model returned malformed output after retry.") from exc

    valid_citations = [c for c in result.citations if 1 <= c <= len(gated)]
    answerable = result.answerable and len(valid_citations) > 0
    answer = result.answer if answerable else UNAVAILABLE_MESSAGE

    sources = _build_evidence(gated, valid_citations)

    return {
        "answer": answer,
        "answerable": answerable,
        "sources": sources,
        "retrieved_count": len(gated),
        "evidence_count": len(valid_citations),
        "distinct_document_count": len({s["document_id"] for s in sources}),
    }


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
