"""Document lifecycle: registry, upload validation, ingestion, listing and deletion.

Everything here owns a document from the moment its bytes arrive to the moment it
is deleted. The query pipeline in `rag.py` never touches this module's functions
directly; it goes through the shared `Service`.

The loaders and splitter live here rather than in `chains.py` because ingestion
is their only caller. `chains.py` keeps the factories that wrap the remote model
and the vector store, which are used from both sides.
"""

import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import chains
from app.config import MAX_UPLOAD_BYTES, Settings
from app.retries import IngestError, embed_with_retry

if TYPE_CHECKING:  # pragma: no cover - import cycle broken by lazy typing only
    from app.rag import Service

logger = logging.getLogger("knowledge_assistant")

ACCEPTED_EXTENSIONS = {".pdf", ".txt", ".md"}

_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


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

    def list(self) -> list[dict]:
        return list(self._by_id.values())

    def remove(self, document_id: str) -> bool:
        return self._by_id.pop(document_id, None) is not None

    def count(self) -> int:
        return len(self._by_id)


def media_type_for(filename: str) -> str | None:
    """Map a filename to its media type, or None when the extension is not accepted."""
    return _MEDIA_TYPES.get(Path(filename).suffix.lower())


def build_loader(file_path: str, extension: str):
    """Return a loader that reads the file into Document objects.

    PDF loading returns one Document per page. Text and Markdown loading return a
    single Document. Encoding is auto detected so that a stray byte order mark or a
    latin-1 source file does not fail an otherwise valid upload.
    """
    if extension == ".pdf":
        return PyPDFLoader(file_path)
    if extension == ".txt":
        return TextLoader(file_path, autodetect_encoding=True)
    if extension == ".md":
        return TextLoader(file_path, autodetect_encoding=True)
    raise ValueError(f"unsupported extension: {extension}")


def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """Return the recursive splitter with exact start offsets recorded per chunk."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def load_and_split(
    file_path: str, extension: str, settings: Settings, metadata: dict
) -> list[Document]:
    """Load the file, split it into chunks and stamp shared metadata onto each one.

    Page numbers are converted to a one based value for PDFs, because that is what a
    reader expects when evidence points at a page. Text and Markdown files carry no
    page number, so the key is absent rather than set to a placeholder.
    """
    documents = build_loader(file_path, extension).load()

    chunks: list[Document] = []
    for document in documents:
        merged = {**document.metadata, **metadata}
        # The loader records the temporary path it read from. It is meaningless
        # once the chunks are indexed, so it is dropped.
        merged.pop("source", None)
        if extension == ".pdf":
            page_value = merged.get("page")
            if page_value is not None:
                merged["page_number"] = int(page_value) + 1
        split = build_splitter(settings).split_documents([document])
        for chunk in split:
            chunk.metadata = {**chunk.metadata, **merged}
            chunks.append(chunk)

    # Chunk numbers run across the whole document, so a PDF cites page and chunk
    # together. The identifier is explicit so that re-ingesting the same document
    # replaces its previous chunks instead of adding duplicates beside them.
    document_id = metadata["document_id"]
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{document_id}::{index}"

    return [chunk for chunk in chunks if chunk.page_content.strip()]


def validate_upload(filename: str, size_bytes: int) -> str:
    """Validate extension and size, returning the media type. Raises IngestError."""
    media_type = media_type_for(filename)
    if media_type is None:
        raise IngestError(415, f"Unsupported file type for {filename!r}.")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise IngestError(
            413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
    return media_type


def ingest_document(content: bytes, filename: str, service: "Service") -> dict:
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
        chunks = load_and_split(tmp_path, extension, service.settings, metadata)
    except Exception as exc:
        raise IngestError(422, "File is corrupted or contains no extractable text.") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not chunks:
        raise IngestError(422, "File contains no extractable text.")

    # Embed first, then delete the previous version, then add. Re-uploading a
    # filename is still idempotent, because the delete clears the superseded
    # chunks and the new ones carry explicit identifiers. Doing the delete last
    # means a quota or rate limit failure while embedding leaves the already
    # indexed version of the document intact, instead of destroying it on the way
    # to a failure it was never going to survive.
    store = service.get_store()
    vectors = embed_with_retry(
        service.get_embeddings(), [chunk.page_content for chunk in chunks], service.settings
    )
    store.delete(where={"document_id": document_id})
    chains.add_chunks(store, chunks, vectors)

    return service.registry.upsert(document_id, filename, media_type, len(chunks))


def documents_in_store(service: "Service") -> list[dict]:
    """Derive the document list from the collection metadata.

    The index outlives the process, so it is the source of truth. The registry
    only fills in a name for a document whose chunks carry no readable metadata.
    """
    store = service.get_store()
    grouped: dict[str, dict[str, Any]] = {}
    for metadata in chains.metadatas_of(store):
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


def list_documents(service: "Service") -> list[dict]:
    """Return every ingested document with its chunk count."""
    return documents_in_store(service)


def delete_document(document_id: str, service: "Service") -> bool:
    """Delete a document and its chunks. Returns False when the id is unknown."""
    store = service.get_store()
    if not chains.has_chunks(store, document_id):
        service.registry.remove(document_id)
        return False

    # Delete forwards keyword arguments to the collection, so the metadata filter
    # removes every chunk of the document in one call.
    store.delete(where={"document_id": document_id})
    service.registry.remove(document_id)
    logger.info("deleted document_id=%s", document_id)
    return True