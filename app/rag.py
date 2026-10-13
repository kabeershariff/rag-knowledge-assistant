"""Ingest, ask, list, delete, grounding and evidence."""

import re
from pathlib import Path
from typing import Any

from app import chains
from app.config import Settings

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

    return registry.upsert(document_id, filename, media_type, len(chunks))
