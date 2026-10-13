"""Request and response models."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    gemini: Literal["configured", "missing"]
    chroma: str
    document_count: int


class DocumentIngestResponse(BaseModel):
    """Returned by POST /api/documents after a successful ingest."""

    document_id: str
    filename: str
    media_type: str
    chunk_count: int


class DocumentSummary(BaseModel):
    """One entry of the document list."""

    document_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class EvidenceLocation(BaseModel):
    """Where an excerpt sits inside its document.

    `page_number` is present only for PDF sources. Offsets are relative to the page
    text rather than the whole file, because each page is split independently.
    """

    page_number: int | None = None
    chunk_index: int
    char_start: int
    char_end: int


class Source(BaseModel):
    """One verbatim piece of evidence, never written by the model."""

    document_name: str
    document_id: str
    excerpt: str
    location: EvidenceLocation
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    sources: list[Source]
    retrieved_count: int
    evidence_count: int
    distinct_document_count: int
    model: str
    latency_ms: float