"""FastAPI application, routes, exception handlers and request logging."""

import logging
import time

from fastapi import Depends, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from app import documents, rag
from app.config import MAX_UPLOAD_BYTES, get_settings
from app.rag import Service
from app.retries import IngestError, ServiceUnavailableError
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentSummary,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge_assistant")

READ_CHUNK_BYTES = 1024 * 1024

app = FastAPI(
    title="Knowledge Assistant",
    description="Answers questions from a document collection with verbatim evidence.",
    version="1.0.0",
)


def get_service() -> Service:
    """Return the shared service, building its objects on first use.

    The offline test suite replaces this dependency to inject fakes.
    """
    return _service


_service = Service(settings=get_settings())


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %s in %.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(IngestError)
async def handle_ingest_error(request: Request, exc: IngestError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(ServiceUnavailableError)
async def handle_service_unavailable(
    request: Request, exc: ServiceUnavailableError
) -> JSONResponse:
    logger.warning("model service unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "The model service is unavailable. Retry shortly or check "
            "the configured API key and quota."
        },
    )


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
def healthz(service: Service = Depends(get_service)) -> HealthResponse:
    settings = get_settings()
    gemini = "configured" if settings.gemini_configured else "missing"

    try:
        documents_list = documents.list_documents(service)
    except Exception as exc:
        logger.warning("vector store unreachable: %s", exc)
        body = HealthResponse(
            status="degraded", gemini=gemini, chroma="unavailable", document_count=0
        )
        return JSONResponse(status_code=503, content=body.model_dump())

    return HealthResponse(
        status="ok" if gemini == "configured" else "degraded",
        gemini=gemini,
        chroma="ok",
        document_count=len(documents_list),
    )


@app.post(
    "/api/documents",
    response_model=DocumentIngestResponse,
    status_code=201,
    tags=["documents"],
)
def ingest_document(
    file: UploadFile = File(..., description="PDF, TXT or Markdown file"),
    service: Service = Depends(get_service),
) -> DocumentIngestResponse:
    filename = file.filename or "document"
    content = _read_upload(file)
    record = documents.ingest_document(content, filename, service)
    return DocumentIngestResponse(**record)


@app.get("/api/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(service: Service = Depends(get_service)) -> DocumentListResponse:
    listed = documents.list_documents(service)
    return DocumentListResponse(documents=[DocumentSummary(**item) for item in listed])


@app.delete("/api/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(
    document_id: str, service: Service = Depends(get_service)
) -> Response:
    if not documents.delete_document(document_id, service):
        raise IngestError(404, f"Unknown document id {document_id!r}.")
    return Response(status_code=204)


@app.post("/api/ask", response_model=AskResponse, tags=["ask"])
def ask(
    payload: AskRequest, service: Service = Depends(get_service)
) -> AskResponse:
    started = time.perf_counter()
    result = rag.ask(payload.question, service, payload.top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return AskResponse(
        answer=result["answer"],
        answerable=result["answerable"],
        sources=result["sources"],
        retrieved_count=result["retrieved_count"],
        evidence_count=result["evidence_count"],
        distinct_document_count=result["distinct_document_count"],
        model=service.settings.gemini_chat_model,
        latency_ms=round(elapsed_ms, 1),
    )


def _read_upload(file: UploadFile) -> bytes:
    """Read the upload in blocks, stopping as soon as the size limit is passed.

    The body is already spooled to disk by the time a plain def route runs, so
    reading the underlying file object is safe and keeps the handler off the
    event loop.
    """
    buffer = bytearray()
    while True:
        block = file.file.read(READ_CHUNK_BYTES)
        if not block:
            break
        buffer.extend(block)
        if len(buffer) > MAX_UPLOAD_BYTES:
            raise IngestError(
                413,
                f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
    return bytes(buffer)