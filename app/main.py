"""FastAPI application, routes, exception handlers and request logging."""

import logging
import time

from fastapi import FastAPI, Request

from app.config import get_settings
from app.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge_assistant")

app = FastAPI(
    title="Knowledge Assistant",
    description="Answers questions from a document collection with verbatim evidence.",
    version="1.0.0",
)


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


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
def healthz() -> HealthResponse:
    settings = get_settings()
    gemini = "configured" if settings.gemini_configured else "missing"
    return HealthResponse(
        status="ok" if gemini == "configured" else "degraded",
        gemini=gemini,
        chroma="ok",
        document_count=0,
    )
