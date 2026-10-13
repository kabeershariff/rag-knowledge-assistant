"""Retry policy for the two remote calls, and the error types the routes map.

Both remote calls can fail the same two ways: the service is unreachable, or it
came back with something unusable. This module owns that distinction once, so
`documents.py` and `rag.py` agree on what counts as retryable and on which
exception the HTTP layer should see.

The dependency direction is deliberate and one-way. `retries` imports nothing
from the application except the settings it reads, so neither caller has to import
the other: `documents.py` uses it for embedding retries and `rag.py` for
generation retries, and neither knows the other exists.
"""

import json
import time

from langchain_core.exceptions import OutputParserException
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from app.config import Settings


class IngestError(Exception):
    """A domain error that maps to an HTTP status code."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ServiceUnavailableError(Exception):
    """Raised when an external service is unavailable after retries."""


RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)
RETRYABLE_HINTS = ("rate", "resource", "unavailable", "timeout", "quota", "429", "500", "502", "503", "504")


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception represents a rate limit or server error."""
    if hasattr(exc, "code"):
        return exc.code in RETRYABLE_STATUS_CODES
    if hasattr(exc, "status_code"):
        return exc.status_code in RETRYABLE_STATUS_CODES
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return any(word in name or word in message for word in RETRYABLE_HINTS)


def embed_with_retry(
    embeddings: Embeddings, texts: list[str], settings: Settings
) -> list[list[float]]:
    """Embed texts with exponential backoff on rate limit and server errors.

    The LangChain embeddings wrapper has no retry setting of its own, so this
    helper wraps the call with retries at 1, 2 and 4 seconds.
    """
    last_exception: Exception | None = None
    for attempt in range(settings.api_max_retries + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exception = exc
            if attempt < settings.api_max_retries:
                time.sleep(2 ** attempt)
    raise ServiceUnavailableError(
        "Embedding service unavailable after retries."
    ) from last_exception


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


def invoke_with_retry(chain, payload: dict, settings: Settings) -> object:
    """Call the answer runnable, retrying with backoff, then classify the failure.

    A parse or validation failure is a malformed model answer and becomes 502.
    Anything else is the model service being unreachable and becomes 503.
    """
    last_exc: Exception | None = None

    for attempt in range(settings.api_max_retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_exc = exc
            if attempt < settings.api_max_retries:
                time.sleep(2 ** attempt)

    if _is_malformed_output(last_exc):
        raise IngestError(502, "Model returned malformed output after retry.") from last_exc
    raise ServiceUnavailableError("Model service unavailable after retries.") from last_exc