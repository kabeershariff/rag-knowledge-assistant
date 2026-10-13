"""LangChain factories, the Chroma adapter and the answer runnable.

Every LangChain object is built by a small factory function that accepts the
settings it needs, and where relevant an alternative implementation. The test
suite swaps deterministic fakes through these same functions.

This module is also the only place that touches Chroma's private collection.
`langchain-chroma` exposes no public API that reads stored documents and metadata
back together, so the four functions below are a documented firewall around that
one gap rather than a set of scattered private accesses.
"""

from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import COLLECTION_NAME, EMBEDDING_DIMENSIONS, MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS, Settings


class StoreAccessError(RuntimeError):
    """Raised when the vector store cannot be read through the adapter."""


def build_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    """Return the Gemini embeddings model with the configured settings."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=EMBEDDING_DIMENSIONS,
        request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
    )


def build_store(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    """Return the Chroma vector store with cosine distance.

    The embedding function is optional. Listing and deleting documents touch only
    the collection and never embed anything, so those routes work without a key.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


# Chroma adapter. Nothing outside this module may reach for `store._collection`.

def collection_of(store: Chroma):
    """Return the underlying Chroma collection, or raise.

    `langchain-chroma` exposes no public API that reads stored documents and
    metadata back together, so the application reaches for the private
    `_collection` attribute in exactly one place: this function. A Chroma release
    that moves the attribute is then a change to this module alone, instead of a
    hunt through the application.
    """
    collection = getattr(store, "_collection", None)
    if collection is None:
        raise StoreAccessError(
            "The installed langchain-chroma does not expose a readable collection. "
            "Check the installed version against the pin in requirements.txt."
        )
    return collection


def read_chunks(
    store: Chroma,
    *,
    ids: list[str] | None = None,
    where: dict | None = None,
    limit: int | None = None,
    include: list[str] | None = None,
) -> dict:
    """Read stored rows out of the collection.

    Returns Chroma's own mapping of `ids`, `documents` and `metadatas`. This is a
    thin, documented pass-through to the private collection, not a reimplementation
    of Chroma's query language.
    """
    return collection_of(store).get(
        ids=ids,
        where=where,
        limit=limit,
        include=include if include is not None else ["documents", "metadatas"],
    )


def metadatas_of(store: Chroma) -> list[dict]:
    """Return every non-empty metadata dict in the collection."""
    records = read_chunks(store, include=["metadatas"])
    return [metadata for metadata in (records.get("metadatas") or []) if metadata]


def has_chunks(store: Chroma, document_id: str) -> bool:
    """True when at least one chunk of the document is indexed."""
    records = read_chunks(store, where={"document_id": document_id}, limit=1, include=[])
    return bool(records.get("ids"))


def add_chunks(store: Chroma, chunks: list[Document], vectors: list[list[float]]) -> None:
    """Add chunks to the store using vectors that were already computed.

    The collection is used directly because the wrapper's add_texts would embed the
    same texts a second time. Chunk identifiers are explicit, and the underlying
    call is an upsert, so re-ingesting a document replaces its previous version
    instead of duplicating it.
    """
    collection_of(store).upsert(
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[dict(chunk.metadata) for chunk in chunks],
        embeddings=vectors,
    )


class Answer(BaseModel):
    """Structured output from the answer model."""

    answer: str
    answerable: bool
    citations: list[int]


def build_answer_chain(settings: Settings, model=None):
    """Compose prompt, model and structured output into the answer runnable.

    The model parameter accepts an alternative implementation for testing.
    The returned runnable takes a dict with context_blocks and question keys.
    """
    if model is None:
        model = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            max_retries=settings.api_max_retries,
        )

    system_message = (
        "Answer only from the CONTEXT provided. Cite supporting context blocks "
        "by their number in square brackets, for example [2]. If the CONTEXT "
        "does not contain the answer, set answerable to false and start the "
        "answer with the word UNAVAILABLE. Never use knowledge from outside "
        "the CONTEXT. Return the answer as plain text."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{context_blocks}\n\n{question}"),
    ])

    return prompt | model.with_structured_output(Answer)