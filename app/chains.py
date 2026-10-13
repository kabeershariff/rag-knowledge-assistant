"""LangChain factories and the answer runnable.

Every LangChain object is built by a small factory function that accepts the
settings it needs, and where relevant an alternative implementation. The test
suite swaps deterministic fakes through these same functions.
"""

import time

from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import COLLECTION_NAME, EMBEDDING_DIMENSIONS, MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS, Settings

ACCEPTED_EXTENSIONS = {".pdf", ".txt", ".md"}

_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def media_type_for(filename: str) -> str | None:
    """Map a filename to its media type, or None when the extension is not accepted."""
    from pathlib import PurePath

    extension = PurePath(filename).suffix.lower()
    return _MEDIA_TYPES.get(extension)


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


class ServiceUnavailableError(Exception):
    """Raised when an external service is unavailable after retries."""


def build_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    """Return the Gemini embeddings model with the configured settings."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=EMBEDDING_DIMENSIONS,
        request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
    )


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
                wait = 2 ** attempt
                time.sleep(wait)
    raise ServiceUnavailableError(
        "Embedding service unavailable after retries."
    ) from last_exception


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception represents a rate limit or server error."""
    if hasattr(exc, "code"):
        return exc.code in (429, 500, 502, 503, 504)
    if hasattr(exc, "status_code"):
        return exc.status_code in (429, 500, 502, 503, 504)
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return any(
        word in name or word in message
        for word in ("rate", "resource", "unavailable", "timeout", "quota", "429", "503", "500", "502", "504")
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


def add_chunks(store: Chroma, chunks: list[Document], vectors: list[list[float]]) -> None:
    """Add chunks to the store using vectors that were already computed.

    The store is reached through its collection because the wrapper's add_texts
    would embed the same texts a second time. Chunk identifiers are explicit, and
    the underlying call is an upsert, so re-ingesting a document replaces its
    previous version instead of duplicating it.
    """
    store._collection.upsert(
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
