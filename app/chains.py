"""LangChain factories and the answer runnable.

Every LangChain object is built by a small factory function that accepts the
settings it needs, and where relevant an alternative implementation. The test
suite swaps deterministic fakes through these same functions.
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

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
        if extension == ".pdf":
            page_value = merged.get("page")
            if page_value is not None:
                merged["page_number"] = int(page_value) + 1
        split = build_splitter(settings).split_documents([document])
        for chunk in split:
            chunk.metadata = {**chunk.metadata, **merged}
            chunks.append(chunk)

    return [chunk for chunk in chunks if chunk.page_content.strip()]
