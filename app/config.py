"""Environment driven settings and fixed constants."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIMENSIONS = 768
COLLECTION_NAME = "knowledge_docs"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 30
UNAVAILABLE_MESSAGE = "The requested information is unavailable in the provided documents."


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str | None = None
    gemini_chat_model: str = "gemini-3.1-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"

    chroma_persist_dir: str = "./data/chroma"

    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    relevance_max_distance: float = 0.33

    llm_temperature: float = 0.0
    api_max_retries: int = 3

    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
