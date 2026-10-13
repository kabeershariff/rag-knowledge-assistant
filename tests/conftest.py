import hashlib
import math

import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic hashed vectors for offline tests.

    Identical text produces identical vectors. Texts sharing words have higher
    cosine similarity. The dimension is fixed at construction time.
    """

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = text.lower().split()
        for word in words:
            hash_bytes = hashlib.sha256(word.encode()).digest()
            for i in range(self.dimensions):
                bit = (hash_bytes[i // 8] >> (i % 8)) & 1
                vector[i] += 1.0 if bit else -1.0

        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: needs GEMINI_API_KEY and network access"
    )
