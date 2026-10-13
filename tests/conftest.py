"""Shared fixtures and offline fakes.

Every test runs without a network call and without an API key. The fake
embeddings produce deterministic hashed vectors, so ranking, offsets and gate
behavior are reproducible. The stub answer runnable is a RunnableLambda that
returns a fixed Answer, so generation can be driven from a test.

Hashed vectors cannot measure semantic similarity. The offline suite verifies
mechanics. Semantic quality is measured by the live evaluation run.
"""

import hashlib
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from app import chains, rag
from app.chains import Answer
from app.config import Settings
from app.main import app, get_service
from app.rag import Service

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"
SAMPLE_FILENAMES = (
    "employee_handbook.pdf",
    "it_security_policy.txt",
    "benefits_guide.md",
    "remote_work_policy.md",
)

FAKE_DIMENSIONS = 64


class FakeEmbeddings(Embeddings):
    """Deterministic hashed vectors for offline tests.

    Identical text produces identical vectors. Texts sharing words have higher
    cosine similarity. The dimension is fixed at construction time.
    """

    def __init__(self, dimensions: int = FAKE_DIMENSIONS):
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append([text])
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


class StubAnswer:
    """A scripted answer runnable.

    The returned object is a RunnableLambda, so it drops into the same
    composition the production chain uses. Every payload it receives is kept in
    `calls`, which lets a test assert that the model was never invoked at all.
    """

    def __init__(
        self,
        answer: str = "A grounded answer [1].",
        answerable: bool = True,
        citations: tuple[int, ...] = (1,),
        fail_with: Exception | None = None,
        failures: int = 0,
    ) -> None:
        self.answer = answer
        self.answerable = answerable
        self.citations = list(citations)
        self.fail_with = fail_with
        self.failures = failures
        self.calls: list[dict] = []

    def runnable(self):
        return RunnableLambda(self._respond)

    def _respond(self, payload: dict) -> Answer:
        self.calls.append(payload)
        if self.fail_with is not None and len(self.calls) <= self.failures:
            raise self.fail_with
        return Answer(
            answer=self.answer,
            answerable=self.answerable,
            citations=list(self.citations),
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


def sample_bytes(filename: str) -> bytes:
    """Return the bytes of one committed sample document."""
    return (SAMPLE_DIR / filename).read_bytes()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway store directory and a fake key.

    Retries default to zero so that a deliberate failure path in a test costs no
    wall clock time. A test that checks the retry count sets the value itself.
    """
    return Settings(
        gemini_api_key="offline-test-key",
        chroma_persist_dir=str(tmp_path / "chroma"),
        api_max_retries=0,
    )


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def stub() -> StubAnswer:
    return StubAnswer()


@pytest.fixture
def service(settings: Settings, embeddings: FakeEmbeddings, stub: StubAnswer) -> Service:
    """A service with hashed embeddings, a temporary store and a stub model."""
    store = chains.build_store(settings, embeddings)
    return Service(
        settings=settings,
        embeddings=embeddings,
        store=store,
        answer_chain=stub.runnable(),
    )


@pytest.fixture
def make_service(settings: Settings):
    """Build extra services against the same temporary store.

    Used to prove the index survives a restart, and to vary one dependency at a
    time in the failure suite.
    """

    def factory(**overrides) -> Service:
        chosen = {**settings.model_dump(), **overrides.pop("settings_overrides", {})}
        resolved = Settings(**chosen)
        local_embeddings = overrides.pop("embeddings", None)
        local_stub = overrides.pop("stub", None)
        local_store = overrides.pop("store", None)
        local_chain = overrides.pop("answer_chain", None)

        if local_store is None:
            local_store = chains.build_store(
                resolved,
                local_embeddings if local_embeddings is not None else FakeEmbeddings(),
            )
        if local_chain is None and local_stub is not None:
            local_chain = local_stub.runnable()

        return Service(
            settings=resolved,
            embeddings=local_embeddings,
            store=local_store,
            answer_chain=local_chain,
        )

    return factory


@pytest.fixture
def client(service: Service):
    """A test client whose routes resolve to the injected service."""
    app.dependency_overrides[get_service] = lambda: service
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def corpus(service: Service) -> Service:
    """A service with all four sample documents ingested."""
    for filename in SAMPLE_FILENAMES:
        rag.ingest_document(sample_bytes(filename), filename, service)
    return service


@pytest.fixture
def chunk_reader(store_of):
    """Read back the indexed chunks of a document.

    Returns a function taking a document id and yielding
    (chunk_id, text, metadata) triples in index order.
    """

    def reader(document_id: str):
        store = store_of()
        records = store._collection.get(
            where={"document_id": document_id}, include=["documents", "metadatas"]
        )
        rows = list(zip(records["ids"], records["documents"], records["metadatas"]))
        return sorted(rows, key=lambda row: row[2]["chunk_index"])

    return reader


@pytest.fixture
def store_of(service: Service):
    """Return the store behind the injected service."""
    return lambda: service.get_store()


@pytest.fixture
def verbatim_query(chunk_reader):
    """Build a question from a chunk's own words.

    Hashed vectors reward word overlap rather than meaning, so a paraphrase does
    not reliably reach the chunk it describes. A query built from the chunk text
    does, which lets the offline suite assert ranking, gate and evidence
    mechanics deterministically. Semantic quality is measured by the live
    evaluation run, not here.
    """

    def query(document_id: str, chunk_index: int, words: int | None = None) -> str:
        rows = chunk_reader(document_id)
        matching = [row for row in rows if row[2]["chunk_index"] == chunk_index]
        assert matching, f"{document_id} has no chunk {chunk_index}"
        tokens = matching[0][1].split()
        if words is not None:
            tokens = tokens[:words]
        return " ".join(tokens)

    return query


@pytest.fixture
def no_sleep(monkeypatch):
    """Remove retry backoff sleeps so failure paths cost no wall clock time.

    Only the sleeping is removed. The retry count and the ordering of attempts
    are still asserted, which is what the retry logic is actually responsible for.
    """
    monkeypatch.setattr("app.rag.time.sleep", lambda seconds: None)
    monkeypatch.setattr("app.chains.time.sleep", lambda seconds: None)


@pytest.fixture
def answerable_question(verbatim_query):
    """A question built from indexed text, which is guaranteed to clear the gate."""

    def build(document_id: str, chunk_index: int) -> str:
        return verbatim_query(document_id, chunk_index)

    return build


OFF_CORPUS_QUESTIONS = (
    "What is the maximum rotor diameter permitted on the coastal wind farm turbines?",
    "How many parking spaces are available in the Munich office garage?",
    "What is the annual bonus target for the sales team in fiscal 2027?",
    "Which airline lounge does the company book for long haul flights?",
    "What is the rivet tolerance on a zeppelin fuselage?",
)


@pytest.fixture
def corpus_client(corpus: Service):
    """A test client over the fully ingested sample corpus."""
    app.dependency_overrides[get_service] = lambda: corpus
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def write_temp(tmp_path: Path):
    """Write text to a file under the test's temporary directory and return its path."""
    counter = {"n": 0}

    def factory(text: str, filename: str) -> str:
        counter["n"] += 1
        suffix = Path(filename).suffix or ".txt"
        path = tmp_path / f"source_{counter['n']}{suffix}"
        path.write_text(text, encoding="utf-8")
        return str(path)

    return factory


@pytest.fixture
def sample_text() -> str:
    """A long synthetic document with clearly separated topics.

    Used where a test needs to know which chunk holds which text without
    depending on the wording of the committed samples. It is long enough to
    split into several overlapping chunks at the configured size.
    """
    sections = [
        (
            "Paid time off accrues monthly at 10 hours per month for every "
            "full time employee, and unused hours carry over up to a maximum "
            "of 120 hours into the next calendar year. A part time employee "
            "accrues on a pro rata basis against contracted hours."
        ),
        (
            "The expense approval chain requires a written manager approval "
            "for any single expense above 500 dollars, and a finance director "
            "approval above 2000 dollars, before reimbursement is issued. A "
            "receipt is required for every expense above 25 dollars."
        ),
        (
            "Onboarding expectations require a new hire to have a company "
            "laptop and a security key delivered within five business days of "
            "the start date, and to complete account setup on the first day. "
            "The induction session covers payroll, benefits and security."
        ),
        (
            "The performance review cycle runs twice a year, with calibration "
            "meetings in the second week of June and the second week of "
            "December, and written feedback delivered within ten days of the "
            "conversation taking place."
        ),
        (
            "Reporting lines run from an individual contributor to a team "
            "lead, then to an engineering manager and finally to a director. "
            "Managers are expected to hold one written one to one meeting with "
            "each report every calendar month."
        ),
        (
            "The probation period lasts three months from the start date. A "
            "probation review meeting is scheduled at the six week mark, and "
            "the written outcome is shared at the end of the period."
        ),
    ]
    paragraphs = []
    for index, body in enumerate(sections, start=1):
        paragraphs.append(f"## Section {index}\n\n{body}\n")
    return "\n".join(paragraphs)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: needs GEMINI_API_KEY and network access"
    )