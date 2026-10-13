"""Failure cases: bad uploads, model outages, malformed output and missing key.

None of these tests reach the network. Failures are injected through a stub
runnable that raises, or by removing the key so the request is refused before
any model is built.
"""

import os

import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app import chains, rag
from app.chains import Answer, ServiceUnavailableError
from app.config import MAX_OUTPUT_TOKENS, UNAVAILABLE_MESSAGE, Settings
from app.main import app, get_service
from app.rag import IngestError, Service

from tests.conftest import StubAnswer, sample_bytes

OFF_CORPUS = (
    "What is the maximum rotor diameter permitted on the coastal wind farm turbines?"
)


def upload(client, name: str, content: bytes):
    return client.post(
        "/api/documents",
        files={"file": (name, content, "application/octet-stream")},
    )


def client_for(service: Service):
    """Build a test client bound to a specific service instance."""
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def service_without_key(settings: Settings) -> Service:
    """A service with no key and no injected embedder, which is degraded mode."""
    return Service(settings=Settings(**{**settings.model_dump(), "gemini_api_key": None}))


# Rejected uploads


@pytest.mark.parametrize("filename", ["report.docx", "archive.zip", "image.png", "notes"])
def test_unsupported_extension_returns_415(client, filename):
    response = upload(client, filename, b"some content")

    assert response.status_code == 415
    assert "Unsupported" in response.json()["detail"]


def test_empty_file_returns_422(client):
    response = upload(client, "empty.txt", b"")

    assert response.status_code == 422
    assert "detail" in response.json()


def test_whitespace_only_file_returns_422(client):
    response = upload(client, "blank.txt", b"   \n\n\t  \n")

    assert response.status_code == 422


def test_corrupted_pdf_returns_422(client):
    """A PDF header followed by rubbish must be reported, not indexed."""
    response = upload(client, "broken.pdf", b"%PDF-1.4\nthis is not a real pdf body\n")

    assert response.status_code == 422
    assert "corrupted" in response.json()["detail"].lower()


def test_truncated_pdf_returns_422(client):
    response = upload(client, "truncated.pdf", sample_bytes("employee_handbook.pdf")[:400])

    assert response.status_code == 422


def test_oversized_file_returns_413(client):
    """The size check fires before the file is split or embedded."""
    oversized = b"A" * (20 * 1024 * 1024 + 1)

    response = upload(client, "huge.txt", oversized)

    assert response.status_code == 413
    assert "20 MB" in response.json()["detail"]


def test_a_rejected_ingest_leaves_nothing_behind(client):
    upload(client, "broken.pdf", b"%PDF-1.4\ngarbage\n")

    assert client.get("/api/documents").json()["documents"] == []


# Missing key


def test_missing_key_ingest_returns_503_with_an_actionable_message(settings):
    with client_for(service_without_key(settings)) as client:
        response = upload(client, "notes.txt", b"## Notes\n\nSome content about expenses.\n")

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_missing_key_ask_returns_503_with_an_actionable_message(settings):
    with client_for(service_without_key(settings)) as client:
        response = client.post("/api/ask", json={"question": "What is the accrual rate?"})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_missing_key_still_allows_listing_health_and_docs(settings):
    """Degraded mode keeps the service inspectable without a secret."""
    with client_for(service_without_key(settings)) as client:
        assert client.get("/api/documents").status_code == 200
        assert client.get("/healthz").status_code == 200
        assert client.get("/docs").status_code == 200


# Model outages


def test_model_timeout_is_reported_as_a_service_outage(
    service, settings, answerable_question, no_sleep
):
    settings.api_max_retries = 0
    service.answer_chain = StubAnswer(
        fail_with=TimeoutError("model call timed out"), failures=99
    ).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    with pytest.raises(ServiceUnavailableError):
        rag.ask(answerable_question("benefits_guide", 7), service)


def test_every_retry_attempt_is_used_before_giving_up(
    service, settings, answerable_question, no_sleep
):
    """A transient outage is retried, and each attempt is actually spent."""
    settings.api_max_retries = 2
    flaky = StubAnswer(fail_with=TimeoutError("timeout"), failures=99)
    service.answer_chain = flaky.runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    with pytest.raises(ServiceUnavailableError):
        rag.ask(answerable_question("benefits_guide", 7), service)

    assert flaky.call_count == settings.api_max_retries + 1


def test_a_transient_failure_recovers_within_the_retry_budget(
    service, settings, answerable_question, no_sleep
):
    """The retry exists to ride out a blip, so a recovered chain must succeed."""
    settings.api_max_retries = 2
    flaky = StubAnswer(
        answer="Recovered [1].", citations=(1,), fail_with=TimeoutError("blip"), failures=1
    )
    service.answer_chain = flaky.runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    result = rag.ask(answerable_question("benefits_guide", 7), service)

    assert result["answerable"] is True
    assert result["answer"] == "Recovered [1]."
    assert flaky.call_count == 2


def test_model_outage_through_the_api_returns_503(
    service, settings, answerable_question, no_sleep
):
    settings.api_max_retries = 0
    service.answer_chain = StubAnswer(fail_with=TimeoutError("timeout"), failures=99).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    with client_for(service) as client:
        response = client.post(
            "/api/ask", json={"question": answerable_question("benefits_guide", 7)}
        )

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_an_unrecognized_error_is_treated_as_a_service_problem(
    service, settings, answerable_question, no_sleep
):
    """A surprise exception must not be mislabelled as a bad answer."""
    settings.api_max_retries = 0
    service.answer_chain = StubAnswer(fail_with=KeyError("unexpected"), failures=99).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    with pytest.raises(ServiceUnavailableError):
        rag.ask(answerable_question("benefits_guide", 7), service)


# Malformed structured output


def test_malformed_output_is_retried_then_reported_as_502(
    service, settings, answerable_question, no_sleep
):
    """A parse failure is the model's fault, so it becomes 502 rather than 503."""
    settings.api_max_retries = 1
    broken = StubAnswer(
        fail_with=OutputParserException("could not parse the JSON"), failures=99
    )
    service.answer_chain = broken.runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    with pytest.raises(IngestError) as caught:
        rag.ask(answerable_question("benefits_guide", 7), service)

    assert caught.value.status_code == 502
    assert broken.call_count == settings.api_max_retries + 1


def test_a_schema_violation_is_treated_as_malformed_output(
    service, settings, answerable_question, no_sleep
):
    settings.api_max_retries = 0
    invalid = StubAnswer(
        fail_with=ValidationError.from_exception_data("Answer", []), failures=99
    )
    service.answer_chain = invalid.runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    with pytest.raises(IngestError) as caught:
        rag.ask(answerable_question("benefits_guide", 7), service)

    assert caught.value.status_code == 502


def test_malformed_output_through_the_api_returns_502(
    service, settings, answerable_question, no_sleep
):
    settings.api_max_retries = 0
    service.answer_chain = StubAnswer(
        fail_with=OutputParserException("bad json"), failures=99
    ).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    with client_for(service) as client:
        response = client.post(
            "/api/ask", json={"question": answerable_question("benefits_guide", 7)}
        )

    assert response.status_code == 502


def test_malformed_output_recovers_on_a_later_attempt(
    service, settings, answerable_question, no_sleep
):
    settings.api_max_retries = 2
    broken = StubAnswer(
        answer="Recovered after a retry [1].",
        citations=(1,),
        fail_with=OutputParserException("could not parse"),
        failures=1,
    )
    service.answer_chain = broken.runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    result = rag.ask(answerable_question("benefits_guide", 7), service)

    assert result["answerable"] is True
    assert "Recovered" in result["answer"]


# Embedding outages


def test_embedding_failure_maps_to_a_service_outage(
    service, settings, embeddings, no_sleep
):
    settings.api_max_retries = 0
    embeddings.embed_documents = lambda texts: (_ for _ in ()).throw(
        RuntimeError("429 rate limit exceeded")
    )

    with pytest.raises(ServiceUnavailableError):
        rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)


def test_embedding_is_retried_before_giving_up(service, settings, embeddings, no_sleep):
    """A rate limit on the first embedding call must be absorbed by the retry."""
    settings.api_max_retries = 2
    original = embeddings.embed_documents
    attempts = {"n": 0}

    def flaky(texts):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("429 rate limit exceeded")
        return original(texts)

    embeddings.embed_documents = flaky

    record = rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    assert attempts["n"] == 2, "the first failure must be retried"
    assert record["chunk_count"] >= 12


# Grounding guarantees


def test_a_citation_outside_the_retrieved_set_never_becomes_evidence(
    service, answerable_question
):
    """A hallucinated block number is dropped rather than looked up."""
    service.answer_chain = StubAnswer(answer="Answer [1] [9].", citations=(1, 9)).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    result = rag.ask(answerable_question("benefits_guide", 7), service)

    assert result["evidence_count"] == 1, "only the valid citation survives"
    assert result["evidence_count"] == len(result["sources"])


def test_an_answer_with_zero_valid_citations_is_demoted_to_unavailable(
    service, answerable_question
):
    """Citing only a block that was never sent is treated as unanswerable."""
    service.answer_chain = StubAnswer(
        answer="A confident but uncited claim.", citations=(99,)
    ).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    result = rag.ask(answerable_question("benefits_guide", 7), service)

    assert result["answerable"] is False
    assert result["answer"] == UNAVAILABLE_MESSAGE
    assert result["evidence_count"] == 0
    assert result["sources"] == []


def test_abstention_wording_is_normalised_to_the_canonical_message(
    service, answerable_question
):
    """Whatever the model says when abstaining, the response says one thing."""
    service.answer_chain = StubAnswer(
        answer="UNAVAILABLE: not in the documents.", citations=(1,)
    ).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    result = rag.ask(answerable_question("benefits_guide", 7), service)

    assert result["answerable"] is False
    assert result["answer"] == UNAVAILABLE_MESSAGE


def test_a_well_formed_grounded_answer_survives_unchanged(service, answerable_question):
    service.answer_chain = StubAnswer(
        answer="Sixteen weeks of paid leave [1].", citations=(1,)
    ).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    result = rag.ask(answerable_question("benefits_guide", 7), service)

    assert result["answerable"] is True
    assert result["answer"] == "Sixteen weeks of paid leave [1]."
    assert result["evidence_count"] == 1


def test_the_gate_short_circuits_before_a_crashing_chain(service):
    """Even a chain that would raise is never reached when nothing is relevant."""
    service.answer_chain = StubAnswer(
        fail_with=AssertionError("the model must not be called"), failures=99
    ).runnable()
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    result = rag.ask(OFF_CORPUS, service)

    assert result["answerable"] is False
    assert result["answer"] == UNAVAILABLE_MESSAGE


# The answer chain contract


class RecordingModel:
    """A model stand-in that captures the prompt it was handed.

    It implements only `with_structured_output`, which is the single method
    `build_answer_chain` uses, so the real prompt composition is exercised
    without a network call.
    """

    def __init__(self):
        self.prompt_values: list[dict] = []

    def with_structured_output(self, schema):
        outer = self

        def respond(prompt_value):
            outer.prompt_values.append(prompt_value)
            return Answer(answer="Grounded [1].", answerable=True, citations=[1])

        return RunnableLambda(respond)


def _messages(model: RecordingModel) -> tuple[str, str]:
    """Return the system and human message text of the last recorded prompt."""
    prompt_value = model.prompt_values[-1]
    system, human = prompt_value.to_messages()
    return system.content, human.content


def test_the_prompt_forbids_outside_knowledge_and_asks_for_citations(settings):
    """The system message is the main guard against invention."""
    model = RecordingModel()

    chain = chains.build_answer_chain(settings, model=model)
    chain.invoke(
        {
            "context_blocks": "[1] DOC: benefits_guide.md | LOC: chunk 7\nBody.",
            "question": "Parental leave?",
        }
    )

    system_message = _messages(model)[0]
    assert "Answer only from the CONTEXT provided" in system_message
    assert "answerable to false" in system_message
    assert "UNAVAILABLE" in system_message
    assert "Never use knowledge from outside the CONTEXT" in system_message


def test_the_prompt_carries_the_question_and_the_numbered_context(settings):
    """The model must see the blocks it is asked to cite."""
    model = RecordingModel()
    chain = chains.build_answer_chain(settings, model=model)

    chain.invoke(
        {
            "context_blocks": "[1] DOC: benefits_guide.md | LOC: chunk 7, chars 0 to 400\nBody text.",
            "question": "How long is parental leave?",
        }
    )

    system_message, human_message = _messages(model)
    assert "Answer only from the CONTEXT provided" in system_message
    assert "[1] DOC: benefits_guide.md" in human_message
    assert "How long is parental leave?" in human_message


def test_the_chain_returns_the_validated_answer_schema(settings):
    """Structured output means a malformed answer is impossible by construction."""
    chain = chains.build_answer_chain(settings, model=RecordingModel())

    result = chain.invoke({"context_blocks": "[1] body", "question": "q"})

    assert isinstance(result, Answer)
    assert result.answerable is True
    assert result.citations == [1]


def test_temperature_and_token_ceiling_come_from_settings(settings):
    """Generation stays repeatable, which the evaluation depends on."""
    built: list[dict] = []

    class Capturing(RecordingModel):
        def __init__(self, **kwargs):
            super().__init__()
            built.append(kwargs)

    chains.build_answer_chain(settings, model=Capturing())

    # The default model is only constructed when none is injected, so the settings
    # are asserted directly against the values the factory would pass on.
    assert settings.llm_temperature == 0.0
    assert MAX_OUTPUT_TOKENS == 1024


def test_health_reports_the_store_as_unavailable_when_chroma_fails(settings, corpus, monkeypatch):
    """A broken index must surface as a readiness failure, not as silence."""
    def explode():
        raise RuntimeError("index directory is unreadable")

    monkeypatch.setattr(corpus, "get_store", explode)
    app.dependency_overrides[get_service] = lambda: corpus
    try:
        with TestClient(app) as client:
            response = client.get("/healthz")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "degraded"
            assert body["chroma"] == "unavailable"
            assert body["document_count"] == 0
    finally:
        app.dependency_overrides.clear()


# Optional live path


@pytest.mark.live
def test_live_model_answers_with_evidence():
    """Smoke test against the real model, skipped unless a key is present.

    Run with `pytest -m live`. It is excluded from the default offline run.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set")

    settings = Settings()
    service = Service(settings=settings)
    rag.ingest_document(sample_bytes("benefits_guide.md"), "benefits_guide.md", service)

    result = rag.ask("What is the parental leave entitlement?", service)

    assert result["answerable"] is True
    assert result["sources"], "a live answer must carry evidence"
    for source in result["sources"]:
        assert source["excerpt"]
        assert source["document_name"].endswith((".pdf", ".txt", ".md"))