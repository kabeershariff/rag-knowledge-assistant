"""API behavior: status codes, response schemas and the endpoint matrix."""

import pytest

from tests.conftest import OFF_CORPUS_QUESTIONS, SAMPLE_FILENAMES, sample_bytes

VALID_TEXT = (
    "## Aurora Labs note\n\n"
    "The internal knowledge base records the expense approval chain for the "
    "company. Any single expense above 500 dollars needs written manager "
    "approval before reimbursement is issued, and a finance director approval "
    "is required above 2000 dollars.\n\n"
    "## Second heading\n\n"
    "Paid time off accrues monthly at 10 hours per month for full time staff, "
    "and unused hours carry over up to 120 hours into the next calendar year.\n"
)


def upload(client, name: str, content: bytes):
    return client.post(
        "/api/documents",
        files={"file": (name, content, "application/octet-stream")},
    )


# Ingest


def test_ingest_returns_201_with_the_document_record(client):
    response = upload(client, "expenses.txt", VALID_TEXT.encode("utf-8"))

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == "expenses"
    assert body["filename"] == "expenses.txt"
    assert body["media_type"] == "text/plain"
    assert body["chunk_count"] >= 1


@pytest.mark.parametrize(
    "filename,expected_media_type",
    [
        ("expenses.txt", "text/plain"),
        ("benefits.md", "text/markdown"),
    ],
)
def test_text_and_markdown_are_accepted(client, filename, expected_media_type):
    response = upload(client, filename, VALID_TEXT.encode("utf-8"))

    assert response.status_code == 201
    assert response.json()["media_type"] == expected_media_type


def test_pdf_is_accepted(client):
    response = upload(client, "handbook_copy.pdf", sample_bytes("employee_handbook.pdf"))

    assert response.status_code == 201
    assert response.json()["media_type"] == "application/pdf"
    assert response.json()["chunk_count"] >= 12


def test_extension_matching_ignores_case(client):
    response = upload(client, "Expenses.TXT", VALID_TEXT.encode("utf-8"))

    assert response.status_code == 201
    assert response.json()["document_id"] == "expenses"


def test_reingesting_the_same_name_replaces_the_previous_version(client):
    first = upload(client, "expenses.txt", VALID_TEXT.encode("utf-8"))
    shorter = b"## Aurora Labs note\n\nOnly one short paragraph about expenses and receipts.\n"
    second = upload(client, "expenses.txt", shorter)

    assert second.status_code == 201
    assert second.json()["document_id"] == first.json()["document_id"]
    assert second.json()["chunk_count"] <= first.json()["chunk_count"]

    listed = client.get("/api/documents").json()["documents"]
    expenses = [item for item in listed if item["document_id"] == "expenses"]
    assert len(expenses) == 1, "re-upload must not create a second document"


# List


def test_list_returns_200_with_every_ingested_document(corpus_client):
    response = corpus_client.get("/api/documents")

    assert response.status_code == 200
    body = response.json()
    names = {item["filename"] for item in body["documents"]}
    assert names == set(SAMPLE_FILENAMES)
    for item in body["documents"]:
        assert set(item) == {"document_id", "filename", "chunk_count"}
        assert item["chunk_count"] >= 12


def test_list_is_empty_for_a_fresh_store(client):
    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": []}


# Delete


def test_delete_returns_204_and_removes_the_document(corpus_client):
    response = corpus_client.delete("/api/documents/benefits_guide")

    assert response.status_code == 204
    assert response.content == b""

    remaining = {item["document_id"] for item in corpus_client.get("/api/documents").json()["documents"]}
    assert "benefits_guide" not in remaining
    assert "employee_handbook" in remaining


def test_delete_of_an_unknown_id_returns_404(corpus_client):
    response = corpus_client.delete("/api/documents/no_such_document")

    assert response.status_code == 404
    assert "detail" in response.json()


def test_delete_is_not_repeatable_after_the_document_is_gone(corpus_client):
    assert corpus_client.delete("/api/documents/benefits_guide").status_code == 204

    assert corpus_client.delete("/api/documents/benefits_guide").status_code == 404


def test_a_deleted_document_no_longer_answers_questions(corpus_client):
    corpus_client.delete("/api/documents/benefits_guide")

    response = corpus_client.post(
        "/api/ask", json={"question": " ".join(sample_bytes("benefits_guide.md").decode("utf-8").split()[:45])}
    )

    assert response.status_code == 200
    for source in response.json()["sources"]:
        assert source["document_id"] != "benefits_guide"


# Ask


def test_ask_returns_200_with_the_full_response_schema(corpus_client, verbatim_query):
    question = verbatim_query("benefits_guide", 7)

    response = corpus_client.post("/api/ask", json={"question": question})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "answer",
        "answerable",
        "sources",
        "retrieved_count",
        "evidence_count",
        "distinct_document_count",
        "model",
        "latency_ms",
    }
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["answerable"], bool)
    assert isinstance(body["sources"], list)
    assert isinstance(body["model"], str) and body["model"]
    assert isinstance(body["latency_ms"], (int, float))
    assert set(body["sources"][0]) == {
        "document_name",
        "document_id",
        "excerpt",
        "location",
        "score",
    }
    assert set(body["sources"][0]["location"]) >= {
        "chunk_index",
        "char_start",
        "char_end",
    }


def test_ask_reports_distinct_document_count(corpus_client, verbatim_query):
    question = verbatim_query("remote_work_policy", 2)

    body = corpus_client.post("/api/ask", json={"question": question}).json()

    assert body["distinct_document_count"] == len(
        {source["document_id"] for source in body["sources"]}
    )


@pytest.mark.parametrize("question", OFF_CORPUS_QUESTIONS)
def test_ask_reports_unavailable_for_an_off_corpus_question(corpus_client, question):
    response = corpus_client.post("/api/ask", json={"question": question})

    assert response.status_code == 200
    body = response.json()
    assert body["answerable"] is False
    assert body["answer"] == (
        "The requested information is unavailable in the provided documents."
    )
    assert body["sources"] == []


def test_ask_accepts_a_top_k_parameter(client):
    upload(client, "expenses.txt", VALID_TEXT.encode("utf-8"))
    records = client.get("/api/documents").json()["documents"]

    assert records, "the fixture upload should be indexed"

    response = client.post("/api/ask", json={"question": VALID_TEXT[:120], "top_k": 1})

    assert response.status_code == 200
    assert response.json()["retrieved_count"] <= 1


# Ask validation


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * 2001},
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "top_k": 21},
        {"question": 12345},
    ],
)
def test_ask_validation_errors_return_422(client, payload):
    response = client.post("/api/ask", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_ask_without_a_body_returns_422(client):
    assert client.post("/api/ask").status_code == 422


# Health and documentation


def test_healthz_reports_ok_when_a_key_is_configured(corpus_client, monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: _settings_with_key())

    response = corpus_client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["gemini"] == "configured"
    assert body["chroma"] == "ok"
    assert body["document_count"] == 4


def test_healthz_reports_degraded_without_a_key(corpus_client, monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: _settings_without_key())

    response = corpus_client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["gemini"] == "missing"


def test_healthz_counts_the_ingested_documents(corpus_client, monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: _settings_with_key())

    body = corpus_client.get("/healthz").json()

    assert body["document_count"] == 4


def test_docs_render(corpus_client):
    response = corpus_client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_schema_is_served(corpus_client):
    response = corpus_client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/api/ask" in schema["paths"]
    assert "/api/documents" in schema["paths"]
    assert "/healthz" in schema["paths"]


# Helpers


def _settings_with_key():
    from app.config import Settings

    return Settings(gemini_api_key="offline-test-key", chroma_persist_dir="./data/chroma")


def _settings_without_key():
    from app.config import Settings

    return Settings(gemini_api_key=None, chroma_persist_dir="./data/chroma")


def test_registry_normalizes_a_filename_into_a_stable_slug():
    from app.documents import DocumentRegistry

    assert DocumentRegistry.normalize("Employee Handbook.pdf") == "employee_handbook"
    assert DocumentRegistry.normalize("benefits_guide.md") == "benefits_guide"
    assert DocumentRegistry.normalize("weird!!name__.txt") == "weird_name"
    assert DocumentRegistry.normalize("") == "document"