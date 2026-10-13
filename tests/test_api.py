from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz_returns_200():
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["gemini"] in {"configured", "missing"}
    assert body["chroma"] == "ok"
    assert isinstance(body["document_count"], int)
