from fastapi.testclient import TestClient

from src.main import app


def test_health_route_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "message" in data