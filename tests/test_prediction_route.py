from fastapi.testclient import TestClient

from src.main import app


def test_predict_route_returns_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41,
                "total_rooms": 880,
                "total_bedrooms": 129,
                "population": 322,
                "households": 126,
                "median_income": 8.3252,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0

def test_predict_route_rejects_invalid_input():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "longitude": "invalid",
                "latitude": 37.88,
                "housing_median_age": 41,
                "total_rooms": 880,
                "total_bedrooms": 129,
                "population": 322,
                "households": 126,
                "median_income": 8.3252,
            },
        )

    assert response.status_code == 422