from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Movie Recommender API is running"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_recommendations():
    response = client.get("/recommendations/1?n=5")

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 5

    for movie in data["recommendations"]:
        assert "movie_id" in movie
        assert "title" in movie
        assert "predicted_rating" in movie
        assert "rating_count" in movie


def test_invalid_user_id():
    response = client.get("/recommendations/0")

    assert response.status_code == 400
    assert response.json()["detail"] == "user_id must be positive"


def test_user_not_found():
    response = client.get("/recommendations/99999")

    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"


def test_invalid_recommendation_count():
    response = client.get("/recommendations/1?n=0")

    assert response.status_code == 400
    assert response.json()["detail"] == "n must be between 1 and 50"


def test_too_many_recommendations():
    response = client.get("/recommendations/1?n=51")

    assert response.status_code == 400
    assert response.json()["detail"] == "n must be between 1 and 50"