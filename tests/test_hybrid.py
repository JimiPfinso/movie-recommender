from src.hybrid import (
    predict_hybrid_rating,
    recommend_movies
)

from src.collaborative import train


def test_hybrid_prediction_range():
    prediction = predict_hybrid_rating(
        user_id=1,
        movie_id=318
    )

    assert 1 <= prediction <= 5


def test_recommendation_count():
    recommendations = recommend_movies(
        user_id=1,
        n=5
    )

    assert len(recommendations) == 5


def test_recommendations_are_sorted():
    recommendations = recommend_movies(
        user_id=1,
        n=10
    )

    ratings = recommendations[
        "predicted_rating"
    ].tolist()

    assert ratings == sorted(
        ratings,
        reverse=True
    )


def test_recommendations_exclude_rated_movies():
    recommendations = recommend_movies(
        user_id=1,
        n=10
    )

    rated_movies = set(
        train[
            train["user_id"] == 1
        ]["movie_id"]
    )

    recommended_movies = set(
        recommendations["movie_id"]
    )

    assert rated_movies.isdisjoint(
        recommended_movies
    )


def test_recommendations_meet_rating_threshold():
    recommendations = recommend_movies(
        user_id=1,
        n=10
    )

    assert (
        recommendations["rating_count"] >= 20
    ).all()