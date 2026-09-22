import pandas as pd

from collaborative import predict_rating
from content_based import predict_content_rating, movies
from collaborative import validation, train, test
from sklearn.metrics import root_mean_squared_error

validation = validation.copy()

print("Generating collaborative predictions...")

validation["collaborative_prediction"] = validation.apply(
    lambda row: predict_rating(
        user_id=row["user_id"],
        movie_id=row["movie_id"],
        k=300
    ),
    axis=1
)

print("Generating content-based predictions...")

validation["content_prediction"] = validation.apply(
    lambda row: predict_content_rating(
        user_id=row["user_id"],
        movie_id=row["movie_id"]
    ),
    axis=1
)

alpha_values = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]

best_alpha = None
best_rmse = float("inf")

for alpha in alpha_values:

    validation["hybrid_prediction"] = (
        alpha * validation["collaborative_prediction"]
        + (1 - alpha) * validation["content_prediction"]
    )

    rmse = root_mean_squared_error(
        validation["rating"],
        validation["hybrid_prediction"]
    )

    print(
        f"alpha={alpha:.1f}: "
        f"Hybrid validation RMSE={rmse:.4f}"
    )

    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha}")
print(f"Best hybrid validation RMSE: {best_rmse:.4f}")

print("\nEvaluating final hybrid model on test set...")

test = test.copy()

test["collaborative_prediction"] = test.apply(
    lambda row: predict_rating(
        user_id=row["user_id"],
        movie_id=row["movie_id"],
        k=300
    ),
    axis=1
)

test["content_prediction"] = test.apply(
    lambda row: predict_content_rating(
        user_id=row["user_id"],
        movie_id=row["movie_id"]
    ),
    axis=1
)

test["hybrid_prediction"] = (
    best_alpha * test["collaborative_prediction"]
    + (1 - best_alpha) * test["content_prediction"]
)

test_rmse = root_mean_squared_error(
    test["rating"],
    test["hybrid_prediction"]
)

print(f"Final hybrid test RMSE: {test_rmse:.4f}")

def recommend_movies(user_id, n=10, k=300, alpha=0.9):
    user_ratings = train[
        train["user_id"] == user_id
    ]

    rated_movie_ids = set(
        user_ratings["movie_id"]
    )

    unseen_movies = movies[
        ~movies["movie_id"].isin(rated_movie_ids)
    ]

    recommendations = []

    for _, movie in unseen_movies.iterrows():
        movie_id = movie["movie_id"]

        collaborative_score = predict_rating(
            user_id=user_id,
            movie_id=movie_id,
            k=k
        )

        content_score = predict_content_rating(
            user_id=user_id,
            movie_id=movie_id
        )

        hybrid_score = (
            alpha * collaborative_score
            + (1 - alpha) * content_score
        )

        recommendations.append({
            "movie_id": movie_id,
            "title": movie["title"],
            "predicted_rating": hybrid_score
        })

    recommendations = pd.DataFrame(recommendations)

    recommendations = recommendations.sort_values(
        "predicted_rating",
        ascending=False
    )

    return recommendations.head(n)

print("\nTop 10 recommendations for User 1:")

top_recommendations = recommend_movies(
    user_id=1,
    n=10
)

print(top_recommendations)