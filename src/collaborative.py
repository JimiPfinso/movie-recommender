import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import root_mean_squared_error

columns = ["user_id", "movie_id", "rating", "timestamp"]

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=columns
)

train, temp = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

validation, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=42
)

user_movie_matrix = train.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
)

user_means = user_movie_matrix.mean(axis=1)

centered_matrix = user_movie_matrix.sub(
    user_means,
    axis=0
)

matrix_filled = centered_matrix.fillna(0)

user_similarity = cosine_similarity(matrix_filled)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)


def predict_rating(user_id, movie_id, k=10):
    global_mean = train["rating"].mean()

    if user_id not in user_movie_matrix.index:
        return global_mean

    if movie_id not in user_movie_matrix.columns:
        return user_means.loc[user_id]

    similar_users = (
        user_similarity_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(k)
    )

    neighbor_ratings = user_movie_matrix.loc[
        similar_users.index,
        movie_id
    ]

    valid_ratings = neighbor_ratings.dropna()

    if valid_ratings.empty:
        return user_means.loc[user_id]

    valid_similarities = similar_users.loc[
        valid_ratings.index
    ]

    if valid_similarities.abs().sum() == 0:
        return user_means.loc[user_id]

    neighbor_means = user_means.loc[valid_ratings.index]
    rating_deviations = valid_ratings - neighbor_means

    weighted_deviation = (
        (rating_deviations * valid_similarities).sum()
        / valid_similarities.abs().sum()
    )

    prediction = user_means.loc[user_id] + weighted_deviation

    prediction = max(1, min(5, prediction))

    return prediction


if __name__ == "__main__":
    print("Training ratings:", len(train))
    print("Validation ratings:", len(validation))
    print("Test ratings:", len(test))
    
    validation = validation.copy()

    k_values = [50, 75, 100, 150, 200, 300, 400, 500]

    best_k = None
    best_rmse = float("inf")

    for k in k_values:
        validation["prediction"] = validation.apply(
            lambda row: predict_rating(
                row["user_id"],
                row["movie_id"],
                k=k
            ),
            axis=1
        )

        rmse = root_mean_squared_error(
            validation["rating"],
            validation["prediction"]
        )

        print(f"k={k}: Validation RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k

    print(f"\nBest k: {best_k}")
    print(f"Best validation RMSE: {best_rmse:.4f}")

    test = test.copy()

    test["prediction"] = test.apply(
        lambda row: predict_rating(
            row["user_id"],
            row["movie_id"],
            k=best_k
        ),
        axis=1
    )

    test_rmse = root_mean_squared_error(
        test["rating"],
        test["prediction"]
    )

    print(f"\nFinal test RMSE with k={best_k}: {test_rmse:.4f}")