import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

genre_columns = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]

movie_columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url"
] + genre_columns

movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    names=movie_columns,
    encoding="latin-1"
)

print(movies.head())
print("\nShape:", movies.shape)

print(
    "\nToy Story genres:"
)

toy_story = movies[
    movies["title"].str.startswith("Toy Story")
]

print(
    toy_story[
        ["title"] + genre_columns
    ]
)

genre_matrix = movies[genre_columns]

movie_similarity = cosine_similarity(genre_matrix)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=movies["movie_id"],
    columns=movies["movie_id"]
)

print("\nMovie similarity matrix shape:")
print(movie_similarity_df.shape)

print("\nSimilarity between first five movies:")
print(movie_similarity_df.iloc[:5, :5])

def get_similar_movies(movie_id, n=10):
    similarities = (
        movie_similarity_df[movie_id]
        .drop(movie_id)
        .sort_values(ascending=False)
        .head(n)
    )

    recommendations = movies[
        movies["movie_id"].isin(similarities.index)
    ][["movie_id", "title"]].copy()

    recommendations["similarity"] = recommendations["movie_id"].map(
        similarities
    )

    recommendations = recommendations.sort_values(
        "similarity",
        ascending=False
    )

    return recommendations


print("\nMovies most similar to Toy Story:")

print(
    get_similar_movies(
        movie_id=1,
        n=10
    )
)

rating_columns = [
    "user_id",
    "movie_id",
    "rating",
    "timestamp"
]

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=rating_columns
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

print("\nTraining ratings:", len(train))
print("Validation ratings:", len(validation))
print("Test ratings:", len(test))

user_id = 1

user_ratings = train[
    train["user_id"] == user_id
]

print("\nUser 1 ratings:")
print(user_ratings.head())

print("\nNumber of movies rated by User 1:")
print(len(user_ratings))

user_movies = user_ratings.merge(
    movies,
    on="movie_id"
)

print("\nUser 1 ratings with genres:")

print(
    user_movies[
        ["title", "rating"] + genre_columns
    ].head()
)

user_mean = user_movies["rating"].mean()

user_movies["rating_deviation"] = (
    user_movies["rating"] - user_mean
)

print("\nUser 1 mean rating:", user_mean)

print(
    user_movies[
        ["title", "rating", "rating_deviation"]
    ].head()
)

genre_preferences = (
    user_movies[genre_columns]
    .multiply(
        user_movies["rating_deviation"],
        axis=0
    )
    .sum()
)

print("\nUser 1 genre preferences:")
print(
    genre_preferences.sort_values(
        ascending=False
    )
)

user_profile = genre_preferences.values.reshape(1, -1)

def build_user_profile(user_id):
    user_ratings = train[
        train["user_id"] == user_id
    ]

    if user_ratings.empty:
        return None, None

    user_movies = user_ratings.merge(
        movies,
        on="movie_id"
    )

    user_mean = user_movies["rating"].mean()

    user_movies["rating_deviation"] = (
        user_movies["rating"] - user_mean
    )

    genre_preferences = (
        user_movies[genre_columns]
        .multiply(
            user_movies["rating_deviation"],
            axis=0
        )
        .sum()
    )

    return genre_preferences, user_mean

def predict_content_rating(user_id, movie_id):
    global_mean = train["rating"].mean()

    genre_preferences, user_mean = build_user_profile(user_id)

    # User wasn't present in training data
    if genre_preferences is None:
        return global_mean

    movie_row = movies[
        movies["movie_id"] == movie_id
    ]

    if movie_row.empty:
        return user_mean

    movie_genres = movie_row[genre_columns].values[0]

    preference_score = (
        genre_preferences.values * movie_genres
    ).sum()

    user_ratings = train[
        train["user_id"] == user_id
    ]

    user_movies = user_ratings.merge(
        movies,
        on="movie_id"
    )

    normalizer = user_movies[genre_columns].sum().sum()

    if normalizer == 0:
        return user_mean

    adjustment = preference_score / normalizer

    prediction = user_mean + adjustment

    return max(1, min(5, prediction))

content_scores = cosine_similarity(
    user_profile,
    genre_matrix
)[0]

movies_scored = movies[
    ["movie_id", "title"]
].copy()

movies_scored["content_score"] = content_scores

rated_movie_ids = set(user_ratings["movie_id"])

recommendations = movies_scored[
    ~movies_scored["movie_id"].isin(rated_movie_ids)
]

recommendations = recommendations.sort_values(
    "content_score",
    ascending=False
)

print("\nTop content-based recommendations for User 1:")
print(recommendations.head(10))

user_1_validation = validation[
    validation["user_id"] == 1
]

print("\nUser 1 validation ratings:")
print(user_1_validation.head())

if not user_1_validation.empty:
    example = user_1_validation.iloc[0]

    movie_id = example["movie_id"]
    actual_rating = example["rating"]

    predicted_rating = predict_content_rating(
        user_id=1,
        movie_id=movie_id
    )

    movie_title = movies.loc[
        movies["movie_id"] == movie_id,
        "title"
    ].iloc[0]

    print("\nExample content-based prediction:")
    print("Movie:", movie_title)
    print("Actual rating:", actual_rating)
    print("Predicted rating:", predicted_rating)

print("\nEvaluating content-based model...")

validation = validation.copy()

validation["content_prediction"] = validation.apply(
    lambda row: predict_content_rating(
        user_id=row["user_id"],
        movie_id=row["movie_id"]
    ),
    axis=1
)

content_rmse = root_mean_squared_error(
    validation["rating"],
    validation["content_prediction"]
)

print(
    f"Content-based validation RMSE: {content_rmse:.4f}"
)