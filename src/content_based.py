import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

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

genre_matrix = movies[genre_columns]

movie_similarity = cosine_similarity(genre_matrix)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=movies["movie_id"],
    columns=movies["movie_id"]
)

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
