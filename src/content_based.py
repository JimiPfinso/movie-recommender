import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

user_id = 1

user_ratings = ratings[
    ratings["user_id"] == user_id
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