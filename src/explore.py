import pandas as pd

columns = ["user_id", "movie_id", "rating", "timestamp"]

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=columns
)

print(ratings.head())
print("\nNumber of ratings:", len(ratings))
print("Number of users:", ratings["user_id"].nunique())
print("Number of movies:", ratings["movie_id"].nunique())

print("\nRating counts:")
print(ratings["rating"].value_counts().sort_index())

print("\nAverage rating:", ratings["rating"].mean())

print("\nMissing values:")
print(ratings.isnull().sum())

movie_columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url"
]

movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    usecols=range(5),
    names=movie_columns
)

print("\nMovies:")
print(movies.head())

ratings_with_titles = ratings.merge(
    movies[["movie_id", "title"]],
    on="movie_id"
)

print("\nRatings with movie titles:")
print(ratings_with_titles.head())

total_possible_ratings = (
    ratings["user_id"].nunique()
    * ratings["movie_id"].nunique()
)

sparsity = 1 - (len(ratings) / total_possible_ratings)

print("\nTotal possible ratings:", total_possible_ratings)
print("Actual ratings:", len(ratings))
print("Sparsity:", round(sparsity * 100, 2), "%")