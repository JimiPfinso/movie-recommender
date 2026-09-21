import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

columns = ["user_id", "movie_id", "rating", "timestamp"]

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=columns
)

train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

user_means = train.groupby("user_id")["rating"].mean()

print(user_means.head())

global_mean = train["rating"].mean()

test = test.copy()

test["prediction"] = test["user_id"].map(user_means)
test["prediction"] = test["prediction"].fillna(global_mean)

movie_means = train.groupby("movie_id")["rating"].mean()

test["movie_prediction"] = test["movie_id"].map(movie_means)
test["movie_prediction"] = test["movie_prediction"].fillna(global_mean)

movie_rmse = root_mean_squared_error(
    test["rating"],
    test["movie_prediction"]
)

print("Movie mean RMSE:", movie_rmse)

print(movie_means.head())

rmse = root_mean_squared_error(
    test["rating"],
    test["prediction"]
)

print("Global mean RMSE: 1.1239")
print("User mean RMSE:", rmse)