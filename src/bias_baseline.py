import pandas as pd
from sklearn.model_selection import train_test_split
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
global_mean = train["rating"].mean()

print("Global mean:", global_mean)

user_means = train.groupby("user_id")["rating"].mean()

user_biases = user_means - global_mean

print("\nUser biases:")
print(user_biases.head())

movie_means = train.groupby("movie_id")["rating"].mean()

movie_biases = movie_means - global_mean

print("\nMovie biases:")
print(movie_biases.head())

test = test.copy()

test["user_bias"] = test["user_id"].map(user_biases)
test["movie_bias"] = test["movie_id"].map(movie_biases)

test["user_bias"] = test["user_bias"].fillna(0)
test["movie_bias"] = test["movie_bias"].fillna(0)

test["prediction"] = (
    global_mean
    + test["user_bias"]
    + test["movie_bias"]
)

print("\nCombined predictions:")
print(
    test[
        ["user_id", "movie_id", "rating",
         "user_bias", "movie_bias", "prediction"]
    ].head()
)

rmse = root_mean_squared_error(
    test["rating"],
    test["prediction"]
)

print("\nCombined bias RMSE:", rmse)