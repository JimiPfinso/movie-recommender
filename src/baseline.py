import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

columns = ["user_id", "movie_id", "rating", "timestamp"]

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=columns
)

print(ratings.head())

train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

print("Training ratings:", len(train))
print("Test ratings:", len(test))

global_mean = train["rating"].mean()

print("Average training rating:", global_mean)

test = test.copy()
test["prediction"] = global_mean

print(test[["user_id", "movie_id", "rating", "prediction"]].head())

rmse = root_mean_squared_error(
    test["rating"],
    test["prediction"]
)

print("Baseline RMSE:", rmse)