from pathlib import Path
import urllib.request
import zipfile


DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "ml-100k"
ZIP_PATH = DATA_DIR / "ml-100k.zip"

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_data():
    if (DATASET_DIR / "u.data").exists() and (DATASET_DIR / "u.item").exists():
        print("MovieLens 100K dataset already exists.")
        return

    DATA_DIR.mkdir(exist_ok=True)

    print("Downloading MovieLens 100K...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    ZIP_PATH.unlink()

    print("MovieLens 100K dataset ready.")


if __name__ == "__main__":
    download_data()
    