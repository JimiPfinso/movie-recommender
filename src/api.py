from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.hybrid import recommend_movies
from src.collaborative import train


app = FastAPI(
    title="Movie Recommender API",
    description="Hybrid movie recommendation system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Movie Recommender API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, n: int = 10):
    if user_id < 1:
        raise HTTPException(
            status_code=400,
            detail="user_id must be positive"
        )

    if user_id not in train["user_id"].values:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    if n < 1 or n > 50:
        raise HTTPException(
            status_code=400,
            detail="n must be between 1 and 50"
        )

    recommendations = recommend_movies(
        user_id=user_id,
        n=n
    )

    return {
        "user_id": user_id,
        "recommendations": recommendations.to_dict(
            orient="records"
        )
    }