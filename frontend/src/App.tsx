import { useState } from "react";
import "./App.css";

type Recommendation = {
  movie_id: number;
  title: string;
  predicted_rating: number;
  rating_count: number;
};

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [userId, setUserId] = useState("1");
  const [count, setCount] = useState("10");
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function getRecommendations() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_URL}/recommendations/${userId}?n=${count}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong");
      }

      setRecommendations(data.recommendations);
    } catch (error) {
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError("Something went wrong");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Movie Recommender</h1>

      <p className="subtitle">
        Hybrid movie recommendations using collaborative and content-based
        filtering.
      </p>

      <div className="search">
        <label htmlFor="userId">MovieLens User ID</label>

        <input
          id="userId"
          type="number"
          value={userId}
          onChange={(event) => setUserId(event.target.value)}
          min="1"
        />

        <label htmlFor="count">Number of movies</label>

        <input
          id="count"
          type="number"
          value={count}
          onChange={(event) => setCount(event.target.value)}
          min="1"
          max="50"
        />

        <button onClick={getRecommendations} disabled={loading}>
          {loading ? "Finding Movies..." : "Get Recommendations"}
        </button>
      </div>

      <div className="info-box">
        <strong>About this recommender</strong>

        <p>
          <strong>MovieLens:</strong> A research dataset containing movie
          ratings from real users. This project uses MovieLens 100K, containing
          100,000 ratings from 943 users across 1,682 movies.
        </p>

        <p>
          <strong>Collaborative filtering:</strong> Recommends movies using
          people with similar rating patterns. If users with tastes similar to
          yours liked a movie, the model is more likely to recommend it to you.
        </p>

        <p>
          <strong>Content-based filtering:</strong> Recommends movies based on
          characteristics of movies you have liked before. In this project,
          those characteristics are genres such as Action, Comedy, Drama, and
          Sci-Fi.
        </p>

        <p>
          <strong>Hybrid model:</strong> Combines collaborative and
          content-based filtering to produce the final predicted rating.
        </p>

        <p>
          <strong>Predicted rating:</strong> The rating the model estimates the
          selected user would give the movie, from 1 to 5.
        </p>

        <p>
          <strong>Training ratings:</strong> The number of ratings for that
          specific movie available in the model's training data.
        </p>
      </div>

      {error && <p className="error">{error}</p>}

      {recommendations.length > 0 && (
        <section>
          <h2>Recommended Movies</h2>

          <div className="recommendations">
            {recommendations.map((movie, index) => (
              <div className="movie-card" key={movie.movie_id}>
                <h3>
                  {index + 1}. {movie.title}
                </h3>

                <p>
                  Predicted rating: {movie.predicted_rating.toFixed(2)} / 5
                </p>

                <p>Training ratings: {movie.rating_count}</p>
              </div>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

export default App;