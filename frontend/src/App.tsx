import { useState } from "react";
import "./App.css";

type Recommendation = {
  movie_id: number;
  title: string;
  predicted_rating: number;
  rating_count: number;
};

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
        `http://localhost:8000/recommendations/${userId}?n=${count}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong");
      }

      setRecommendations(data.recommendations);
    } catch (error) {
      if (error instanceof Error) {
        setError(error.message);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Movie Recommender</h1>

      <p className="subtitle">
        Hybrid recommendations using collaborative and content-based filtering.
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