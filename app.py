from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("movies.csv")

# Clean genres
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Index mapping
movie_indices = pd.Series(movies.index, index=movies["title"])

def get_best_match(movie_name):
    """Find closest matching movie title"""
    matches = movies[movies["title"].str.contains(movie_name, case=False, na=False)]
    if matches.empty:
        return None
    return matches.iloc[0]["title"]

def recommend_movies(movie_title, top_n=6):
    idx = movie_indices[movie_title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    movie_ids = [i[0] for i in scores[1:top_n+1]]
    return movies.iloc[movie_ids]

@app.route("/", methods=["GET", "POST"])
def index():
    search_movie = request.form.get("movie", "").strip()
    genre = request.form.get("genre", "")
    sort_by = request.form.get("sort", "")

    results = movies.copy()
    recommendations = pd.DataFrame()
    base_movie = None

    # SEARCH & RECOMMEND
    if search_movie:
        base_movie = get_best_match(search_movie)

        if base_movie:
            recommendations = recommend_movies(base_movie)
            results = movies[movies["title"].str.contains(search_movie, case=False)]
        else:
            results = movies[movies["title"].str.contains(search_movie, case=False)]

    # GENRE FILTER
    if genre:
        results = results[results["genres"].str.contains(genre, case=False)]

    # SORT
    if sort_by == "az":
        results = results.sort_values("title")
    elif sort_by == "za":
        results = results.sort_values("title", ascending=False)

    results = results.head(20)

    genres_list = sorted(set(" ".join(movies["genres"]).split()))

    return render_template(
        "index.html",
        movies=results,
        recommendations=recommendations,
        genres=genres_list,
        base_movie=base_movie
    )

if __name__ == "__main__":
    app.run(debug=True)
