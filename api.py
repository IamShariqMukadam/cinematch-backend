from datetime import datetime
from fastapi import FastAPI, Query
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
#to check if this works
# print("TMDB KEY LOADED:", bool(TMDB_API_KEY))


# --------------------
# Load resources ONCE
# --------------------
DATA_DIR = "data"
DATA_FILE = "Movie_Recommendation_dataset_cleaned.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
HF_DATASET_URL = os.getenv("HF_DATASET_URL")

VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
MATRIX_PATH = "models/tfidf_matrix.joblib"

download_dataset()
df = pd.read_csv(DATA_PATH, keep_default_na=False)

df["release_year"] = pd.to_numeric(
    df.get("release_year"), errors="coerce"
)

df["vote_count"] = pd.to_numeric(
    df.get("vote_count"), errors="coerce"
).fillna(0)

df["vote_average"] = pd.to_numeric(
    df.get("vote_average"), errors="coerce"
).fillna(0)

df["popularity"] = pd.to_numeric(
    df.get("popularity"), errors="coerce"
).fillna(0)

df["release_date"] = pd.to_datetime(
    df.get("release_date"), errors="coerce"
).fillna(0)

tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
tfidf_matrix = joblib.load(MATRIX_PATH)

app = FastAPI(
    title="CineMatch Recommendation API",
    description="Content-based movie recommendation engine using ML similarity",
    version="1.0.0"
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_dataset():
    if os.path.exists(DATA_PATH):
        print("✅ Dataset already present")
        return

    if not HF_DATASET_URL:
        raise RuntimeError("HF_DATASET_URL not set")

    os.makedirs(DATA_DIR, exist_ok=True)

    print("⬇️ Downloading dataset from Hugging Face...")
    with requests.get(HF_DATASET_URL, stream=True) as r:
        r.raise_for_status()
        with open(DATA_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("✅ Dataset download complete")


def get_movie_index(data, title):
    title = title.lower().strip()

    # 1️⃣ exact match
    exact = data[data['title'].str.lower() == title]
    if not exact.empty:
        return exact.index[0]

    # 2️⃣ contains match
    contains = data[data['title'].str.lower().str.contains(title, regex=False)]
    if not contains.empty:
        return (
            contains
            .sort_values(by=['popularity', 'vote_count'], ascending=False)
            .index[0]
        )

    return None

def recommend_movies_internal(title, top_n=12):
    idx = get_movie_index(df, title)

    if idx is None:
        return []

    cosine_sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_indices = cosine_sim.argsort()[::-1][1:101]
    candidates = df.iloc[top_indices].copy()

    # 1️⃣ strict quality filter
    strict = candidates[
        (candidates['vote_count'] >= 100) &
        (candidates['vote_average'] >= 7.0)
    ]

    if not strict.empty:
        final = strict.sort_values(
            by=['vote_average', 'popularity'],
            ascending=False
        )
    else:
        # 2️⃣ relaxed fallback
        relaxed = candidates[candidates['vote_count'] >= 20]

        if not relaxed.empty:
            final = relaxed.sort_values(by='popularity', ascending=False)
        else:
            # 3️⃣ absolute fallback
            final = candidates.sort_values(by='popularity', ascending=False)

    results = []
    for _, row in final.head(top_n).iterrows():
        poster = fetch_poster(row["title"], row["release_year"])

        results.append({
            "title": row["title"],
            "release_year": int(row["release_year"]) if not pd.isna(row["release_year"]) else None,
            "vote_average": float(row["vote_average"]),
            "popularity": float(row["popularity"]),
            "poster_path": fetch_poster(row["title"], row["release_year"])
        })


    return results


    # return (
    #     final[['title', 'release_year', 'vote_average', 'popularity']]
    #     .head(top_n)
    #     .to_dict(orient="records")
    # )

def search_suggestions(query, top_n=12):
    query = query.lower().strip()

    if len(query) < 2:
        return []


    matches = df[df['title'].str.lower().str.contains(query, regex=False)]

    if matches.empty:
        return []

    # ensure columns exist safely
    result_df = matches.sort_values(
        by=['popularity', 'vote_count'],
        ascending=False
    ).head(top_n)

    response = []

    for _, row in result_df.iterrows():
        response.append({
            "title": row["title"],
            "release_year": int(row["release_year"]) if not pd.isna(row["release_year"]) else None,
                # "poster_path": fetch_poster(
                #     row["title"],
                #     row["release_year"]
                # )
                "poster_path": fetch_poster(row["title"], row["release_year"])

        })

    return response

    # return (
    #     matches
    #     .sort_values(by=['popularity', 'vote_count'], ascending=False)
    #     .head(top_n)[['title', 'poster_path', 'release_year']]
    #     .to_dict(orient="records")
    # )

def fetch_poster(title: str, year=None):
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
        }

        # year may be float or NaN → clean safely
        if year and not pd.isna(year):
            params["year"] = int(float(year))

        res = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=3
        )
        data = res.json()

        if data.get("results"):
            return data["results"][0].get("poster_path")

        return None
    except Exception:
        return None



@app.get("/")
def home():
    return {"status": "CineMatch Recommendation API is running"}

@app.get("/search")
def search_movies(query: str = Query(..., min_length=1)):
    return search_suggestions(query)

@app.get("/recommend")
def recommend_movies(movie: str,top_n: int = 12):
    results = recommend_movies_internal(movie, top_n)

    if not results:
        return {"error": f"Movie {movie} not found."}

    return {
        "input_movie": movie,
        "recommendations": results
    }

@app.get("/genre")
def genre_movies(genre: str, top_n: int = 12):
    if genre.lower() == "trending":
        return []

    normalized = genre.lower().replace("-", " ")
    matches = df[df["genres"].str.lower().str.replace("-", " ").str.contains(normalized, regex=False)]

    top = (
        matches[matches["vote_count"] >= 100]
        .sort_values(by=["vote_average", "popularity"], ascending=False)
        .head(top_n)
    )

    results = []
    for _, row in top.iterrows():
        results.append({
            "title": row["title"],
            "release_year": int(row["release_year"]) if not pd.isna(row["release_year"]) else None,
            "vote_average": float(row["vote_average"]),
            "popularity": float(row["popularity"]),
            "poster_path": fetch_poster(row["title"], row["release_year"])
        })

    return results

@app.get("/top-rated")
def top_rated_movies(top_n: int = 12):
    movies = df[
        df["vote_count"] >= 500
    ].sort_values(
        by=["vote_average", "vote_count"],
        ascending=False
    ).head(top_n)

    results = []
    for _, row in movies.iterrows():
        results.append({
            "title": row["title"],
            "release_year": int(row["release_year"]) if not pd.isna(row["release_year"]) else None,
            "vote_average": float(row["vote_average"]),
            "popularity": float(row["popularity"]),
            "poster_path": fetch_poster(row["title"], row["release_year"])
        })

    return results



@app.get("/latest")
def latest_movies(top_n: int = 10):
    latest_df = df.copy()

    # 🔒 SAFE integer year
    latest_df["release_year"] = pd.to_numeric(
        latest_df["release_year"],
        errors="coerce"
    ).astype("Int64")

    movies = (
        latest_df[
            (latest_df["release_year"] == 2025) &
            (latest_df["vote_count"] >= 2)
        ]
        .sort_values(
            by=["popularity", "vote_average"],
            ascending=False
        )
        .head(top_n)
    )


    results = []
    for _, row in movies.iterrows():
        results.append({
            "title": row["title"],
            "release_year": int(row["release_year"]),
            "vote_average": float(row["vote_average"]),
            "popularity": float(row["popularity"]),
            "poster_path": fetch_poster(
                row["title"],
                row["release_year"]
            )
        })

    return results


#Steps to run the FASTApi:
#1.Change directory : cd "E:\Python Programming Files\ML Projects\Movie_Recommnedation_System"
#2.Run python -m uvicorn api:app --reload
#3.go on the site address http://127.0.0.1:8000
#4.test