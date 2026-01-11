import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/Final_TMDB_movie_dataset_3.csv"
TFIDF_MATRIX_PATH = "models/tfidf_matrix.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

def load_data(path=DATA_PATH):
    data = pd.read_csv(path, low_memory=False)
    return data

def build_combined_features(data):
    TEXT_COLS = ['overview', 'genres', 'keywords']
    data[TEXT_COLS] = data[TEXT_COLS].fillna('')
    return (
        data['overview'] + " " +
        data['genres'] + " " +
        data['keywords']
    )

def vectorize_text(text_series):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=30000,   # VERY IMPORTANT
        ngram_range=(1, 1)
    )
    tfidf_matrix = vectorizer.fit_transform(text_series)
    return tfidf_matrix, vectorizer


def get_movie_index(data, title):
    title = title.lower()
    # matches = data[data['title'] == title]

    # 1️⃣ exact match
    exact = data[data['title'] == title]
    if not exact.empty:
        return exact.index[0]

    # 2️⃣ contains match (most common fix)
    contains = data[data['title'].str.contains(title, regex=False, na=False)]
    if not contains.empty:
        # pick most popular version
        return contains.sort_values(
            by=['popularity', 'vote_count'],
            ascending=False
        ).index[0]

    # if matches.empty:
    #     raise ValueError("Movie not found in dataset")

    # return matches.index[0]

    # 3️⃣ not found
    return None


def recommend_movies(title, data, tfidf_matrix, top_n=10):
    idx = get_movie_index(data, title)

    if idx is None:
        return f"❌ Movie '{title}' not found in dataset."

    cosine_sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # get top 50 first (not just 10)
    top_indices = cosine_sim.argsort()[::-1][1:101]

    candidates = data.iloc[top_indices].copy()

    # # quality filter
    # candidates = candidates[
    #     (candidates['vote_count'] >= 100) &
    #     (candidates['vote_average'] >= 6.0)
    # ]

    # # final ranking
    # candidates = candidates.sort_values(
    #     by=['vote_average', 'popularity'],
    #     ascending=False
    # )

    # this is used to not return empty list and recommend atleast something.
    # 1️⃣ strict filter
    strict = candidates[
        (candidates['vote_count'] >= 100) &
        (candidates['vote_average'] >= 6.0)
        ]

    if not strict.empty:
        return (
            strict
            .sort_values(by=['vote_average', 'popularity'], ascending=False)
            [['title', 'release_year', 'vote_average', 'popularity']]
            .head(top_n)
        )

    # 2️⃣ relaxed filter (fallback)
    relaxed = candidates[
        candidates['vote_count'] >= 20
        ]

    if not relaxed.empty:
        return (
            relaxed
            .sort_values(by=['popularity'], ascending=False)
            [['title', 'release_year', 'vote_average', 'popularity']]
            .head(top_n)
        )

    # 3️⃣ absolute fallback (no filtering)
    return (
        candidates
        .sort_values(by=['popularity'], ascending=False)
        [['title', 'release_year', 'vote_average', 'popularity']]
        .head(top_n)
    )
    return candidates[['title', 'release_year', 'vote_average', 'popularity']].head(top_n)

def search_suggestions(query, data, top_n=10):
    query = query .lower().strip()

    if len(query) < 2:
        return []

    matches = data[data['title'].str.contains(query, regex=False, na=False)]

    if matches.empty:
        return []

    suggestion = (matches.sort_values(by=['popularity', 'vote_count'], ascending=False).head(top_n)['title'].tolist())

    return suggestion


def main():
    print("Loading data...")
    data = load_data()

    print("Building combined features...")
    combined_features = build_combined_features(data)

    print("Vectorizing text...")
    tfidf_matrix, vectorizer = vectorize_text(combined_features)

    print("Ready to recommend!")

    result = [
        (recommend_movies(title="inception", data=data, tfidf_matrix=tfidf_matrix)),
        (recommend_movies(title="avengers", data=data, tfidf_matrix=tfidf_matrix)),
        (recommend_movies(title="spiderman", data=data, tfidf_matrix=tfidf_matrix)),
        (recommend_movies("empire strikes back", data, tfidf_matrix))
    ]

    for i, res in enumerate(result, start=1):
        print(f"\n=== Result {i} ===")
        print(res)
    # result = recommend_movies(title="inception",data=data,tfidf_matrix=tfidf_matrix)
    # print(result)

    print(type(result))

    result1 = (search_suggestions("spid", data))
    result2 = (search_suggestions("harry", data))
    result3 = (search_suggestions("lord", data))

    for i, movie in enumerate(result1, start=1):
        print(f"{i}. {movie.title()}")
    for i, movie in enumerate(result2, start=1):
        print(f"{i}. {movie.title()}")
    for i, movie in enumerate(result3, start=1):
        print(f"{i}. {movie.title()}")

if __name__ == "__main__":
    main()


