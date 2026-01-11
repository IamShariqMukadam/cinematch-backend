import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# paths
DATA_PATH = "data/Movie_Recommendation_dataset_cleaned.csv"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
MATRIX_PATH = "models/tfidf_matrix.joblib"

# load data
data1 = pd.read_csv("data/Movie_Recommendation_dataset_cleaned.csv")
print(data1.shape)
print(data1.columns)
print(data1[['release_date', 'release_year']].head(10))
print(data1[data1['release_year'] == 2025][['title', 'release_date', 'popularity']].head(20))
print(data1.head(10).to_string())
print(data1.isna().sum())
# fill text columns used for ML
data1['title'] = data1['title'].fillna('')
data1['overview'] = data1['overview'].fillna('')
data1['genres'] = data1['genres'].fillna('')
data1['keywords'] = data1['keywords'].fillna('')
data1['runtime'] = data1['runtime'].fillna(data1['runtime'].median())
print(data1.isna().sum())

# combine text fields
data1['combined_text'] = (
    data1['title'] + " " +
    data1['overview'] + " " +
    data1['genres'] + " " +
    data1['keywords']
)

# build vectorizer
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

tfidf_matrix = tfidf_vectorizer.fit_transform(data1['combined_text'])

# save models
joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
joblib.dump(tfidf_matrix, MATRIX_PATH)

print("✅ TF-IDF models saved successfully")