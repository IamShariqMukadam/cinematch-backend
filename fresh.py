import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/Movie_Recommendation_dataset_cleaned.csv",
    keep_default_na=True
)

TEXT_COLS = ['title', 'overview', 'genres', 'keywords']

# convert fake NaNs → real NaNs
df[TEXT_COLS] = df[TEXT_COLS].replace(
    ['nan', 'NaN', 'None', 'null', 'NULL', ' '],
    np.nan
)

# now fill them
df[TEXT_COLS] = df[TEXT_COLS].fillna('')

df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
df['runtime'] = df['runtime'].fillna(df['runtime'].median())

print(df[TEXT_COLS + ['runtime']].isna().sum())

df.to_csv(
    "data/Movie_Recommendation_dataset_cleaned_FINAL.csv",
    index=False
)
