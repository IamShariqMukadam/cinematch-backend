import pandas as pd

df = pd.read_csv("data/Movie_Recommendation_dataset_cleaned.csv",  keep_default_na=False)

print(df[['title','overview','genres','keywords','runtime']].isna().sum())
print()


