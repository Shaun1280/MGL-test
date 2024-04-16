import pandas as pd
import numpy as np
import os


path = os.path.join(os.path.dirname(__file__), "ratings.dat")

# don't include the first line as header
df = pd.read_csv(path, header=None, sep='::', encoding="latin-1", engine="python")
df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

print(len(df))
print("Distinct user_id count: ", df['user_id'].nunique())
print("Distinct movie_id count: ", df['movie_id'].nunique())