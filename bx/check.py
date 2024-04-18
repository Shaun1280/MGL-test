import pandas as pd
import numpy as np
import os


path = os.path.join(os.path.dirname(__file__), "BX_Book_Ratings.csv")

df = pd.read_csv(path, sep=';', encoding="latin-1")
# df.columns = ['index', 'user_id', 'isbn', 'rating']
df.columns = ['user_id', 'isbn', 'rating']
print(len(df))


books_path = os.path.join(os.path.dirname(__file__), "BX_Books.csv")
books = pd.read_csv(books_path, sep=';', encoding="latin-1")
books.columns = ['isbn', 'title', 'author', 'year', 'publisher', 'img_url_s', 'img_url_m', 'img_url_l']

print(len(books))

print(len(pd.merge(df, books, on='isbn')))

# print("Distinct user_id count: ", len(set(df['user_id'])))
# print("Distinct isbn count: ", df['isbn'].nunique())