import pandas as pd
import os

rating_file_path = os.path.join(os.path.dirname(__file__), 'ratings_wo_duplicates.csv')
user_file_path = os.path.join(os.path.dirname(__file__), 'BX_Users.csv')
item_file_path = os.path.join(os.path.dirname(__file__), 'BX_Books.csv')
output_path = os.path.join(os.path.dirname(__file__), 'output')


def transform_rating():
    # ignore index
    rating_file = pd.read_csv(
        rating_file_path,
        sep=';',
        names=['index', 'userid', 'itemid', 'score'],
        header=None,
        low_memory=False
    )

    # drop index
    rating_file.drop(columns=['index'], inplace=True)

    # add timestamp
    rating_file['timestamp'] = 0

    # save to output/rating.txt
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    rating_file.to_csv(
        os.path.join(output_path, 'rating.txt'),
        sep='\t',
        header=False,
        index=False
    )

    print('rating.txt saved')

def transform_user():
    # read user.csv
    user_file = pd.read_csv(
        user_file_path,
        sep=';',
        header=None,
        low_memory=False,
        encoding="latin-1",
    )
    # keep only the first 5 columns
    # rename columns
    user_file.columns = ['user', 'location', 'age']

    # fill age na with 0
    user_file['age'].fillna("0", inplace=True)

    # save to output/rating.txt
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # to pkl
    user_file.to_pickle(os.path.join(output_path, 'user_feature.pkl'))
    print('user_feature.pkl saved')

def transform_item():
    # read item.csv
    item_file = pd.read_csv(
        item_file_path,
        sep=';',
        header=None,
        low_memory=False,
        encoding="latin-1",
    )
    # keep only the first 5 columns
    item_file = item_file.iloc[1:, :5]
    # rename columns
    item_file.columns = ['item', 'title', 'Book-Author', 'Year-Of-Publication', 'Publisher']
    item_file['Publisher'].fillna('unknown', inplace=True)
    item_file['Book-Author'].fillna('unknown', inplace=True)

    # save to output/rating.txt
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # to pkl
    item_file.to_pickle(os.path.join(output_path, 'item_feature.pkl'))
    print('item_feature.pkl saved')

# transform_rating()
# transform_user()
# transform_item()

user_feature = pd.read_pickle(os.path.join(output_path, 'user_feature.pkl'))
print(user_feature.head())
print(user_feature.info())

item_feature = pd.read_pickle(os.path.join(output_path, 'item_feature.pkl'))
print(item_feature.head())
print(item_feature.info())


