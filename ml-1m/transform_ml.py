import pandas as pd
import os

rating_file_path = os.path.join(os.path.dirname(__file__), 'ratings.dat')
user_file_path = os.path.join(os.path.dirname(__file__), 'users.dat')
item_file_path = os.path.join(os.path.dirname(__file__), 'movies.dat')
output_path = os.path.join(os.path.dirname(__file__), 'output')


def transform_rating():
    # ignore index
    rating_file = pd.read_csv(
        rating_file_path,
        sep='::',
        names=['userid', 'itemid', 'score', 'timestamp'],
        header=None,
        encoding="latin-1",
        engine='python'
    )

    # save to output/rating.txt
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    rating_file.to_csv(
        os.path.join(output_path, 'ratings.txt'),
        sep='\t',
        header=False,
        index=False
    )

    print('ratings.txt saved')

def transform_user():
    # read user.csv
    user_file = pd.read_csv(
        user_file_path,
        sep='::',
        header=None,
        encoding="latin-1",
        engine="python"
    )

    # rename columns
    user_file.columns = ['user', 'gender', 'age', 'occupation', 'zip-code']

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
        sep='::',
        header=None,
        encoding="latin-1",
        engine="python"
    )

    # rename columns
    item_file.columns = ['item', 'title', 'genres']

    # split title into title and year
    item_file['year'] = item_file['title'].apply(lambda x: x[-5:-1])
    item_file['title'] = item_file['title'].apply(lambda x: x[:-7])

    # save to output/rat)ing.txt
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


