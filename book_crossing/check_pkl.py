import pandas as pd
import numpy as np
import os

def check_pkl(path):
    data = pd.read_pickle(path)
    print('head:\n', data.head(10))
    print('----------------------------')
    print('info:\n', data.info())
    print('----------------------------')
    print('describe:\n', data.describe())
    print('----------------------------')
    print('isnull..sum:\n', data.isnull().sum())
    print('----------------------------')
    print('columns:\n', data.columns)
    print('----------------------------')
    print('dtypes:\n', data.dtypes)
    print('----------------------------')
    print('shape:\n', data.shape)
    print('----------------------------')
    print('nunique:\n', data.nunique())
    # print(data['userid'].nunique())
    # print(data['itemid'].nunique())
    # print(data['rating'].nunique())
    # print(data['timestamp'].nunique())
    # print(data['timestamp'].max())
    # print(data['timestamp'].min())
    # print(data['rating'].value_counts())
    # print(data['userid'].value_counts())
    # print(data['itemid'].value_counts())
    # print(data['timestamp'].value_counts())

check_pkl(os.path.join(os.path.dirname(__file__), 'item_feature.pkl'))