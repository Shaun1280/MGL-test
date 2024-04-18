import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def data_process(dataset_name, split_rate=0.9, user_freq_threshold=None,
                 item_freq_threshold=None, shuffle_split=True, with_time=False, leave_out=None):
    save_dir = os.path.join(os.path.dirname(__file__), "dataset", dataset_name)

    if not os.path.exists(save_dir):
        print("dataset is not exist!!!!")
        return

    interact = pd.read_table(
        os.path.join(save_dir, "ratings.txt"),
        sep="\t",
        header= None,
        names= ['userid', 'itemid', 'score', 'timestamp']
    )

    # Encode user and item ids
    if with_time == True:
        interact = interact.sort_values("timestamp")
    
    # Filter out users that do not meet the frequency threshold
    if user_freq_threshold is not None:
        user_counts = interact['userid'].value_counts()
        interact['user_count'] = interact['userid'].apply(lambda x: user_counts[x])
        interact = interact[interact['user_count'] > user_freq_threshold]
        interact.drop(columns=['user_count'], inplace=True)

    # Filter out items that do not meet the frequency threshold
    if item_freq_threshold is not None:
        item_counts = interact['itemid'].value_counts()
        interact['item_count'] = interact['itemid'].apply(lambda x: item_counts[x])
        interact = interact[interact['item_count'] > item_freq_threshold]
        interact.drop(columns=['item_count'], inplace=True)

    # user id label encoder 
    user_id_encoder = LabelEncoder()
    user_id_encoder.fit(interact['userid'])
    interact['userid'] = user_id_encoder.transform(interact['userid'])
    
    # save user encoder map
    user_encoder_map = pd.DataFrame({
        'encoded': np.arange(len(user_id_encoder.classes_)),
        'user': user_id_encoder.classes_,
    })
    user_encoder_map.to_csv(os.path.join(save_dir, 'user_encoder_map.csv'), index=False)

    # item id label encoder
    item_id_encoder = LabelEncoder()
    item_id_encoder.fit(interact['itemid'])
    interact['itemid'] = item_id_encoder.transform(interact['itemid']) 
    
    # save item encoder map
    item_encoder_map = pd.DataFrame({
        'encoded': np.arange(len(item_id_encoder.classes_)),
        'item': item_id_encoder.classes_,
    })
    item_encoder_map.to_csv(os.path.join(save_dir, 'item_encoder_map.csv'), index=False)
    
    # print(interact.head(20))
    # print(interact.describe())

    

if __name__ == '__main__':
    data_process('bx',
                 user_freq_threshold=None,
                 item_freq_threshold=None);


