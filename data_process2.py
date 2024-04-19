import numpy as np
import pandas as pd
import os
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
        'user': user_id_encoder.classes_.astype(str),
    })
    user_encoder_map.to_csv(os.path.join(save_dir, 'user_encoder_map.csv'), index=False)

    # save encoded user feature
    user_feature = pd.read_pickle(os.path.join(save_dir, 'user_feature.pkl'))
    user_feature['user'] = user_feature['user'].astype(str)
    user_feature = pd.merge(
        user_encoder_map,
        user_feature,
        on='user'
    )
    user_feature.to_pickle(os.path.join(save_dir, 'encoded_user_feature.pkl'))

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
    
    # save encoded item feature
    item_feature = pd.read_pickle(os.path.join(save_dir, 'item_feature.pkl'))
    item_feature['item'] = item_feature['item'].astype(str)
    item_feature = pd.merge(
        item_encoder_map,
        item_feature,
        on='item'
    )
    item_feature.to_pickle(os.path.join(save_dir, 'encoded_item_feature.pkl'))

    # Split the data into train and test sets
    if leave_out == None:
        interact_train, interact_test = \
            train_test_split(interact, train_size=split_rate, random_state=5, shuffle=shuffle_split)
    else:
        # Leave out the last 'leave_out' interactions for each user
        interact_train = []
        interact_test = []

        for _, group in interact.groupby('userid'):
            if len(group) > leave_out:
                interact_train.append(group[:-leave_out])
                interact_test.append(group[-leave_out:])
            else:
                interact_train.append(group)

        interact_train = pd.concat(interact_train, ignore_index=True)
        interact_test = pd.concat(interact_test, ignore_index=True)

    # get all user keeps
    def get_all_user_keeps(interact_test, history_users):
        user_keeps = interact_test['userid'].apply(lambda x: x in history_users)
        return user_keeps

    # get all item keeps
    def get_all_item_keeps(interact_test, history_items):
        item_keeps = interact_test['itemid'].apply(lambda x: x in history_items)
        return item_keeps
    
    # Filter out users and items that do not appear in the training set
    history_users = set(interact_train['userid'])
    history_items = set(interact_train['itemid'])
    user_keeps = get_all_user_keeps(interact_test, history_users)
    item_keeps = get_all_item_keeps(interact_test, history_items)
    interact_test = interact_test[user_keeps & item_keeps]

    # Save the train and test sets
    interact_train.to_pickle(os.path.join(save_dir, "interact_train.pkl"))
    interact_test.to_pickle(os.path.join(save_dir, "interact_test.pkl"))

if __name__ == '__main__':
    data_process('bx',
                 user_freq_threshold=None,
                 item_freq_threshold=None,
                 leave_out=1
                );


