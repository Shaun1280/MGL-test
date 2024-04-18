import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

core = cpu_count()

def parallelize(data, func, num_of_processes=core):
    data_split = np.array_split(data, num_of_processes)
    with Pool(num_of_processes) as pool:
        data_list = pool.map(func, data_split)
    data = pd.concat(data_list)
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=core):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

def get_item_count(item_list, row):
    item_count = item_list.count(row.itemid)
    return item_count

def get_all_item_counts(data, item_list):
    item_counts = parallelize_on_rows(data, partial(get_item_count, deepcopy(item_list)))
    return item_counts

def get_user_count(user_list, row):
    user_count = user_list.count(row.userid)
    return user_count

def get_all_user_counts(data, user_list):
    user_counts = parallelize_on_rows(data, partial(get_user_count, deepcopy(user_list)))
    return user_counts

def get_user_keep(history_users, row):
    return (row.userid in history_users)

def get_all_user_keeps(data, history_users):
    user_keeps = parallelize_on_rows(data, partial(get_user_keep, deepcopy(history_users)))
    return user_keeps



def get_item_keep(history_items, row):
    return (row.itemid in history_items)

def get_all_item_keeps(data, history_items):
    item_keeps = parallelize_on_rows(data, partial(get_item_keep, deepcopy(history_items)))
    return item_keeps


def data_process(dataset_name, split_rate=0.9, user_fre_threshold=None,
                 item_fre_threshold=None, shuffle_split=True, with_time=False, leave_out=None):
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

    print(interact.head())

    

if __name__ == '__main__':
    data_process('bx');


