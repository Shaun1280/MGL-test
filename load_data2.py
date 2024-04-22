import torch
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import numpy as np
import pandas as pd
from random import choice
import os
from sklearn.preprocessing import LabelEncoder

class Data(object):
    def __init__(self, opt):
        # load data
        self.interact_train, self.interact_test = None, None
        self.user_num, self.item_num = None, None
        self.user_feature, self.item_feature = None, None
        self.__data_load(opt.dataset_name, bottom=opt.implcit_bottom)

        self.user_list = list(range(self.user_num))
        self.item_list = list(range(self.item_num))

        # preprocess features
        self.user_feature_list = None
        self.item_feature_list = None
        self.user_feature_matrix = None
        self.__preprocess_features()

        # generate train and test set (maps)
        self.trainset_user = defaultdict(dict)
        self.trainset_item = defaultdict(dict)
        self.testset_user = defaultdict(dict)
        self.testset_item = defaultdict(dict)
        self.__generate_set()

        # compute means, degrees, probs, and global mean
        self.user_means = {} # mean values of users's ratings
        self.item_means = {} # mean values of items's ratings
        self.user_degrees = {} # users' degrees
        self.item_degrees = {} # items' degrees
        self.user_probs = {} # probability of being selected by the user
        self.item_probs = {} # probability of being selected
        self.global_mean = 0
        self.__compute_statistics()

        # create datasets
        self.train_dataset = TrainDataset(self.interact_train, self.item_num, self.trainset_user)
        self.test_dataset = TestDataset(self.testset_user, self.item_num)

        # create mask for user's historical interactions
        user_historical_mask = np.ones((self.user_num, self.item_num))
        # Iterate over users and items
        for user, items in self.trainset_user.items():
            if items:
                # Convert keys to list and update mask
                item_list = list(items.keys())
                user_historical_mask[user, item_list] = 0
        # Convert numpy array to PyTorch tensor
        self.user_historical_mask = torch.from_numpy(user_historical_mask)

    # load processed data
    def __data_load(self, dataset_name, bottom=None):
        save_dir = os.path.join(os.path.dirname(__file__), "dataset/" + dataset_name)
        if not os.path.exists(save_dir):
            print("dataset is not exist!!!!")
            return None

        if os.path.exists(save_dir + '/encoded_user_feature.pkl'):
            self.user_feature = pd.read_pickle(save_dir + '/encoded_user_feature.pkl')
            print("encoded user_feature loaded", self.user_feature.shape)
        else:
            print("user_feature is not exist!!!!")

        if os.path.exists(save_dir + '/encoded_item_feature.pkl'):
            self.item_feature = pd.read_pickle(save_dir + '/encoded_item_feature.pkl')
            print("encoded item_feature loaded", self.item_feature.shape)
        else:
            print("item_feature is not exist!!!!")

        self.interact_train = pd.read_pickle(os.path.join(save_dir, 'interact_train.pkl'))
        self.interact_test = pd.read_pickle(os.path.join(save_dir, 'interact_test.pkl'))

        self.item_encoder_map = pd.read_csv(os.path.join(save_dir, 'item_encoder_map.csv'))
        self.user_encoder_map = pd.read_csv(os.path.join(save_dir, 'user_encoder_map.csv'))
        self.item_num, self.user_num = len(self.item_encoder_map), len(self.user_encoder_map)

        # filter the data by bottom (e.g. implicit feedback = 0)
        if bottom is not None:
            self.interact_train = self.interact_train[self.interact_train['score'] > bottom]
            self.interact_test = self.interact_test[self.interact_test['score'] > bottom]  

    # generate train and test set (maps)
    def __generate_set(self):
        def process_interactions(interactions, user_set, item_set):
            for row in interactions.itertuples(index=False):
                user_set[row.userid][row.itemid] = row.score
                item_set[row.itemid][row.userid] = row.score

        process_interactions(self.interact_train, self.trainset_user, self.trainset_item)
        process_interactions(self.interact_test, self.testset_user, self.testset_item)
        
    def __compute_statistics(self):
        def compute_mean(keys, trainset, means, degrees, probs):
            EPSILON = 0.00000001
            for key in keys:
                values = trainset[key].values()
                means[key] = sum(values) / (len(values) + EPSILON)
                degrees[key] = len(trainset[key])
                probs[key] = len(values) / len(self.interact_train)

        compute_mean(self.user_list, self.trainset_user, self.user_means, self.user_degrees, self.user_probs)
        compute_mean(self.item_list, self.trainset_item, self.item_means, self.item_degrees, self.item_probs)

        self.global_mean = sum(self.user_means.values()) / len(self.user_means) if self.user_means else 0

    def __preprocess_features(self, remove_cols=['user', 'item', 'encoded']):
        # process user feature
        user_feature_name_list = [col for col in self.user_feature.columns if col not in remove_cols]
        print(user_feature_name_list)
        
        # user_feature is a dataframe with columns: user, feature1, feature2, ...,
        self.user_feature_list = []
        for f in user_feature_name_list:
            encoder = LabelEncoder()
            self.user_feature[f] = encoder.fit_transform(self.user_feature[f])
            feature_dim = len(encoder.classes_)
            # feature_dim is the number of unique values in the feature 
            self.user_feature_list.append({'feature_name':f, 'feature_dim':feature_dim})

        self.user_feature_list.append({'feature_name':'encoded', 'feature_dim':self.user_num})
        # (one hot encoding)
        self.user_feature_matrix = torch.tensor(self.user_feature[[f['feature_name'] for f in self.user_feature_list]].values)

        
        # process item feature
        item_feature_name_list = [col for col in self.item_feature.columns if col not in remove_cols]
        print(item_feature_name_list)

        self.item_feature_list = []
        self.dense_f_list_transforms = {}
        for f in item_feature_name_list:
            if isinstance(self.item_feature[f][0], list):
                dense_f_list = self.item_feature[f].values.tolist()
                vocab = []
                for i in dense_f_list:
                    if i:  # Check if the feature is not empty
                        vocab += i
                    else:
                        print('empty feature')

                vocab = list(set(vocab)) # remove duplicates
                vocab_dict = {word: idx for idx, word in enumerate(vocab)}  # Create a dictionary for faster lookup

                dense_f_transform = []
                for t in dense_f_list:
                    dense_f_idx = torch.zeros(1, len(vocab)).long()
                    if t:  # Check if the feature is not empty
                        for w in t:
                            idx = vocab_dict.get(w)  # Get the index from the dictionary
                            if idx is not None:  # Check if the word is in the vocabulary
                                dense_f_idx[0, idx] = 1
                        dense_f_transform.append(dense_f_idx)

                self.dense_f_list_transforms[f] = torch.cat(dense_f_transform, dim=0)
            else:
                encoder = LabelEncoder()
                self.item_feature[f] = encoder.fit_transform(self.item_feature[f])
                feature_dim = len(encoder.classes_)
                # feature_dim is the number of unique values in the feature 
                self.item_feature_list.append({'feature_name':f, 'feature_dim':feature_dim})

        # (one hot encoding)
        self.item_feature_list.append({'feature_name':'encoded', 'feature_dim':self.item_num})
        self.item_feature_matrix = torch.from_numpy(self.item_feature[[f['feature_name'] for f in self.item_feature_list]].values)

class TrainDataset(Dataset):
    def __init__(self, interact_train, item_num, trainset_user):
        super().__init__()
        self.interact_train = interact_train
        self.item_list = list(range(item_num))
        self.trainset_user = trainset_user

    def __len__(self):
        return len(self.interact_train)

    def __getitem__(self, idx):
        entry = self.interact_train.iloc[idx]
        user = entry.userid
        # positive item: the user has interacted with
        pos_item = entry.itemid
        # negative item: the user has not interacted with
        neg_items = [item for item in self.item_list if item not in self.trainset_user[user]]
        neg_item = choice(neg_items)

        return user, pos_item, neg_item

class TestDataset(Dataset):
    def __init__(self, testset_user, item_num):
        super().__init__()
        self.testset_user = testset_user
        self.user_list = list(testset_user.keys())
        self.item_num = item_num

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        # get the items the user has interacted with
        item_list = torch.tensor(list(self.testset_user[user].keys()))
        # create a tensor with 1s at the indices of the items the user has interacted with
        tensor = torch.zeros(self.item_num).scatter(0, item_list, 1)
        return user, tensor
