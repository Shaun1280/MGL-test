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

        self.feature_extract()

        self.userMeans = {} #mean values of users's ratings
        self.itemMeans = {} #mean values of items's ratings
        self.userDegrees = {} #users' degrees
        self.itemDegrees = {} #items' degrees
        self.userProbs = {} #probability of being selected by the user
        self.itemProbs = {} #probability of being selected

        self.globalMean = 0

        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self.testSet_i = defaultdict(dict)

        self.__generateSet()
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()

        self.train_dataset = Train_dataset(self.interact_train, self.item_num, self.trainSet_u)
        self.test_dataset = Test_dataset(self.testSet_u, self.item_num)
        self.test_dataset_one_plus_all = Test_dataset_one_plus_all(self.interact_test)


        user_historical_mask = np.ones((self.user_num, self.item_num))
        for uuu in self.trainSet_u.keys():
            item_list = list(self.trainSet_u[uuu].keys())
            if len(item_list) != 0:
                user_historical_mask[uuu, item_list] = 0
        

        self.user_historical_mask = torch.from_numpy(user_historical_mask)

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

        if bottom is not None:
            self.interact_train = self.interact_train[self.interact_train['score'] > bottom]
            self.interact_test = self.interact_test[self.interact_test['score'] > bottom]  

    def __generateSet(self):
        for row in self.interact_train.itertuples(index=False):
            userName = row.userid
            itemName = row.itemid
            rating = row.score
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating


        for row in self.interact_test.itertuples(index=False):
            userName = row.userid
            itemName = row.itemid
            rating = row.score
            self.testSet_u[userName][itemName] = rating
            self.testSet_i[itemName][userName] = rating


    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user_list:
            self.userMeans[u] = sum(self.trainSet_u[u].values())/(len(self.trainSet_u[u]) + 0.00000001)
            self.userDegrees[u] = len(list(self.trainSet_u[u].keys()))
            self.userProbs[u] = len(self.trainSet_u[u].values()) /len(self.interact_train)

    def __computeItemMean(self):
        for c in self.item_list:
            self.itemMeans[c] = sum(self.trainSet_i[c].values())/(len(self.trainSet_i[c])+0.00000001)
            self.itemDegrees[c] = len(list(self.trainSet_i[c].keys()))
            self.itemProbs[c] = len(self.trainSet_i[c].values()) /len(self.interact_train)

    def userRated(self,u):
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def itemRated(self,i):
        return list(self.trainSet_i[i].keys()),list(self.trainSet_i[i].values())


    def feature_extract(self):
        try:
            user_feature_name_list = list(self.user_feature.columns)
            user_feature_name_list.remove("user")
            user_feature_name_list.remove("encoded")

            self.user_feature_list = []
            for f in user_feature_name_list:
                encoder = LabelEncoder()
                encoder.fit(self.user_feature[f])
                self.user_feature[f] = encoder.transform(self.user_feature[f])
                feature_dim = len(encoder.classes_)
                self.user_feature_list.append({'feature_name':f, 'feature_dim':feature_dim})

            self.user_feature_list.append({'feature_name':'encoded', 'feature_dim':self.user_num})
            self.user_feature_matrix = torch.from_numpy(self.user_feature[[f['feature_name'] for f in self.user_feature_list]].values)
        except:
            self.user_feature_list = None
            self.user_feature_matrix = None


        self.dense_f_list_transforms = {}

        item_feature_name_list = list(self.item_feature.columns)
        print(item_feature_name_list)
        item_feature_name_list.remove("item")
        item_feature_name_list.remove("encoded")

        self.item_feature_list = []
        for f in item_feature_name_list:
            if type(self.item_feature[f][0]) == list:
                dense_f_list = self.item_feature[f].values.tolist()
                vocab = []
                for i in dense_f_list:
                    try:
                        vocab += i
                    except:
                        print('empty feature')
                        continue
                vocab = list(set(vocab))
                vocab_len = len(vocab)

                dense_f_transform = []
                for t in dense_f_list:
                    dense_f_idx = torch.zeros(1, vocab_len).long()
                    try:
                        for w in t:
                            idx = vocab.index(w)
                            dense_f_idx[0, idx] = 1
                        dense_f_transform.append(dense_f_idx)
                    except:
                        continue

                self.dense_f_list_transforms[f] = torch.cat(dense_f_transform, dim=0)

            else:
                encoder = LabelEncoder()
                encoder.fit(self.item_feature[f].fillna('NA'))
                self.item_feature[f] = encoder.transform(self.item_feature[f])
                feature_dim = len(encoder.classes_)
                self.item_feature_list.append({'feature_name':f, 'feature_dim':feature_dim})


        self.item_feature_list.append({'feature_name':'encoded', 'feature_dim':self.item_num})


        self.item_feature_matrix = torch.from_numpy(self.item_feature[[f['feature_name'] for f in self.item_feature_list]].values)




class Train_dataset(Dataset):
    def __init__(self, interact_train, item_num, trainSet_u):
        super(Train_dataset, self).__init__()
        self.interact_train = interact_train
        self.item_list = list(range(item_num))
        self.trainSet_u = trainSet_u

    def __len__(self):
        return len(self.interact_train)

    def __getitem__(self, idx):
        entry = self.interact_train.iloc[idx]

        user = entry.userid
        pos_item = entry.itemid
        neg_item = choice(self.item_list)
        while neg_item in self.trainSet_u[user]:
            neg_item = choice(self.item_list)

        return user, pos_item, neg_item

    


class Test_dataset_one_plus_all(Dataset):
    def __init__(self, interact_test):
        super(Test_dataset_one_plus_all, self).__init__()

        self.interact_test = interact_test

    def __len__(self):
        return len(self.interact_test)

    def __getitem__(self, idx):
        entry = self.interact_test.iloc[idx]

        user = entry.userid
        item = entry.itemid

        return user, item


class Test_dataset(Dataset):
    def __init__(self, testSet_u, item_num):
        super(Test_dataset, self).__init__()

        self.testSet_u = testSet_u
        self.user_list = list(testSet_u.keys())
        self.item_num = item_num

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item_list = torch.tensor(list(self.testSet_u[user].keys()))
        tensor = torch.zeros(self.item_num).scatter(0, item_list, 1)
        return user, tensor
