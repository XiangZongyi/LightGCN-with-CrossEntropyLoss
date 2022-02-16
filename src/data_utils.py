import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import random
import torch
import copy


def create_dataset(args):
    return ML_100K_Dataset(args)


def create_dataloader(dataset, batch_size, training=False):
    if training:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class ML_100K_Dataset(object):
    def __init__(self, args):
        self.args = args

        self.dataset = self.args.dataset
        self.data_path = self.args.data_path

        self.inter_feat = self._load_dataframe()  # user-item-interaction DataFrame: user, item, rating, timestamp

        # user_id, item_id, rating, timestamp
        self.uid_field, self.iid_field, self.rating_field, self.timestamp = self.inter_feat.columns

        self.user_num = self.inter_feat[self.uid_field].max() + 1
        self.item_num = self.inter_feat[self.iid_field].max() + 1

        self.train_inter_feat, self.test_inter_feat = self._split_inter_feat()  # DataFrame: user, pos_item_list

        # add negative item in train_inter_feat
        self.user_pool = set(self.inter_feat[self.uid_field])
        self.item_pool = set(self.inter_feat[self.iid_field])
        self.train_inter_add_negative = self._sample_negative()  # DataFrame: user, pos_item_list, neg_item_list

        # the positive item num and negative item num of each user
        self.train_items_num = [len(i) for i in self.train_inter_feat['train_items']]
        self.neg_items_num = [self.args.neg_sample_num * len(i) for i in self.train_inter_add_negative['train_items']]

    def _load_dataframe(self):
        # '../data/ml-100k/ml-100k.inter'
        inter_feat_path = os.path.join(self.data_path + '/' + self.dataset + '/' + f'{self.dataset}.inter')
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f'File {inter_feat_path} not exist.')

        # create DataFrame
        columns = []
        usecols = []
        dtype = {}
        with open(inter_feat_path, 'r') as f:
            head = f.readline()[:-1]
        for field_type in head.split('\t'):
            field, ftype = field_type.split(':')
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == 'float' else int
        df = pd.read_csv(
            inter_feat_path, delimiter='\t', usecols=usecols, dtype=dtype
        )
        df.columns = columns
        df['rating'] = 1.0  # implicit feedback

        # reset user(item) id from 1-943(1-1682) to 0-942(0-1681)
        if df[columns[0]].min() > 0:
            df[columns[0]] = df[columns[0]].apply(lambda x: x - 1)
            df[columns[1]] = df[columns[1]].apply(lambda x: x - 1)

        return df

    def _split_inter_feat(self):
        interact_status = self.inter_feat.groupby(self.uid_field)[self.iid_field].apply(set).reset_index().rename(
            columns={self.iid_field: 'interacted_items'}
        )  # user-item_dic-interaction DataFrame: user, interacted_items

        # split train and test randomly by args.data_split_ratio
        interact_status['train_items'] = interact_status['interacted_items'].\
            apply(lambda x: set(random.sample(x, round(len(x) * self.args.data_split_ratio[0]))))
        interact_status['test_items'] = interact_status['interacted_items'] - interact_status['train_items']
        interact_status['train_items'] = interact_status['train_items'].apply(list)
        interact_status['test_items'] = interact_status['test_items'].apply(list)

        train_inter_feat = interact_status[[self.uid_field, 'train_items']]
        test_inter_feat = interact_status[[self.uid_field, 'test_items']]

        return train_inter_feat, test_inter_feat

    def _sample_negative(self):
        interact_status = self.inter_feat.groupby(self.uid_field)[self.iid_field].apply(set).reset_index().rename(
            columns={self.iid_field: 'interacted_items'}
        )  # user-item_dic-interaction DataFrame: user, interacted_items
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        train_inter_add_negative = pd.merge(self.train_inter_feat, interact_status, on=self.uid_field)

        neg_items_num = [self.args.neg_sample_num * len(i) for i in train_inter_add_negative['train_items']]
        neg_sample_items = []
        for i in range(len(neg_items_num)):
            neg_sample_items.append(random.sample(train_inter_add_negative['negative_items'][i], neg_items_num[i]))
        train_inter_add_negative['negative_samples'] = neg_sample_items
        return train_inter_add_negative[[self.uid_field, 'train_items', 'negative_samples']]

    def get_train_dataset(self):
        # get train_dataset
        users, items, ratings = [], [], []

        for row in self.train_inter_add_negative.itertuples():
            index = getattr(row, 'Index')
            # add positive data
            for i in range(self.train_items_num[index]):
                users.append(int(getattr(row, self.uid_field)))
                items.append(int(getattr(row, 'train_items')[i]))
                ratings.append(float(1))
            # add negative data
            for j in range(self.neg_items_num[index]):
                users.append(int(getattr(row, self.uid_field)))
                items.append(int(getattr(row, 'negative_samples')[j]))
                ratings.append(float(0))

        train_dataset = TorchDataset(user=torch.LongTensor(users),
                                     item=torch.LongTensor(items),
                                     rating=torch.FloatTensor(ratings))
        return train_dataset

    def get_test_data(self):
        test_users = list(self.test_inter_feat[self.uid_field])
        ground_true_items = list(self.test_inter_feat['test_items'])  # list like [[],[],...,[]] len: n_users
        return test_users, ground_true_items

    def create_mask_matrix(self):
        mask_matrix = torch.zeros(self.user_num, self.item_num)
        for row in self.train_inter_feat.itertuples():
            user = getattr(row, self.uid_field)
            index = getattr(row, 'Index')
            for i in range(self.train_items_num[index]):
                item = getattr(row, 'train_items')[i]
                mask_matrix[user][item] = -np.inf
        return mask_matrix

    def inter_matrix(self, form='coo'):
        row = []
        col = []
        for row_ in self.train_inter_feat.itertuples():
            index = getattr(row_, 'Index')
            row.extend([getattr(row_, self.uid_field)] * self.train_items_num[index])
            col.extend(getattr(row_, 'train_items'))
        data = np.ones(len(row))

        mat = sp.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')


class TorchDataset(Dataset):
    def __init__(self, user, item, rating):
        super(Dataset, self).__init__()

        self.user = user
        self.item = item
        self.rating = rating

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.rating[idx]

