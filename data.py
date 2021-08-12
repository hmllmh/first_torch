import torch
import numpy as np
import pandas as pd
import json
import pickle
import scipy.sparse
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, user_content, item_content, data):
        self.user_content = user_content
        self.item_content = item_content
        self.data = data

    def __getitem__(self, index):
        uid = int(self.data['uid'].values[index])
        iid = int(self.data['iid'].values[index])
        target = self.data['target'].values[index]
        target = target.astype(np.float32)
        user_c = self.user_content[uid, :].toarray()
        item_c = self.item_content[iid, :].toarray()
        return uid, user_c, iid, item_c, target

    def __len__(self):
        return len(self.data)

def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def negative_sample(train, neg):
    user_neg = np.tile(train['uid'], neg)
    item_neg = np.random.choice(train['iid'].unique(), size=user_neg.shape[0], replace=True)
    target_neg = np.zeros_like(item_neg)
    neg = pd.DataFrame({'uid': user_neg, 'iid': item_neg, 'target':target_neg})
    return pd.concat([train, neg])

def read_csv(name):
    data = pd.read_csv(name)
    data['target'] = np.ones(data.shape[0])
    return data

def get_batch_dataset(user_content, item_content, data, batch_size):
    dataset = MyDataset(user_content, item_content, data)
    batch_set = DataLoader(dataset, batch_size, shuffle=True)
    return batch_set

def load_data(infop, user_contentp, item_contentp, trainp, vali_itemp, vali_userp, vali_user_itemp, batch_size, neg):
    dat = {}
    info = read_pickle(infop)
    dat['user_num'] = info['num_user']
    dat['item_num'] = info['num_item']
    user_content = scipy.sparse.load_npz(user_contentp)
    item_content = scipy.sparse.load_npz(item_contentp)
    dat['duser_c'] = user_content.shape[1]
    dat['ditem_c'] = item_content.shape[1]
    train = read_csv(trainp)
    train = negative_sample(train, neg)  # 负采样，将负样本与正样本拼接后形成训练数据
    dat['tr_set'] = get_batch_dataset(user_content, item_content, train, batch_size)
    vali_item = read_csv(vali_itemp)
    dat['va_item_set'] = get_batch_dataset(user_content, item_content, vali_item, batch_size)
    vali_user = read_csv(vali_userp)
    dat['va_user_set'] = get_batch_dataset(user_content, item_content, vali_user, batch_size)
    vali_user_item = read_csv(vali_user_itemp)
    dat['va_user_item_set'] = get_batch_dataset(user_content, item_content, vali_user_item, batch_size)

    return dat
