import torch
import numpy as np
import json
import pickle
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, userp, itemp, negap):
        user = self.read_pickle(userp)
        item = self.read_pickle(itemp)
        nega = self.read_pickle(negap)
        self.user_num = len(user)
        self.item_num = len(item)
        self.user = []
        self.item = []
        self.y    = []
        # y = 1, 喜欢的item
        for u in range(self.user_num):
            for i in range(len(user[u])):
                self.user.append(u)
                self.item.append(user[u][i])
                self.y.append(1)
        # y = 0, 不喜欢的item
        for u in range(self.user_num):
            for i in range(len(nega[u])):
                self.user.append(u)
                self.item.append(nega[u][i])
                self.y.append(0)

    def __getitem__(self, index):
        user_id = torch.from_numpy(np.array(self.user[index]))
        item_id = torch.from_numpy(np.array(self.item[index]))
        y = torch.from_numpy(np.array(self.y[index])).float()
        return user_id, item_id, y

    def __len__(self):
        return len(self.user)

    def read_pickle(self, name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data

