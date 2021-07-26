import torch
import numpy as np
import json
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, path, is_train, item_num=0):
        with open(path, 'r') as f:
            self.data = json.load(f)
            self.is_train = is_train
            if is_train:
                self.user_num = self.data[-1]['user_num']
                self.item_num = self.data[-1]['item_num']
                self.data = self.data[:-1]
            else:
                self.item_num = item_num

    def __getitem__(self, index):
        user = torch.from_numpy(np.array(self.data[index]['pid']))
        user_aux = torch.from_numpy(np.array([self.data[index]['gender'], self.data[index]['age'], self.data[index]['model'], self.data[index]['make']])).float()
        item_aux = torch.Tensor([0, 0, 0, 0])
        tagid = torch.from_numpy(np.array(self.data[index]['tagid']))
        tagid = tagid.unsqueeze(0)
        item = torch.zeros(tagid.size(0), self.item_num).scatter_(1, tagid, 1).squeeze()  # multi-hot
        if self.is_train:
            y = torch.from_numpy(np.array(self.data[index]['label'])).float()
            return user, user_aux, item, item_aux, y
        return user, user_aux, item, item_aux

    def __len__(self):
        return len(self.data)

