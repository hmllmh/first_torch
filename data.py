import torch
import numpy as np
import json
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, path, is_train):
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.is_train = is_train
        self.dtagid = 204572
        self.dlocate = 2
        self.dmobile = 2

    def __getitem__(self, index):
        index = str(index)
        if type(self.data[index]) == type('a'):
            self.data[index] = json.loads(self.data[index])
        locate = torch.from_numpy(np.array([self.data[index]['province'], self.data[index]['city']])).float()
        mobile = torch.from_numpy(np.array([self.data[index]['model'], self.data[index]['make']])).float()
        tagid = torch.from_numpy(np.array(self.data[index]['tagid']))
        tagid = tagid.unsqueeze(0)
        temp = torch.zeros(tagid.size(0), self.dtagid)  # 若tagid为空,则multi-hot全0
        tagid = temp.squeeze() if tagid.shape[1] == 0 else temp.scatter_(1, tagid, 1).squeeze()  # multi-hot
        if self.is_train:
            y = torch.from_numpy(np.array(self.data[index]['buy'])).float()
            return tagid, locate, mobile, y
        user = torch.tensor(int(self.data[index]['pid']))
        return tagid, locate, mobile, user

    def __len__(self):
        return len(self.data)

