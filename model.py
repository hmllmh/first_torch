#!/user/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, duser, duser_c, ditem, ditem_c, duser_emb, duser_c_emb, ditem_emb, ditem_c_emb, dhidden):
        super(MyModel, self).__init__()
        self.nuser_embed = nn.Embedding(duser, duser_emb)
        self.nitem_embed = nn.Embedding(ditem, ditem_emb)
        self.uc = nn.Sequential(
                nn.Linear(duser_c, duser_c_emb),
                nn.Sigmoid(),
                )
        self.ic = nn.Sequential(
                nn.Linear(ditem_c, ditem_c_emb),
                nn.Sigmoid(),
                )
        self.net = nn.Sequential(
                nn.Linear(duser_emb + duser_c_emb + ditem_emb + ditem_c_emb, dhidden),
                nn.Sigmoid(),
                nn.Linear(dhidden, dhidden),
                nn.Sigmoid(),
                nn.Linear(dhidden, 1),
                nn.Sigmoid(),
                )

    def forward(self, user_id, user_c, item_id, item_c):
        user_embed = self.nuser_embed(user_id)
        user_c_embed = self.uc(user_c).squeeze()
        item_embed = self.nitem_embed(item_id)
        item_c_embed = self.ic(item_c).squeeze()
        v = torch.cat((user_embed, user_c_embed, item_embed, item_c_embed), dim=1)
        result = self.net(v)

        return result

