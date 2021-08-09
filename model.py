#!/user/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, duser, ditem, duser_emb, ditem_emb, dhidden):
        super(MyModel, self).__init__()
        self.nuser_embed = nn.Embedding(duser, duser_emb)
        self.nitem_embed = nn.Embedding(ditem, ditem_emb)
        self.net = nn.Sequential(
                nn.Linear(duser_emb + ditem_emb, dhidden),
                nn.Sigmoid(),
                nn.Linear(dhidden, dhidden),
                nn.Sigmoid(),
                nn.Linear(dhidden, 1),
                nn.Sigmoid(),
                )

    def forward(self, user_id, item_id):
        user_embed = self.nuser_embed(user_id)
        item_embed = self.nitem_embed(item_id)
        v = torch.cat((user_embed, item_embed), dim=1)
        result = self.net(v)

        return result

