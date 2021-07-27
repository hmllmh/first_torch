#!/user/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, user_dim, uid_dim, user_aux_dim, uad_dim, item_dim, vid_dim, item_aux_dim, vad_dim):
        """
        user_dim:  user one-hot编码的维度
        uid_dim:   user embedding的维度
        user_aux_dim  user辅助信息的维度
        uad_dim       user辅助信息embedding的维度
        item_dim   item multi-hot编码的维度
        vid_dim    item embedding的维度
        item_aux_dim  item辅助信息的维度
        vad_dim       item辅助信息embedding的维度
        """
        self.uid_dim = uid_dim
        self.vad_dim = vad_dim
        super(MyModel, self).__init__()
        self.user_embed = nn.Embedding(user_dim, uid_dim)
        self.item_embed = nn.Embedding(item_dim, vid_dim)
        self.user_aux_net = nn.Sequential(
                nn.Linear(user_aux_dim, uad_dim),
                nn.Sigmoid(),
                )
        self.item_aux_net = nn.Sequential(
                nn.Linear(item_aux_dim, vad_dim),
                nn.Sigmoid(),
                )

    def forward(self, user, user_aux, item, item_aux):
        # self.uid = self.user_embed(user)
        self.uid = torch.zeros(self.uid_dim)
        self.uad = self.user_aux_net(user_aux)
        self.vid = torch.mm(item, self.item_embed.weight)  # batch_size*embed_dim
        # self.vad = self.item_aux_net(item_aux)
        self.vad = torch.zeros(self.vad_dim)
        self.u = self.uid + self.uad
        self.v = self.vid + self.vad
        s = nn.Sigmoid()
        result = s(torch.sum(self.u * self.v, dim=1))
        return result

