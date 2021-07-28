#!/user/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dtagid, dlocate, dmobile, dtagid_emb, dlocate_emb, dmobile_emb, dhidden):
        super(MyModel, self).__init__()
        self.tagid_embed = nn.Embedding(dtagid, dtagid_emb)
        self.locate_embed = nn.Sequential(
                nn.Linear(dlocate, dlocate_emb),
                nn.Sigmoid(),
                )
        self.mobile_embed = nn.Sequential(
                nn.Linear(dmobile, dmobile_emb),
                nn.Sequential(),
                )
        self.rating = nn.Sequential(
                nn.Linear(dtagid_emb + dlocate_emb + dmobile_emb, dhidden),
                nn.Sigmoid(),
                nn.Linear(dhidden, 1),
                nn.Sigmoid(),
                )

    def forward(self, tagid, locate, mobile):
        tagid_emb = torch.mm(tagid, self.tagid_embed.weight)  # batch_size*embed_dim
        locate_emb = self.locate_embed(locate)
        mobile_emb = self.mobile_embed(mobile)
        result = self.rating(torch.cat((tagid_emb, locate_emb, mobile_emb), 1)).squeeze()
        return result

