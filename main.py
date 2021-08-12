#!/usr/bin/python
# -*-coding:utf-8-*-

from tqdm import tqdm
import json
import pandas as pd
import torch
import torch.nn as nn
import model
import data

'''hyper parameter'''
batch_size = 1024
lr = 0.1
n_epochs = 200
duser_emb = 200
ditem_emb = 200
duser_c_emb = 200
ditem_c_emb = 200
stop_cnt = 50                  # 超过stop_cnt epoch模型 loss没有降低则提前退出training
dhidden = 200                  # 隐藏层的神经元数量
neg = 5                        # train dataset 负采样负样本为正样本的neg倍

'''data path'''
xing = 'data/XING/'
infop = xing + 'info.pkl'
user_contentp = xing + 'user_content.npz'
item_contentp = xing + 'item_content.npz'
trainp = xing + 'train.csv'
vali_itemp = xing + 'vali_item.csv'
vali_userp = xing + 'vali_user.csv'
vali_user_itemp = xing + 'vali_user_item.csv'
loss_json = "data/loss.json"   # 记录每个epoch的训练误差与泛化误差
model_path = "data/model"      # 记录模型训练的结果

record = {"train_loss": [], "validate_loss": [], "F1": [], "acc": []}
dat = data.load_data(infop, user_contentp, item_contentp, trainp, vali_itemp, vali_userp, vali_user_itemp, batch_size, neg)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
mymodel = model.MyModel(dat['user_num'], dat['duser_c'], dat['item_num'], dat['ditem_c'], duser_emb, duser_c_emb, ditem_emb, ditem_c_emb,  dhidden)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(mymodel.parameters(), lr)
print(mymodel, criterion, optimizer)

# 用验证集计算loss
def get_va_loss(va_set):
    validate_loss = 0
    mymodel.eval()
    for uid, user_c, iid, item_c, target in tqdm(va_set):
        with torch.no_grad():
            pred = mymodel(uid, user_c, iid, item_c).squeeze()
            loss = criterion(pred, target)
        validate_loss += loss.cpu().detach().item() * len(target)
    validate_loss = validate_loss / len(va_set.dataset)
    return validate_loss

if __name__ == "__main__":
    # training
    min_loss = 1000
    for epoch in range(n_epochs):
        mymodel.train()
        for uid, user_c, iid, item_c, target in tqdm(dat['tr_set']):
            optimizer.zero_grad()
            pred = mymodel(uid, user_c, iid, item_c).squeeze()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            record['train_loss'].append("{:.4f}".format(loss.detach().cpu().item()))  # detach阻断反向传播,返回值仍为tensor; item返回tensor的值
        va_item_loss = get_va_loss(dat['va_item_set'])
        va_user_loss = get_va_loss(dat['va_user_set'])
        va_user_item_loss = get_va_loss(dat['va_user_item_set'])
        validate_loss = va_item_loss + va_user_loss + va_user_item_loss
        if validate_loss < min_loss:
            # 若模型loss减小了,则存储模型
            min_loss = validate_loss
            print("[Epoch {:d}] min validate loss: {:.4f} Saving model".format(epoch, min_loss))
            torch.save(mymodel.state_dict(), model_path)
            non_update_cnt = 0
        else:
            non_update_cnt += 1
        record['validate_loss'].append("{:.4f}".format(validate_loss))
        # 若模型已经多次epoch都没有提升,则break
        if non_update_cnt > stop_cnt:
            print("[Epoch {:d}] min validate loss: {:.4f} Finished training".format(epoch, min_loss))
            break
        print('[Epoch {:d}] train_loss: {:.4f} va_item_loss: {:.4f} va_user_loss: {:.4f} va_user_item_loss: {:.4f}'\
                .format(epoch, loss, va_item_loss, va_user_loss, va_user_item_loss))
        '''
        # test data计算F1 score与正确率
        mymodel.eval()
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for user_id, item_id, y in tqdm(te_set):
            with torch.no_grad():
                pred = mymodel(user_id, item_id)
                pred = (pred >= 0.5).detach().cpu()
                TP += ((pred == 1) & (y == 1)).cpu().sum()
                TN += ((pred == 0) & (y == 0)).cpu().sum()
                FN += ((pred == 0) & (y == 1)).cpu().sum()
                FP += ((pred == 1) & (y == 0)).cpu().sum()
        precision = TP / (TP + FP)
        recall = TP / (TP +FN)
        F1 = 2 * precision * recall / (precision + recall)
        acc = (TP +TN) / (TP + TN + FP + FN)
        print("[Epoch {:d}] F1 score: {:.4f} accuracy: {:.4f}".format(epoch, F1, acc))
        '''

