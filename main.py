#!/usr/bin/python
# -*-coding:utf-8-*-

from tqdm import tqdm
import json
import pandas as pd
import torch
import torch.nn as nn
import model
import data
from torch.utils.data import DataLoader

if __name__ == "__main__":
    batch_size = 100
    lr = 0.1
    n_epochs = 200
    duser_emb = 200
    ditem_emb = 200
    stop_cnt = 50                  # 超过stop_cnt epoch模型 loss没有降低则提前退出training
    dhidden = 200                  # 隐藏层的神经元数量
    record = {"train_loss": [], "validate_loss": [], "F1": [], "acc": []}
    userp = "data/peo2book_id.pkl" # keys表示用户的集合，不重复。每个用户喜欢哪些item
    itemp = "data/book2peo_id.pkl" # keys表示item的集合，不重复。每个item被哪些用户喜欢
    negap = "data/book_nega.pkl"    # keys表示用户的集合，不重复。每个用户不喜欢哪99本书
    loss_json = "data/loss.json"   # 记录每个epoch的训练误差与泛化误差
    model_path = "data/model"      # 记录模型训练的结果
    dataset = data.MyDataset(userp, itemp, negap)
    user_num = dataset.user_num
    item_num = dataset.item_num
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mymodel = model.MyModel(user_num, item_num, duser_emb, ditem_emb,  dhidden).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(mymodel.parameters(), lr)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    vali_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - vali_size
    train_dataset, vali_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, vali_size, test_size])
    tr_set = DataLoader(train_dataset, batch_size, shuffle=True)
    va_set = DataLoader(vali_dataset, batch_size, shuffle=True)
    te_set = DataLoader(test_dataset, batch_size, shuffle=True)
    # training
    min_loss = 1000
    for epoch in range(n_epochs):
        mymodel.train()
        for user_id, item_id, y in tqdm(tr_set):
            optimizer.zero_grad()
            pred = mymodel(user_id, item_id).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            record['train_loss'].append("{:.4f}".format(loss.detach().cpu().item()))  # detach阻断反向传播,返回值仍为tensor; item返回tensor的值
        print("[Epoch {:d}] train loss: {:.4f}".format(epoch, loss))

        # 用验证集计算loss
        validate_loss = 0
        mymodel.eval()
        for user_id, item_id, y in tqdm(va_set):
            with torch.no_grad():
                pred = mymodel(user_id, item_id).squeeze()
                loss = criterion(pred, y)
            validate_loss += loss.cpu().detach().item() * len(y)
        validate_loss = validate_loss / len(va_set.dataset)
        if validate_loss < min_loss:
            # 若模型loss减小了,则存储模型
            min_loss = validate_loss
            print("[Epoch {:d}] validate loss: {:.4f} Saving model".format(epoch, min_loss))
            torch.save(mymodel.state_dict(), model_path)
            non_update_cnt = 0
        else:
            non_update_cnt += 1
        record['validate_loss'].append("{:.4f}".format(validate_loss))
        # 若模型已经多次epoch都没有提升,则break
        if non_update_cnt > stop_cnt:
            print("[Epoch {:d}] min loss: {:.4f} Finished training".format(epoch, min_loss))
            break

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

