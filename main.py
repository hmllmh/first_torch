#!/usr/bin/python
# -*-coding:utf-8-*-

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import model
import data
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_path = 'data/train.json'  # 训练数据
    test_path = 'data/test.json'    # 测试数据
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr_dataset = data.MyDataset(train_path, is_train=True)
    lr = 0.1
    n_epochs = 120
    user_dim = tr_dataset.user_num  # user数量=user one-hot embedding维度
    uid_dim = 200
    user_aux_dim = 4                # user辅助信息维度
    uad_dim = 200
    item_dim = tr_dataset.item_num  # item数量=item multi-hot embedding维度
    vid_dim = 200
    item_aux_dim = 4                # 置为0
    vad_dim = 200
    record = {"train_loss": [], "validate_loss": []}
    loss_json = "data/loss.json"  # 记录每个epoch的训练误差与泛化误差
    model_path = "data/model"     # 记录模型训练的结果
    test_result = "data/result.csv"  # 记录测试结果

    mymodel = model.MyModel(user_dim, uid_dim, user_aux_dim, uad_dim, item_dim, vid_dim, item_aux_dim, vad_dim).to(device)

    # 划分训练集和验证集
    train_size = int(0.7 * len(tr_dataset))
    vali_size = len(tr_dataset) - train_size
    train_dataset, vali_dataset = torch.utils.data.random_split(tr_dataset, [train_size, vali_size])
    tr_set = DataLoader(train_dataset, batch_size, shuffle=True)
    va_set = DataLoader(vali_dataset, batch_size, shuffle=True)
    # train&validate
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(mymodel.parameters(), lr)

    for epoch in range(n_epochs):
        mymodel.train()
        for user, user_aux, item, item_aux, y in tqdm(tr_set):
            user, user_aux, item, item_aux= user.to(device), user_aux.to(device), item.to(device), item_aux.to(device)
            optimizer.zero_grad()
            pred = mymodel(user, user_aux, item, item_aux)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # 用训练集计算loss
        train_loss = 0
        mymodel.eval()
        for user, user_aux, item, item_aux, y in tqdm(tr_set):
            user, user_aux, item, item_aux= user.to(device), user_aux.to(device), item.to(device), item_aux.to(device)
            with torch.no_grad():
                pred = mymodel(user, user_aux, item, item_aux)
                loss = criterion(pred, y)
            train_loss += loss.cpu()
        
        # 用验证集计算loss
        validate_loss = 0
        mymodel.eval()
        for user, user_aux, item, item_aux, y in tqdm(va_set):
            user, user_aux, item, item_aux= user.to(device), user_aux.to(device), item.to(device), item_aux.to(device)
            with torch.no_grad():
                pred = mymodel(user, user_aux, item, item_aux)
                loss = criterion(pred, y)
            validate_loss += loss.cpu()

        train_loss = train_loss / len(tr_set)
        validate_loss = validate_loss / len(va_set)
        record['train_loss'].append(train_loss)
        record['validate_loss'].append(validate_loss)
        print("[Epoch %d]  train_loss: %.2f  validate_loss: %.2f" % (epoch, train_loss, validate_loss))

    with open(loss_json, 'w') as f:
        json.dump(record, f)

    # 存储模型
    torch.save(mymodel.state_dict(), model_path)

    # test
    te_dataset = data.MyDataset(test_path, is_train=False, item_num=item_dim)
    te_set = DataLoader(te_dataset, batch_size, shuffle=True)
    mymodel.eval()
    preds = []
    users = []
    for user, user_aux, item, item_aux in tqdm(te_set):
        user, user_aux, item, item_aux= user.to(device), user_aux.to(device), item.to(device), item_aux.to(device)
        with torch.no_grad():
            pred = mymodel(user, user_aux, item, item_aux)
            preds += (pred >= 0.5).long().tolist()
        users += user.long().tolist()

    with open(test_result, "w"):
        pass
    data = pd.read_csv(test_result, names=['user_id', 'category_id'])
    data['user_id'] = users
    data['category_id'] = preds
    data.to_csv(test_result, index=False)

