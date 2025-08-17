import os
import re
import numpy as np
import json
from collections import Counter
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_predict
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import R2Score
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics, neighbors
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
# np.set_printoptions(threshold=np.inf)
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv("D:\Beidou_cleaned_data.csv")
# print(data)
# print(data.info())
data = data.dropna()
a = list(set(data['年份']))
b = list(set(data['總坪數']))
c = sorted(list(set(data['價格'])))
d = list(set(data['屋齡']))
e = list(set(data['主建築物佔比']))
f = list(set(data['用途']))
g = list(set(data['房子類型']))
h = list(set(data['車位']))
i = list(set(data['區域']))
j = list(set(data['電梯']))
# print(a)
print(b)
# print(c)
print(d)
# print(e)
# print(f)
# print(g)
# print(h)
# print(i)
# print(j)
# print(k)
# print(m)
# print(n)
# data = data[data['縣市'] == 'Kaohsiung']
data = data[data['價格'] > 1000000]
data = data[data['價格'] < 60000001]
# data = data.sample(frac=1).reset_index(drop=True)
data = data.reset_index(drop=True)

data = data.drop(columns=['主建築物佔比'])
data = data.drop(columns=['用途'])
data = data.drop(columns=['區域'])
# data = data.drop(columns=['縣市'])
data['房子類型'] = data['房子類型'].replace({'華廈(10層含以下有電梯)' : 1, '店面（店舖)' : 2, '透天厝' : 3, '公寓(5樓含以下無電梯)' : 4, '住宅大樓(11層含以上有電梯)' : 5, '農舍' : 6, '工廠' : 7, '其他' : 8, '倉庫' : 9, '廠辦' : 10, '辦公商業大樓' : 11, '套房(1房(1廳)1衛)' : 12})

data['價格'] = data['價格'].map(lambda x: x // 500000 * 500000 + 250000)
# data['總坪數'] = data['總坪數'].map(lambda x: x // 5 * 5 + 2.5)
# data['屋齡'] = data['屋齡'].map(lambda x: x // 5 * 5 + 2.5)

# data = data.drop(index = [172039, 162573, 151864, 108245, 86547, 128575, 69246, 15802, 56671, 67008, 89453, 10587,  69858,  79217,  80148, 143341, 155063], axis = 0)
# data = data.reset_index(drop=True)

# data_year = data["年份"]
# data = data.drop(columns=['年份'])
# scale = StandardScaler()
# data = scale.fit_transform(data)
# data = pd.DataFrame(data)
# data[6] = data_year

drop_list = []
for i in list(set(data['價格'])):
    # print(np.where(data['價格'] == i))
    # print(len(np.where(data['價格'] == i)[0].tolist()))
    if len(np.where(data['價格'] == i)[0].tolist()) < 5:
        for j in np.where(data['價格'] == i)[0].tolist():
            # print(j)
            drop_list.append(j)
data = data.drop(index = drop_list, axis = 0)
data = data.reset_index(drop=True)
print(len(list(set(data['價格']))))

price_list = []
for i in range(len(data['價格'])):
    if data['價格'][i] not in price_list:
        price_list.append(data['價格'][i])
    # print(price_list)
    data['價格'][i] = int(np.where(price_list == data['價格'].iloc[i])[0])
print(data['價格'])
# print(price_list)
print(len(list(set(data['價格']))))

data = data.sample(frac=1).reset_index(drop=True)
data = data.reset_index(drop=True)

print(data)
data = data.rename(columns = {'年份':'year', '總坪數':'size', '屋齡':'age', '房子類型':'type', '車位':'park', '電梯':'elevator', '價格':'price'})
data_year = data["year"]

corr = data.corr("pearson")
fig, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(corr, cmap = 'RdBu', annot = True)
plt.show()

y = data['price']
x = data.drop(columns = 'price')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int)
y_test = np.array(y_test, dtype=np.int)
y_train = np.reshape(y_train, [y_train.shape[0], 1])
y_test = np.reshape(y_test, [y_test.shape[0], 1])
X = np.hstack([x_train, y_train])
Y = np.hstack([x_test, y_test])
# X = [x_train, y_train]
# Y = [x_test, y_test]
# print(np.array(X).shape)
# print(x_train.shape)
# print(y_train.shape)
batch_size = 128
# transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(dataset=X, batch_size=batch_size, shuffle=True)
print(len(train_loader))
test_loader = DataLoader(dataset=Y, batch_size=batch_size, shuffle=False)

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.dropout = nn.Dropout(0.1).to(device)
        self.linear0 = nn.Linear(input_size, hidden_size).to(device)
        self.batch1 = nn.BatchNorm1d(hidden_size).to(device)
        self.linear = nn.Linear(hidden_size, hidden_size // 2).to(device)
        self.batch2 = nn.BatchNorm1d(hidden_size // 2).to(device)
        self.linear2 = nn.Linear(hidden_size // 2, hidden_size // 4).to(device)
        self.batch3 = nn.BatchNorm1d(hidden_size // 4).to(device)
        self.linear3 = nn.Linear(hidden_size // 4, hidden_size // 8).to(device)
        self.batch4 = nn.BatchNorm1d(hidden_size // 8).to(device)
        self.linear4 = nn.Linear(hidden_size // 8, num_classes).to(device)
        self.linear5 = nn.Linear(num_classes, 1).to(device)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        #
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # print(x.dtype)
        # print(h0.dtype)
        # print(c0.dtype)
        # print((h0, c0).dtype)
        # output, (h_n, h_c) = self.lstm(x, (h0, c0))
        # # out = self.linear(out[:, -1, :].to(device))
        #
        # h_n = h_n.view(-1, self.hidden_size)
        h_n = self.linear0(x)
        out = self.batch1(h_n)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.batch2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.batch3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear3(out)
        out = self.batch4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear4(out)

        # out = self.relu(out)
        # out = self.linear5(out)
        return out
model = LSTM_Model(x_train.shape[1], 256, 1, len(list(set(data['price']))))
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=1e-5)#
metric = R2Score().to(device)
epochs = 1000
train_acc = []
test_acc = []
train_loss_all = []
test_loss_all = []
for epoch in range(epochs):
    model.train()
    correct = 0
    total_train = 0
    train_loss = []
    cnt_tr = 0
    for i, (factor) in enumerate(train_loader):
        label = factor[:, -1].to(device)
        label = torch.as_tensor(label, dtype=torch.long)
        factor = factor[:, 0:6].to(device)
        # print(factor.shape)
        # print(label.shape)
        # factor = torch.reshape(factor, [factor.size(0), 1, factor.size(-1)]).to(device)
        output = model(factor)
        # print(output.shape)
        loss = criterion(output, label)
        train_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        total_train += label.size(0)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label.to(device)).sum().item()
        # print(output.shape)
        cnt_tr += 1
    print(f'acc_train:{correct / total_train}')
    train_acc.append(correct / total_train)
    train_loss_all.append(np.sum(train_loss) / cnt_tr)

    model.eval()
    correct2 = 0
    total_test = 0
    preds_all = []
    labels_all = []
    test_loss = []
    cnt_te = 0
    with torch.no_grad():
        for i, (factor) in enumerate(test_loader):
            label = factor[:, -1].to(device)
            label = torch.as_tensor(label, dtype=torch.long)
            factor = factor[:, 0:6].to(device)
            # factor = torch.reshape(factor, [factor.size(0), 1, factor.size(-1)]).to(device)
            output = model(factor)
            # print(label.shape)
            # print(output.shape)
            preds_all.append(output)
            labels_all.append(torch.unsqueeze(label, 1))
            loss = criterion(output, label)
            test_loss.append(loss)
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            total_test += label.size(0)
            _, predicted = torch.max(output.data, 1)
            correct2 += (predicted == label.to(device)).sum().item()
            cnt_te += 1
        # preds_all = torch.cat(preds_all)
        # labels_all = torch.cat(labels_all)
        # print(preds_all.shape)
        # print(labels_all.shape)
        # r2 = metric(preds_all, labels_all)
        # print(f'r2:{r2}')
        print(f'acc_test:{correct2 / total_test}')
        test_acc.append(correct2 / total_test)
        test_loss_all.append(np.sum(test_loss) / cnt_te)
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.legend()
plt.show()
plt.clf()
plt.plot(train_loss_all, label='train')
plt.plot(test_loss_all, label='test')
plt.legend()
plt.show()