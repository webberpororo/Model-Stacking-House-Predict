import os
import re
import numpy as np
import time
import json
from collections import Counter
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_predict
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchmetrics import R2Score
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn import metrics, neighbors
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, plot_importance, XGBRegressor, XGBRFRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# np.set_printoptions(threshold=np.inf)
# import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gzip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv("D:\downloads\House_analysis_data_cv.csv")
print(data)
# print(data.info())
# data = data.dropna()
a = list(set(data['年份']))
# b = list(set(data['總坪數']))
# c = sorted(list(set(data['價格'])))
# d = list(set(data['屋齡']))
# e = list(set(data['主建築物佔比']))
# f = list(set(data['用途']))
# g = list(set(data['房子類型']))
# h = list(set(data['車位']))
i = list(set(data['區域']))
# j = list(set(data['電梯']))
# k = list(set(data['縣市']))
# m = list(set(data['縣市編號']))
# n = list(set(data['區域編號']))

# o = list(set(data['地址']))
# p = list(set(data['售出時間']))
q = list(set(data['層']))
print(q)
# r = list(set(data['警衛']))
# print(r)
# s = list(set(data['價格/坪']))
# print(s)
t = list(set(data['格局']))
print(t)
# w = list(set(data['車位價格']))
# print(w)
# x = list(set(data['備註']))
# print(x)

print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)
# print(g)
# print(h)
print(i)
print(len(i))
# print(j)
# print(k)
# print(m)
# print(n)
datass = data
# places = []
# not_enough = 0
# for place in i:
#     datas = data[data['區域'] == place]
#     # print(len(datas))
#     places.append(len(datas))
#     if len(datas) < 1000:
#         not_enough += 1
# print(sorted(places))
# print(sum(places))
# print(not_enough)

# price_list = []
meta_train_list = []
meta_test_list = []
y_train_list = []
y_test_list = []

regressor_RF = RandomForestRegressor(n_estimators=400,
                                      n_jobs=-1,
                                      criterion='squared_error',
                                      max_depth=None,
                                      min_samples_split=2,
                                      max_features='sqrt',
                                      min_samples_leaf=5,
                                      random_state = 42,
                                      bootstrap=True,
                                      oob_score=True)

regressor_KNN = neighbors.KNeighborsRegressor(n_neighbors=18,
                                                 weights='distance',
                                                 algorithm='auto',
                                                 leaf_size=30,
                                                 p=2,
                                                 metric='euclidean',
                                                 metric_params=None,
                                                 n_jobs=None)

# regressor_XGB = CatBoostRegressor(
#     iterations=1000,
#     learning_rate=0.05,
#     depth=6,
#     loss_function='RMSE',
#     eval_metric='RMSE',
#     random_seed=42,
#
# # 過擬合控制
#     l2_leaf_reg=3,
#     random_strength=1.0,
#     bootstrap_type='Bayesian',
#     bagging_temperature=1.0,
#
# # 類別特徵處理
#     one_hot_max_size=10,
#
# # Early stopping
#     use_best_model=True,
#     od_type='Iter',
#     od_wait=50,
#
# # 計算資源
#     thread_count=4,
#     verbose=100
# )

regressor_XGB = LGBMRegressor(
    learning_rate=0.1,
    n_estimators=600,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# regressor_XGB = XGBRegressor(learning_rate = 0.01,
#                           n_estimators = 400,
#                           max_depth = 8,
#                           min_child_weight = 7,
#                           gamma = 0.1,
#                           subsample = 0.8,
#                           colsample_bytree = 0.8,
#                           scale_pos_weight = 1,
#                           random_state = 42,
#                           verbosity = 0,
#                           booster = 'gbtree',
#                           n_jobs = 1,
#                           max_delta_step = 0,
#                           colsample_bylevel = 0.8,
#                           colsample_bynode = 1,
#                           base_score = 0.5)



# lin_reg = LinearRegression()
#
# lin_reg2 = LinearRegression()
#
for i in range(109, 113):
    # datas = data[data['年份'] == i]
    # print(len(datas))
    # price_list.append(len(datas))
    # standard = sorted(price_list)[0]
    # print(standard)

    data = datass[datass['年份'] == i]
    # data = data[data['縣市'] == 'Kaohsiung']
    data = data[data['價格'] > 1000000]
    data = data[data['價格'] < 60000001]
    data['價格'] = data['價格'].map(lambda x: x / 1000000)
    # data = data.sample(frac=1).reset_index(drop=True)
    data = data.reset_index(drop=True)

    # data = data.drop(columns=['屋齡'])

    data = data.drop(columns=['地址'])
    data = data.drop(columns=['電梯'])
    data = data.drop(columns=['車位'])
    data = data.drop(columns=['車位價格']) #
    # data = data.dropna(axis = 'index', how = 'all', subset = ['路/街'])
    data = data.dropna(axis='index', how='all', subset=['路/街CV值'])
    # data = data.drop(columns=['路/街CV值'])
    year_list = data['年份']
    print('year', year_list)

    # data = data.drop(columns=['區域編號'])
    # data = data.drop(columns=['縣市編號'])
    # data = data.drop(columns=['主建築物佔比'])
    # data = data.drop(columns=['屋齡'])
    # data['房子類型'] = data['房子類型'].replace({'華廈(10層含以下有電梯)' : 1, '店面（店舖)' : 2, '透天厝' : 3, '公寓(5樓含以下無電梯)' : 4, '住宅大樓(11層含以上有電梯)' : 5, '農舍' : 6, '工廠' : 7, '其他' : 8, '倉庫' : 9, '廠辦' : 10, '辦公商業大樓' : 11, '套房(1房(1廳)1衛)' : 12})

    # data['價格'] = data['價格'].map(lambda x: x // 500000 * 500000 + 250000)
    # data['總坪數'] = data['總坪數'].map(lambda x: x // 5 * 5 + 2.5)
    # data['屋齡'] = data['屋齡'].map(lambda x: x // 5 * 5 + 2.5)

    # data = data.drop(index = [172039, 162573, 151864, 108245, 86547, 128575, 69246, 15802, 56671, 67008, 89453, 10587,  69858,  79217,  80148, 143341, 155063], axis = 0)
    # data = data.reset_index(drop=True)
    print(data)
    # print(len(np.where(data['價格'] == 6750000.0)[0].tolist()))
    # drop_list = []
    # for i in list(set(data['價格'])):
    #     # print(np.where(data['價格'] == i))
    #     print(len(np.where(data['價格'] == i)[0].tolist()))
    #     if len(np.where(data['價格'] == i)[0].tolist()) < 10:
    #         for j in np.where(data['價格'] == i)[0].tolist():
    #             # print(j)
    #             drop_list.append(j)
    # data = data.drop(index = drop_list, axis = 0)
    # data = data.reset_index(drop=True)
    print(len(list(set(data['價格']))))

    # price_list = []
    # for i in range(len(data['價格'])):
    #     if data['價格'][i] not in price_list:
    #         price_list.append(data['價格'][i])
    #     # print(price_list)
    #     data['價格'][i] = int(np.where(price_list == data['價格'].iloc[i])[0])
    # print(data['價格'])
    # # print(price_list)
    num_classes = len(list(set(data['價格'])))
    # print(len(list(set(data['價格']))))
    # print(list(set(data['價格'])))

    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_features = data[['區域', '縣市', '房子類型']] # '格局', '路/街', '用途', , '警衛'
    # categorical_features = ohe.fit_transform(categorical_features)
    data = data.drop(columns=['售出時間'])
    data = data.drop(columns=['用途'])
    data = data.drop(columns=['區域'])
    data = data.drop(columns=['縣市'])
    data = data.drop(columns=['房子類型'])
    data = data.drop(columns=['警衛'])
    data = data.drop(columns=['層'])
    data = data.drop(columns=['價格/坪']) #
    data = data.drop(columns=['格局']) #
    data = data.drop(columns=['備註']) #
    data = data.drop(columns=['路/街'])  #
    # categorical_features = pd.get_dummies(data, columns=categorical_features)
    # print(categorical_features)
    categorical_features = categorical_features.apply(le.fit_transform)
    data = np.column_stack((data, categorical_features))
    data = pd.DataFrame(data)

    data = data.sample(frac=1).reset_index(drop=True)
    data = data.reset_index(drop=True)
    # data = data.rename(columns = {'年份':'year', '總坪數':'size', '屋齡':'age', '房子類型':'type', '車位':'park', '電梯':'elevator', '縣市':'city', '區域':'town', '價格':'price', '用途':'ways'})

    # corr = data.corr("pearson")
    # fig, ax = plt.subplots(figsize = (15, 15))
    # sns.heatmap(corr, cmap = 'RdBu', annot = True)
    # plt.show()
    print(data)

    y = data[2] # 2 '價格'
    data = data.drop(columns=[2])
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    data = pd.DataFrame(data)

    # data = np.random.shuffle(data)
    # y = data['price']
    # data = data.drop(columns=['price'])
    # y = data[2]
    # data = data.drop(columns=[2])
    x = data
    print(x)
    print(y)


    # print(len(dataset))
    # x = dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # mean, std = np.array(x_train).mean(), np.array(x_train).std()
    # x_train = (np.array(x_train) - mean) / std
    # x_test = (np.array(x_test) - mean) / std
    x_train = np.column_stack((x_train, year_list[:x_train.shape[0]]))
    x_test = np.column_stack((x_test, year_list[:x_test.shape[0]]))
    scaler = StandardScaler()
    x_trains = scaler.fit_transform(x_train)
    x_tests = scaler.transform(x_test)

    x_trains = np.array(x_trains, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int)
    x_tests = np.array(x_tests, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int)
    print("x_train mean:", np.mean(x_trains), "std:", np.std(x_trains))
    print("x_test mean:", np.mean(x_tests), "std:", np.std(x_tests))

    regressor_RF.fit(x_trains, y_train)
    y_pred = regressor_RF.predict(x_tests)
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R2:', r2_score(y_test, y_pred))

    regressor_KNN.fit(x_trains, y_train)
    y_pred2 = regressor_KNN.predict(x_tests)
    print('MAE:', mean_absolute_error(y_test, y_pred2))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred2)))
    print('R2:', r2_score(y_test, y_pred2))

    regressor_XGB.fit(x_trains, y_train)
    y_pred3 = regressor_XGB.predict(x_tests)
    print('MAE:', mean_absolute_error(y_test, y_pred3))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred3)))
    print('R2:', r2_score(y_test, y_pred3))

    # lin_reg.fit(x_trains, y_train)
    # y_pred4 = lin_reg.predict(x_tests)
    # print('MAE:', mean_absolute_error(y_test, y_pred4))
    # print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred4)))
    # print('R2:', r2_score(y_test, y_pred4))

    # # poly_reg = PolynomialFeatures(degree=13)
    # # x_poly = poly_reg.fit_transform(x_trains)
    # lin_reg.fit(x_trains, y_train)
    # y_pred4 = lin_reg.predict(x_tests)
    # print('MAE:', mean_absolute_error(y_test, y_pred4))
    # print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred4)))
    # print('R2:', r2_score(y_test, y_pred4))

    y_train_2 = cross_val_predict(regressor_RF, x_trains, y_train, cv=5, method='predict')
    y_train_3 = cross_val_predict(regressor_KNN, x_trains, y_train, cv=5, method='predict')
    y_train_4 = cross_val_predict(regressor_XGB, x_trains, y_train, cv=5, method='predict')
    # y_train_5 = cross_val_predict(lin_reg, x_trains, y_train, cv=5, method='predict')
    print(y_train_4)
    print(y_train_4.shape)

    y_test_2 = regressor_RF.predict(x_tests)
    y_test_3 = regressor_KNN.predict(x_tests)
    y_test_4 = regressor_XGB.predict(x_tests)
    # y_test_5 = lin_reg.predict(x_tests)
    print(y_test_4)
    print(y_test_4.shape)

    # ###
    # scaler2 = StandardScaler()
    #
    # meta_test = np.column_stack((y_test_2, y_test_3, y_test_4, x_test))  # 將測試集的預測結果拼接
    # meta_train = np.column_stack((y_train_2, y_train_3, y_train_4, x_train))
    # # y_train = np.hstack((y_train, y_train, y_train))
    # # y_test = np.hstack((y_test, y_test, y_test))
    # print(meta_train.shape)
    # print(meta_test.shape)
    # meta_trains = scaler2.fit_transform(meta_train)
    # meta_tests = scaler2.fit_transform(meta_test)
    #
    # regressor_RF.fit(meta_trains, y_train)
    # y_pred = regressor_RF.predict(meta_tests)
    # print('MAE:', mean_absolute_error(y_test, y_pred))
    # print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    # print('R2:', r2_score(y_test, y_pred))
    #
    # regressor_KNN.fit(meta_trains, y_train)
    # y_pred2 = regressor_KNN.predict(meta_tests)
    # print('MAE:', mean_absolute_error(y_test, y_pred2))
    # print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred2)))
    # print('R2:', r2_score(y_test, y_pred2))
    #
    # regressor_XGB.fit(meta_trains, y_train)
    # y_pred3 = regressor_XGB.predict(meta_tests)
    # print('MAE:', mean_absolute_error(y_test, y_pred3))
    # print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred3)))
    # print('R2:', r2_score(y_test, y_pred3))
    # ###

    # y_train_2 = y_train_2 - y_train
    # y_train_3 = y_train_3 - y_train
    # y_train_4 = y_train_4 - y_train
    #
    # y_test_2 = y_test_2 - y_test
    # y_test_3 = y_test_3 - y_test
    # y_test_4 = y_test_4 - y_test

    scaler2 = StandardScaler()

    meta_test = np.column_stack((y_test_2, y_test_3, y_test_4, x_test))  # 將測試集的預測結果拼接
    meta_train = np.column_stack((y_train_2, y_train_3, y_train_4, x_train))
    # y_train = np.hstack((y_train, y_train, y_train))
    # y_test = np.hstack((y_test, y_test, y_test))
    print(meta_train.shape)
    print(meta_test.shape)
    # meta_train = meta_train.reshape(-1, 1)
    # meta_test = meta_test.reshape(-1, 1)
    meta_trains = scaler2.fit_transform(meta_train)
    meta_tests = scaler2.transform(meta_test)

    regressor_RF.fit(meta_trains, y_train)
    y_pred = regressor_RF.predict(meta_tests)
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R2:', r2_score(y_test, y_pred))

    regressor_KNN.fit(meta_trains, y_train)
    y_pred2 = regressor_KNN.predict(meta_tests)
    print('MAE:', mean_absolute_error(y_test, y_pred2))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred2)))
    print('R2:', r2_score(y_test, y_pred2))

    regressor_XGB.fit(meta_trains, y_train)
    y_pred3 = regressor_XGB.predict(meta_tests)
    print('MAE:', mean_absolute_error(y_test, y_pred3))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred3)))
    print('R2:', r2_score(y_test, y_pred3))

    # # poly_reg = PolynomialFeatures(degree=5)
    # # x_poly = poly_reg.fit_transform(meta_train)
    # lin_reg.fit(meta_trains, y_train)
    # y_pred4 = lin_reg.predict(meta_tests)
    # print('MAE:', mean_absolute_error(y_test, y_pred4))
    # print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred4)))
    # print('R2:', r2_score(y_test, y_pred4))

    # poly_reg = PolynomialFeatures(degree=3)
    # x_poly = poly_reg.fit_transform(meta_train)
    # lin_reg2.fit(x_poly, y_train)
    # x_poly_test = poly_reg.transform(meta_tests)
    # y_pred5 = lin_reg2.predict(x_poly_test)
    # print('MAE:', mean_absolute_error(y_test, y_pred5))
    # print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred5)))
    # print('R2:', r2_score(y_test, y_pred5))

    meta_train_list.append(meta_train)
    meta_test_list.append(meta_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)

meta_train = np.vstack([meta_train_list[t] for t in range(len(meta_train_list))])
meta_test = np.vstack([meta_test_list[t] for t in range(len(meta_test_list))])
scaler3 = StandardScaler()
meta_trains = scaler3.fit_transform(meta_train)
meta_tests = scaler3.transform(meta_test)
print(meta_train.shape)
print(np.array(y_train_list).shape)
y_train = np.hstack([y_train_list[t] for t in range(len(y_train_list))])
y_test = np.hstack([y_test_list[t] for t in range(len(y_test_list))])
y_train = np.transpose(y_train)
y_test = np.transpose(y_test)
print(y_train.shape)

# train_data = np.vstack((meta_trains, meta_tests))
# test_data = np.hstack((y_train, y_test))
# meta_trains, meta_tests, y_train, y_test = train_test_split(train_data, test_data, test_size=0.2, random_state=42)

train_datas = np.column_stack((meta_trains, y_train))
test_datas = np.column_stack((meta_tests, y_test))
train_datas = np.random.permutation(train_datas)
test_datas = np.random.permutation(test_datas)
print(train_datas)
y_train = train_datas[:, -1]
meta_trains = train_datas[:, 0:train_datas.shape[1] - 1]
y_test = test_datas[:, -1]
meta_tests = test_datas[:, 0:train_datas.shape[1] - 1]

RF_start = time.time()
regressor_RF.fit(meta_trains, y_train)
y_pred = regressor_RF.predict(meta_tests)
print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))
RF_end = time.time()

with gzip.GzipFile("RF_pre_model.pgz", 'wb') as f:
    pickle.dump(regressor_RF, f)

KNN_start = time.time()
regressor_KNN.fit(meta_trains, y_train)
y_pred2 = regressor_KNN.predict(meta_tests)
print('MAE:', mean_absolute_error(y_test, y_pred2))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred2)))
print('R2:', r2_score(y_test, y_pred2))
KNN_end = time.time()

with gzip.GzipFile("KNN_pre_model.pgz", 'wb') as f:
    pickle.dump(regressor_KNN, f)

GBM_start = time.time()
regressor_XGB.fit(meta_trains, y_train)
y_pred3 = regressor_XGB.predict(meta_tests)
print('MAE:', mean_absolute_error(y_test, y_pred3))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred3)))
print('R2:', r2_score(y_test, y_pred3))
GBM_end = time.time()

with gzip.GzipFile("GBM_pre_model.pgz", 'wb') as f:
    pickle.dump(regressor_XGB, f)

# lin_reg.fit(meta_trains, y_train)
# y_pred4 = lin_reg.predict(meta_tests)
# print('MAE:', mean_absolute_error(y_test, y_pred4))
# print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred4)))
# print('R2:', r2_score(y_test, y_pred4))

# shuffle
# train_data = np.vstack((meta_trains, meta_tests))
# test_data = np.hstack((y_train, y_test))
# meta_trains, meta_tests, y_train, y_test = train_test_split(train_data, test_data, test_size=0.2, random_state=42)

# meta_trains = np.delete(meta_trains, 17, axis = 1)
# meta_tests = np.delete(meta_tests, 17, axis = 1)

temp_data = np.column_stack((meta_trains, y_train))
temp_data = pd.DataFrame(data = temp_data)
corr = temp_data.corr("pearson")
fig, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(corr, cmap = 'RdBu', annot = True)
plt.show()

meta_train = torch.tensor(meta_trains, dtype=torch.float).to(device)
meta_test = torch.tensor(meta_tests, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

class Meta_Classifier(nn.Module):
    def __init__(self, input_shape, num_layers, classes = 64):
        super(Meta_Classifier, self).__init__()
        self.fc0 = nn.Linear(in_features=input_shape, out_features=2048)
        self.batch1 = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.batch2 = nn.BatchNorm1d(1024)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=input_shape, out_features=512)
        self.batch3 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.batch4 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.batch5 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(in_features=128, out_features=64)
        self.batch6 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(in_features=64, out_features=1)
        self.drop = nn.Dropout(0.2)
        # self.num_layers = num_layers

    def forward(self, x):
        # x = self.fc0(x)
        # x = self.batch1(x)
        # x = self.act(x)
        # x = self.drop(x)
        # x = self.fc1(x)
        # x = self.batch2(x)
        # x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.batch3(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.batch4(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc4(x)
        x = self.batch5(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc5(x)
        x = self.batch6(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc6(x)

        return x

batch_size = 512
model2 = Meta_Classifier(input_shape=meta_train.shape[1], num_layers=2).to(device)
epochs = 20
optimizer = optim.AdamW(model2.parameters(), lr = 0.00001, weight_decay=1e-5)
# criterion = nn.CrossEntropyLoss()
train_loss = []
test_loss = []
r2_list = []
criterion = nn.MSELoss()
metric = R2Score().to(device)
MLP_start = time.time()
model2.train()
for epoch in range(epochs):
    for i in range(len(meta_train) // batch_size):
        # outputs = model2(meta_train[i * batch_size:(i + 1) * batch_size])
        outputs = model2(meta_train[i * batch_size:(i + 1) * batch_size]).flatten()
        # print(type(y_train[i * batch_size:(i + 1) * batch_size]))
        # print(type(outputs.to(dtype = torch.int64).argmax(dim=1).cpu().numpy()))
        # outputs = int()
        # print('acc', metrics.accuracy_score(y_train[i * batch_size:(i + 1) * batch_size].detach().cpu().numpy(), (outputs.argmax(dim=1).cpu().numpy())))
        # loss = criterion(outputs, y_train[i * batch_size:(i + 1) * batch_size + 1])

        # log_probs = F.log_softmax(outputs, dim=1)  # 手動應用 log-softmax
        # loss = F.nll_loss(log_probs, y_train[i * batch_size:(i + 1) * batch_size], reduction='none')
        # loss = torch.mean(loss)
        # print(outputs.shape)
        # print(y_train[i * batch_size:(i + 1) * batch_size].shape)
        # print(outputs.shape)
        # print(y_train[i * batch_size:(i + 1) * batch_size].shape)
        loss = criterion(outputs, y_train[i * batch_size:(i + 1) * batch_size]).flatten()
        # print(loss)
        # r2 = metric(outputs, y_train[i * batch_size:(i + 1) * batch_size])
        # print(f'train r2:{r2}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if(epoch + 1) % 1 == 0:

        # print('loss', loss)
    train_loss.append(loss)
    print(train_loss[-1])

    model2.eval()
    with torch.no_grad():
        # print(meta_test.shape)
        # output = model2(meta_test[0].unsqueeze(0)).argmax(dim=1).cpu().numpy()
        # y_true = [y_test[0].item()]
        # print('acc', metrics.accuracy_score(y_true, output))

        # output = model2(meta_test).argmax(dim=1).cpu().numpy()
        # print('acc2', metrics.accuracy_score(y_test, output))

        output = model2(meta_test).flatten()
        # output = model2(meta_test)
        loss = criterion(output, y_test.flatten())
        print('loss', loss)
        test_loss.append(loss)
        r2 = metric(output, y_test)
        r2_list.append(r2)
        print(f'r2:{r2}')

MLP_end = time.time()
# plt.plot(train_loss, label='train')
# plt.plot(test_loss, label='test')
plt.plot(r2_list[0:-1], label = 'r2')
plt.legend()
plt.show()

torch.save(model2.state_dict(), "House_MLP.pt")

print("RF time:", RF_end - RF_start)
print("KNN time:", KNN_end - KNN_start)
print("GBM time:", GBM_end - GBM_start)
print("MLP time:", MLP_end - MLP_start)

# # future predict
# # meta_train_list = []
# # meta_test_list = []
# # y_train_list = []
# # y_test_list = []
# for i in range(113, 114):
#     data = datass[datass['年份'] == i]
#     # data = data[data['縣市'] == 'Kaohsiung']
#     data = data[data['價格'] > 1000000]
#     data = data[data['價格'] < 60000001]
#     data['價格'] = data['價格'].map(lambda x: x / 1000000)
#     # data = data.sample(frac=1).reset_index(drop=True)
#     data = data.reset_index(drop=True)
#
#     # data = data.drop(columns=['屋齡'])
#
#     data = data.drop(columns=['地址'])
#     data = data.drop(columns=['電梯'])
#     data = data.drop(columns=['車位'])
#     data = data.drop(columns=['車位價格'])  #
#     # data = data.dropna(axis = 'index', how = 'all', subset = ['路/街'])
#     data = data.dropna(axis='index', how='all', subset=['路/街CV值'])
#     # data = data.drop(columns=['路/街CV值'])
#     year_list = data['年份']
#     print('year', year_list)
#
#     le = LabelEncoder()
#     ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
#     categorical_features = data[['區域', '縣市', '房子類型']]  # '格局', '路/街', '用途', , '警衛'
#     # categorical_features = ohe.fit_transform(categorical_features)
#     data = data.drop(columns=['售出時間'])
#     data = data.drop(columns=['用途'])
#     data = data.drop(columns=['區域'])
#     data = data.drop(columns=['縣市'])
#     data = data.drop(columns=['房子類型'])
#     data = data.drop(columns=['警衛'])
#     data = data.drop(columns=['層'])
#     data = data.drop(columns=['價格/坪'])  #
#     data = data.drop(columns=['格局'])  #
#     data = data.drop(columns=['備註'])  #
#     data = data.drop(columns=['路/街'])  #
#     # categorical_features = pd.get_dummies(data, columns=categorical_features)
#     # print(categorical_features)
#     categorical_features = categorical_features.apply(le.fit_transform)
#     data = np.column_stack((data, categorical_features))
#     data = pd.DataFrame(data)
#
#     data = data.sample(frac=1).reset_index(drop=True)
#     data = data.reset_index(drop=True)
#     # data = data.rename(columns = {'年份':'year', '總坪數':'size', '屋齡':'age', '房子類型':'type', '車位':'park', '電梯':'elevator', '縣市':'city', '區域':'town', '價格':'price', '用途':'ways'})
#
#     # corr = data.corr("pearson")
#     # fig, ax = plt.subplots(figsize = (15, 15))
#     # sns.heatmap(corr, cmap = 'RdBu', annot = True)
#     # plt.show()
#     print(data)
#
#     y = data[2]  # 2 '價格'
#     data = data.drop(columns=[2])
#     # scaler = StandardScaler()
#     # data = scaler.fit_transform(data)
#     data = pd.DataFrame(data)
#     x = data
#
#     print(x)
#     print(y)
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
#     x_train = np.column_stack((x_train, year_list[:x_train.shape[0]]))
#     x_test = np.column_stack((x_test, year_list[:x_test.shape[0]]))
#     scaler = StandardScaler()
#     x_trains = scaler.fit_transform(x_train)
#     x_tests = scaler.transform(x_test)
#
#     x_trains = np.array(x_trains, dtype=np.float32)
#     y_train = np.array(y_train, dtype=np.int)
#     x_tests = np.array(x_tests, dtype=np.float32)
#     y_test = np.array(y_test, dtype=np.int)
#     print("x_train mean:", np.mean(x_trains), "std:", np.std(x_trains))
#     print("x_test mean:", np.mean(x_tests), "std:", np.std(x_tests))
#
#     regressor_RF.fit(x_trains, y_train)
#     y_pred = regressor_RF.predict(x_tests)
#     print('MAE:', mean_absolute_error(y_test, y_pred))
#     print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
#     print('R2:', r2_score(y_test, y_pred))
#
#     regressor_KNN.fit(x_trains, y_train)
#     y_pred2 = regressor_KNN.predict(x_tests)
#     print('MAE:', mean_absolute_error(y_test, y_pred2))
#     print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred2)))
#     print('R2:', r2_score(y_test, y_pred2))
#
#     regressor_XGB.fit(x_trains, y_train)
#     y_pred3 = regressor_XGB.predict(x_tests)
#     print('MAE:', mean_absolute_error(y_test, y_pred3))
#     print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred3)))
#     print('R2:', r2_score(y_test, y_pred3))
#
#     y_train_2 = cross_val_predict(regressor_RF, x_trains, y_train, cv=5, method='predict')
#     y_train_3 = cross_val_predict(regressor_KNN, x_trains, y_train, cv=5, method='predict')
#     y_train_4 = cross_val_predict(regressor_XGB, x_trains, y_train, cv=5, method='predict')
#     # y_train_5 = cross_val_predict(lin_reg, x_trains, y_train, cv=5, method='predict')
#     print(y_train_4)
#     print(y_train_4.shape)
#
#     y_test_2 = regressor_RF.predict(x_tests)
#     y_test_3 = regressor_KNN.predict(x_tests)
#     y_test_4 = regressor_XGB.predict(x_tests)
#
#     scaler2 = StandardScaler()
#
#     meta_test = np.column_stack((y_test_2, y_test_3, y_test_4, x_test))  # 將測試集的預測結果拼接
#     meta_train = np.column_stack((y_train_2, y_train_3, y_train_4, x_train))
#     # y_train = np.hstack((y_train, y_train, y_train))
#     # y_test = np.hstack((y_test, y_test, y_test))
#     print(meta_train.shape)
#     print(meta_test.shape)
#     # meta_train = meta_train.reshape(-1, 1)
#     # meta_test = meta_test.reshape(-1, 1)
#     meta_trains = scaler2.fit_transform(meta_train)
#     meta_tests = scaler2.transform(meta_test)
#
#     regressor_RF.fit(meta_trains, y_train)
#     y_pred = regressor_RF.predict(meta_tests)
#     print('MAE:', mean_absolute_error(y_test, y_pred))
#     print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
#     print('R2:', r2_score(y_test, y_pred))
#
#     regressor_KNN.fit(meta_trains, y_train)
#     y_pred2 = regressor_KNN.predict(meta_tests)
#     print('MAE:', mean_absolute_error(y_test, y_pred2))
#     print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred2)))
#     print('R2:', r2_score(y_test, y_pred2))
#
#     regressor_XGB.fit(meta_trains, y_train)
#     y_pred3 = regressor_XGB.predict(meta_tests)
#     print('MAE:', mean_absolute_error(y_test, y_pred3))
#     print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred3)))
#     print('R2:', r2_score(y_test, y_pred3))
#
#     # poly_reg = PolynomialFeatures(degree=5)
#     # x_poly = poly_reg.fit_transform(meta_train)
#     # lin_reg.fit(meta_trains, y_train)
#     # y_pred4 = lin_reg.predict(meta_tests)
#     # print('MAE:', mean_absolute_error(y_test, y_pred4))
#     # print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred4)))
#     # print('R2:', r2_score(y_test, y_pred4))
#
#     meta_train_list.append(meta_train)
#     meta_test_list.append(meta_test)
#     y_train_list.append(y_train)
#     y_test_list.append(y_test)
#
# meta_train = np.vstack([meta_train_list[t] for t in range(len(meta_train_list))])
# meta_test = np.vstack([meta_test_list[t] for t in range(len(meta_test_list))])
# scaler3 = StandardScaler()
# meta_trains = scaler3.fit_transform(meta_train)
# meta_tests = scaler3.fit_transform(meta_test)
# print(meta_train.shape)
# print(np.array(y_train_list).shape)
# y_train = np.hstack([y_train_list[t] for t in range(len(y_train_list))])
# y_test = np.hstack([y_test_list[t] for t in range(len(y_test_list))])
# y_train = np.transpose(y_train)
# y_test = np.transpose(y_test)
# print(y_train.shape)
#
# train_datas = np.column_stack((meta_trains, y_train))
# test_datas = np.column_stack((meta_tests, y_test))
# train_datas = np.random.permutation(train_datas)
# test_datas = np.random.permutation(test_datas)
# print(train_datas)
# y_train = train_datas[:, -1]
# meta_trains = train_datas[:, 0:train_datas.shape[1] - 1]
# y_test = test_datas[:, -1]
# meta_tests = test_datas[:, 0:train_datas.shape[1] - 1]
#
# with gzip.open("D:\pycharmprojectnew\RF_pre_model.pgz", 'r') as f:
#     regressor_RF_input = pickle.load(f)
# # regressor_RF.fit(meta_trains, y_train)
# y_pred = regressor_RF_input.predict(meta_tests)
# print('MAE:', mean_absolute_error(y_test, y_pred))
# print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
# print('R2:', r2_score(y_test, y_pred))
#
# with gzip.open("D:\pycharmprojectnew\KNN_pre_model.pgz", 'r') as f:
#     regressor_KNN_input = pickle.load(f)
# # regressor_KNN.fit(meta_trains, y_train)
# y_pred2 = regressor_KNN_input.predict(meta_tests)
# print('MAE:', mean_absolute_error(y_test, y_pred2))
# print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred2)))
# print('R2:', r2_score(y_test, y_pred2))
#
# with gzip.open("D:\pycharmprojectnew\GBM_pre_model.pgz", 'r') as f:
#     regressor_GBM_input = pickle.load(f)
# # regressor_XGB.fit(meta_trains, y_train)
# y_pred3 = regressor_GBM_input.predict(meta_tests)
# print('MAE:', mean_absolute_error(y_test, y_pred3))
# print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred3)))
# print('R2:', r2_score(y_test, y_pred3))
#
# class Meta_Classifier2(nn.Module):
#     def __init__(self, input_shape, num_layers, classes = 64):
#         super(Meta_Classifier2, self).__init__()
#         self.fc0 = nn.Linear(in_features=input_shape, out_features=2048)
#         self.batch1 = nn.BatchNorm1d(2048)
#         self.fc1 = nn.Linear(in_features=2048, out_features=1024)
#         self.batch2 = nn.BatchNorm1d(1024)
#         self.act = nn.ReLU()
#         self.fc2 = nn.Linear(in_features=input_shape, out_features=512)
#         self.batch3 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(in_features=512, out_features=256)
#         self.batch4 = nn.BatchNorm1d(256)
#         self.fc4 = nn.Linear(in_features=256, out_features=128)
#         self.batch5 = nn.BatchNorm1d(128)
#         self.fc5 = nn.Linear(in_features=128, out_features=64)
#         self.batch6 = nn.BatchNorm1d(64)
#         self.fc6 = nn.Linear(in_features=64, out_features=1)
#         self.drop = nn.Dropout(0.2)
#         # self.num_layers = num_layers
#
#     def forward(self, x):
#         # x = self.fc0(x)
#         # x = self.batch1(x)
#         # x = self.act(x)
#         # x = self.drop(x)
#         # x = self.fc1(x)
#         # x = self.batch2(x)
#         # x = self.act(x)
#         # x = self.drop(x)
#         x = self.fc2(x)
#         x = self.batch3(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         x = self.batch4(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc4(x)
#         x = self.batch5(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc5(x)
#         x = self.batch6(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc6(x)
#
#         return x
#
# # meta_trains = np.delete(meta_trains, 14, axis = 1)
# # meta_tests = np.delete(meta_tests, 14, axis = 1)
#
# meta_train = torch.tensor(meta_trains, dtype=torch.float).to(device)
# meta_test = torch.tensor(meta_tests, dtype=torch.float).to(device)
# y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
# y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
#
# model2 = Meta_Classifier2(input_shape=meta_train.shape[1], num_layers=2).to(device)
# model2.load_state_dict(torch.load("D:\pycharmprojectnew\House_MLP.pt"))
# test_loss = []
# r2_list = []
# criterion = nn.MSELoss()
# metric = R2Score().to(device)
#
# model2.eval()
# with torch.no_grad():
#     output = model2(meta_test).flatten()
#     loss = criterion(output, y_test.flatten())
#     print('loss', loss)
#     test_loss.append(loss)
#     r2 = metric(output, y_test)
#     r2_list.append(r2)
#     print(f'r2:{r2}')
