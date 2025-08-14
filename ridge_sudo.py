# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import pickle
import os
import sys
import math
import time
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

#from correlation import correlation_c

#相関係数の計算
def correlation_c(y_test_T,y_pre_T):
    corr_list = []
    for j in range(y_test_T.shape[0]):
        corr = np.corrcoef(y_test_T[j, :],y_pre_T[j, :])[0,1]
        corr_list.append(corr)
    #ave_corr = np.average(corr_list)
    #print("相関係数: {}".format(corr_list))

    return corr_list

begin = time.time()

r=4#run数
width = 5#時間幅
print("width:",width)
width = width - 1

#alpha = 100.0

#a_list = [0.5,1.0,5.0]

#正則化項
a_list = [0.5,1.0,5.0,10.0,10.0**2,10.0**3,10.0**4,2.5*(10.0**4),5.0*(10.0**4),10.0**5,10.0**6,10.0**7]

alpha1 = (0.5)
alpha2= (1.0)
alpha3 = (5.0)
alpha4 = (10.0)
alpha5 = (10.0**2)
alpha6 = (10.0**3)
alpha7 = (10.0**4)
alpha8 = 2.5*(10.0**4)
alpha9 = 5.0*(10.0**4)
alpha10 = (10.0**5)
alpha11 = (10.0**6)
alpha12 = (10.0**7)

#print("alpha:",alpha)

data2=[]
bert_data = []

#BERT特徴量のロード
with open("../../dvd-prepro/data/pickle_data/Mentalist_sentence.pickle",'rb') as f1:
    data1 = pickle.load(f1)

data1 = data1[:-4,:]
#bert_data = data1[:-4,:]
#print(data1[2])
#print(type(data1[2][0]))

"""
for i in range(len(data1)-width):
    if np.all(data1[i]!=0.)== True and np.all(data1[i+1]!=0.)== True:
        data2.append(data1[i])
        data2.append(data1[i+1])
        #data2.append(data1[i+2])
        #data2.append(data1[i+3])
        #data2.append(data1[i+4])
        data2 = np.array(data2)
        data2 = data2.flatten()
        #print(data2.shape)
        bert_data.append(data2)
        data2=[]
"""

for i in range(len(data1)-width):
    data2.append(data1[i])
    data2.append(data1[i+1])
    data2.append(data1[i+2])
    data2.append(data1[i+3])
    data2.append(data1[i+4])
    data2 = np.array(data2)
    data2 = data2.flatten()
    bert_data.append(data2)
    data2=[]

bert_data = np.array(bert_data)
#bert_data = bert_data.T
#bert_data = bert_data[:-2,:]#なんか微妙に合わない…
print("bert_dataの形状:",bert_data.shape)
#print("word2vec_dataの形状:",bert_data.shape)


#脳データのロード
for i in range(r):
    with open("../data/DM01/Mentalist/width" + str(1) + "/Mentalist_brain_run" + str(i+1)+".pickle",'rb') as f2:
        data = pickle.load(f2)
        #print(data.shape)

    if i+1 == 1:
        brain_data = data[8:,:]
        #print(brain_data.shape)

    #print(data.shape)
    else:
        brain_data = np.vstack([brain_data,data])
        #print(brain_data.shape)

#brain_data = brain_data.T
#brain_data = brain_data[23:-23,:]#なんか微妙に合わない…
print("brain_dataの形状:",brain_data.shape)

#データを分ける
X_train, X_test, y_train, y_test = train_test_split(bert_data, brain_data, train_size=0.75, random_state=0)

print("X_train shape:",X_train.shape)
print("y_train shape:",y_train.shape)
print("X_test shape:",X_test.shape)
print("y_test shape:",y_test.shape)

data_loading_end = time.time()

print("データロード時間:",data_loading_end - begin)
ridge_start = time.time()



# 交差検証
def cross_validate(train_x_all,train_y_all,a_,split_size=5):
  results = [0 for _ in range(train_y_all.shape[1])]
  kf = KFold(n_splits=split_size)
  for train_idx, val_idx in kf.split(train_x_all, train_y_all):
    train_x = train_x_all[train_idx]
    train_y = train_y_all[train_idx]
    val_x = train_x_all[val_idx]
    val_y = train_y_all[val_idx]

    reg = Ridge(alpha=a_).fit(train_x,train_y)
    pre_y = reg.predict(val_x)

    y_val_T = val_y.T
    y_pre_T = pre_y.T

    #print("y_test_T shape:",y_val_T.shape)
    #print("y_pre_T shape:",y_pre_T.shape)

    k_fold_r = correlation_c(y_val_T,y_pre_T)
    results = [x + y for (x, y) in zip(results, k_fold_r)]

  results = map(lambda x : x/5,results)
  results = list(results)
  #print(results)
  return results



best_co = [-100 for _ in range(y_train.shape[1])]
best_parameters = [0 for _ in range(y_train.shape[1])]

for a in a_list:
    print("alpha:",a)
    c = cross_validate(X_train,y_train,a,5)
    print(len(c))
    for i in range(len(c)):
        if c[i] > best_co[i]:
            best_co[i] = c[i]
            best_parameters[i] = a
            #print(best_co[i])
            #print(best_parameters[i])

#print(best_parameters)
print(len(best_parameters))
print(len(best_co))

reg1 = Ridge(alpha=alpha1).fit(X_train, y_train)
reg2 = Ridge(alpha=alpha2).fit(X_train, y_train)
reg3 = Ridge(alpha=alpha3).fit(X_train, y_train)
reg4 = Ridge(alpha=alpha4).fit(X_train, y_train)
reg5 = Ridge(alpha=alpha5).fit(X_train, y_train)
reg6 = Ridge(alpha=alpha6).fit(X_train, y_train)
reg7 = Ridge(alpha=alpha7).fit(X_train, y_train)
reg8 = Ridge(alpha=alpha8).fit(X_train, y_train)
reg9 = Ridge(alpha=alpha9).fit(X_train, y_train)
reg10 = Ridge(alpha=alpha10).fit(X_train, y_train)
reg11 = Ridge(alpha=alpha11).fit(X_train, y_train)
reg12 = Ridge(alpha=alpha12).fit(X_train, y_train)


y_pre1=reg1.predict(X_test)
y_pre2=reg2.predict(X_test)
y_pre3=reg3.predict(X_test)
y_pre4=reg4.predict(X_test)
y_pre5=reg5.predict(X_test)
y_pre6=reg6.predict(X_test)
y_pre7=reg7.predict(X_test)
y_pre8=reg8.predict(X_test)
y_pre9=reg9.predict(X_test)
y_pre10=reg10.predict(X_test)
y_pre11=reg11.predict(X_test)
y_pre12=reg12.predict(X_test)



print('Completed training regression.')

ridge_end = time.time()

print("回帰時間:",ridge_end - ridge_start)



calculation_start = time.time()



y_test_T = y_test.T

y_pre1_T = y_pre1.T
y_pre2_T = y_pre2.T
y_pre3_T = y_pre3.T
y_pre4_T = y_pre4.T
y_pre5_T = y_pre5.T
y_pre6_T = y_pre6.T
y_pre7_T = y_pre7.T
y_pre8_T = y_pre8.T
y_pre9_T = y_pre9.T
y_pre10_T = y_pre10.T
y_pre11_T = y_pre11.T
y_pre12_T = y_pre12.T

counter = 0
y_pred_all = []
corr_list = []

for i in best_parameters:
    if i == alpha1:
        corr = np.corrcoef(y_test_T[counter, :],y_pre1_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha2:
        corr = np.corrcoef(y_test_T[counter, :],y_pre2_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha3:
        corr = np.corrcoef(y_test_T[counter, :],y_pre3_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1

    elif i == alpha4:
        corr = np.corrcoef(y_test_T[counter, :],y_pre4_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha5:
        corr = np.corrcoef(y_test_T[counter, :],y_pre5_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha6:
        corr = np.corrcoef(y_test_T[counter, :],y_pre6_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha7:
        corr = np.corrcoef(y_test_T[counter, :],y_pre7_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha8:
        corr = np.corrcoef(y_test_T[counter, :],y_pre8_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha9:
        corr = np.corrcoef(y_test_T[counter, :],y_pre9_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha10:
        corr = np.corrcoef(y_test_T[counter, :],y_pre10_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha11:
        corr = np.corrcoef(y_test_T[counter, :],y_pre11_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1
    elif i == alpha12:
        corr = np.corrcoef(y_test_T[counter, :],y_pre12_T[counter, :])[0,1]
        corr_list.append(corr)
        counter = counter + 1

corr_list= np.array(corr_list)
print(corr_list)
print(len(corr_list))

calculation_end = time.time()
print("計算時間:",calculation_end - calculation_start)

"""
ridge_cv = RidgeCV(alphas=[alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10,alpha11,alpha12], cv=5)
reg = ridge_cv.fit(X_train, y_train)
print("alpha:",ridge_cv.alpha_)

pred = reg.predict(X_test)

corr_list = correlation_c(y_test.T,pred.T)
corr_list = np.array(corr_list)
print(corr_list.shape)
#print(reg.score(X_train,Y_train))
#print(reg.coef_)
#print(reg.intercept_)
"""

save_start = time.time()

#結果をセーブ
np.save('../src/DM01/corr/Mentalist_correlation',corr_list)
#np.savetxt('../src/DM09/相関係数/GIS2_correlation.txt',correlation)
#np.save('../src/DM01/Y_pred/Mentalist_pred',  y_pre)

save_end = time.time()

print("保存時間:",save_end - save_start)

print("全工程時間:",save_end - begin)
