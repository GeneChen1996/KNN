# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections, numpy
from itertools import combinations

#-----讀取資料
dataset_dataframe = pd.read_csv('C:/Users/Gene/Desktop/程式/iris_data.csv')
dataset_array=dataset_dataframe.values
RawData = dataset_array[:,0:4]
label = dataset_array[:,4]

Knn = 3
error = 0
accuracy= 0 
feature = ['花萼長度', '花萼寬度', '花瓣長度', '花瓣寬度'] #4種特徵
feature_combinations =  np.array(list(combinations(feature,2))) #4種特徵兩兩不重複組合


# -----Scatter Plot 繪出特徵1與特徵2的訓練集數據分布
plt.style.use("ggplot")     # 使用ggplot主題樣式
plt.xlabel('Sepal_length', fontweight = "bold")                  #設定x座標標題及粗體
plt.ylabel('Sepal_width', fontweight = "bold")   #設定y座標標題及粗體
plt.title('Sepal_length & Sepal_width',
          fontsize = 15, fontweight = "bold")        #設定標題、字大小及粗體

plt.scatter(RawData[0:25,0],     # x軸資料
            RawData[0:25,1],     # y軸資料
            c = "m",                                  # 點顏色
            s = 50,                                   # 點大小
            alpha = 0.5,                               # 透明度
            marker = "s")                             # 點樣式

plt.scatter(RawData[50:75,0],     # x軸資料
            RawData[50:75,1],     # y軸資料
            c = "r",                                  # 點顏色
            s = 50,                                   # 點大小
            alpha = 0.5,                               # 透明度
            marker = "s")                             # 點樣式

plt.scatter(RawData[100:125,0],     # x軸資料
            RawData[100:125,1],     # y軸資料
            c = "b",                                  # 點顏色
            s = 50,                                   # 點大小
            alpha = 0.5,                               # 透明度
            marker = "s")                             # 點樣式

#-----計算測試集與訓練集的歐式距離
trainset = np.vstack((RawData[0:25,0:1],RawData[50:75,0:1],RawData[100:125,0:1]))   #將每一類別的前25筆資料當作訓練集
testset = np.vstack((RawData[25:50,0:1],RawData[75:100,0:1],RawData[125:150,0:1]))  #將每一類別的後25筆資料當作測試集
trainset_row = np.size(trainset,0)
trainset_column = np.size(trainset,1)
testset_row = np.size(testset,0)
testset_column = np.size(testset,1)
distanceVector = np.zeros((testset_row,1))
distanceVectorw = 0

for i in range(testset_row):
    for j in range(trainset_row):
        distanceVector[j]=0;
        for k in range(trainset_column):
             distanceVector[j]=distanceVector[j]+(testset[i,k]-trainset[j,k])**2
        distanceVector[j]=np.sqrt(distanceVector[j])
        
        # distanceVector = distanceVector[:,0]
        # distanceVector_argsort = np.argsort(distanceVector)
        # distanceVector_sort = np.sort(distanceVector)
        
        # M = np.argmax(label(distanceVector_argsort[1:K]))
    distanceVectorw = distanceVector[:,0]
    distanceVector_argsort = np.argsort(distanceVectorw)
    # print(label[distanceVector_argsort[0:K]])
    knn = label[distanceVector_argsort[0:Knn]]
    # print(knn)
    knn = collections.Counter(knn)
    print("---------------------")
    print(knn)
    print("預測結果"+knn.most_common(1)[0][0])
    print("實際標籤"+label[j])
    if str(knn.most_common(1)[0][0]) != str(label[j]):
        error += 1
    accuracy = (len(label)-error)/len(label)
print("")
print("準確率 = "+ str(accuracy))
    
    
        
    
