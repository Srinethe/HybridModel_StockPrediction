# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:20:30 2020

@author: srine
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from xgboost import plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from IPython.display import display
from datetime import datetime, timedelta
import time
import random
from sklearn.metrics import mean_absolute_error
pd.set_option('display.max_columns', 50)

#from pandas import read_csv, set_option
from sklearn.model_selection import train_test_split
batch_size=50

stock_prices = "C:/Users/srine/OneDrive/Desktop/FYP/JPM-XGBoost.csv"
stock_data = pd.read_csv(stock_prices, parse_dates=[0])

merged_dataframe = stock_data[['Date','Open', 'High', 'Adj Close', 'Volume','Low' ,'Close','EPS','PE Ratio']]
pd.options.mode.chained_assignment = None

merged_dataframe['Year'] = pd.DatetimeIndex(merged_dataframe['Date']).year
merged_dataframe['Day'] = pd.DatetimeIndex(merged_dataframe['Date']).month
merged_dataframe['Month'] = pd.DatetimeIndex(merged_dataframe['Date']).day


#Data Preprocessing
merged_dataframe['Adj Factor'] = merged_dataframe['Adj Close'] / merged_dataframe['Close']
merged_dataframe['Open'] = merged_dataframe['Open'] / merged_dataframe['Adj Factor']
merged_dataframe['High'] = merged_dataframe['High'] / merged_dataframe['Adj Factor']
merged_dataframe['Low'] = merged_dataframe['Low'] / merged_dataframe['Adj Factor']
merged_dataframe['Volume'] = merged_dataframe['Volume'] / merged_dataframe['Adj Factor']
merged_dataframe['Adj Close shift'] = merged_dataframe['Adj Close'].shift(-1)
merged_dataframe['Open shift'] = merged_dataframe['Open'].shift(-1)
merged_dataframe['high_diff'] = merged_dataframe['High'] - merged_dataframe['Adj Close shift']
merged_dataframe['low_diff'] = merged_dataframe['Low'] - merged_dataframe['Adj Close shift']
merged_dataframe['close_diff'] = merged_dataframe['Adj Close'] - merged_dataframe['Adj Close shift']
merged_dataframe['open_diff'] = merged_dataframe['Open shift'] - merged_dataframe['Adj Close shift']

# Separate the dataframe for input(X) and output variables(y)
X = merged_dataframe[['Open', 'High', 'Adj Close', 'Volume','Low','Day','Year','Month','Adj Factor','Adj Close shift','Open shift','high_diff','low_diff','close_diff','open_diff','EPS','PE Ratio']]
y = merged_dataframe.loc[:,'Close']

#print(merged_dataframe['Date'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train=np.array(X_train)
y_train=np.array(y_train)
y_test=np.array(y_test)
X_test=np.array(X_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)
X_test = X_test.astype(float)


model = XGBRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)


imp = model.feature_importances_
imp_list = imp.tolist()
list(np.float_(imp_list))

label = ['Open', 'High', 'Adj Close', 'Volume','Low','Day','Year','Month','Adj Factor','Adj Close shift','Open shift','high_diff','low_diff','close_diff','open_diff','EPS','PE Ratio']

i=0
selected=[]
for val in imp_list:
    print(val)
    if(val>0.01):
        selected.append(label[i])
    i=i+1
   
print(selected)

#Kalman Filter Logic

stock_dict = {}

stock_dict['HDFCBANK'] = pd.read_csv('C:/Users/srine/OneDrive/Desktop/FYP/JPM-XGBoost.csv')
feature_var = {'Open':None,'High':None,'Low':None,'Close':None,'Volume':None,
              'Adj Close':None,'adj_factor':None,'Adj Close shift':None,'Open shift':None,'high_diff':None,
              'low_diff':None,'close_diff':None,'open_diff':None,'EPS':None,'PE Ratio':None}

sample = selected

from os import listdir
from copy import deepcopy

def mean_absolute_percentage_error(y_true, y_pred):
    return 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adjust_price_volume(df):   
    df['adj_factor'] = df['Adj Close'] / df['Close']
    df['Open'] = df['Open'] * df['adj_factor']
    df['High'] = df['High'] * df['adj_factor']
    df['Low'] = df['Low'] * df['adj_factor']
    df['Volume'] = df['Volume'] / df['adj_factor']
    
def create_label_column(df):   
    df['Adj Close shift'] = df['Adj Close'].shift(-1)

def create_next_day_open(df):   
    df['Open shift'] = df['Open'].shift(-1)

def create_diffs(df):   
    df['high_diff'] = df['High'] - df['Adj Close shift']
    df['low_diff'] = df['Low'] - df['Adj Close shift']
    df['close_diff'] = df['Adj Close'] - df['Adj Close shift']
    df['open_diff'] = df['Open shift'] - df['Adj Close shift']

def preprocess(dataframe):
    df = deepcopy(dataframe)   
    adjust_price_volume(df)
    create_label_column(df)
    create_next_day_open(df)
    create_diffs(df)   
    df.dropna(inplace=True)
    return df

stock_dict['HDFCBANK'] = preprocess(stock_dict['HDFCBANK'])
print(stock_dict['HDFCBANK'])

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)    
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  

    def update(self, z,r):
        y = z - np.dot(self.H, self.x)      
        S = r + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)  
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)  


kalman_dict = {}

dt = 1.0/60
F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
H = np.array([1, 0, 0]).reshape(1, 3)
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
R = np.array([0.5]).reshape(1, 1)
stock_cur = np.array(stock_dict['HDFCBANK'])
#print(stock_cur)
init_price = stock_cur[0,4]
#print(init_price)
kalman_dict['HDFCBANK'] = init_price, KalmanFilter(F = F, H = H, Q = Q, R = R)

prediction_dict = {}    
stock = stock_dict['HDFCBANK']
stock_cur = np.array(stock_dict['HDFCBANK'])
print(stock[0:126])
kalman = kalman_dict['HDFCBANK'][1]
   
prediction = []
counter = 0
for i, r in stock.iterrows():
    (u,) = stock.index.get_indexer_for([i])
   
    counter += 1  #to keep track of no.of days

    if counter > 126:
        # Variance calculation
        feature_var['Open'] = stock_cur[u-126:u,1].std()**2
        feature_var['High'] = stock_cur[u-126:u,2].std()**2
        feature_var['Low'] = stock_cur[u-126:u,3].std()**2
        feature_var['Close'] = stock_cur[u-126:u,4].std()**2
        feature_var['Volume'] = stock_cur[u-126:u,5].std()**2
        feature_var['Adj Close'] = stock_cur[u-126:u,6].std()**2
        feature_var['adj_factor'] = stock_cur[u-126:u,7].std()**2
        feature_var['Adj Close shift'] = stock_cur[u-126:u,8].std()**2
        feature_var['Open shift'] = stock_cur[u-126:u,9].std()**2
        feature_var['high_diff'] = stock_cur[u-126:u,10].std()**2
        feature_var['low_diff'] = stock_cur[u-126:u,11].std()**2
        feature_var['close_diff'] = stock_cur[u-126:u,12].std()**2
        feature_var['open_diff'] = stock_cur[u-126:u,13].std()**2
        feature_var['EPS']=stock_cur[:,7].std()**2
        feature_var['PE Ratio']=stock_cur[:,8].std()**2

        kalman.predict()
        for i in range(len(sample)):
            kalman.update(r[sample[i]],feature_var[sample[i]])
        
        prediction.append(kalman.x[0])

stock = np.array(stock)
#1886,1500
measurements = stock[1600:2000,1] #1783 125:1925,1  1500:1886,1
prediction = np.array(prediction)
prediction_test = prediction[1470:1870] # 0:1800 1470:1870 1370:1756
test_data = stock[1784:2000,4]      #1784:2000,4

plt.title('HDFC Bank')
plt.plot(range(len(measurements)), measurements, label = 'Ground Truth')
plt.plot(range(len(prediction_test)), np.array(prediction_test), label = 'Kalman Filter Prediction')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()

accuracy = mean_absolute_percentage_error(measurements, prediction_test)
print('Accuracy = '+str(accuracy) + ' %')

x = int(input("Enter no.of days : "))
print('Actual Price : ',measurements[x])
print('Predicted Price : ',float(prediction_test[x]))