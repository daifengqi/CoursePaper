#!/usr/bin/env python
# coding: utf-8

# In[37]:


import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import backend as K  # Keras解决OOM超内存问题
from keras.layers.noise import GaussianDropout
# SSwitchable Normalization
from switchnorm import SwitchNormalization


# SVR
import numpy as np
from sklearn.svm import SVR
# BP
from sklearn.neural_network import MLPRegressor
# RNN
from keras.layers.recurrent import SimpleRNN
# GRU
from keras.layers import GRU
# LSTM
from keras.layers import LSTM, Dense

np.random.seed(1)


# In[48]:


class model():
    
    def __init__(self, dataset = None, look_back = None, train_ratio = 0.85, datestart = None, epochs = 100,
                 batch_size = 100, isdropout = False, plt_title = 'Please Add the Title'):
    # 通用参数
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.datestart = datestart
        self.plt_title = plt_title
        # 神经网络参数
        self.epochs = epochs
        self.batch_size = batch_size
        self.look_back = look_back
        self.isdropout = isdropout
        self.feature_num = dataset.shape[1]
        # 划分
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
    
    # 根据窗宽lookback交叉对应
    def create_dataset(self, dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0:dataset.shape[1]]
            dataX.append(a)
            # Y放在第一列
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    
    # 数据集切分：普通
    def split_dataset(self):
        train_size = int(len(self.dataset) * self.train_ratio)
        train_data = self.dataset[0:train_size, :]
        test_data = self.dataset[train_size - self.look_back - 1:len(self.dataset), :]

        # Data set detail/具体分割后数据集
        self.x_train, self.y_train = self.create_dataset(train_data, self.look_back)
        self.x_test, self.y_test = self.create_dataset(test_data, self.look_back)
        
        # Reshape
        self.x_train = np.array(self.x_train)[:,0,:]
        self.x_test = np.array(self.x_test)[:,0,:]
    
    # 数据集切分：深度学习
    def split_dataset_rnn(self):
        train_size = int(len(self.dataset) * self.train_ratio)
        train_data = self.dataset[0:train_size, :]
        test_data = self.dataset[train_size - self.look_back - 1:len(self.dataset), :]

        # Data set detail/具体分割后数据集
        x_train, self.y_train = self.create_dataset(train_data, self.look_back)
        x_test, self.y_test = self.create_dataset(test_data, self.look_back)

        # Reshape input to be [samples, feature_num, features]/整理特征数据的格式
        self.x_train = np.reshape(x_train, (x_train.shape[0], self.feature_num, x_train.shape[1]))
        self.x_test = np.reshape(x_test, (x_test.shape[0], self.feature_num, x_test.shape[1]))
    
    # 模型一：Support Vector Regression
    def svr(self):
        start_cr_a_fit_net = time.time()
        self.split_dataset()
        svr = SVR(kernel='linear', gamma=0.1)
        svr.fit(self.x_train, self.y_train)
        end_cr_a_fit_net = time.time() - start_cr_a_fit_net
        print('Running time of creating and fitting the SVR model: %.2f Seconds' % (end_cr_a_fit_net))
        # SVR进行预测
        trainPredict = svr.predict(self.x_train)
        testPredict = svr.predict(self.x_test)
        return trainPredict, testPredict, self.y_train, self.y_test
    
    # 模型二：Back Propagation Network
    def bp_network(self):
        start_cr_a_fit_net = time.time()
        self.split_dataset()
        mlp = MLPRegressor(hidden_layer_sizes=(15,10), max_iter=500)
        mlp.fit(self.x_train, self.y_train)
        end_cr_a_fit_net = time.time() - start_cr_a_fit_net
        print('Running time of creating and fitting the BP network model: %.2f Seconds' % (end_cr_a_fit_net))
        # SVR进行预测
        trainPredict = mlp.predict(self.x_train)
        testPredict = mlp.predict(self.x_test)
        return trainPredict, testPredict, self.y_train, self.y_test  
    
    # 模型三：Recurrent Neural Network
    def rnn(self):
        start_cr_a_fit_net = time.time()
        self.split_dataset_rnn()

        rnn_model = Sequential()
        
        # RNN层设计
        rnn_model.add(SimpleRNN(15, input_shape=(None, self.look_back), return_sequences=True))
        rnn_model.add(SimpleRNN(10, input_shape=(None, self.look_back), return_sequences=True))
        # SN层
        if self.isdropout:
            rnn_model.add(SwitchNormalization(axis=-1))
        rnn_model.add(SimpleRNN(15, input_shape=(None, self.look_back), return_sequences=True))
        rnn_model.add(SimpleRNN(10, input_shape=(None, self.look_back)))
        rnn_model.add(Dense(1))
        # dropout层
        if self.isdropout:
            rnn_model.add(GaussianDropout(0.2))
            
        rnn_model.summary()
        rnn_model.compile(loss='mean_squared_error', optimizer='adam')
        rnn_model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        end_cr_a_fit_net = time.time() - start_cr_a_fit_net
        print('Running time of creating and fitting the RNN network: %.2f Seconds' % (end_cr_a_fit_net))

        # LSTM prediction/LSTM进行预测
        trainPredict = rnn_model.predict(self.x_train)  # Predict by training data set/训练集预测
        testPredict = rnn_model.predict(self.x_test)  # Predict by test data set/测试集预测
        return trainPredict, testPredict, self.y_train, self.y_test
    
    # 模型三：Long Short Term Memory
    def lstm(self):
        start_cr_a_fit_net = time.time()
        self.split_dataset_rnn()

        rnn_model = Sequential()
        
        # LSTM层设计
        rnn_model.add(LSTM(15, input_shape=(None, self.look_back), return_sequences=True))
        rnn_model.add(LSTM(10, input_shape=(None, self.look_back), return_sequences=True))
        # SN层
        if self.isdropout:
            rnn_model.add(SwitchNormalization(axis=-1))
        rnn_model.add(LSTM(15, input_shape=(None, self.look_back), return_sequences=True))
        rnn_model.add(LSTM(10, input_shape=(None, self.look_back)))
        rnn_model.add(Dense(1))
        # dropout层
        if self.isdropout:
            rnn_model.add(GaussianDropout(0.2))
            
        rnn_model.summary()
        rnn_model.compile(loss='mean_squared_error', optimizer='adam')
        rnn_model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        end_cr_a_fit_net = time.time() - start_cr_a_fit_net
        print('Running time of creating and fitting the LSTM network: %.2f Seconds' % (end_cr_a_fit_net))

        # LSTM prediction/LSTM进行预测
        trainPredict = rnn_model.predict(self.x_train)  # Predict by training data set/训练集预测
        testPredict = rnn_model.predict(self.x_test)  # Predict by test data set/测试集预测
        return trainPredict, testPredict, self.y_train, self.y_test
    
    # 模型四：Gated Recurrent Unit
    def gru(self):
        start_cr_a_fit_net = time.time()
        self.split_dataset_rnn()

        rnn_model = Sequential()
        
        # GRU层设计
        rnn_model.add(GRU(15, input_shape=(None, self.look_back), return_sequences=True))
        rnn_model.add(GRU(10, input_shape=(None, self.look_back), return_sequences=True))
        # SN层
        if self.isdropout:
            rnn_model.add(SwitchNormalization(axis=-1))
        rnn_model.add(GRU(15, input_shape=(None, self.look_back), return_sequences=True))
        rnn_model.add(GRU(10, input_shape=(None, self.look_back)))
        rnn_model.add(Dense(1))
        # dropout层
        if self.isdropout:
            rnn_model.add(GaussianDropout(0.2))
            
        rnn_model.summary()
        rnn_model.compile(loss='mean_squared_error', optimizer='adam')
        rnn_model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        end_cr_a_fit_net = time.time() - start_cr_a_fit_net
        print('Running time of creating and fitting the GRU network: %.2f Seconds' % (end_cr_a_fit_net))

        # LSTM prediction/LSTM进行预测
        trainPredict = rnn_model.predict(self.x_train)  # Predict by training data set/训练集预测
        testPredict = rnn_model.predict(self.x_test)  # Predict by test data set/测试集预测
        return trainPredict, testPredict, self.y_train, self.y_test
    
    # 结果评估
    def mape(self, scaler, trainPredict, testPredict):
        # 将预测值转换为正常数值
        # 创建一个空的数组, 结构同dataset
        trainPredict_dataset_like = np.zeros(shape=(len(trainPredict), self.dataset.shape[1]))
        testPredict_dataset_like = np.zeros(shape=(len(testPredict), self.dataset.shape[1]))
        # 将预测值填充进新建数组
        try:
            trainPredict_dataset_like[:, 0] = trainPredict[:, 0]
            testPredict_dataset_like[:, 0] = testPredict[:, 0]
        except:
            trainPredict_dataset_like[:, 0] = trainPredict
            testPredict_dataset_like[:, 0] = testPredict       
        # 数据转换
        trainPredict = scaler.inverse_transform(trainPredict_dataset_like)[:, 0]
        testPredict = scaler.inverse_transform(testPredict_dataset_like)[:, 0]
        y_train_dataset_like = np.zeros(shape=(len(self.y_train), self.dataset.shape[1]))
        
        y_train_dataset_like[:, 0] = self.y_train
        self.y_train = scaler.inverse_transform(y_train_dataset_like)[:, 0]
        y_test_dataset_like = np.zeros(shape=(len(self.y_test), self.dataset.shape[1]))
        y_test_dataset_like[:, 0] = self.y_test
        self.y_test = scaler.inverse_transform(y_test_dataset_like)[:, 0]
        # Invert predictions end/数据转换结束

        # Calculate root mean squared error and MAPE/计算RMSE和误差率MAPE
        train_RMSE = math.sqrt(mean_squared_error(self.y_train, trainPredict))
        test_RMSE = math.sqrt(mean_squared_error(self.y_test, testPredict))
        trainMAPE = np.mean(np.abs(self.y_train - trainPredict) / self.y_train)
        testMAPE = np.mean(np.abs(self.y_test - testPredict) / self.y_test)

        print("Train RMSE: " + str(round(train_RMSE, 2)) + '  ' + "Train MAPE: " + str(round(trainMAPE * 100, 2)))
        print("Test RMSE: " + str(round(test_RMSE, 2)) + '  ' + "Test MAPE: " + str(round(testMAPE * 100, 2)))
        return trainMAPE, testMAPE, trainPredict, testPredict
    
    # 结果可视化
    def plot(self, scaler, trainPredict, testPredict):
        # 转换数据结构用于作图-训练预测结果
        sub_traindataset = [[data] for data in self.dataset[:, 0]]
        trainPredictPlot = np.empty_like(sub_traindataset)
        trainPredictPlot[:, 0] = np.nan
        trainPredictPlot[self.look_back:len(trainPredict) + self.look_back, 0] = trainPredict

        # 转换数据结构用于作图-测试预测结果
        sub_testdataset = [[data] for data in self.dataset[:, 0]]
        testPredictPlot = np.empty_like(sub_testdataset)
        testPredictPlot[:] = np.nan
        testPredictPlot[len(trainPredict) + self.look_back - 1:len(self.dataset), 0] = testPredict

        # 作图
        datasety_like = np.zeros(shape=(self.dataset.shape[0], self.dataset.shape[1]))
        datasety_like[:, 0] = self.dataset[:, 0]
        y = scaler.inverse_transform(datasety_like)[:, 0]
        dates = pd.date_range(self.datestart[0], periods=len(y), freq=self.datestart[1])
        xs = [datetime.strptime(str(d)[0:7], '%Y-%m').date() for d in dates]

        A, = plt.plot(xs, y[0:len(y)], linewidth='2', color='red')  # 真实值
        B, = plt.plot(xs, trainPredictPlot, linewidth='1.5', color='lightblue')  # LSTM训练集结果
        C, = plt.plot(xs, testPredictPlot, linewidth='1.5', color='blue')  # LSTM测试集结果

        train_size = int(len(self.dataset) * self.train_ratio)
        plt.axvline(xs[train_size], linewidth='2', color='black')  # 画直线区分训练部分与测试部分
        plt.legend((A, B, C), ('real_value', 'train_data', 'test_data'), loc='best')
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记

        plt.xlabel('Date', family='Times New Roman', fontsize=16)  # X轴
        plt.ylabel('CS', family='Times New Roman', fontsize=16)  # Y轴
        plt.title(self.plt_title, family='Times New Roman', fontsize=16)  # 添加标题

        plt.show()
        del trainPredictPlot, testPredictPlot


# In[ ]:


# JANET


# In[47]:


# test
# 导入数据集
file = r'USdata.xlsx'
dataframe = pd.read_excel(file, sheet_name=0, header=0, index_col=None)
dataset = dataframe.iloc[:, [18, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
dataset = dataset.fillna(method="ffill").values
dataset = dataset.astype('float32')

# 标准化数据集
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 设定超参数
look_back = 7
epochs = 20
batch_size = 100
train_ratio = 0.7
datestart = ['2005-01-03', 'D']


Model = model(dataset = dataset, look_back = look_back, train_ratio = train_ratio, datestart = datestart, epochs = epochs,
              batch_size = batch_size, isdropout = True, plt_title = 'RNN model')

trainPredict, testPredict, y_train, y_test = Model.rnn()
trainMAPE, testMAPE, trainPredict, testPredict = Model.mape(scaler, trainPredict, testPredict)
Model.plot(scaler, trainPredict, testPredict)

# 关掉内存中神经网络
K.clear_session()

