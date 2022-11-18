import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

dkt = pd.read_csv('/opt/ml/input/data/test_data.csv')

dkt = dkt[['userID','assessmentItemID','testId','Timestamp','KnowledgeTag','answerCode']]

dkt_list = list(dkt['userID'].unique())

answer = []

for qq in dkt_list:
    
    
    ddkt = dkt[dkt['userID']==qq]
    a = int(int(len(ddkt))*0.8)
    b = int(int(len(ddkt))*0.2)
#     print(a,b, a+b, len(ddkt))
    if a+b != len(ddkt):
        a += 1
#     print(a,b, a+b, len(ddkt))
    
    # dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

    dataset_train = ddkt[:a]
    dataset_test = ddkt[a:-1]
#     print(len(dataset_test))
    training_set = dataset_train.iloc[:, -1:].values
#     print(dataset_train)
#     print(training_set)


    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    
    X_train = []
    y_train = []
    c = int(a/10)
    for i in range(c, a):
        X_train.append(training_set_scaled[i-c:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
#     print(c)


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_train.shape


    regressor = Sequential()
    
    
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    
    regressor.add(Dense(units = 1))
    
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    regressor.fit(X_train, y_train, epochs = 20, batch_size = 1024)
    
    # dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

    real_stock_price = dataset_test.iloc[:, -1:].values
    real_stock_price
    
    dataset_total = pd.concat((dataset_train.iloc[:, -1:], dataset_test.iloc[:, -1:]), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - c:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
#     print(inputs)
#     print(len(inputs))
    X_test = []
    for i in range(c, len(inputs)):
        X_test.append(inputs[i-c:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    answer.append(predicted_stock_price[-1])

print(answer)