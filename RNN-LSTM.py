# Recurrent neural network


# part 1 - Data preprocessing
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_train.csv')
training_set = dataset_train.iloc[: , 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
 
# creating a data structure with 60 timesteps and 1 output 
X_train = []
y_train = []
for i in range(60, 1258) :
    X_train.append(training_set_scaled[i-60:i , 0])
    y_train.append(training_set_scaled[i , 0])
X_train , y_train = np.array(X_train) , np.array(y_train)


# reshaping
X_train = np.reshape(X_train , (X_train.shape[0] , X_train.shape[1] , 1))    


# part 2 - Building a RNN
# importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initializing the RNN
regressor = Sequential()

# adding first LSTM layer and some dropout reguralization
regressor.add(LSTM(units = 50 , return_sequences = True , input_shape = (X_train.shape[1] , 1)))
regressor.add(Dropout(0.2))

# adding second LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50 , return_sequences = True))
regressor.add(Dropout(0.2))


# adding third LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50 , return_sequences = True))
regressor.add(Dropout(0.2))


# adding fourth LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# adding the output layer
regressor.add(Dense(units = 1))


# compiling the RNN
regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error')


# fitting the RNN to the training set
regressor.fit(X_train , y_train , epochs = 100 , batch_size = 32)


# part 3 - Makking the predictions and visualizing the data

# getting real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price  = dataset_test.iloc[: , 1:2].values


# getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'] , dataset_test['Open']) , axis =0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60 : ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80) :
    X_test.append(inputs[i-60:i , 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test , (X_test.shape[0] , X_test.shape[1] , 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualizing the data
plt.plot(real_stock_price , color = 'red' , label = 'real google stock price')
plt.plot(predicted_stock_price , color = 'blue' , label = 'predicted google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))


print("this is just a test line added")