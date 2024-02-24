# Import the libraries
import math
import datetime as dt
import pandas_datareader as web

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')

scaler = MinMaxScaler(feature_range=(0,1))

def GOOG_model():
    end_date = dt.datetime.now()
    ten_years_ago = end_date - dt.timedelta(days=365 * 10)
    df = web.DataReader(name='GOOG', data_source = 'stooq', start=ten_years_ago, end=end_date)
    df = df.iloc[::-1]

    #Get the number of rows and the columns in the dataset
    df.shape

    print(df)

    #Visualize the closing price history
    # plt.figure(figsize=(24,8))
    # plt.title('Close Price History')
    # plt.plot(df['Close'])
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Close Price USD ($)', fontsize=18)
    # plt.show()

    #Create a new dataframe with only the 'Close column'
    data = df.filter(['Close'])
    #Convert the dataframe to a numpy array
    dataset = data.values
    #Get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * .8)

    training_data_len

    #Scale the data
    scaled_data = scaler.fit_transform(dataset)

    scaled_data

    #Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    #Split the data into X_train and Y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<=61:
            print(x_train)
            print(y_train)
            print()

    #Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build the LSTM model
    model = Sequential()
    model.add(LSTM(30, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(30, return_sequences=False))
    model.add(Dense(10))
    model.add(Dense(1))

    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    #Create the testing data set
    #Create a new array containing scaled values from index 3517 to 4471
    test_data = scaled_data[training_data_len-60: , :]
    #Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len: , :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    #Convert the data to a numpy array
    x_test = np.array(x_test)

    #Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    #Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean( predictions - y_test )**2)
    rmse

    #Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #Visualize the data
    plt.figure(figsize=(32,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    # plt.plot(train[['Close']])
    # plt.plot(valid[['Predictions']]) # Only shows predictions
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    #Show the valid and predicted prices
    valid

    return model
