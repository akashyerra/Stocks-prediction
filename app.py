from flask import Flask, render_template, request
import requests
import os

# Import the libraries
import math
import datetime as dt
import pandas_datareader as web

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
plt.style.use('fivethirtyeight')
from models import GOOG_model

# ============================================Imported from Prediction==========================

scaler = MinMaxScaler(feature_range=(0,1))

def GOOG_model():
    print("THIS IS UNDERGOING TRAINING OF GOOG MODEL")
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
    # model = Sequential()
    # model.add(LSTM(30, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    # model.add(LSTM(30, return_sequences=False))
    # model.add(Dense(10))
    # model.add(Dense(1))

    model = Sequential()
    model.add(LSTM(30, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Add dropout layer with a dropout rate of 0.2
    model.add(LSTM(30, return_sequences=False))
    model.add(Dropout(0.2))  # Add another dropout layer with a dropout rate of 0.2
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

def AAPL_model():
    print("THIS IS UNDERGOING TRAINING OF AAPL MODEL")
    end_date = dt.datetime.now()
    ten_years_ago = end_date - dt.timedelta(days=365 * 10)
    df = web.DataReader(name='AAPL', data_source = 'stooq', start=ten_years_ago, end=end_date)
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
















GOOG = AAPL_model()
AAPL = MSFT = SNX = TSLA = BTUSD = UBER = ETHEUR = GOOG
# AAPL = AAPL_model()
# MSFT = AAPL_model()
# SNX = AAPL_model()
# TSLA = AAPL_model()
# BTUSD = AAPL_model()
# UBER = AAPL_model()
# ETHEUR = AAPL_model()


models = {
    "GOOG": GOOG,
    "AAPL": AAPL,
    "MSFT": MSFT,
    "SNX": SNX,
    "TSLA": TSLA,
    "BTUSD": BTUSD,
    "UBER": UBER,
    "ETHEUR": ETHEUR,
}

def predict_value(days, stock_name):
    #Get the quote
    new_end_date = dt.datetime.now()
    ten_years_ago = new_end_date - dt.timedelta(days=365 * 10)
    google_quote = web.DataReader(stock_name, 'stooq', start=ten_years_ago, end=new_end_date)

    #Create a new dataframe
    new_df = google_quote.filter(['Close'])
    new_df = new_df.iloc[::-1]

    temp1_arr = new_df[:]
    # print(temp1_arr)
    #Get the last 60day Closing price values and convert the dataframe to an array


    pred_df = []

    pred_df = new_df[:].values

    future_df = []

    # print(pred_df)
    pred_price = 0
    last_day = 0
    n=days
    
    predicted_sentence = ''
    # scaler = MinMaxScaler(feature_range=(0,1))
    # scaler = GOOG_model.scaler

    for i in range(1, n):
        last_60_days = pred_df[-60:]
        last_day = pred_df[-1:]
        #Scale the data to be values between 0 and 1
        last_60_days_scaled = scaler.transform(last_60_days)
        #Create an empty list
        X_test = []
        #Append the past 60days
        X_test.append(last_60_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #Get the predicted scaled price
        pred_price = models[stock_name].predict(X_test)
        #Undo the scaling
        pred_price = scaler.inverse_transform(pred_price)

        # print("Last Day Predicted Price was: " + str(last_day))
        # print("Next Day Predicted Price is: " + str(pred_price))


        pred_df = pred_df.tolist()
        pred_df.append(pred_price)
        future_df.append(pred_price)
        pred_df=np.array(pred_df, dtype=object)


    last_day = new_df[-1:].values
    if(pred_price>last_day):
        # print("The Price of GOOG stocks is RAISING for the next " +str(n)+"days")
        predicted_sentence = "The Price of " + stock_name + " stocks is RAISING for the next " +str(n)+"days"
    else:
        # print("The Price of GOOG stocks is DECLAINING for the next "+str(n)+ "days")
        predicted_sentence = "The Price of " + stock_name + " stocks is DECLINING for the next " +str(n)+"days"

    # Create a new plot
    plt.figure(figsize=(32,8))
    plt.title('Stock Price Prediction', fontsize=18, fontweight='light')  # Decrease font weight
    plt.xlabel('Date', fontsize=18, fontweight='light')  # Decrease font weight
    plt.ylabel('Close Price USD ($)', fontsize=18)


    # print(new_df)
    # plotting_data = new_df
    # plotting_data = new_df['Close'].values + future_df
    # print(plotting_data)
    # plt.plot(plotting_data)

    # Plot the historical data (current values) in blue
    plt.plot(new_df.index, new_df['Close'], label='Actual', color='blue')


    # Plot the predicted values for each day in red
    for i in range(1, n):
    # Plot the predicted value for the corresponding date
        plt.scatter(new_end_date + pd.DateOffset(days=i), future_df[i - 1], color='red', label='Predicted')
    #  plotting_data +=

    # Add a legend
    plt.legend(['Actual', 'Predicted'], loc='upper left')

    # Show the plot

    plt.savefig('static/predicted_graph.png', dpi=300)  # You can change the file format and resolution as needed

    plt.show()


    return predicted_sentence
#  ========================================close===============================================
# AAPL = AAPL_model()

app = Flask(__name__)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # Load the graph and HTML template here
    if request.method == "GET":
        return render_template("graph.html", graph_available=False)

    if request.method == "POST":
        days = request.form.get("days")
        stock_name = request.form.get("stock")
        output = predict_value((int)(days), stock_name)
        return render_template("graph.html", graph_available=True, Output=output)


@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html", graph_available=False)

if __name__ == "__main__":
    app.run()