import math
import datetime
import pandas as pd
import yfinance as yf
import numpy as np
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import streamlit as st
import time

# Download stock data
def download_stock_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    return df

# Main function
def main():
    st.title('Stock Price Prediction')

    # Sidebar
    st.sidebar.title('Options')
    start_date = st.sidebar.date_input('Start Date', datetime.date(2005, 1, 13))
    end_date = st.sidebar.date_input('End Date', datetime.date(2023, 5, 30))
    stock_symbol = st.sidebar.text_input('Stock Symbol', 'AAPL')

    # Download data
    @st.experimental_singleton
    def download_data():
        return download_stock_data(stock_symbol, start_date, end_date)

    df = download_data()

    # Plot closing price history
    st.subheader('Close Price History')
    plt.figure(figsize=(16, 8))
    plt.plot(df['Close'], alpha=1, color='deeppink')
    plt.title('Close Price History', fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Close Price USD($)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    st.pyplot()

    # Preprocess data
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Training the model
    training_data_len = math.ceil(len(dataset) * 0.8)
    train_data = scaled_data[0:training_data_len]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Predictions
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(predictions - dataset[training_data_len:, 0]) ** 2)

    # Plotting
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    st.subheader('Model Prediction')
    plt.figure(figsize=(16, 8))
    plt.plot(train['Close'], label='Train')
    plt.plot(valid[['Close', 'Predictions']])
    plt.title('Model Prediction', fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Close Price USD($)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(['Train', 'Validation', 'Predictions'], loc='upper left')
    st.pyplot()

    st.write(f'Root Mean Squared Error (RMSE): {rmse}')

    # Predict next day's price
    st.subheader('Predict Next Day Price')

    last_60_days = scaled_data[-60:]
    x_test = np.array([last_60_days])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred_price = model.predict(x_test)
    pred_price = scaler.inverse_transform(pred_price)

    st.write(f'Predicted Price for Next Day: {pred_price[0][0]}')

    # Rerun the app every second to update the data
    st.experimental_rerun()

if __name__ == '__main__':
    main()
