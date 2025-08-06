import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

model = load_model("Stock Prediction Model.keras")

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')

start = '2012-01-01'
end = '2024-12-31'

@st.cache_data
def load_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

try:
    data = load_stock_data(stock, start, end)
    if data.empty:
        st.error("Failed to load stock data. Yahoo Finance may have rate-limited the request. Please try again later.")
        st.stop()
except Exception as e:
    st.error(f"Error downloading stock data: {e}")
    st.stop()

st.subheader('Raw Stock Data')
st.write(data)

data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full)

st.subheader('Price vs 50-day MA')
ma50 = data['Close'].rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma50, 'r', label='MA50')
plt.plot(data['Close'], 'g', label='Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma100 = data['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma50, 'r', label='MA50')
plt.plot(ma100, 'b', label='MA100')
plt.plot(data['Close'], 'g', label='Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma200 = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma100, 'r', label='MA100')
plt.plot(ma200, 'b', label='MA200')
plt.plot(data['Close'], 'g', label='Price')
plt.legend()
st.pyplot(fig3)

x_test = []
y_test = []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

predictions = model.predict(x_test)

scale = 1 / scaler.scale_[0]
predictions = predictions * scale
y_test = y_test * scale

st.subheader('Actual Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y_test, 'r', label='Actual Price')
plt.plot(predictions, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
