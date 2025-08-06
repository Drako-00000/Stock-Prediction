# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Download stock data
start = '2024-12-31'
end = '2025-08-01'
stock = 'GOOG'
data = yf.download(stock, start, end)

# Moving averages (Visualization)
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

plt.figure(figsize=(10,6))
plt.plot(data['Close'], label='Original Price', color='green')
plt.plot(ma_100_days, label='100-day MA', color='red')
plt.plot(ma_200_days, label='200-day MA', color='blue')
plt.title('Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Drop NA values (from rolling)
data.dropna(inplace=True)

# Split into training and testing data
data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scaled = scaler.fit_transform(data_train)

# Prepare training sequences
x_train = []
y_train = []
for i in range(100, len(data_train_scaled)):
    x_train.append(data_train_scaled[i-100:i])
    y_train.append(data_train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape input to be 3D [samples, time steps, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

# Prepare test data (with previous 100 days for continuity)
past_100_days = data_train.tail(100)
final_test_data = pd.concat([past_100_days, data_test], ignore_index=True)
final_test_scaled = scaler.transform(final_test_data)

# Create test sequences
x_test = []
y_test = []
for i in range(100, len(final_test_scaled)):
    x_test.append(final_test_scaled[i-100:i])
    y_test.append(final_test_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Predictions
y_predicted = model.predict(x_test)

# Inverse transform to get actual prices
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(10,8))
plt.plot(y_test, color='green', label='Actual Price')
plt.plot(y_predicted, color='red', label='Predicted Price')
plt.title('Stock Price Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

#Save the model
model.save('Stock Prediction Model.keras')
