# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Override Yahoo Finance
yf.pdr_override()

# Streamlit UI
st.title('📈 Stock Closing Price Prediction using LSTM')
user_input = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL, MSFT)', 'GOOGL')

start = "2009-01-01"
end = "2023-01-01"

# Fetch data
try:
    df = pdr.get_data_yahoo(user_input, start, end)
except Exception as e:
    st.error(f"Error fetching data for {user_input}: {e}")
    st.stop()

st.subheader('Dated from 1st Jan, 2009 to 1st Jan, 2023')
st.write(df.describe())

# Plot 1: Closing Price
st.subheader('Closing Price vs Time')
fig1 = plt.figure(figsize=(12,6))
plt.plot(df.Close, label='Closing Price')
plt.legend()
st.pyplot(fig1)

# Moving averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Plot 2: 100-Day Moving Average
st.subheader('Closing Price with 100-Day Moving Average')
fig2 = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'r', label='Closing Price')
plt.plot(ma100, 'g', label='100-Day MA')
plt.legend()
st.pyplot(fig2)

# Plot 3: 100-Day & 200-Day MA
st.subheader('Closing Price with 100 & 200-Day Moving Average')
fig3 = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'r', label='Closing Price')
plt.plot(ma100, 'g', label='100-Day MA')
plt.plot(ma200, 'b', label='200-Day MA')
plt.legend()
st.pyplot(fig3)

# Split data
train_df = pd.DataFrame(df['Close'][0:int(len(df)*0.85)])
test_df = pd.DataFrame(df['Close'][int(len(df)*0.85):])

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
train_df_arr = scaler.fit_transform(train_df)

# Load model
try:
    model = load_model('keras_model.h5')
except Exception as e:
    st.error(f"Model file not found or invalid: {e}")
    st.stop()

# Prepare test data
past_100_days = train_df.tail(100)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)

input_data = scaler.fit_transform(final_df)
x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_pred = model.predict(x_test)
scale = scaler.scale_
scale_factor = 1 / scale[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Plot final comparison
st.subheader('Predicted vs Original Closing Price')
fig4 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.legend()
st.pyplot(fig4)

st.success("✅ Prediction complete!")
