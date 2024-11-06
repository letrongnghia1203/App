import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load LSTM model and data only once
@st.cache_resource
def load_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@st.cache_resource
def load_data():
    gdown.download("https://drive.google.com/uc?export=download&id=1x1CkrJRe6PTOdWouYLhqG3f8MEXP-kbl", "VN2023-data-Ticker.csv", quiet=False)
    gdown.download("https://drive.google.com/uc?export=download&id=1M9GA96Zhoj9HzqMPIlfnMeK7pob1bv2z", "VN2023-data-Info.csv", quiet=False)
    df_ticker = pd.read_csv("VN2023-data-Ticker.csv", low_memory=False)
    df_info = pd.read_csv("VN2023-data-Info.csv", low_memory=False)
    df_info.columns = [col.replace('.', '_') for col in df_info.columns]
    df_joined = pd.merge(df_info, df_ticker, on="Name", how="inner")
    ticker_date_columns = [col for col in df_ticker.columns if '/' in col]

    df_vietnam = df_joined.melt(
        id_vars=list(df_info.columns) + ["Code"],
        value_vars=ticker_date_columns,
        var_name="Ngày", value_name="Giá đóng cửa"
    )
    df_vietnam["Symbol"] = df_vietnam["Symbol"].str.replace("^VT:", "", regex=True)
    df_vietnam["Ngày"] = pd.to_datetime(df_vietnam["Ngày"], format='%m/%d/%Y', errors='coerce')
    df_vietnam["Giá đóng cửa"] = pd.to_numeric(df_vietnam["Giá đóng cửa"].str.replace(',', '.'), errors='coerce')
    return df_vietnam.dropna(subset=["Giá đóng cửa"])

# Cache data and model in session state
if 'df_vietnam' not in st.session_state:
    st.session_state.df_vietnam = load_data()
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = load_lstm_model()

# Stock Price Prediction Section
st.title("Stock Market Data Visualization with Extended LSTM Predictions")
symbol_price = st.text_input("Enter stock symbol for extended price prediction:")

if symbol_price:
    df_filtered = st.session_state.df_vietnam[st.session_state.df_vietnam['Symbol'] == symbol_price.upper()]

    if not df_filtered.empty:
        st.write(f"Historical data for stock symbol {symbol_price.upper()}:")
        st.write(df_filtered.head(1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['Ngày'], y=df_filtered['Giá đóng cửa'], mode='lines+markers', name='Historical Prices'))

        # Prepare and scale data for prediction
        prices = df_filtered[['Giá đóng cửa']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # Create sequences for LSTM
        seq_length = 60
        X = np.array([prices_scaled[i:i + seq_length] for i in range(len(prices_scaled) - seq_length)])
        y = prices_scaled[seq_length:]

        # Train the LSTM model
        X = np.expand_dims(X, axis=2)  # Add feature dimension
        st.session_state.lstm_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)

        # Dự đoán giá trong khoảng thời gian hiện có
        predictions = st.session_state.lstm_model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        prediction_dates = df_filtered['Ngày'].iloc[seq_length:].values
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='LSTM Predictions'))

        # Dự đoán cho 1 năm tiếp theo
        future_prices = prices_scaled[-seq_length:]  # Lấy đoạn chuỗi cuối cùng để bắt đầu dự đoán
        future_predictions = []

        for _ in range(252):  # Giả sử có 252 ngày giao dịch trong một năm
            # Đảm bảo rằng future_prices có kích thước đúng (1, seq_length, 1)
            future_seq = np.expand_dims(future_prices, axis=0)  # (seq_length,) -> (1, seq_length)
            future_seq = future_seq.reshape(1, seq_length, 1)   # Đảm bảo kích thước (1, seq_length, 1)
            
            # Predict the next price
            next_price = st.session_state.lstm_model.predict(future_seq)
            future_predictions.append(next_price[0, 0])  # Lưu giá trị dự đoán
            
            # Cập nhật future_prices để tạo chuỗi mới cho lần dự đoán kế tiếp
            future_prices = np.append(future_prices[1:], next_price[0, 0])

        # Sau khi dự đoán xong, chuyển đổi lại dự đoán về thang đo gốc
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Tạo danh sách ngày cho dự đoán 1 năm tiếp theo
        last_date = df_filtered['Ngày'].max()
        future_dates = pd.date_range(start=last_date, periods=252, freq='B')  # Chỉ tính các ngày làm việc

        # Vẽ biểu đồ dự đoán
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Next Year Predictions'))

        fig.update_layout(title=f'Giá Đóng Cửa Cổ Phiếu {symbol_price.upper()} với Dự Đoán LSTM cho 1 Năm Tiếp Theo',
                          xaxis_title='Ngày', yaxis_title='Giá Đóng Cửa (VND)', template='plotly_white')
        
        st.plotly_chart(fig)
    else:
        st.write("No data available for this stock symbol.")
else:
    st.write("Please enter a stock symbol for extended price prediction.")
