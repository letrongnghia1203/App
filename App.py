import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import gdown
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

analyzer = SentimentIntensityAnalyzer()

@st.cache_resource
def load_lstm_model():
    model_id = '1-2diAZCXfnoe38o21Vv5Sx8wmre1IceY'
    gdown.download(f'https://drive.google.com/uc?export=download&id={model_id}', 'best_lstm_model.keras', quiet=False)
    return load_model('best_lstm_model.keras')

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

if 'df_vietnam' not in st.session_state:
    st.session_state.df_vietnam = load_data()
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = load_lstm_model()

st.title("Stock Market Data Visualization with LSTM Predictions")
symbol_price = st.text_input("Enter stock symbol for price prediction:")

if symbol_price:
    df_filtered = st.session_state.df_vietnam[st.session_state.df_vietnam['Symbol'] == symbol_price.upper()]

    if not df_filtered.empty:
        st.write(f"Detailed information for stock symbol {symbol_price.upper()}:")
        st.write(df_filtered.head(1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['Ngày'], y=df_filtered['Giá đóng cửa'], mode='lines+markers', name='Giá Đóng Cửa'))
        
        prices = df_filtered[['Giá đóng cửa']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        seq_length = 5
        X = np.array([prices_scaled[i:i + seq_length] for i in range(len(prices_scaled) - seq_length)])

        if len(X) > 0:
            predictions = st.session_state.lstm_model.predict(X)
            predictions = scaler.inverse_transform(predictions)
            prediction_dates = df_filtered['Ngày'].iloc[seq_length:].values

            fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Dự đoán'))

        fig.update_layout(title=f'Giá Đóng Cửa Cổ Phiếu {symbol_price.upper()} với Dự Đoán LSTM',
                          xaxis_title='Ngày', yaxis_title='Giá Đóng Cửa (VND)', template='plotly_white')
        
        st.plotly_chart(fig)
    else:
        st.write("No data available for this stock symbol.")
else:
    st.write("Please enter a stock symbol for price prediction.")

st.title("Sentiment Analysis of Stock News")

symbol_sentiment = st.text_input("Enter stock symbol for sentiment analysis:")

def get_introduction(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")
        introduction = soup.find("h2", {"class": "intro"})
        return introduction.text.strip() if introduction else "No introduction"
    except Exception as e:
        st.write(f"Error fetching introduction: {e}")
        return "No introduction"

def get_latest_articles(symbol, limit=20):
    data_rows = []
    try:
        url = f'https://s.cafef.vn/Ajax/Events_RelatedNews_New.aspx?symbol={symbol}&floorID=0&configID=0&PageIndex=1&PageSize={limit}&Type=2'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")
        data = soup.find("ul", {"class": "News_Title_Link"})
        if not data:
            st.write("No news data available.")
            return pd.DataFrame()
        
        raw = data.find_all('li')
        for row in raw:
            news_date = row.span.text.strip()
            title = row.a.text.strip()
            article_url = "https://s.cafef.vn/" + str(row.a['href'])
            introduction = get_introduction(article_url)
            data_rows.append({"news_date": news_date, "title": title, "url": article_url, "symbol": symbol, "introduction": introduction})
            if len(data_rows) >= limit:
                break
    except Exception as e:
        st.write(f"Error fetching articles: {e}")
    return pd.DataFrame(data_rows)

def translate_text(text):
    try:
        return GoogleTranslator(source='vi', target='en').translate(text)
    except Exception as e:
        st.write(f"Translation error: {e}")
        return text

def vader_analyze(row):
    combined_text = f"{row['title_en']} {row['introduction_en']}".strip() if row['introduction_en'] != "No introduction" else row['title_en']
    sentiment_score = analyzer.polarity_scores(combined_text)
    score = (sentiment_score['compound'] + 1) * 50
    return pd.Series([score, "NEGATIVE" if score < 34 else "NEUTRAL" if score < 67 else "POSITIVE"])

if symbol_sentiment:
    df_pandas_news = get_latest_articles(symbol_sentiment, limit=20)
    
    if not df_pandas_news.empty:
        df_pandas_news['title_en'] = df_pandas_news['title'].apply(translate_text)
        df_pandas_news['introduction_en'] = df_pandas_news['introduction'].apply(translate_text)
        
        df_pandas_news[['article_score', 'article_sentiment']] = df_pandas_news.apply(vader_analyze, axis=1)
        df_pandas_news['news_date'] = pd.to_datetime(df_pandas_news['news_date'], dayfirst=True)

        st.write("### News Articles")
        for index, row in df_pandas_news.iterrows():
            st.markdown(f"**Date**: {row['news_date'].strftime('%d/%m/%Y')} | **Title**: [{row['title']}]({row['url']}) | **Sentiment**: {row['article_sentiment']} | **Score**: {row['article_score']:.2f}")

        average_sentiment = df_pandas_news['article_score'].mean()
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=average_sentiment,
            title={'text': "Overall Sentiment Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [{'range': [0, 33], 'color': '#FF4C4C'}, {'range': [33, 66], 'color': '#FFDD57'}, {'range': [66, 100], 'color': '#4CAF50'}],
            }
        ))
        st.plotly_chart(fig)

        st.write("### Sentiment Over Time")
        df_pandas_news = df_pandas_news.sort_values(by='news_date')
        st.line_chart(df_pandas_news.set_index('news_date')['article_score'])

        st.write("### Sentiment Distribution")
        sentiment_counts = df_pandas_news['article_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        color_discrete_map = {'NEGATIVE': 'red', 'NEUTRAL': 'yellow', 'POSITIVE': 'green'}

        fig_plotly = px.bar(
            sentiment_counts, 
            x='Sentiment', 
            y='Count', 
            color='Sentiment',
            color_discrete_map=color_discrete_map, 
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_plotly)

        df_pandas_news['cleaned_title_en'] = df_pandas_news['title_en'].str.replace(r'\W', ' ', regex=True)
        df_pandas_news['cleaned_introduction_en'] = df_pandas_news['introduction_en'].str.replace(r'\W', ' ', regex=True)
        
        text_data = df_pandas_news['cleaned_title_en'] + " " + df_pandas_news['cleaned_introduction_en']
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        df_pandas_news['cluster_id'] = kmeans.fit_predict(tfidf_matrix)
        
        densest_cluster_id = df_pandas_news['cluster_id'].value_counts().idxmax()
        densest_cluster_text = " ".join(
            df_pandas_news[df_pandas_news['cluster_id'] == densest_cluster_id]['cleaned_title_en'] + " " +
            df_pandas_news[df_pandas_news['cluster_id'] == densest_cluster_id]['cleaned_introduction_en']
        )
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(densest_cluster_text)
        
        st.write(f"### Word Cloud for Densest Cluster (Cluster ID: {densest_cluster_id})")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("No news data available for this stock symbol.")

