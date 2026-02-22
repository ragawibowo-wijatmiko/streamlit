# app.py
# Dashboard Sentiment Analysis Streamlit (Dark Theme + Visual Dashboard)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# ======================
# Custom CSS (Dark UI)
# ======================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.sidebar .sidebar-content {
    background-color: #0b1320;
}
.block-container {
    padding-top: 1.5rem;
}
.metric-box {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.title {
    color: #60a5fa;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ======================
# Sidebar
# ======================
st.sidebar.title("📊 Sentiment Dashboard")
menu = st.sidebar.radio("Menu", ["Dashboard Utama", "Perbandingan Model", "Distribusi Sentimen"])

# ======================
# Load Data
# ======================
@st.cache_data
def load_metrics():
    return pd.read_csv("model_metrics.csv")

@st.cache_data
def load_sentiment():
    return pd.read_csv("sentiment_label_output.csv")

# ======================
# Dashboard Utama
# ======================
if menu == "Dashboard Utama":
    st.markdown("<h1 class='title'>📌 Dashboard Analisis Sentimen</h1>", unsafe_allow_html=True)

    df = load_sentiment()

    pos = (df['sentiment'] == 'positive').sum()
    neg = (df['sentiment'] == 'negative').sum()
    neu = (df['sentiment'] == 'neutral').sum()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"<div class='metric-box'><h3>😊 Positive</h3><h1>{pos}</h1></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><h3>😡 Negative</h3><h1>{neg}</h1></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><h3>😐 Neutral</h3><h1>{neu}</h1></div>", unsafe_allow_html=True)

    # Pie Chart
    st.subheader("📊 Distribusi Sentimen")
    fig, ax = plt.subplots()
    ax.pie([pos, neg, neu], labels=['Positive','Negative','Neutral'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# ======================
# Perbandingan Model
# ======================
elif menu == "Perbandingan Model":
    st.markdown("<h1 class='title'>📈 Perbandingan Model</h1>", unsafe_allow_html=True)

    dfm = load_metrics().set_index("Model")

    st.dataframe(dfm)

    st.subheader("📊 Grafik Batang Perbandingan Model")
    fig, ax = plt.subplots()
    dfm[['Accuracy','Precision','Recall','F1']].plot(kind='bar', ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

# ======================
# Distribusi Sentimen
# ======================
elif menu == "Distribusi Sentimen":
    st.markdown("<h1 class='title'>📊 Distribusi Sentimen Dataset</h1>", unsafe_allow_html=True)

    df = load_sentiment()

    st.dataframe(df.head(100))

    sent_count = df['sentiment'].value_counts()

    fig, ax = plt.subplots()
    sent_count.plot(kind='bar', ax=ax)
    st.pyplot(fig)
