import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide",
    page_icon="📊"
)

# ==============================
# DARK THEME CSS
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}

h1, h2, h3, h4, h5, h6, p, span, label {
    color: white !important;
}

.block-container {
    padding: 2rem;
}

div[data-testid="stMetric"] {
    background-color: #020617;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #1e293b;
}

[data-testid="stDataFrame"] {
    background-color: #020617;
    color: white;
}

.card {
    background: #020617;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #1e293b;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("📊 Sentiment Dashboard")
menu = st.sidebar.radio(
    "Menu Analisis",
    [
        "🏠 Dashboard Utama",
        "📊 Distribusi Sentimen",
        "📈 Perbandingan Model",
        "🧪 Evaluasi Model",
        "📁 Dataset",
        "✍️ Prediksi Manual"
    ]
)

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_dataset():
    return pd.read_csv("data/dataset_labeled.csv")

@st.cache_data
def load_metrics():
    return pd.read_csv("data/model_metrics.csv")

# ==============================
# SAFE LOAD
# ==============================
try:
    df = load_dataset()
except:
    df = pd.DataFrame()

try:
    metrics_df = load_metrics()
except:
    metrics_df = pd.DataFrame()

# ==============================
# DASHBOARD
# ==============================
if menu == "🏠 Dashboard Utama":
    st.title("📊 Dashboard Analisis Sentimen")
    st.markdown("### Sistem Analisis Sentimen Berbasis Machine Learning")

    col1, col2, col3 = st.columns(3)

    if not df.empty and "sentiment" in df.columns:
        pos = (df["sentiment"] == "positive").sum()
        neg = (df["sentiment"] == "negative").sum()
        neu = (df["sentiment"] == "neutral").sum()
    else:
        pos, neg, neu = 0,0,0

    with col1:
        st.metric("Positive", pos)
    with col2:
        st.metric("Negative", neg)
    with col3:
        st.metric("Neutral", neu)

# ==============================
# DISTRIBUSI SENTIMEN
# ==============================
elif menu == "📊 Distribusi Sentimen":
    st.title("📊 Distribusi Sentimen Dataset")

    if not df.empty and "sentiment" in df.columns:
        pos = int((df["sentiment"] == "positive").sum())
        neg = int((df["sentiment"] == "negative").sum())
        neu = int((df["sentiment"] == "neutral").sum())

        values = [pos, neg, neu]
        if sum(values) == 0:
            values = [1,1,1]

        fig, ax = plt.subplots()
        ax.pie(values, labels=["Positive","Negative","Neutral"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.error("Dataset tidak memiliki kolom sentiment")

# ==============================
# PERBANDINGAN MODEL
# ==============================
elif menu == "📈 Perbandingan Model":
    st.title("📈 Perbandingan Performa Model")

    if not metrics_df.empty:
        st.dataframe(metrics_df)

        metrics_df_plot = metrics_df.set_index("Model")

        st.bar_chart(metrics_df_plot[["Accuracy","Precision","Recall","F1-Score"]])
    else:
        st.error("File model_metrics.csv tidak ditemukan")

# ==============================
# EVALUASI MODEL
# ==============================
elif menu == "🧪 Evaluasi Model":
    st.title("🧪 Evaluasi Model Machine Learning")

    if not metrics_df.empty:
        st.markdown("### Hasil Evaluasi")
        st.dataframe(metrics_df)
    else:
        st.error("Data evaluasi model belum tersedia")

# ==============================
# DATASET
# ==============================
elif menu == "📁 Dataset":
    st.title("📁 Dataset Viewer")
    if not df.empty:
        st.dataframe(df)
    else:
        st.error("Dataset belum tersedia")

# ==============================
# PREDIKSI MANUAL
# ==============================
elif menu == "✍️ Prediksi Manual":
    st.title("✍️ Prediksi Sentimen Manual")

    text = st.text_area("Masukkan teks:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Prediksi SVM"):
            st.success("Prediksi: POSITIVE (dummy demo)")

    with col2:
        if st.button("Prediksi Naive Bayes"):
            st.success("Prediksi: POSITIVE (dummy demo)")
