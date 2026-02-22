import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# =========================
# 🎨 PREMIUM DARK STYLE
# =========================
st.markdown("""
<style>

html, body, .stApp {
    background: linear-gradient(135deg,#0B1E3C,#071633);
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #071633;
}

h1, h2, h3 {
    color: #4FD1C5;
}

div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#112D4E,#1A3A6E);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.4);
}

.stButton>button {
    background: linear-gradient(90deg,#1F4068,#1B262C);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#4FD1C5,#38B2AC);
    color: black;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD FILES
# =========================
svm_model = joblib.load("svm_model.pkl")
nb_model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

metrics = pd.read_csv("model_metrics.csv")
df = pd.read_csv("dataset.csv")

# Ambil metrics per model
svm_metrics = metrics[metrics["Model"] == "SVM"].iloc[0]
nb_metrics = metrics[metrics["Model"] == "Naive Bayes"].iloc[0]

# =========================
# HEADER
# =========================
st.markdown("## 🚀 Sentiment Analysis Comparison Dashboard")
st.markdown("Perbandingan performa model SVM dan Naive Bayes pada data uji")

st.divider()

# =========================
# KPI METRICS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.metric("SVM Accuracy", svm_metrics["Accuracy"])
    st.metric("SVM Precision", svm_metrics["Precision"])
    st.metric("SVM Recall", svm_metrics["Recall"])
    st.metric("SVM F1-Score", svm_metrics["F1-score"])

with col2:
    st.metric("Naive Bayes Accuracy", nb_metrics["Accuracy"])
    st.metric("Naive Bayes Precision", nb_metrics["Precision"])
    st.metric("Naive Bayes Recall", nb_metrics["Recall"])
    st.metric("Naive Bayes F1-Score", nb_metrics["F1-score"])

st.divider()

# =========================
# 📊 BAR COMPARISON
# =========================
st.subheader("📊 Perbandingan Performa Model (Data Uji)")

comparison_df = pd.DataFrame({
    "Metric": ["Accuracy","Precision","Recall","F1-score"],
    "SVM": [
        svm_metrics["Accuracy"],
        svm_metrics["Precision"],
        svm_metrics["Recall"],
        svm_metrics["F1-score"]
    ],
    "Naive Bayes": [
        nb_metrics["Accuracy"],
        nb_metrics["Precision"],
        nb_metrics["Recall"],
        nb_metrics["F1-score"]
    ]
})

fig_bar = px.bar(
    comparison_df,
    x="Metric",
    y=["SVM","Naive Bayes"],
    barmode="group",
    template="plotly_dark",
    text_auto=True
)

st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# 🍩 DONUT CHART SENTIMENT
# =========================
st.subheader("🍩 Distribusi Sentiment Dataset")

sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment","Count"]

fig_donut = go.Figure(data=[go.Pie(
    labels=sentiment_counts["Sentiment"],
    values=sentiment_counts["Count"],
    hole=.6
)])

fig_donut.update_layout(
    template="plotly_dark",
    showlegend=True
)

st.plotly_chart(fig_donut, use_container_width=True)

# =========================
# 🔍 PREDICTION SECTION
# =========================
st.divider()
st.subheader("🔍 Coba Prediksi Sentiment")

user_input = st.text_area("Masukkan komentar")

if st.button("Prediksi"):
    text_vector = vectorizer.transform([user_input])

    svm_pred = svm_model.predict(text_vector)[0]
    nb_pred = nb_model.predict(text_vector)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"SVM Prediction: {svm_pred}")

    with col2:
        st.info(f"Naive Bayes Prediction: {nb_pred}")

st.divider()

# =========================
# 📋 DATASET PREVIEW
# =========================
st.subheader("📋 Preview Dataset")
st.dataframe(df.head())
