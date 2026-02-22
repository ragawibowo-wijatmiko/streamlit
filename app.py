import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# =========================
# 🎨 PREMIUM CSS STYLE
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
metrics = joblib.load("model_metrics.csv")
df = pd.read_csv("dataset.csv")

# =========================
# HEADER
# =========================
st.markdown("## 🚀 Sentiment Analysis Comparison Dashboard")
st.markdown("Perbandingan performa model SVM dan Naive Bayes pada data uji")

# =========================
# KPI METRICS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.metric("SVM Accuracy", metrics["SVM"]["accuracy"])
    st.metric("SVM F1-Score", metrics["SVM"]["f1"])

with col2:
    st.metric("NB Accuracy", metrics["Naive Bayes"]["accuracy"])
    st.metric("NB F1-Score", metrics["Naive Bayes"]["f1"])

st.divider()

# =========================
# 📊 BAR COMPARISON
# =========================
st.subheader("📊 Perbandingan Performa Model")

comparison_df = pd.DataFrame({
    "Metric": ["Accuracy","Precision","Recall","F1-score"],
    "SVM": [
        metrics["SVM"]["accuracy"],
        metrics["SVM"]["precision"],
        metrics["SVM"]["recall"],
        metrics["SVM"]["f1"]
    ],
    "Naive Bayes": [
        metrics["Naive Bayes"]["accuracy"],
        metrics["Naive Bayes"]["precision"],
        metrics["Naive Bayes"]["recall"],
        metrics["Naive Bayes"]["f1"]
    ]
})

fig_bar = px.bar(
    comparison_df,
    x="Metric",
    y=["SVM","Naive Bayes"],
    barmode="group",
    template="plotly_dark"
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

# =========================
# 📋 DATASET PREVIEW
# =========================
st.divider()
st.subheader("📋 Preview Dataset")
st.dataframe(df.head())

