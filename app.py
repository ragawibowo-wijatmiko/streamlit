import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# =========================
# LOAD FILES
# =========================
svm_model = joblib.load("svm_model.pkl")
nb_model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

metrics = pd.read_csv("model_metrics.csv")
df = pd.read_csv("dataset.csv")

# 🔥 Bersihkan nama kolom (anti spasi & beda format)
metrics.columns = metrics.columns.str.strip().str.lower()

# Ambil baris model
svm_metrics = metrics[metrics["model"].str.lower() == "svm"].iloc[0]
nb_metrics = metrics[metrics["model"].str.lower() == "naive bayes"].iloc[0]

# =========================
# HEADER
# =========================
st.title("🚀 Sentiment Analysis Dashboard")
st.markdown("Perbandingan performa model pada data uji")

st.divider()

# =========================
# KPI
# =========================
col1, col2 = st.columns(2)

with col1:
    st.metric("SVM Accuracy", svm_metrics["accuracy"])
    st.metric("SVM Precision", svm_metrics["precision"])
    st.metric("SVM Recall", svm_metrics["recall"])
    st.metric("SVM F1 Score", svm_metrics["f1"])

with col2:
    st.metric("NB Accuracy", nb_metrics["accuracy"])
    st.metric("NB Precision", nb_metrics["precision"])
    st.metric("NB Recall", nb_metrics["recall"])
    st.metric("NB F1 Score", nb_metrics["f1"])

st.divider()

# =========================
# BAR CHART
# =========================
comparison_df = pd.DataFrame({
    "Metric": ["Accuracy","Precision","Recall","F1"],
    "SVM": [
        svm_metrics["accuracy"],
        svm_metrics["precision"],
        svm_metrics["recall"],
        svm_metrics["f1"]
    ],
    "Naive Bayes": [
        nb_metrics["accuracy"],
        nb_metrics["precision"],
        nb_metrics["recall"],
        nb_metrics["f1"]
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
# DONUT CHART
# =========================
st.subheader("Distribusi Sentiment")

sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment","Count"]

fig_donut = go.Figure(data=[go.Pie(
    labels=sentiment_counts["Sentiment"],
    values=sentiment_counts["Count"],
    hole=.6
)])

fig_donut.update_layout(template="plotly_dark")

st.plotly_chart(fig_donut, use_container_width=True)

st.divider()

# =========================
# PREDICTION
# =========================
st.subheader("Coba Prediksi")

user_input = st.text_area("Masukkan komentar")

if st.button("Prediksi"):
    text_vector = vectorizer.transform([user_input])
    svm_pred = svm_model.predict(text_vector)[0]
    nb_pred = nb_model.predict(text_vector)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"SVM: {svm_pred}")

    with col2:
        st.info(f"Naive Bayes: {nb_pred}")

st.divider()

st.subheader("Preview Dataset")
st.dataframe(df.head())
