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

# Ambil baris sesuai model
svm_metrics = metrics[metrics["Model"] == "SVM"].iloc[0]
nb_metrics = metrics[metrics["Model"] == "Naive Bayes"].iloc[0]

# =========================
# HEADER
# =========================
st.title("🚀 Sentiment Analysis Dashboard")
st.markdown("Perbandingan performa model pada data uji")

st.divider()

# =========================
# KPI SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("SVM Performance")
    st.metric("Accuracy", round(svm_metrics["Accuracy"],4))
    st.metric("Precision", round(svm_metrics["Precision"],4))
    st.metric("Recall", round(svm_metrics["Recall"],4))
    st.metric("F1-Score", round(svm_metrics["F1-Score"],4))

with col2:
    st.subheader("Naive Bayes Performance")
    st.metric("Accuracy", round(nb_metrics["Accuracy"],4))
    st.metric("Precision", round(nb_metrics["Precision"],4))
    st.metric("Recall", round(nb_metrics["Recall"],4))
    st.metric("F1-Score", round(nb_metrics["F1-Score"],4))

st.divider()

# =========================
# BAR CHART
# =========================
comparison_df = pd.DataFrame({
    "Metric": ["Accuracy","Precision","Recall","F1-Score"],
    "SVM": [
        svm_metrics["Accuracy"],
        svm_metrics["Precision"],
        svm_metrics["Recall"],
        svm_metrics["F1-Score"]
    ],
    "Naive Bayes": [
        nb_metrics["Accuracy"],
        nb_metrics["Precision"],
        nb_metrics["Recall"],
        nb_metrics["F1-Score"]
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
st.subheader("Distribusi Sentiment Dataset")

sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment","Count"]

fig_donut = go.Figure(data=[go.Pie(
    labels=sentiment_counts["Sentiment"],
    values=sentiment_counts["Count"],
    hole=0.6
)])

fig_donut.update_layout(template="plotly_dark")

st.plotly_chart(fig_donut, use_container_width=True)

st.divider()

# =========================
# PREDICTION SECTION
# =========================
st.subheader("Coba Prediksi Sentiment")

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

st.subheader("Preview Dataset")
st.dataframe(df.head())
