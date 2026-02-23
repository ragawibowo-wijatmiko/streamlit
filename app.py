import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Sentiment Analysis ML - Raga", layout="wide", page_icon="💎")

# =========================
# CUSTOM CSS (SAMA SEPERTI TEMANMU)
# =========================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }

div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.05);
}

.insight-card {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1e3a8a;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.snow()

# =========================
# LOAD FILES
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    metrics = pd.read_csv("model_metrics.csv")
    df.columns = [c.strip() for c in df.columns]
    metrics.columns = [c.strip() for c in metrics.columns]
    return df, metrics

df, metrics = load_data()

svm_model = joblib.load("svm_model.pkl")
nb_model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

svm_metrics = metrics[metrics["Model"] == "SVM"].iloc[0]
nb_metrics = metrics[metrics["Model"] == "Naive Bayes"].iloc[0]

# =========================
# SIDEBAR PREDIKSI ML
# =========================
with st.sidebar:
    st.header("🔬 Lab Prediksi ML")
    teks_uji = st.text_area("Masukkan Kalimat")

    if st.button("Prediksi"):
        vector = vectorizer.transform([teks_uji])
        pred_svm = svm_model.predict(vector)[0]
        pred_nb = nb_model.predict(vector)[0]

        st.success(f"SVM: {pred_svm}")
        st.info(f"Naive Bayes: {pred_nb}")

    st.divider()
    st.caption("Raga Wijatmiko - Skripsi 2026")

# =========================
# MAIN TITLE
# =========================
st.title("💎 Intelligent Sentiment Analysis Dashboard (Machine Learning)")

# =========================
# KPI METRICS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔷 SVM Performance")
    st.metric("Accuracy", round(svm_metrics["Accuracy"],4))
    st.metric("Precision", round(svm_metrics["Precision"],4))
    st.metric("Recall", round(svm_metrics["Recall"],4))
    st.metric("F1-Score", round(svm_metrics["F1-Score"],4))

with col2:
    st.subheader("🔶 Naive Bayes Performance")
    st.metric("Accuracy", round(nb_metrics["Accuracy"],4))
    st.metric("Precision", round(nb_metrics["Precision"],4))
    st.metric("Recall", round(nb_metrics["Recall"],4))
    st.metric("F1-Score", round(nb_metrics["F1-Score"],4))

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Perbandingan Model", "🍩 Distribusi Sentiment", "☁️ WordCloud", "📁 Data Explorer"])

# =========================
# TAB 1 - BAR CHART
# =========================
with tab1:
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
        text_auto=True
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# TAB 2 - DONUT
# =========================
with tab2:
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment","Count"]

    fig_donut = go.Figure(data=[go.Pie(
        labels=sentiment_counts["Sentiment"],
        values=sentiment_counts["Count"],
        hole=0.6
    )])

    st.plotly_chart(fig_donut, use_container_width=True)

# =========================
# TAB 3 - WORDCLOUD
# =========================
with tab3:
    sentiment_option = st.selectbox("Pilih Sentiment", df["sentiment"].unique())

    text_data = " ".join(df[df["sentiment"] == sentiment_option]["stemming"].astype(str))

    if text_data.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

# =========================
# TAB 4 - DATA EXPLORER
# =========================
with tab4:
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Dataset", csv, "dataset_labeled.csv", "text/csv")
