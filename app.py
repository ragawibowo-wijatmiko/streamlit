import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# THEME CSS
# =========================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}
[data-testid="stAppViewContainer"]{
    background: radial-gradient(circle at top, #0f172a, #020617);
}
.card {
    background: rgba(15,23,42,0.8);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 0 25px rgba(0,255,255,0.05);
}
.title {
    font-size: 40px;
    font-weight: 700;
}
.subtitle {
    font-size: 20px;
    color: #cbd5f5;
}
.metric-card {
    background: linear-gradient(135deg, #020617, #0f172a);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.05);
}
.metric-title {
    font-size: 18px;
    color: #94a3b8;
}
.metric-value {
    font-size: 36px;
    font-weight: bold;
}
hr {
    border: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_dataset():
    df = pd.read_csv("sentiment_label_output.csv")
    # auto detect kolom sentimen
    for c in df.columns:
        if "sentiment" in c.lower() or "label" in c.lower():
            df["sentiment"] = df[c].astype(str).str.lower()
            return df
    df["sentiment"] = "neutral"
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv("model_metrics.csv")

@st.cache_resource
def load_models():
    with open("nb_model.pkl", "rb") as f:
        nb = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        svm = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return nb, svm, vec

df = load_dataset()
metrics_df = load_metrics()
nb_model, svm_model, vectorizer = load_models()

# =========================
# HEADER
# =========================
st.markdown("<div class='title'>📊 Dashboard Analisis Sentimen</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Sistem Analisis Sentimen Berbasis Machine Learning (Naive Bayes & SVM)</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# =========================
# DISTRIBUSI SENTIMEN
# =========================
pos = (df["sentiment"]=="positive").sum()
neg = (df["sentiment"]=="negative").sum()
neu = (df["sentiment"]=="neutral").sum()

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Positive</div>
        <div class='metric-value'>{pos}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Negative</div>
        <div class='metric-value'>{neg}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Neutral</div>
        <div class='metric-value'>{neu}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# PIE CHART SENTIMEN
# =========================
st.markdown("## 📌 Distribusi Sentimen")

fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.pie(
    [pos, neg, neu],
    labels=["Positive","Negative","Neutral"],
    autopct='%1.1f%%',
    startangle=90
)
ax1.axis('equal')
st.pyplot(fig1)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================
# PERBANDINGAN MODEL
# =========================
st.markdown("## 📈 Perbandingan Performa Model")

metrics_df_plot = metrics_df.set_index("Model")

fig2, ax2 = plt.subplots(figsize=(10,6))
metrics_df_plot[["Accuracy","Precision","Recall","F1-Score"]].plot(kind="bar", ax=ax2)
ax2.set_ylabel("Score")
ax2.set_title("Perbandingan Naive Bayes vs SVM")
ax2.grid(axis="y", alpha=0.3)
st.pyplot(fig2)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================
# PREDIKSI MANUAL
# =========================
st.markdown("## ✍️ Prediksi Sentimen Manual")

text_input = st.text_area("Masukkan teks komentar:")

if st.button("Prediksi Sentimen"):
    if text_input.strip() != "":
        X = vectorizer.transform([text_input])
        nb_pred = nb_model.predict(X)[0]
        svm_pred = svm_model.predict(X)[0]

        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Naive Bayes</div>
                <div class='metric-value'>{nb_pred.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>SVM</div>
                <div class='metric-value'>{svm_pred.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Masukkan teks terlebih dahulu")

# =========================
# FOOTER
# =========================
st.markdown("<br><br><center style='color:#64748b;'>Dashboard Analisis Sentimen | Machine Learning | Final Sidang</center>", unsafe_allow_html=True)
