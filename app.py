import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sistem Analisis Sentimen",
    page_icon="📊",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #020617, #020617);
    color: white;
}
.block-container {
    padding-top: 1rem;
}
.card {
    background: rgba(15,23,42,0.95);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
    text-align: center;
}
.big {
    font-size: 42px;
    font-weight: bold;
}
.sidebar .sidebar-content {
    background: #020617;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# LOAD MODEL (SAFE)
# =========================
@st.cache_resource
def load_models():
    try:
        with open("models/nb_model.pkl", "rb") as f:
            nb = pickle.load(f)
        with open("models/svm_model.pkl", "rb") as f:
            svm = pickle.load(f)
        with open("models/vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)
        return nb, svm, vec, True
    except:
        return None, None, None, False

nb_model, svm_model, vectorizer, model_status = load_models()

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio("📂 Menu Sistem", [
    "🏠 Dashboard",
    "✍️ Prediksi Sentimen",
    "📊 Perbandingan Model",
    "📈 Distribusi Sentimen",
    "ℹ️ Tentang Sistem"
])

# =========================
# DASHBOARD
# =========================
if menu == "🏠 Dashboard":
    st.markdown("## 📊 Dashboard Analisis Sentimen")
    st.markdown("### Sistem Analisis Sentimen Berbasis Machine Learning")

    pos = sum(1 for x in st.session_state.history if x=="positive")
    neg = sum(1 for x in st.session_state.history if x=="negative")
    neu = sum(1 for x in st.session_state.history if x=="neutral")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='card'><div>Positive</div><div class='big'>{pos}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><div>Negative</div><div class='big'>{neg}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><div>Neutral</div><div class='big'>{neu}</div></div>", unsafe_allow_html=True)

# =========================
# PREDIKSI
# =========================
elif menu == "✍️ Prediksi Sentimen":
    st.markdown("## ✍️ Input Teks Ulasan")

    text_input = st.text_area("Masukkan teks ulasan:")

    if st.button("🔍 Analisis Sentimen"):
        if model_status:
            X = vectorizer.transform([text_input])
            pred = svm_model.predict(X)[0]
            st.session_state.history.append(pred)

            st.success(f"Hasil Sentimen: **{pred.upper()}**")
        else:
            st.error("Model tidak termuat")

# =========================
# PERBANDINGAN MODEL
# =========================
elif menu == "📊 Perbandingan Model":
    st.markdown("## 📊 Perbandingan Performa Model")

    metrics = pd.DataFrame({
        "Model": ["Naive Bayes", "SVM"],
        "Accuracy": [0.62, 0.80],
        "Precision": [0.60, 0.80],
        "Recall": [0.61, 0.80],
        "F1-Score": [0.53, 0.80]
    })

    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(metrics["Model"]))
    w = 0.18

    ax.bar(x - 1.5*w, metrics["Accuracy"], w, label="Accuracy")
    ax.bar(x - 0.5*w, metrics["Precision"], w, label="Precision")
    ax.bar(x + 0.5*w, metrics["Recall"], w, label="Recall")
    ax.bar(x + 1.5*w, metrics["F1-Score"], w, label="F1-Score")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Model"])
    ax.set_ylabel("Score")
    ax.set_title("Perbandingan Kinerja Model")
    ax.legend()

    st.pyplot(fig)

# =========================
# DISTRIBUSI
# =========================
elif menu == "📈 Distribusi Sentimen":
    st.markdown("## 📈 Distribusi Hasil Prediksi")

    pos = st.session_state.history.count("positive")
    neg = st.session_state.history.count("negative")
    neu = st.session_state.history.count("neutral")

    df = pd.DataFrame({
        "Sentimen": ["Positive", "Negative", "Neutral"],
        "Jumlah": [pos, neg, neu]
    })

    fig, ax = plt.subplots()
    ax.bar(df["Sentimen"], df["Jumlah"])
    ax.set_title("Distribusi Sentimen")
    ax.set_ylabel("Jumlah Prediksi")

    st.pyplot(fig)

# =========================
# ABOUT
# =========================
elif menu == "ℹ️ Tentang Sistem":
    st.markdown("## ℹ️ Tentang Sistem")
    st.write("""
Sistem ini merupakan aplikasi analisis sentimen berbasis machine learning 
menggunakan algoritma Naive Bayes dan Support Vector Machine (SVM).

Tahapan sistem:
1. Preprocessing teks
2. Vectorisasi TF-IDF
3. Training model
4. Evaluasi model
5. Prediksi sentimen
6. Visualisasi hasil

Sistem ini dikembangkan sebagai bagian dari penelitian skripsi.
""")
