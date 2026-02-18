import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ================= LOAD MODEL =================
nb_model = joblib.load("nb_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ================= CONFIG =================
st.set_page_config(
    page_title="Sistem Analisis Sentimen Cybercrime",
    layout="wide"
)

# ================= SIDEBAR =================
st.sidebar.title("Menu Sistem")
menu = st.sidebar.radio("Navigasi", [
    "Beranda",
    "Dataset",
    "Prediksi Sentimen",
    "Evaluasi Model",
    "Perbandingan Model",
    "Tentang Penelitian"
])

# ================= BERANDA =================
if menu == "Beranda":
    st.title("Sistem Analisis Sentimen Cybercrime")
    st.subheader("Komentar YouTube Menggunakan Naive Bayes dan SVM")

    st.markdown("""
    **Judul Penelitian:**  
    *Analisis sentimen kasus cybercrime pada kolom komentar YouTube menggunakan Naive Bayes dan SVM*

    **Tujuan Sistem:**  
    Sistem ini dikembangkan untuk menganalisis opini publik terhadap kasus cybercrime 
    berdasarkan komentar YouTube menggunakan metode Machine Learning.

    **Metode:**
    - TF-IDF
    - Naive Bayes
    - Support Vector Machine (SVM)

    **Output Sistem:**
    - Klasifikasi sentimen
    - Prediksi otomatis
    - Visualisasi performa model
    - Perbandingan algoritma
    """)

# ================= DATASET =================
elif menu == "Dataset":
    st.title("Dataset Komentar YouTube")
    uploaded_file = st.file_uploader("Upload dataset (Excel/CSV)", type=["xlsx","csv"])

    if uploaded_file:
        if uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.subheader("Preview Data")
        st.dataframe(df.head())

        st.write("Jumlah Data:", df.shape[0])
        st.write("Jumlah Kolom:", df.shape[1])

# ================= PREDIKSI =================
elif menu == "Prediksi Sentimen":
    st.title("Prediksi Sentimen Komentar")

    text_input = st.text_area("Masukkan komentar YouTube:")

    if st.button("Prediksi Sentimen"):
        if text_input.strip() != "":
            X_input = vectorizer.transform([text_input])

            nb_pred = nb_model.predict(X_input)[0]
            svm_pred = svm_model.predict(X_input)[0]

            st.subheader("Hasil Prediksi")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Naive Bayes: {nb_pred}")
            with col2:
                st.info(f"SVM: {svm_pred}")
        else:
            st.warning("Masukkan teks terlebih dahulu")

# ================= EVALUASI =================
elif menu == "Evaluasi Model":
    st.title("Evaluasi Model")

    st.info("Evaluasi model dilakukan pada tahap training menggunakan data uji.")

    st.markdown("""
    Parameter evaluasi:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix
    """)

    st.write("Silakan lihat hasil evaluasi pada laporan skripsi.")

# ================= PERBANDINGAN =================
elif menu == "Perbandingan Model":
    st.title("Perbandingan Naive Bayes vs SVM")

    metrics = {
        "Accuracy": [0.0, 0.0],   # nanti bisa isi manual dari hasil skripsi
        "Precision": [0.0, 0.0],
        "Recall": [0.0, 0.0],
        "F1-Score": [0.0, 0.0]
    }

    df_metrics = pd.DataFrame(metrics, index=["Naive Bayes", "SVM"])
    st.dataframe(df_metrics)

    st.bar_chart(df_metrics)

# ================= TENTANG =================
elif menu == "Tentang Penelitian":
    st.title("Tentang Penelitian")

    st.markdown("""
    **Judul:**  
    Analisis sentimen kasus cybercrime pada kolom komentar YouTube menggunakan Naive Bayes dan SVM

    **Peneliti:**  
    Raga Wibowo

    **Bidang:**  
    Sistem Informasi / Data Science / Machine Learning

    **Metodologi:**
    - Scraping komentar YouTube
    - Preprocessing teks
    - TF-IDF
    - Training model
    - Evaluasi model
    - Deployment sistem berbasis web

    **Tools:**
    - Python
    - Google Colab
    - Scikit-learn
    - Streamlit
    - GitHub
    """)
