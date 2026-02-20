import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="Analisis Sentimen Masyarakat Pada Kolom Komentar Youtube Menggunakan Algoritma SVM dan Naive Bayes",
    layout="wide"
)

# ================= LOAD FILES =================
@st.cache_resource
def load_models():
    nb = joblib.load("nb_model.pkl")
    svm = joblib.load("svm_model.pkl")
    vec = joblib.load("vectorizer.pkl")
    return nb, svm, vec

nb_model, svm_model, vectorizer = load_models()

# ================= SIDEBAR =================
st.sidebar.title("📚 Menu Sistem")
menu = st.sidebar.radio("Navigasi", [
    "Beranda",
    "Prediksi Manual",
    "Prediksi Dataset (CSV)",
    "Evaluasi Model",
    "Perbandingan Model",
    "Tentang Penelitian"
])

# ================= BERANDA =================
if menu == "Beranda":
    st.title("📊 Sistem Analisis Sentimen Cybercrime")
    st.subheader("Komentar YouTube Menggunakan Naive Bayes dan SVM")

    st.markdown("""
    ### 🎓 Judul Penelitian  
    **Analisis sentimen kasus cybercrime pada kolom komentar YouTube menggunakan Naive Bayes dan SVM**

    ### 🎯 Tujuan Sistem  
    Sistem ini bertujuan untuk menganalisis opini publik terhadap kasus cybercrime 
    berdasarkan komentar YouTube menggunakan metode Machine Learning.

    ### 🧠 Metode:
    - Scraping komentar YouTube  
    - Preprocessing teks  
    - TF-IDF  
    - Naive Bayes  
    - Support Vector Machine (SVM)  

    ### 📌 Output Sistem:
    - Prediksi sentimen otomatis  
    - Analisis dataset  
    - Evaluasi model  
    - Perbandingan algoritma  
    - Visualisasi performa model  
    """)

# ================= PREDIKSI MANUAL =================
elif menu == "Prediksi Manual":
    st.title("📝 Prediksi Sentimen Manual")

    text_input = st.text_area("Masukkan komentar YouTube:")

    if st.button("Prediksi Sentimen"):
        if text_input.strip() != "":
            X_input = vectorizer.transform([text_input])

            nb_pred = nb_model.predict(X_input)[0]
            svm_pred = svm_model.predict(X_input)[0]

            st.subheader("📌 Hasil Prediksi")
            col1, col2 = st.columns(2)

            with col1:
                st.success(f"Naive Bayes: {nb_pred}")
            with col2:
                st.info(f"SVM: {svm_pred}")
        else:
            st.warning("Masukkan teks terlebih dahulu")

# ================= PREDIKSI DATASET =================
elif menu == "Prediksi Dataset (CSV)":
    st.title("📂 Prediksi Sentimen Dataset")

    uploaded_file = st.file_uploader("Upload CSV (kolom: text)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV harus memiliki kolom bernama 'text'")
        else:
            X_data = vectorizer.transform(df["text"])

            df["NB_Prediction"] = nb_model.predict(X_data)
            df["SVM_Prediction"] = svm_model.predict(X_data)

            st.subheader("📄 Hasil Prediksi")
            st.dataframe(df.head(20))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil Prediksi",
                csv,
                "hasil_prediksi_sentimen.csv",
                "text/csv"
            )

# ================= EVALUASI MODEL =================
elif menu == "Evaluasi Model":
    st.title("📊 Evaluasi Model (Confusion Matrix + Report)")

    uploaded_test = st.file_uploader("Upload data uji CSV (kolom: text,label)", type=["csv"])

    if uploaded_test:
        df_test = pd.read_csv(uploaded_test)

        if not {"text","label"}.issubset(df_test.columns):
            st.error("CSV harus memiliki kolom: text dan label")
        else:
            X_test = vectorizer.transform(df_test["text"])
            y_true = df_test["label"]

            nb_pred = nb_model.predict(X_test)
            svm_pred = svm_model.predict(X_test)

            labels = sorted(list(set(y_true)))

            # ===== Confusion Matrix NB =====
            st.subheader("📌 Confusion Matrix Naive Bayes")
            cm_nb = confusion_matrix(y_true, nb_pred)
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels, ax=ax1)
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("Actual")
            st.pyplot(fig1)

            # ===== Confusion Matrix SVM =====
            st.subheader("📌 Confusion Matrix SVM")
            cm_svm = confusion_matrix(y_true, svm_pred)
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens",
                        xticklabels=labels, yticklabels=labels, ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)

            # ===== Classification Report =====
            st.subheader("📄 Classification Report Naive Bayes")
            st.text(classification_report(y_true, nb_pred))

            st.subheader("📄 Classification Report SVM")
            st.text(classification_report(y_true, svm_pred))

# ================= PERBANDINGAN MODEL =================
elif menu == "Perbandingan Model":
    st.title("📈 Perbandingan Performa Model")

    try:
        df_metrics = pd.read_csv("model_metrics.csv").set_index("Model")

        st.subheader("📊 Hasil Evaluasi Model (Data Asli Training)")
        st.dataframe(df_metrics)

        # === Diagram Batang (Simple Akademik) ===
        st.subheader("📉 Diagram Perbandingan Performa Model")

        fig, ax = plt.subplots()
        df_metrics.plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_title("Perbandingan Naive Bayes vs SVM")
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ File model_metrics.csv tidak ditemukan atau format salah")

# ================= TENTANG =================
elif menu == "Tentang Penelitian":
    st.title("📘 Tentang Penelitian")

    st.markdown("""
    ### 🎓 Judul  
    Analisis sentimen kasus cybercrime pada kolom komentar YouTube menggunakan Naive Bayes dan SVM

    ### 👨‍🎓 Peneliti  
    **Raga Wibowo**

    ### 📚 Bidang  
    Sistem Informasi / Data Science / Machine Learning

    ### 🧠 Metodologi:
    - Scraping komentar YouTube  
    - Preprocessing teks  
    - TF-IDF  
    - Training model  
    - Evaluasi model  
    - Deployment sistem berbasis web  

    ### 🛠 Tools:
    - Python  
    - Google Colab  
    - Scikit-learn  
    - Streamlit  
    - GitHub  
    """)


