import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ================= NLTK SETUP =================
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# ================= PREPROCESS FUNCTION =================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

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
st.sidebar.title("📊 Menu Sistem")
menu = st.sidebar.radio("Navigasi", [
    "Beranda",
    "Prediksi Manual",
    "Prediksi Dataset (CSV)",
    "Perbandingan Model",
    "Tentang Penelitian"
])

# ================= BERANDA =================
if menu == "Beranda":
    st.title("📌 Sistem Analisis Sentimen Cybercrime")
    st.subheader("Komentar YouTube Menggunakan Naive Bayes dan SVM")

    st.markdown("""
    **Judul Penelitian:**  
    *Analisis sentimen kasus cybercrime pada kolom komentar YouTube menggunakan Naive Bayes dan SVM*

    **Alur Sistem:**
    1. Scraping komentar YouTube  
    2. Preprocessing teks  
    3. TF-IDF Vectorization  
    4. Training model  
    5. Evaluasi model  
    6. Deployment sistem berbasis web  

    **Metode:**
    - TF-IDF
    - Naive Bayes
    - Support Vector Machine (SVM)
    - Ensemble System

    **Output Sistem:**
    - Prediksi sentimen
    - Confidence score
    - Prediksi massal dataset
    - Perbandingan model
    """)

# ================= PREDIKSI MANUAL =================
elif menu == "Prediksi Manual":
    st.title("🧠 Prediksi Sentimen Manual")

    text_input = st.text_area("Masukkan komentar YouTube:")

    if st.button("Prediksi Sentimen"):
        if text_input.strip() != "":
            clean_text = preprocess_text(text_input)
            X_input = vectorizer.transform([clean_text])

            nb_pred = nb_model.predict(X_input)[0]
            svm_pred = svm_model.predict(X_input)[0]

            nb_prob = nb_model.predict_proba(X_input).max()
            svm_prob = svm_model.predict_proba(X_input).max()

            # ===== Ensemble Logic =====
            if nb_pred == svm_pred:
                final_pred = nb_pred
            else:
                # voting sederhana → pilih confidence terbesar
                final_pred = nb_pred if nb_prob >= svm_prob else svm_pred

            st.subheader("📊 Hasil Prediksi")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.success(f"Naive Bayes: {nb_pred}")
                st.write(f"Confidence: {nb_prob:.2f}")

            with col2:
                st.info(f"SVM: {svm_pred}")
                st.write(f"Confidence: {svm_prob:.2f}")

            with col3:
                st.warning(f"Ensemble: {final_pred}")

        else:
            st.warning("Masukkan teks terlebih dahulu")

# ================= PREDIKSI DATASET =================
elif menu == "Prediksi Dataset (CSV)":
    st.title("📂 Prediksi Sentimen Dataset")

    uploaded_file = st.file_uploader("Upload file CSV (kolom harus bernama: comment)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "comment" not in df.columns:
            st.error("CSV harus memiliki kolom bernama: comment")
        else:
            st.subheader("Preview Data")
            st.dataframe(df.head())

            # Preprocessing
            df["clean_comment"] = df["comment"].astype(str).apply(preprocess_text)

            X_data = vectorizer.transform(df["clean_comment"])

            df["NB_Prediction"] = nb_model.predict(X_data)
            df["SVM_Prediction"] = svm_model.predict(X_data)

            # Ensemble
            ensemble_preds = []
            for i in range(X_data.shape[0]):
                nb_p = df["NB_Prediction"].iloc[i]
                svm_p = df["SVM_Prediction"].iloc[i]
                if nb_p == svm_p:
                    ensemble_preds.append(nb_p)
                else:
                    ensemble_preds.append(svm_p)
            df["Ensemble_Prediction"] = ensemble_preds

            st.subheader("📊 Hasil Prediksi Dataset")
            st.dataframe(df[["comment", "NB_Prediction", "SVM_Prediction", "Ensemble_Prediction"]])

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Hasil Prediksi",
                csv,
                "hasil_prediksi_sentimen.csv",
                "text/csv"
            )

# ================= PERBANDINGAN =================
elif menu == "Perbandingan Model":
    st.title("📈 Perbandingan Model")

    try:
        df_metrics = pd.read_csv("model_metrics.csv")

        st.subheader("📊 Hasil Evaluasi Model (Data Asli Training)")
        st.dataframe(df_metrics)

        # ================= BAR CHART (ILMIAH STYLE) =================
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

        nb_values = df_metrics[df_metrics["Model"]=="Naive Bayes"][metrics].values.flatten()
        svm_values = df_metrics[df_metrics["Model"]=="SVM"][metrics].values.flatten()

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(x - width/2, nb_values, width, label="Naive Bayes")
        ax.bar(x + width/2, svm_values, width, label="SVM")

        ax.set_ylabel("Score")
        ax.set_title("Perbandingan Performa Model")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error("❌ File model_metrics.csv tidak ditemukan atau format salah")
        st.code(str(e))


# ================= TENTANG =================
elif menu == "Tentang Penelitian":
    st.title("📚 Tentang Penelitian")

    st.markdown("""
    **Judul:**  
    Analisis sentimen kasus cybercrime pada kolom komentar YouTube menggunakan Naive Bayes dan SVM

    **Peneliti:**  
    Raga Wibowo

    **Bidang:**  
    Sistem Informasi / Data Science / Machine Learning

    **Metodologi Penelitian:**
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


