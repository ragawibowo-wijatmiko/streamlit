import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================
# LOAD MODEL
# =====================
nb_model = joblib.load("nb_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Dashboard Skripsi",
    page_icon="📊",
    layout="wide"
)

# =====================
# SIDEBAR
# =====================
menu = st.sidebar.selectbox(
    "📌 Menu",
    [
        "Dashboard",
        "Dataset",
        "Prediksi Sentimen",
        "Evaluasi Model",
        "Tentang Penelitian"
    ]
)

# =====================
# DASHBOARD
# =====================
if menu == "Dashboard":
    st.title("📊 Dashboard Analisis Sentimen Cybercrime")
    st.subheader("Kasus Cybercrime pada Kolom Komentar YouTube")
    st.write("Sistem analisis sentimen berbasis Machine Learning menggunakan algoritma Naive Bayes dan Support Vector Machine (SVM).")

    st.markdown("---")

    uploaded = st.file_uploader("Upload Dataset (Excel/CSV)", type=["xlsx", "csv"])

    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success("Dataset berhasil dimuat")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data", len(df))
        col2.metric("Jumlah Kolom", df.shape[1])
        col3.metric("Jumlah Baris", df.shape[0])

        st.subheader("Preview Data")
        st.dataframe(df.head())

# =====================
# DATASET
# =====================
elif menu == "Dataset":
    st.title("📂 Dataset Penelitian")
    uploaded = st.file_uploader("Upload dataset", type=["xlsx", "csv"])

    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.dataframe(df)

# =====================
# PREDIKSI
# =====================
elif menu == "Prediksi Sentimen":
    st.title("🔍 Prediksi Sentimen Komentar")

    teks = st.text_area("Masukkan teks komentar YouTube:")

    model_choice = st.selectbox("Pilih Model", ["Naive Bayes", "SVM"])

    if st.button("Prediksi"):
        if teks.strip() == "":
            st.warning("Teks tidak boleh kosong")
        else:
            vect_text = vectorizer.transform([teks])

            if model_choice == "Naive Bayes":
                pred = nb_model.predict(vect_text)[0]
            else:
                pred = svm_model.predict(vect_text)[0]

            st.success(f"Hasil Prediksi Sentimen: **{pred}**")

# =====================
# EVALUASI
# =====================
elif menu == "Evaluasi Model":
    st.title("📈 Evaluasi Model")

    st.subheader("Perbandingan Model")
    data_eval = {
        "Model": ["Naive Bayes", "SVM"],
        "Akurasi": [0.76, 0.78],
        "Presisi": [0.64, 0.78],
        "Recall": [0.59, 0.76],
        "F1-Score": [0.53, 0.77]
    }

    df_eval = pd.DataFrame(data_eval)

    st.dataframe(df_eval)

    st.subheader("Grafik Perbandingan")
    fig, ax = plt.subplots()
    df_eval.set_index("Model").plot(kind="bar", ax=ax)
    st.pyplot(fig)

# =====================
# TENTANG
# =====================
elif menu == "Tentang Penelitian":
    st.title("📄 Tentang Penelitian")

    st.markdown("""
    ### Judul:
    **Analisis Sentimen Kasus Cybercrime pada Kolom Komentar YouTube Menggunakan Naive Bayes dan SVM**

    ### Tujuan:
    Mengembangkan sistem analisis sentimen berbasis web untuk mengklasifikasikan opini masyarakat terhadap kasus cybercrime.

    ### Metode:
    - Scraping komentar YouTube
    - Preprocessing teks
    - TF-IDF Vectorization
    - Naive Bayes
    - Support Vector Machine
    - Evaluasi model

    ### Output Sistem:
    - Dashboard interaktif
    - Prediksi sentimen otomatis
    - Visualisasi hasil analisis
    """)
