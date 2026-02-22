import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Dashboard Analisis Sentimen Cybercrime",
                   page_icon="📊",
                   layout="wide")

# ==============================
# LOAD DATA & MODEL
# ==============================

@st.cache_resource
def load_model():
    svm = joblib.load("svm_model.pkl")
    nb = joblib.load("nb_model.pkl")
    tfidf = joblib.load("vectorizer.pkl")
    return svm, nb, tfidf

@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

svm_model, nb_model, tfidf = load_model()
df = load_data()

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Dashboard Utama",
     "Visualisasi Sentimen",
     "Perbandingan Model",
     "Prediksi Komentar"]
)

st.sidebar.markdown("---")
st.sidebar.info("Skripsi Raga Wibowo Wijatmiko\nAnalisis Sentimen Cybercrime\nSVM vs Naive Bayes")

# ==============================
# DASHBOARD UTAMA
# ==============================

if menu == "Dashboard Utama":

    st.title("📊 Dashboard Analisis Sentimen Komentar YouTube")
    st.markdown("### Kasus Cybercrime (Bjorka)")

    total = len(df)
    positif = len(df[df['sentimen'] == 'positif'])
    negatif = len(df[df['sentimen'] == 'negatif'])
    netral = len(df[df['sentimen'] == 'netral'])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Komentar", total)
    col2.metric("Positif", positif)
    col3.metric("Negatif", negatif)
    col4.metric("Netral", netral)

    st.markdown("---")

    st.subheader("📄 Preview Dataset")
    st.dataframe(df.head())

# ==============================
# VISUALISASI SENTIMEN
# ==============================

elif menu == "Visualisasi Sentimen":

    st.title("📈 Distribusi Sentimen")

    sentiment_counts = df['sentimen'].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah Komentar")
    ax.set_title("Distribusi Sentimen Komentar YouTube")

    st.pyplot(fig)

# ==============================
# PERBANDINGAN MODEL
# ==============================

elif menu == "Perbandingan Model":

    st.title("⚖️ Perbandingan Model SVM vs Naive Bayes")

    X = tfidf.transform(df['clean_text'])
    y = df['sentimen']

    svm_pred = svm_model.predict(X)
    nb_pred = nb_model.predict(X)

    svm_report = classification_report(y, svm_pred, output_dict=True)
    nb_report = classification_report(y, nb_pred, output_dict=True)

    metrics = ["accuracy", "weighted avg"]

    svm_accuracy = svm_report["accuracy"]
    nb_accuracy = nb_report["accuracy"]

    st.subheader("📌 Akurasi Model")
    col1, col2 = st.columns(2)
    col1.metric("SVM Accuracy", f"{svm_accuracy:.2f}")
    col2.metric("Naive Bayes Accuracy", f"{nb_accuracy:.2f}")

    st.markdown("---")

    st.subheader("📊 Grafik Perbandingan Akurasi")

    fig, ax = plt.subplots()
    ax.bar(["SVM", "Naive Bayes"], [svm_accuracy, nb_accuracy])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Perbandingan Akurasi Model")

    st.pyplot(fig)

# ==============================
# PREDIKSI KOMENTAR BARU
# ==============================

elif menu == "Prediksi Komentar":

    st.title("🧠 Prediksi Sentimen Komentar Baru")

    user_input = st.text_area("Masukkan komentar di sini:")

    model_choice = st.selectbox("Pilih Model", ["SVM", "Naive Bayes"])

    if st.button("Prediksi"):

        if user_input.strip() == "":
            st.warning("Komentar tidak boleh kosong!")
        else:
            vector = tfidf.transform([user_input])

            if model_choice == "SVM":
                prediction = svm_model.predict(vector)[0]
            else:
                prediction = nb_model.predict(vector)[0]

            if prediction == "positif":
                st.success(f"Sentimen: {prediction}")
            elif prediction == "negatif":
                st.error(f"Sentimen: {prediction}")
            else:
                st.info(f"Sentimen: {prediction}")


