import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    page_icon="📊",
    layout="wide"
)

# ======================
# DARK MODE STYLE
# ======================

st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stMetric {background-color: #1E222B; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL & DATA
# ======================

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

# ======================
# SIDEBAR
# ======================

st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Dashboard Utama",
     "Visualisasi Sentimen",
     "Perbandingan Model",
     "Prediksi Komentar"]
)

st.sidebar.markdown("---")
st.sidebar.info("Skripsi Raga Wibowo Wijatmiko\nAnalisis Sentimen Cybercrime\nSVM dan Naive Bayes")

# ======================
# DASHBOARD UTAMA
# ======================

if menu == "Dashboard Utama":

    st.title("📊 Dashboard Analisis Sentimen Komentar YouTube")

    total = len(df)
    positif = len(df[df['sentiment'] == 'positif'])
    negatif = len(df[df['sentiment'] == 'negatif'])
    netral = len(df[df['sentiment'] == 'netral'])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Komentar", total)
    col2.metric("Positif", positif)
    col3.metric("Negatif", negatif)
    col4.metric("Netral", netral)

    st.markdown("---")
    st.subheader("📄 Preview Dataset")
    st.dataframe(df[['comment','stemming','sentiment']].head())

# ======================
# VISUALISASI
# ======================

elif menu == "Visualisasi Sentimen":

    st.title("📈 Distribusi Sentimen")

    sentiment_counts = df['sentiment'].value_counts()

    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Sentimen")
    st.pyplot(fig)

# ======================
# PERBANDINGAN MODEL
# ======================

elif menu == "Perbandingan Model":

    st.title("⚖️ Perbandingan Model SVM vs Naive Bayes")

    X = tfidf.transform(df['stemming'])
    y = df['sentiment']

    svm_pred = svm_model.predict(X)
    nb_pred = nb_model.predict(X)

    svm_report = classification_report(y, svm_pred, output_dict=True)
    nb_report = classification_report(y, nb_pred, output_dict=True)

    svm_accuracy = svm_report["accuracy"]
    nb_accuracy = nb_report["accuracy"]

    col1, col2 = st.columns(2)
    col1.metric("SVM Accuracy", f"{svm_accuracy:.2f}")
    col2.metric("Naive Bayes Accuracy", f"{nb_accuracy:.2f}")

    st.markdown("---")

    fig, ax = plt.subplots()
    ax.bar(["SVM", "Naive Bayes"], [svm_accuracy, nb_accuracy])
    ax.set_ylim(0, 1)
    ax.set_title("Perbandingan Akurasi Model")
    st.pyplot(fig)

# ======================
# PREDIKSI
# ======================

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
