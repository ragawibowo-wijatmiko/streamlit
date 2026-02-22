import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    page_icon="📊",
    layout="wide"
)

# ==============================
# DARK NAVY PREMIUM STYLE
# ==============================

st.markdown("""
<style>

/* BACKGROUND */
html, body, .stApp {
    background-color: #0B1E3C;
    color: white;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #071633;
}

/* SIDEBAR TITLE */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #4FD1C5;
}

/* METRIC CARD */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #112D4E, #1A3A6E);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

/* METRIC LABEL */
div[data-testid="metric-container"] label {
    color: #A8DADC;
    font-weight: 600;
}

/* METRIC VALUE */
div[data-testid="metric-container"] div {
    color: white;
    font-size: 24px;
    font-weight: bold;
}

/* BUTTON STYLE */
.stButton>button {
    background: linear-gradient(90deg, #1F4068, #1B262C);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #4FD1C5, #38B2AC);
    color: black;
}

/* TABLE */
div[data-testid="stDataFrame"] {
    background-color: #102A43;
    border-radius: 10px;
}

/* HEADINGS */
h1, h2, h3 {
    color: #4FD1C5;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD FILES
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

@st.cache_data
def load_metrics():
    return pd.read_csv("model_metrics.csv")

svm_model, nb_model, tfidf = load_model()
df = load_data()
metrics_df = load_metrics()

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

    st.title("📊 Dashboard Analisis Sentimen")

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
    st.subheader("Preview Dataset")
    st.dataframe(df[['comment','stemming','sentiment']].head())

# ==============================
# VISUALISASI
# ==============================

elif menu == "Visualisasi Sentimen":

    st.title("📈 Distribusi Sentimen")

    counts = df['sentiment'].value_counts()

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Sentimen Komentar")
    st.pyplot(fig)

# ==============================
# PERBANDINGAN MODEL
# ==============================

elif menu == "Perbandingan Model":

    st.title("⚖️ Perbandingan Performa Model (Data Uji)")
    st.dataframe(metrics_df)

    # Ambil nilai berdasarkan nama kolom CSV kamu
    svm_row = metrics_df[metrics_df['Model'] == 'SVM'].iloc[0]
    nb_row = metrics_df[metrics_df['Model'] == 'Naive Bayes'].iloc[0]

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    svm_scores = [svm_row[m] for m in metrics]
    nb_scores = [nb_row[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(x - width/2, svm_scores, width, label="SVM")
    ax.bar(x + width/2, nb_scores, width, label="Naive Bayes")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0,1)
    ax.set_ylabel("Nilai Skor")
    ax.set_title("Perbandingan Performa Algoritma")
    ax.legend()

    for i in range(len(metrics)):
        ax.text(x[i] - width/2, svm_scores[i] + 0.02, f"{svm_scores[i]:.2f}", ha='center')
        ax.text(x[i] + width/2, nb_scores[i] + 0.02, f"{nb_scores[i]:.2f}", ha='center')

    st.pyplot(fig)

# ==============================
# PREDIKSI KOMENTAR
# ==============================

elif menu == "Prediksi Komentar":

    st.title("🧠 Prediksi Sentimen Komentar")

    user_input = st.text_area("Masukkan komentar di sini:")

    if st.button("Prediksi"):

        if user_input.strip() == "":
            st.warning("Komentar tidak boleh kosong!")
        else:
            vector = tfidf.transform([user_input])

            svm_pred = svm_model.predict(vector)[0]
            nb_pred = nb_model.predict(vector)[0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Hasil SVM")
                if svm_pred == "positif":
                    st.success(svm_pred)
                elif svm_pred == "negatif":
                    st.error(svm_pred)
                else:
                    st.info(svm_pred)

            with col2:
                st.subheader("Hasil Naive Bayes")
                if nb_pred == "positif":
                    st.success(nb_pred)
                elif nb_pred == "negatif":
                    st.error(nb_pred)
                else:
                    st.info(nb_pred)
