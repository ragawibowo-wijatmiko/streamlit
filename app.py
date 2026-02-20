import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Sistem Analisis Sentimen Cybercrime",
    layout="wide",
    page_icon="📊"
)

# ================= CUSTOM STYLE =================
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
.main {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
}
h1, h2, h3 {
    color: #0b3c5d;
}
.sidebar .sidebar-content {
    background: linear-gradient(to bottom, #141e30, #243b55);
    color: white;
}
.css-1d391kg {
    background: linear-gradient(to bottom, #141e30, #243b55);
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
nb_model = joblib.load("nb_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ================= LOAD KAMUS =================
positive_df = pd.read_csv("kamus/positive.csv")
negative_df = pd.read_csv("kamus/negative.csv")
alay_df = pd.read_csv("kamus/kamusalay.csv")
slang_df = pd.read_csv("kamus/slangword.csv")

positive_words = set(positive_df.iloc[:,0].str.lower())
negative_words = set(negative_df.iloc[:,0].str.lower())

alay_dict = dict(zip(
    alay_df.iloc[:,0].str.lower(),
    alay_df.iloc[:,1].str.lower()
))

slang_dict = dict(zip(
    slang_df.iloc[:,0].str.lower(),
    slang_df.iloc[:,1].str.lower()
))

# ================= FUNCTIONS =================
def normalize_text(text):
    words = text.lower().split()
    result = []
    for w in words:
        if w in alay_dict:
            result.append(alay_dict[w])
        elif w in slang_dict:
            result.append(slang_dict[w])
        else:
            result.append(w)
    return " ".join(result)

def lexicon_sentiment(text):
    words = text.lower().split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)

    if pos > neg:
        return "positif"
    elif neg > pos:
        return "negatif"
    else:
        return "netral"

# ================= SIDEBAR =================
st.sidebar.title("📚 Sistem Analisis Sentimen")
menu = st.sidebar.radio("Navigasi Menu", [
    "🏠 Beranda",
    "🧠 Prediksi Manual",
    "📊 Evaluasi Model",
    "📈 Perbandingan Model",
    "ℹ️ Tentang Sistem"
])

# ================= BERANDA =================
if menu == "🏠 Beranda":
    st.title("📊 Sistem Analisis Sentimen Cybercrime")
    st.subheader("Hybrid System: Lexicon + Naive Bayes + SVM")

    st.markdown("""
    ### 🎯 Fitur Sistem
    - Analisis sentimen komentar YouTube
    - Hybrid approach:
        - Lexicon-Based (kamus)
        - Machine Learning
    - TF-IDF Vectorization
    - Naive Bayes Classifier
    - Support Vector Machine (SVM)
    - Evaluasi model ilmiah
    - Visualisasi performa

    ### 🧠 Metode:
    - Preprocessing teks
    - Normalisasi slang & alay
    - TF-IDF
    - Supervised Learning
    - Hybrid sentiment system
    """)

# ================= PREDIKSI MANUAL =================
elif menu == "🧠 Prediksi Manual":
    st.title("🧠 Prediksi Sentimen Komentar")

    text = st.text_area("Masukkan komentar YouTube:")

    if st.button("🔍 Prediksi Sentimen"):
        if text.strip() != "":
            norm_text = normalize_text(text)

            # Lexicon
            lex_result = lexicon_sentiment(norm_text)

            # ML
            X = vectorizer.transform([norm_text])
            nb_pred = nb_model.predict(X)[0]
            svm_pred = svm_model.predict(X)[0]

            st.subheader("📌 Hasil Prediksi")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"📘 Lexicon: {lex_result}")
            with col2:
                st.info(f"🤖 Naive Bayes: {nb_pred}")
            with col3:
                st.warning(f"🧠 SVM: {svm_pred}")

            st.markdown("---")
            st.caption("Hasil dapat berbeda karena perbedaan pendekatan lexicon-based dan machine learning.")

        else:
            st.warning("Masukkan teks terlebih dahulu!")

# ================= EVALUASI MODEL =================
elif menu == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Model")

    uploaded_test = st.file_uploader("Upload CSV evaluasi (text,label)", type=["csv"])

    if uploaded_test:
        df_test = pd.read_csv(uploaded_test)

        X_test = vectorizer.transform(df_test["text"])
        y_true = df_test["label"]

        nb_pred = nb_model.predict(X_test)
        svm_pred = svm_model.predict(X_test)

        acc_nb = accuracy_score(y_true, nb_pred)
        acc_svm = accuracy_score(y_true, svm_pred)

        st.subheader("🎯 Akurasi Model")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Naive Bayes Accuracy", round(acc_nb, 4))
        with col2:
            st.metric("SVM Accuracy", round(acc_svm, 4))

        st.markdown("---")

        labels = sorted(list(set(y_true)))

        st.subheader("📌 Confusion Matrix Naive Bayes")
        cm_nb = confusion_matrix(y_true, nb_pred)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax1)
        st.pyplot(fig1)

        st.subheader("📌 Confusion Matrix SVM")
        cm_svm = confusion_matrix(y_true, svm_pred)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=labels, yticklabels=labels, ax=ax2)
        st.pyplot(fig2)

        st.subheader("📄 Classification Report Naive Bayes")
        st.text(classification_report(y_true, nb_pred))

        st.subheader("📄 Classification Report SVM")
        st.text(classification_report(y_true, svm_pred))

# ================= PERBANDINGAN MODEL =================
elif menu == "📈 Perbandingan Model":
    st.title("📈 Perbandingan Model")

    try:
        df_metrics = pd.read_csv("model_metrics.csv").set_index("Model")

        st.subheader("📊 Performa Model (Hasil Training)")
        st.dataframe(df_metrics)

        st.bar_chart(df_metrics)

    except:
        st.error("File model_metrics.csv tidak ditemukan.")

# ================= TENTANG =================
elif menu == "ℹ️ Tentang Sistem":
    st.title("ℹ️ Tentang Sistem")

    st.markdown("""
    **Sistem:**  
    Sistem Analisis Sentimen Cybercrime Berbasis Hybrid

    **Metode:**  
    - Lexicon-Based Sentiment  
    - TF-IDF  
    - Naive Bayes  
    - Support Vector Machine (SVM)

    **Fitur Utama:**
    - Hybrid classification
    - Evaluasi ilmiah model
    - Visualisasi performa
    - Implementasi web-based
    - NLP system

    **Teknologi:**
    - Python  
    - Streamlit  
    - Scikit-learn  
    - Pandas  
    - Machine Learning  
    - NLP  
    """)
