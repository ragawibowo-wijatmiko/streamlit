import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Skripsi", layout="centered")

st.title("📊 Dashboard Skripsi")
st.write("Aplikasi Streamlit untuk hasil scraping & analisis data")

st.subheader("Upload Data")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx","csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File berhasil diupload")
    st.dataframe(df.head(50))
