import streamlit as st
import pandas as pd
import numpy as np

st.title("Süper Loto Tahmin Botu")

uploaded_file = st.file_uploader("CSV dosyanızı seçin", type=["csv"])

def calculate_probabilities(df):
    # 1-60 arası sayıların çıkma frekansını hesapla
    counts = np.zeros(60, dtype=int)
    for col in ['number1','number2','number3','number4','number5','number6']:
        counts += df[col].value_counts().reindex(range(1,61), fill_value=0).values
    probabilities = counts / counts.sum()
    return probabilities

def generate_prediction(probabilities, n=6):
    # Olasılıklara göre 6 sayı tahmin et (tekrar olmadan)
    numbers = np.arange(1,61)
    prediction = np.random.choice(numbers, size=n, replace=False, p=probabilities)
    prediction.sort()
    return prediction

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dosya başarıyla yüklendi.")
    st.dataframe(df.head())

    # İstatistikleri hesapla
    probabilities = calculate_probabilities(df)

    if st.button("Tahmin Üret"):
        prediction = generate_prediction(probabilities)
        st.write("Tahmin edilen sayılar:", ", ".join(map(str, prediction)))
else:
    st.info("Lütfen süper loto sonuçlarını içeren CSV dosyanızı yükleyin.")
