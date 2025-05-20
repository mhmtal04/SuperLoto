import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

st.title("Süper Loto Analiz ve Tahmin Botu - Milli Piyango")

@st.cache_data
def fetch_data():
    url = "https://www.millipiyango.gov.tr/cekilisler/super-loto-sonuclari"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    numbers = []
    # Milli Piyango sitesinde çekiliş sonuçlarını içeren tablo yapısını inceleyelim
    # Örneğin, çekiliş sonuçları 'table' etiketi içinde ve her satır 'tr' ile olabilir
    for row in soup.select("table tbody tr"):
        row_numbers = []
        for cell in row.select("td"):
            try:
                num = int(cell.text.strip())
                row_numbers.append(num)
            except:
                continue
        if len(row_numbers) == 6:
            numbers.append(row_numbers)
    return pd.DataFrame(numbers, columns=[f"Sayı{i}" for i in range(1,7)])

df = fetch_data()

if df.empty:
    st.error("Veri alınamadı veya site yapısı değişmiş olabilir.")
else:
    all_numbers = df.values.flatten()
    freq = pd.Series(all_numbers).value_counts().sort_index()

    st.subheader("En Çok Çıkan Sayılar")
    st.bar_chart(freq)

    weights = freq / freq.sum()

    st.subheader("Tahmin Üret")

    def weighted_random_choice(weights, n=6):
        return np.random.choice(weights.index, size=n, replace=False, p=weights.values)

    if st.button("Tahmin Üret"):
        tahmin = weighted_random_choice(weights)
        st.write("Tahmin edilen sayılar:", sorted(tahmin))
