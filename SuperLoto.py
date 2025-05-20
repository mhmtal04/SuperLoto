import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

st.title("Süper Loto Tahmin Botu (Mobil Uyumlu)")

@st.cache_data
def fetch_milliyet_results():
    url = "https://www.milliyet.com.tr/super-loto-sonuclari"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    draw_data = []

    for result in soup.select("ul.loto-number"):
        numbers = result.get_text(strip=True).split()
        try:
            nums = [int(n) for n in numbers if n.isdigit()]
            if len(nums) == 6:
                draw_data.append(nums)
        except:
            continue

    return pd.DataFrame(draw_data, columns=[f"Sayı{i+1}" for i in range(6)])

df = fetch_milliyet_results()

if df.empty:
    st.error("Sonuçlar alınamadı. Site yapısı değişmiş olabilir.")
else:
    all_numbers = df.values.flatten()
    freq = pd.Series(all_numbers).value_counts().sort_index()

    st.subheader("En Çok Çıkan Sayılar")
    st.bar_chart(freq)

    weights = freq / freq.sum()

    def weighted_choice(weights, n=6):
        return np.random.choice(weights.index, size=n, replace=False, p=weights.values)

    st.subheader("Tahmin Üret")
    if st.button("Tahmin Üret"):
        tahmin = weighted_choice(weights)
        st.success("Tahmin edilen sayılar: " + ", ".join(map(str, sorted(tahmin))))
