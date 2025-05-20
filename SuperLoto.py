import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

st.title("Süper Loto Tahmin Botu (Milli Piyango - Selenium)")

@st.cache_data
def fetch_super_loto_data():
    options = Options()
    options.add_argument('--headless')  # Tarayıcıyı görünmez çalıştırır
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = "https://www.millipiyangoonline.com/super-loto/sonuclar"
    driver.get(url)
    driver.implicitly_wait(5)  # Sayfanın yüklenmesi için biraz bekle

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Çekilişleri bulalım
    results = []
    for draw in soup.select(".list-number"):
        nums = draw.get_text(strip=True).replace('–', '').split()
        if len(nums) >= 6:
            try:
                results.append([int(n) for n in nums[:6]])
            except:
                continue

    return pd.DataFrame(results, columns=[f"Sayı{i+1}" for i in range(6)])

df = fetch_super_loto_data()

if df.empty:
    st.error("Veriler alınamadı. Site yapısı değişmiş olabilir.")
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
        st.write("Tahmin edilen sayılar:", sorted(tahmin))
