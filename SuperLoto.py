import streamlit as st
import pandas as pd
import random
from collections import Counter

st.title("Süper Loto Tahmin Botu")

uploaded_file = st.file_uploader("CSV dosyasını yükleyin (Tarih,Num1,Num2,...,Num6 formatında)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Veri Önizlemesi (İlk 5 Satır):")
    st.write(df.head())

    # Tüm sayıların frekansını hesapla
    numbers = df.iloc[:, 1:].values.flatten()
    freq = Counter(numbers)

    # Ağırlıklı sayı havuzu oluştur (frekansa göre)
    weighted_numbers = []
    for num, count in freq.items():
        weighted_numbers.extend([num] * count)

    # Sayı aralığı kategorisi fonksiyonu
    def get_category(n):
        if n <= 20:
            return 'low'
        elif n <= 40:
            return 'mid'
        else:
            return 'high'

    # Kombinasyon geçerlilik kontrolü
    def is_valid_combination(nums):
        cats = [get_category(n) for n in nums]
        evens = sum(1 for n in nums if n % 2 == 0)
        return (cats.count('low') == 2 and
                cats.count('mid') == 2 and
                cats.count('high') == 2 and
                evens in [2, 3, 4])

    # Geçmiş çekilişlere benzerlik kontrolü (3 veya daha fazla ortak sayı varsa benzer kabul)
    past_draws = df.iloc[:, 1:].values.tolist()
    def is_similar_to_past(nums, past_draws):
        for past in past_draws:
            if len(set(nums) & set(past)) >= 3:
                return True
        return False

    # Tahmin oluşturma fonksiyonu
    def generate_prediction(max_tries=1000):
        tries = 0
        while tries < max_tries:
            candidate = sorted(random.sample(weighted_numbers, 6))
            if is_valid_combination(candidate) and not is_similar_to_past(candidate, past_draws):
                return candidate
            tries += 1
        return None

    if st.button("Tahmin Üret"):
        prediction = generate_prediction()
        if prediction:
            st.success(f"Tahmin Edilen Sayılar: {prediction}")
        else:
            st.error("Uygun tahmin bulunamadı, lütfen tekrar deneyin.")
else:
    st.info("Lütfen tahmin yapmak için CSV dosyasını yükleyin.")
