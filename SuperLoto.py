import streamlit as st
import pandas as pd
import random
from collections import Counter

st.title("Süper Loto Tahmin Botu")

uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Tarih, Num1, Num2, ..., Num6 formatında)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Yüklenen Dosyanın Tüm Verisi:")
        st.write(df)

        # Sayıları frekanslarına göre al
        numbers = df.iloc[:, 1:].values.flatten()
        freq = Counter(numbers)

        # Ağırlıklı sayı havuzu oluştur
        weighted_numbers = []
        for num, count in freq.items():
            weighted_numbers.extend([num] * count)

        # Sayı kategorisi (düşük, orta, yüksek)
        def get_category(n):
            if n <= 20:
                return 'low'
            elif n <= 40:
                return 'mid'
            else:
                return 'high'

        # Geçerli kombinasyon mu?
        def is_valid_combination(nums):
            cats = [get_category(n) for n in nums]
            evens = sum(1 for n in nums if n % 2 == 0)
            return (cats.count('low') == 2 and
                    cats.count('mid') == 2 and
                    cats.count('high') == 2 and
                    evens in [2, 3, 4])

        # Geçmişle benzer mi?
        past_draws = df.iloc[:, 1:].values.tolist()
        def is_similar_to_past(nums, past_draws):
            for past in past_draws:
                if len(set(nums) & set(past)) >= 3:
                    return True
            return False

        # Tahmin üret
        def generate_prediction(max_tries=1000):
            tries = 0
            while tries < max_tries:
                candidate = sorted(random.sample(weighted_numbers, 6))
                if is_valid_combination(candidate) and not is_similar_to_past(candidate, past_draws):
                    return [int(n) for n in candidate]  # np.int64 yerine int
                tries += 1
            return None

        if st.button("Tahmin Üret"):
            prediction = generate_prediction()
            if prediction:
                st.success(f"Tahmin Edilen Sayılar: {prediction}")
            else:
                st.error("Uygun tahmin bulunamadı, lütfen tekrar deneyin.")

    except Exception as e:
        st.error(f"Dosya işlenirken hata oluştu: {e}")

else:
    st.info("Lütfen tahmin yapmak için bir CSV dosyası yükleyin.")
