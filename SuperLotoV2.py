import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import itertools

st.title("Gelişmiş Süper Loto Tahmin Botu")

uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Tarih, Num1, Num2, ..., Num6 formatında)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Yüklenen Dosyanın Tüm Verisi:")
        st.write(df)
        
        # --- Temel Hazırlık ---
        # Varsayıyoruz ki CSV dosyasında ilk sütun tarih, geri kalanlar çekiliş numaraları
        total_draws = len(df)
        draws = df.iloc[:, 1:].values.tolist()  # her çekilişteki 6 sayı

        # 1. Frekans Hesabı
        freq = Counter()
        # 1a. Her sayının en son hangi çekilişte çıktığını tutalım (indeks bazında)
        last_occurrence = {}
        for idx, row in enumerate(draws):
            for num in row:
                freq[num] += 1
                last_occurrence[num] = idx  # sıralı okuduğumuz için sonuncusu kalır

        # 2. Zaman Ağırlıklı Frekans: 
        # Formül: base_score(n) = f(n) * (1/(total_draws - last_occurrence(n) + 1))
        base_score = {}
        for num, f in freq.items():
            # Eğer en son index çok yakınsa (yeni çekilişte), pay daha yüksek olur.
            base_score[num] = f * (1 / (total_draws - last_occurrence[num] + 1))
        
        # 3. Koşullu Olasılık (Pair Bonus) hesaplaması:
        # Tüm çekilişlerde birlikte çıkan sayı çiftleri için frekans sayısı
        pair_freq = Counter()
        for row in draws:
            # Sıraya göre (küçükten büyüğe) çiftler oluşturalım
            for pair in itertools.combinations(sorted(row), 2):
                pair_freq[pair] += 1

        # Koşullu bonus: Örneğin, bonus(a,b) = (pair_freq((a,b))/freq[a] + pair_freq((a,b))/freq[b]) / 2
        def conditional_bonus(candidate):
            bonus = 0
            for a, b in itertools.combinations(sorted(candidate), 2):
                pair = (a, b)
                if pair in pair_freq:
                    bonus += (pair_freq[pair] / freq[a] + pair_freq[pair] / freq[b]) / 2
            return bonus

        # Toplam sayıları (1-60) üzerinden çalışacağımızı varsayalım.
        all_numbers = list(range(1, 61))
        # Eğer dosyanızda eksik veya sadece belirli sayılar varsa, frekans dict’inden alabilirsiniz.
        # Burada, yalnızca daha önce çekilmiş sayıları göz önüne alıyoruz:
        available_numbers = list(base_score.keys())

        # Normalize edilmiş olasılık dağılımı: sadece mevcut sayılara göre
        scores = np.array([base_score[num] for num in available_numbers], dtype=float)
        prob = scores / scores.sum()
        
        # --- Kural Kontrolleri ---
        # a) Kategori (Low-Mid-High): 
        # low: 1-20, mid: 21-40, high: 41-60; her kategoriden 2 sayı
        def get_category(n):
            if n <= 20:
                return 'low'
            elif n <= 40:
                return 'mid'
            else:
                return 'high'
        
        def valid_category(candidate):
            cats = [get_category(n) for n in candidate]
            return cats.count('low') == 2 and cats.count('mid') == 2 and cats.count('high') == 2

        # b) Çift-Tek Dengesi: En az 2, en fazla 4 çift sayı
        def valid_even_odd(candidate):
            evens = sum(1 for n in candidate if n % 2 == 0)
            return evens in [2, 3, 4]

        # c) Sayısal Ortalama Kontrolü: Ortalama 25-35 aralığında olsun
        def valid_average(candidate):
            avg = sum(candidate) / 6
            return 25 <= avg <= 35

        # d) Geçmişle Benzerlik Kontrolü: Herhangi çekilişle 3 veya daha fazla ortak sayı varsa reddedilsin
        def not_similar_to_past(candidate, past_draws):
            for past in past_draws:
                if len(set(candidate) & set(past)) >= 3:
                    return False
            return True

        # fonskiyon: Tüm kuralları kontrol eden
        def is_valid_candidate(candidate):
            return (valid_category(candidate) and 
                    valid_even_odd(candidate) and 
                    valid_average(candidate) and 
                    not_similar_to_past(candidate, draws))
        
        # --- Tahmin Üretim Fonksiyonu ---
        def generate_prediction(max_tries=10000):
            best_candidate = None
            best_score = -np.inf
            for _ in range(max_tries):
                # Ağırlıklı seçim: np.random.choice ile replace=False
                candidate = np.random.choice(available_numbers, 6, replace=False, p=prob)
                candidate = sorted(candidate.tolist())
                if not is_valid_candidate(candidate):
                    continue
                # Toplam puan: sum(base_score) + conditional bonus
                candidate_score = sum(base_score[n] for n in candidate) + conditional_bonus(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = candidate
            return best_candidate

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
