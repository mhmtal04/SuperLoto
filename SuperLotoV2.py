import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
import itertools

st.title("Gelişmiş Süper Loto Tahmin Botu")

uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Tarih, Num1, Num2, ..., Num6 formatında)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Yüklenen Dosya:")
        st.write(df)

        total_draws = len(df)
        draws = df.iloc[:, 1:].values.tolist()

        freq = Counter()
        last_occurrence = {}

        for idx, row in enumerate(draws):
            for num in row:
                freq[num] += 1
                last_occurrence[num] = idx

        base_score = {}
        for num, f in freq.items():
            base_score[num] = f * (1 / (total_draws - last_occurrence[num] + 1))

        pair_freq = Counter()
        for row in draws:
            for pair in itertools.combinations(sorted(row), 2):
                pair_freq[pair] += 1

        def conditional_bonus(candidate):
            bonus = 0
            for a, b in itertools.combinations(sorted(candidate), 2):
                pair = (a, b)
                if pair in pair_freq:
                    bonus += (pair_freq[pair] / freq[a] + pair_freq[pair] / freq[b]) / 2
            return bonus

        available_numbers = list(base_score.keys())
        scores = np.array([base_score[num] for num in available_numbers], dtype=float)
        prob = scores / scores.sum()

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

        def valid_even_odd(candidate):
            evens = sum(1 for n in candidate if n % 2 == 0)
            return evens in [2, 3, 4]

        def valid_average(candidate):
            avg = sum(candidate) / 6
            return 25 <= avg <= 35

        def not_similar_to_past(candidate, past_draws):
            for past in past_draws:
                if len(set(candidate) & set(past)) >= 3:
                    return False
            return True

        def is_valid_candidate(candidate):
            return (valid_category(candidate) and
                    valid_even_odd(candidate) and
                    valid_average(candidate) and
                    not_similar_to_past(candidate, draws))

        def generate_prediction(max_tries=10000):
            best_candidate = None
            best_score = -np.inf
            for _ in range(max_tries):
                candidate = np.random.choice(available_numbers, 6, replace=False, p=prob)
                candidate = sorted(candidate.tolist())
                if not is_valid_candidate(candidate):
                    continue
                candidate_score = sum(base_score[n] for n in candidate) + conditional_bonus(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = candidate
            return best_candidate

        if st.button("4 Tahmin Üret"):
            predictions = []
            tries = 0
            while len(predictions) < 4 and tries < 100:
                pred = generate_prediction()
                if pred and pred not in predictions:
                    predictions.append(pred)
                tries += 1
            if predictions:
                st.session_state['predictions'] = predictions
            else:
                st.session_state['predictions'] = None

        if 'predictions' in st.session_state:
            if st.session_state['predictions']:
                for i, pred in enumerate(st.session_state['predictions'], 1):
                    st.success(f"Tahmin {i}: {', '.join(str(n) for n in pred)}")
            else:
                st.error("Uygun tahmin bulunamadı, lütfen tekrar deneyin.")

    except Exception as e:
        st.error(f"Dosya işlenirken hata oluştu: {e}")
else:
    st.info("Lütfen tahmin yapmak için bir CSV dosyası yükleyin.")
