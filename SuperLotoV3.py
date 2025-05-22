import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime

# Ağırlık hesaplama (zaman bazlı)
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    weights = (max_days - days_ago) / max_days
    return weights

# Tekil sayıların ağırlıklı olasılıkları
def weighted_single_probabilities(df):
    weights = get_weights(df['Date'])
    total_weight = weights.sum()

    freq = pd.Series(0, index=range(1, 61), dtype=float)

    for idx, row in df.iterrows():
        numbers = row['Numbers']
        w = weights[idx]
        for n in numbers:
            freq[n] += w

    prob = freq / total_weight
    return prob

# İkili sayıların frekansı
def pair_frequencies(df):
    pair_freq = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    weights = get_weights(df['Date'])

    for idx, row in df.iterrows():
        numbers = row['Numbers']
        w = weights[idx]
        for a, b in combinations(numbers, 2):
            pair_freq.at[a, b] += w
            pair_freq.at[b, a] += w

    return pair_freq

# Koşullu olasılıklar
def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    for a in range(1, 61):
        freq_a = single_prob[a]
        if freq_a > 0:
            for b in range(1, 61):
                cond_prob.at[a, b] = pair_freq.at[a, b] / freq_a
        else:
            cond_prob.loc[a, :] = 0
    return cond_prob

# Kısıt: en az 2 tek ve 2 çift sayı
def check_constraints(numbers):
    odd_count = sum(n % 2 == 1 for n in numbers)
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

# Kombinasyon olasılığı
def combo_probability(numbers, single_prob, cond_prob):
    prob = 1.0
    for n in numbers:
        prob *= single_prob[n]
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            prob *= cond_prob.at[numbers[i], numbers[j]]
    return prob

# Monte Carlo ile tahmin üret
def generate_predictions(single_prob, cond_prob, n_preds=1, n_numbers=6, trials=10000):
    predictions = []
    numbers_list = list(range(1, 61))
    single_probs_list = single_prob.values

    for _ in range(n_preds):
        best_combo = None
        best_prob = 0

        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=n_numbers, replace=False, p=single_probs_list / single_probs_list.sum())
            chosen = np.sort(chosen)
            if not check_constraints(chosen):
                continue
            p = combo_probability(chosen, single_prob, cond_prob)
            if p > best_prob:
                best_prob = p
                best_combo = chosen

        if best_combo is not None:
            predictions.append((best_combo, best_prob))
        else:
            chosen = np.random.choice(numbers_list, size=n_numbers, replace=False)
            predictions.append((np.sort(chosen), 0))

    return predictions

# Ana Streamlit arayüzü
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")
    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

        st.success(f"Veriler yüklendi. Toplam satır sayısı: {len(df)}")
        st.write(df)  # TÜM satırları göster

        with st.spinner("Olasılıklar hesaplanıyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)

        st.success("Olasılıklar hesaplandı!")

        n_preds = st.number_input("Kaç tahmin üretmek istiyorsunuz?", min_value=1, max_value=10, value=1, step=1)
        if st.button("Tahmin Üret"):
            with st.spinner("Tahminler hesaplanıyor (biraz zaman alabilir)..."):
                preds = generate_predictions(single_prob, cond_prob, n_preds=n_preds)
            st.success("Tahminler hazır!")
            for i, (combo, prob) in enumerate(preds, 1):
                st.write(f"Tahmin {i}: {', '.join(map(str, combo))}  (Olasılık: {prob:.6e})")

if __name__ == "__main__":
    main()
