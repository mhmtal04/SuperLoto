
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime

# Ağırlık hesaplama (zaman ağırlıklı frekanslar için)
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    weights = (max_days - days_ago) / max_days
    return weights

# Tekil sayıların ağırlıklı olasılığı
def weighted_single_probabilities(df):
    weights = get_weights(df['Date'])
    total_weight = weights.sum()
    freq = pd.Series(0, index=range(1, 61), dtype=float)

    for idx, row in df.iterrows():
        w = weights[idx]
        for n in row['Numbers']:
            freq[n] += w

    prob = freq / total_weight
    return prob

# İkili sayıların beraber çıkma frekansı
def pair_frequencies(df):
    weights = get_weights(df['Date'])
    pair_freq = pd.DataFrame(0, index=range(1,61), columns=range(1,61), dtype=float)

    for idx, row in df.iterrows():
        w = weights[idx]
        for a, b in combinations(row['Numbers'], 2):
            pair_freq.at[a,b] += w
            pair_freq.at[b,a] += w

    return pair_freq

# Koşullu olasılık: P(b|a)
def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1,61), columns=range(1,61), dtype=float)

    for a in range(1, 61):
        freq_a = single_prob[a]
        if freq_a > 0:
            for b in range(1, 61):
                cond_prob.at[a, b] = pair_freq.at[a, b] / freq_a
        else:
            cond_prob.loc[a, :] = 0

    return cond_prob

# Skor hesaplayan gelişmiş formül
def scored_combo_probability(numbers, single_prob, cond_prob):
    prob = 1.0
    for n in numbers:
        prob *= single_prob[n]
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            prob *= cond_prob.at[numbers[i], numbers[j]]

    # Yumuşak ceza: en az 2 tek ve 2 çift sayı şartı
    odd = sum(n % 2 == 1 for n in numbers)
    even = len(numbers) - odd
    penalty = 1.0 if odd >= 2 and even >= 2 else (odd/6)*(even/6)
    return prob * penalty

# Monte Carlo tahmin üretici
def generate_predictions(single_prob, cond_prob, n_preds=1, n_numbers=6, trials=10000):
    predictions = []
    numbers_list = list(range(1, 61))
    single_probs_list = single_prob.values

    for _ in range(n_preds):
        best_combo = None
        best_score = 0

        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=n_numbers, replace=False, p=single_probs_list/single_probs_list.sum())
            chosen = np.sort(chosen)
            score = scored_combo_probability(chosen, single_prob, cond_prob)
            if score > best_score:
                best_score = score
                best_combo = chosen

        if best_combo is not None:
            predictions.append((best_combo, best_score))
        else:
            chosen = np.random.choice(numbers_list, size=n_numbers, replace=False)
            predictions.append((np.sort(chosen), 0))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# Streamlit arayüzü
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")
    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

        st.success("Veriler yüklendi. İlk 20 satır gösteriliyor:")
        st.dataframe(df.head(20))

        with st.spinner("Olasılıklar hesaplanıyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)

        st.success("Olasılıklar hesaplandı!")

        n_preds = st.number_input("Kaç tahmin üretmek istiyorsunuz?", min_value=1, max_value=10, value=1)
        if st.button("Tahmin Üret"):
            with st.spinner("Tahminler oluşturuluyor..."):
                preds = generate_predictions(single_prob, cond_prob, n_preds=n_preds)
            st.success("Tahminler hazır!")
            for i, (combo, score) in enumerate(preds, 1):
                st.write(f"**Tahmin {i}:** {', '.join(map(str, combo))} — Skor: `{score:.6e}`")

if __name__ == "__main__":
    main()
