import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB

# --- Yardımcı Fonksiyonlar ---
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    return (max_days - days_ago) / max_days

def check_constraints(numbers):
    odd_count = sum(n % 2 == 1 for n in numbers)
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

# --- Olasılık Hesaplamaları ---
def weighted_single_probabilities(df):
    weights = get_weights(df['Date'])
    total_weight = weights.sum()
    freq = pd.Series(0, index=range(1, 61), dtype=float)
    for idx, row in df.iterrows():
        w = weights[idx]
        for n in row['Numbers']:
            freq[n] += w
    return freq / total_weight

def pair_frequencies(df):
    pair_freq = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    weights = get_weights(df['Date'])
    for idx, row in df.iterrows():
        w = weights[idx]
        for a, b in combinations(row['Numbers'], 2):
            pair_freq.at[a, b] += w
            pair_freq.at[b, a] += w
    return pair_freq

def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    for a in range(1, 61):
        if single_prob[a] > 0:
            cond_prob.loc[a] = pair_freq.loc[a] / single_prob[a]
    return cond_prob

# --- Makine Öğrenimi Modelleri ---
def train_naive_bayes(df):
    X = np.repeat(df.index.values.reshape(-1, 1), 6, axis=0)
    y = np.array([n for row in df['Numbers'] for n in row])
    model = GaussianNB()
    model.fit(X, y)
    return model

def train_gradient_boost(df):
    X = np.repeat(df.index.values.reshape(-1, 1), 6, axis=0)
    y = np.array([n for row in df['Numbers'] for n in row])
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

def markov_chain(df):
    transitions = np.zeros((61, 61))
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]['Numbers']
        curr = df.iloc[i]['Numbers']
        for a in prev:
            for b in curr:
                transitions[a][b] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums!=0)
    return transition_probs

# --- Yeni Matematiksel Formül Skoru ---
def custom_formula_score(combo, single_prob, pair_freq):
    # Skor = Π(P(n_i)) * Π(F(n_i, n_j))
    prod_single = np.prod([single_prob[n] for n in combo])
    prod_pair = 1.0
    for a, b in combinations(combo, 2):
        freq_ab = pair_freq.at[a, b]
        prod_pair *= freq_ab if freq_ab > 0 else 1e-6
    return prod_single * prod_pair

# --- Tahmin Üret ---
def generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_probs, pair_freq, n_preds=1, trials=5000):
    predictions = []
    numbers_list = list(range(1, 61))
    single_probs_list = single_prob.values

    for _ in range(n_preds):
        best_combo = None
        best_score = -1
        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=6, replace=False, p=single_probs_list / single_probs_list.sum())
            chosen = np.sort(chosen)
            if not check_constraints(chosen):
                continue

            # Kombine skorlama (frekans & koşul)
            combo_score = 1.0
            for i in range(6):
                combo_score *= single_prob[chosen[i]]
                for j in range(i+1, 6):
                    combo_score *= cond_prob.at[chosen[i], chosen[j]]

            # Naive Bayes skoru
            X_test = np.array([[len(df) + 1]])
            classes = nb_model.classes_
            probs = nb_model.predict_proba(X_test)[0]
            nb_score = np.mean([probs[np.where(classes == n)[0][0]] if n in classes else 0 for n in chosen])

            # Gradient Boost skoru
            gb_pred = gb_model.predict(X_test)[0]

            # Markov skoru
            markov_score = np.mean([markov_probs[a].mean() if a < markov_probs.shape[0] else 0 for a in chosen])

            # Yeni formül skoru
            custom_score = custom_formula_score(chosen, single_prob, pair_freq)

            # Nihai skor
            final_score = (combo_score
                           * (1 + nb_score)
                           * (1 + gb_pred / 60.0)
                           * (1 + markov_score)
                           * (1 + custom_score))

            if final_score > best_score:
                best_score = final_score
                best_combo = chosen

        if best_combo is not None:
            predictions.append((best_combo, best_score))

    return predictions

# --- Streamlit Arayüz ---
def main():
    st.title("Süper Loto | Gelişmiş Tahmin Botu v6")

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

        st.success(f"Veriler yüklendi. Toplam satır: {len(df)}")
        st.write(df)

        with st.spinner("Model eğitiliyor ve olasılıklar hesaplanıyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)
            nb_model = train_naive_bayes(df)
            gb_model = train_gradient_boost(df)
            markov_probs = markov_chain(df)

        n_preds = st.number_input("Kaç tahmin üretmek istiyorsunuz?", min_value=1, max_value=10, value=3, step=1)

        if st.button("Tahminleri Hesapla"):
            with st.spinner("Tahminler hesaplanıyor..."):
                preds = generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_probs, pair_freq, n_preds=n_preds)
            st.success("Tahminler hazır!")
            for i, (comb, _) in enumerate(preds):
                st.write(f"{i+1}. Tahmin: {', '.join(map(str, comb))}")

if __name__ == "__main__":
    main()
