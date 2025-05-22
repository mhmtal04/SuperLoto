import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
import xgboost as xgb

# --- Ağırlık hesaplama (zaman bazlı) ---
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    weights = (max_days - days_ago) / max_days
    return weights

# --- Tekil sayıların ağırlıklı olasılıkları ---
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

# --- İkili sayıların frekansı ---
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

# --- Koşullu olasılıklar ---
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

# --- Kısıt: en az 2 tek ve 2 çift sayı ---
def check_constraints(numbers):
    odd_count = sum(n % 2 == 1 for n in numbers)
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

# --- Kombinasyon olasılığı ---
def combo_probability(numbers, single_prob, cond_prob):
    prob = 1.0
    for n in numbers:
        prob *= single_prob[n]
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            prob *= cond_prob.at[numbers[i], numbers[j]]
    return prob

# --- Monte Carlo ile tahmin üret ---
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

# --- Bayesian Model ---
def bayesian_model(single_prob, n_numbers=6):
    # En yüksek tekil olasılıkları seç
    numbers = single_prob.sort_values(ascending=False).index[:n_numbers]
    return np.array(numbers)

# --- Markov Zinciri Model ---
def markov_chain_model(df, n_numbers=6):
    transition = pd.DataFrame(0, index=range(1,61), columns=range(1,61), dtype=float)
    total_transitions = pd.Series(0, index=range(1,61), dtype=float)

    draws = df['Numbers'].tolist()
    for i in range(len(draws)-1):
        current_draw = draws[i]
        next_draw = draws[i+1]
        for num in current_draw:
            total_transitions[num] += 1
            for nxt in next_draw:
                transition.at[num, nxt] += 1

    for num in range(1,61):
        if total_transitions[num] > 0:
            transition.loc[num] /= total_transitions[num]

    last_draw = draws[-1]
    scores = pd.Series(0, index=range(1,61), dtype=float)
    for num in last_draw:
        scores += transition.loc[num]
    scores = scores / len(last_draw)

    pred = scores.sort_values(ascending=False).index[:n_numbers]
    return np.array(pred)

# --- XGBoost Model Eğitimi ---
def xgboost_model(df):
    df = df.reset_index(drop=True)
    X = []
    y = []

    NUMBERS_RANGE = 60
    NUMBERS_DRAWN = 6

    for idx in range(len(df) - 1):
        feature = np.zeros(NUMBERS_RANGE, dtype=int)
        # Set 1 for drawn numbers in idx-th draw
        for num in df.loc[idx]['Numbers']:
            feature[num-1] = 1
        X.append(feature)
        target = np.array([num-1 for num in df.loc[idx + 1]['Numbers']])
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    models = []
    for i in range(NUMBERS_DRAWN):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
        target = y[:, i].astype(int)
        model.fit(X, target)
        models.append(model)
    return models

# --- XGBoost Tahmin ---
def predict_xgboost(models, last_draw):
    feature = np.zeros(60, dtype=int)
    for num in last_draw:
        feature[num-1] = 1
    preds = []
    for m in models:
        pred = m.predict_proba(feature.reshape(1, -1))[0]
        preds.append(pred)
    return np.array(preds)

# --- Modellerin Kombinasyonu ---
def combined_predictions(df, single_prob, cond_prob, n_preds=1, n_numbers=6):
    preds = []

    # Bayesian
    bayes_pred = bayesian_model(single_prob, n_numbers)

    # Markov
    markov_pred = markov_chain_model(df, n_numbers)

    # XGBoost
    xgb_models = xgboost_model(df)
    last_draw = df['Numbers'].iloc[-1]
    xgb_probs = predict_xgboost(xgb_models, last_draw)

    xgb_scores = xgb_probs.sum(axis=0)

    combined_scores = pd.Series(0, index=range(1,61), dtype=float)

    # Bayesian ve Markov puanları (1 puan)
    combined_scores[bayes_pred] += 1.0
    combined_scores[markov_pred] += 1.0

    # XGBoost skorlarını normalize edip ekle
    if xgb_scores.max() - xgb_scores.min() > 0:
        xgb_scores_norm = (xgb_scores - xgb_scores.min()) / (xgb_scores.max() - xgb_scores.min())
    else:
        xgb_scores_norm = xgb_scores
    combined_scores += xgb_scores_norm

    top_numbers = combined_scores.sort_values(ascending=False).index[:n_numbers]

    if check_constraints(top_numbers):
        preds.append((np.array(top_numbers), None))
    else:
        chosen = np.random.choice(range(1,61), size=n_numbers, replace=False)
        preds.append((np.sort(chosen), None))

    return preds

# --- Ana Streamlit Arayüzü ---
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

        st.success(f"Veriler yüklendi. Toplam satır sayısı: {len(df)}")
        st.write(df)  # CSV'deki tüm veriler gösteriliyor

        with st.spinner("Olasılıklar hesaplanıyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)

        st.success("Olasılıklar hesaplandı!")

        n_preds = st.number_input("Kaç tahmin üretmek istiyorsunuz?", min_value=1, max_value=10, value=1, step=1)

        if st.button("Tahmin Üret"):
            with st.spinner("Tahminler hesaplanıyor (biraz zaman alabilir)..."):
                preds = combined_predictions(df, single_prob, cond_prob, n_preds=n_preds)

            st.success("Tahminler hazır!")
            for
