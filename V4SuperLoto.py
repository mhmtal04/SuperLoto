import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from collections import defaultdict
from datetime import datetime

np.random.seed(42)

# --- 1. Veri Ön İşleme ve Ağırlıklandırma ---
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    numbers = df.loc[:, 'Num1':'Num6'].values
    max_date = df['Date'].max()
    days_diff = (max_date - df['Date']).dt.days
    time_weights = 1 / (1 + days_diff)  # Daha yeni çekiliş daha yüksek ağırlık

    freq_weighted = defaultdict(float)
    for i, row in enumerate(numbers):
        for num in row:
            freq_weighted[num] += time_weights.iloc[i]
    total_weight = sum(freq_weighted.values())
    freq_weighted = {k: v / total_weight for k, v in freq_weighted.items()}

    pair_freq = defaultdict(float)
    for i, row in enumerate(numbers):
        row_sorted = sorted(row)
        for i1 in range(len(row_sorted)):
            for i2 in range(i1 + 1, len(row_sorted)):
                pair = (row_sorted[i1], row_sorted[i2])
                pair_freq[pair] += time_weights.iloc[i]
    total_pair_weight = sum(pair_freq.values())
    pair_freq = {k: v / total_pair_weight for k, v in pair_freq.items()}

    cond_prob = {}
    for (a, b), freq_ab in pair_freq.items():
        if freq_weighted.get(a, 0) > 0:
            cond_prob[(a, b)] = freq_ab / freq_weighted[a]
        else:
            cond_prob[(a, b)] = 0.0
        if freq_weighted.get(b, 0) > 0:
            cond_prob[(b, a)] = freq_ab / freq_weighted[b]
        else:
            cond_prob[(b, a)] = 0.0

    return numbers, freq_weighted, pair_freq, cond_prob, time_weights

# --- 2. Bayesian Model ---
def bayesian_model(freq_weighted, cond_prob):
    numbers = np.array(list(freq_weighted.keys()))
    probs = np.array(list(freq_weighted.values()))
    probs = probs / probs.sum()
    chosen = np.random.choice(numbers, size=6, replace=False, p=probs)
    return sorted(chosen)

# --- 3. Markov Zinciri Modeli ---
def markov_chain_model(numbers, time_weights):
    transition_counts = np.zeros((60, 60))
    for row in numbers:
        sorted_row = sorted(row)
        for i in range(len(sorted_row) - 1):
            transition_counts[sorted_row[i] - 1, sorted_row[i + 1] - 1] += 1
    row_sums = transition_counts.sum(axis=1)
    transition_probs = np.divide(
        transition_counts,
        row_sums[:, None],
        out=np.zeros_like(transition_counts),
        where=row_sums[:, None] != 0,
    )
    start_probs = row_sums / row_sums.sum()
    start = np.random.choice(np.arange(60), p=start_probs)
    result = [start + 1]
    for _ in range(5):
        next_prob = transition_probs[result[-1] - 1]
        if next_prob.sum() == 0:
            next_num = np.random.choice(60)
        else:
            next_num = np.random.choice(np.arange(60), p=next_prob)
        result.append(next_num + 1)
    return sorted(result)

# --- 4. XGBoost Modeli ---
def xgboost_model(df):
    features = []
    for _, row in df.iterrows():
        feature = np.zeros(60)
        numbers = row.loc['Num1':'Num6'].values
        feature[numbers - 1] = 1
        features.append(feature)
    X = np.array(features[:-1])
    y = np.array(features[1:])
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    return model

# --- 5. Kısıtlar ---
def check_constraints(nums):
    odd_count = sum(n % 2 == 1 for n in nums)
    even_count = 6 - odd_count
    return odd_count >= 2 and even_count >= 2

# --- 6. Ensemble Tahmin ---
def ensemble_prediction(freq_weighted, cond_prob, numbers, time_weights, model, n_preds=1):
    preds = []
    attempts = 0
    while len(preds) < n_preds and attempts < n_preds * 10:
        bayes_pred = bayesian_model(freq_weighted, cond_prob)
        markov_pred = markov_chain_model(numbers, time_weights)

        feature = np.zeros(60)
        feature[bayes_pred - 1] = 1

        xgb_pred_probs = model.predict_proba(feature.reshape(1, -1))[0]
        xgb_pred = np.argsort(xgb_pred_probs)[-6:] + 1

        combined = sorted(set(bayes_pred) | set(markov_pred) | set(xgb_pred))

        if len(combined) > 6:
            combined = np.random.choice(combined, 6, replace=False).tolist()

        if check_constraints(combined):
            preds.append(sorted(combined))
        attempts += 1

    return preds

# --- 7. Streamlit Arayüzü ---
def main():
    st.title(
        "Süper Loto Gelişmiş Tahmin Botu (Bayesian - Markov - XGBoost - Koşullu Olasılık - Kısıtlar)"
    )
    uploaded_file = st.file_uploader(
        "CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Veri yüklendi!")

        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1)

        with st.spinner("Tahminler hesaplanıyor..."):
            numbers, freq_weighted, pair_freq, cond_prob, time_weights = preprocess_data(df)
            model = xgboost_model(df)
            preds = ensemble_prediction(freq_weighted, cond_prob, numbers, time_weights, model, n_preds=n_preds)

        st.subheader("Tahminler")
        for i, pred in enumerate(preds):
            st.write(f"{i+1}. Tahmin: {sorted(pred)}")

if __name__ == "__main__":
    main()
