import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pymc as pm
import arviz as az

# Zaman bazlı ağırlık
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    weights = (max_days - days_ago) / max_days
    return weights

# Tekil sayı olasılığı
def weighted_single_probabilities(df):
    weights = get_weights(df['Date'])
    total_weight = weights.sum()

    freq = pd.Series(0, index=range(1, 61), dtype=float)
    for idx, row in df.iterrows():
        for n in row['Numbers']:
            freq[n] += weights[idx]
    prob = freq / total_weight
    return prob

# İkili sayı frekansı
def pair_frequencies(df):
    pair_freq = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    weights = get_weights(df['Date'])
    for idx, row in df.iterrows():
        w = weights[idx]
        for a, b in combinations(row['Numbers'], 2):
            pair_freq.at[a, b] += w
            pair_freq.at[b, a] += w
    return pair_freq

# Koşullu olasılık
def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    for a in range(1, 61):
        for b in range(1, 61):
            if single_prob[a] > 0:
                cond_prob.at[a, b] = pair_freq.at[a, b] / single_prob[a]
    return cond_prob

# Bayesian model ile olasılık tahmini
def bayesian_number_model(df):
    freqs = weighted_single_probabilities(df)
    with pm.Model() as model:
        probs = pm.Dirichlet("probs", a=np.ones(60))
        observations = df['Numbers'].explode() - 1
        pm.Categorical("obs", p=probs, observed=observations)
        trace = pm.sample(1000, tune=1000, chains=2, progressbar=False)
    mean_probs = np.mean(trace.posterior["probs"].stack(sample=("chain", "draw")).values, axis=1)
    return pd.Series(mean_probs, index=range(1, 61))

# Markov zinciri geçiş olasılıkları
def markov_chain_model(df):
    transitions = defaultdict(lambda: np.zeros(60))
    for _, row in df.iterrows():
        sorted_nums = sorted(row["Numbers"])
        for i in range(len(sorted_nums) - 1):
            transitions[sorted_nums[i] - 1][sorted_nums[i + 1] - 1] += 1
    for key in transitions:
        total = transitions[key].sum()
        if total > 0:
            transitions[key] /= total
    return transitions

# XGBoost veri hazırlığı
def prepare_xgb_data(df):
    rows = []
    labels = []
    for _, row in df.iterrows():
        for n in range(1, 61):
            feature = [int(n in row['Numbers'])]
            for a in row['Numbers']:
                feature.append(int(n == a))
            rows.append(feature)
            labels.append(int(n in row['Numbers']))
    return np.array(rows), np.array(labels)

# XGBoost tahmini
def xgboost_model(df):
    X, y = prepare_xgb_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(np.eye(60))[:, 1]
    return pd.Series(pred_proba, index=range(1, 61))

# Kombinasyon olasılığı
def combo_probability(numbers, single_prob, cond_prob):
    prob = 1.0
    for n in numbers:
        prob *= single_prob[n]
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            prob *= cond_prob.at[numbers[i], numbers[j]]
    return prob

# Kısıt kontrol
def check_constraints(numbers):
    odd_count = sum(n % 2 == 1 for n in numbers)
    return odd_count >= 2 and (len(numbers) - odd_count) >= 2

# Tahmin üretici
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
    return predictions

# Streamlit UI
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu (Bayesian - Markov - XGBoost)")
    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

        st.success("Veri yüklendi!")
        st.write(df)

        with st.spinner("Olasılıklar hesaplanıyor..."):
            bayes_probs = bayesian_number_model(df)
            xgb_probs = xgboost_model(df)
            combined_probs = (bayes_probs + xgb_probs) / 2
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(combined_probs, pair_freq)
            markov_model = markov_chain_model(df)

        st.success("Modeller tamamlandı!")

        n_preds = st.number_input("Kaç tahmin üretmek istersiniz?", min_value=1, max_value=10, value=1)

        if st.button("Tahminleri Üret"):
            with st.spinner("Tahminler hazırlanıyor..."):
                preds = generate_predictions(combined_probs, cond_prob, n_preds=n_preds)
            st.success("Tahminler üretildi!")
            for i, (combo, prob) in enumerate(preds, 1):
                st.write(f"Tahmin {i}: {', '.join(map(str, combo))}  (Olasılık: {prob:.6e})")

if __name__ == "__main__":
    main()
