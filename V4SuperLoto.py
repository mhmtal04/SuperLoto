import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import pymc as pm
from sklearn.model_selection import train_test_split
import xgboost as xgb
import random

# --- Zaman bazlı ağırlık ---
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    weights = (max_days - days_ago) / max_days
    return weights

# --- Tekil sayı ağırlıklı olasılık ---
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

# --- İkili frekanslar ---
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
            cond_prob.loc[a] = pair_freq.loc[a] / freq_a
        else:
            cond_prob.loc[a] = 0
    return cond_prob

# --- Kısıtlar (en az 2 tek, 2 çift) ---
def check_constraints(numbers):
    odd_count = sum(n % 2 == 1 for n in numbers)
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

# --- Monte Carlo tahmin ---
def monte_carlo_predict(single_prob, cond_prob, n_preds=1, n_numbers=6, trials=10000):
    numbers_list = list(range(1, 61))
    single_probs_list = single_prob.values
    predictions = []

    for _ in range(n_preds):
        best_combo = None
        best_prob = 0
        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=n_numbers, replace=False, p=single_probs_list / single_probs_list.sum())
            chosen = np.sort(chosen)
            if not check_constraints(chosen):
                continue
            p = 1
            for n in chosen:
                p *= single_prob[n]
            for i in range(n_numbers):
                for j in range(i+1, n_numbers):
                    p *= cond_prob.at[chosen[i], chosen[j]]
            if p > best_prob:
                best_prob = p
                best_combo = chosen
        if best_combo is not None:
            predictions.append((best_combo, best_prob))
        else:
            chosen = np.random.choice(numbers_list, size=n_numbers, replace=False)
            predictions.append((np.sort(chosen), 0))
    return predictions

# --- Bayesian model ile olasılık tahmini ---
def bayesian_probabilities(df, weights):
    freqs = weighted_single_probabilities(df)
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(60))
        observed = pm.Multinomial('obs', n=6, p=p, observed=np.random.multinomial(6, freqs.values))
        trace = pm.sample(1000, tune=1000, chains=2, progressbar=False)
    bayes_probs = trace.posterior['p'].mean(dim=['chain', 'draw']).values
    return pd.Series(bayes_probs, index=range(1, 61))

# --- Markov Zinciri ile tahmin üretme ---
def markov_predict(df, weights, cond_prob, n_preds=1, n_numbers=6):
    single_prob = weighted_single_probabilities(df)
    predictions = []
    numbers_list = list(range(1, 61))
    for _ in range(n_preds):
        seq = []
        first = np.random.choice(numbers_list, p=single_prob.values / single_prob.sum())
        seq.append(first)
        for _ in range(n_numbers-1):
            last = seq[-1]
            next_probs = cond_prob.loc[last].copy()
            for s in seq:
                next_probs[s] = 0  # Aynı sayıyı seçme
            if next_probs.sum() == 0:
                choices = [x for x in numbers_list if x not in seq]
                next_num = random.choice(choices)
            else:
                next_probs /= next_probs.sum()
                next_num = np.random.choice(numbers_list, p=next_probs.values)
            seq.append(next_num)
        if check_constraints(seq):
            predictions.append((np.array(seq), 0))
        else:
            # Basit random fallback
            predictions.append((np.random.choice(numbers_list, size=n_numbers, replace=False), 0))
    return predictions

# --- XGBoost Model Eğitimi ---
def train_xgboost(df):
    X, y = [], []
    for _, row in df.iterrows():
        feature = np.zeros(60)
        for num in row['Numbers']:
            feature[num - 1] = 1
        X.append(feature)
        y.append(feature)
    X = np.array(X)
    y = np.array(y)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    models = []
    for i in range(60):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train[:, i])
        models.append(model)
    return models

# --- XGBoost ile tahmin üret ---
def xgboost_predict(models, son_cekilis):
    feature = np.zeros(60)
    for num in son_cekilis:
        feature[num - 1] = 1
    preds = np.array([model.predict_proba(feature.reshape(1, -1))[0][1] for model in models])
    preds /= preds.sum()
    tahmin = preds.argsort()[-6:][::-1] + 1
    return tahmin

# --- Tahminleri Birleştir ---
def combine_predictions(monte_carlo_preds, bayesian_probs, markov_preds, xgb_preds, n_preds):
    combined = []
    for i in range(n_preds):
        mc_nums, _ = monte_carlo_preds[i] if i < len(monte_carlo_preds) else ([], 0)
        mk_nums, _ = markov_preds[i] if i < len(markov_preds) else ([], 0)
        xb_nums = xgb_preds if xgb_preds is not None else []

        # Ortalama al, benzersiz sayılar al
        all_nums = np.concatenate([mc_nums, mk_nums, xb_nums])
        unique_nums = np.unique(all_nums)
        if len(unique_nums) >= 6:
            chosen = np.random.choice(unique_nums, 6, replace=False)
        else:
            chosen = unique_nums
        if not check_constraints(chosen):
            # Gerekiyorsa tekrar random üret
            chosen = np.random.choice(range(1, 61), 6, replace=False)
        combined.append(np.sort(chosen))
    return combined

# --- Streamlit arayüzü ---
def main():
    st.title("Süper Loto Çok Katmanlı Tahmin Botu")

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

        st.success(f"Veriler yüklendi. Toplam çekiliş: {len(df)}")

        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1, step=1)

        with st.spinner("Modeller eğitiliyor ve tahminler üretiliyor..."):
            weights = get_weights(df['Date'])
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)

            monte_preds = monte_carlo_predict(single_prob, cond_prob, n_preds=n_preds)
            bayes_probs = bayesian_probabilities(df, weights)
            markov_preds = markov_predict(df, weights, cond_prob, n_preds=n_preds)
            xgb_models = train_xgboost(df)
            son_cekilis = df.iloc[-1]['Numbers']
            xgb_pred = x
