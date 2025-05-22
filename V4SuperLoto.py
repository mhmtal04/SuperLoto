import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import pymc as pm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# --- Yardımcı Fonksiyonlar ---

def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    weights = (max_days - days_ago) / max_days
    return weights

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
    prob = prob.clip(lower=1e-6)  # Sıfır olmasın
    return prob

def pair_frequencies(df):
    pair_freq = pd.DataFrame(1, index=range(1, 61), columns=range(1, 61), dtype=float)  # smoothing=1
    weights = get_weights(df['Date'])
    for idx, row in df.iterrows():
        numbers = row['Numbers']
        w = weights[idx]
        for a, b in combinations(numbers, 2):
            pair_freq.at[a, b] += w
            pair_freq.at[b, a] += w
    return pair_freq

def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    for a in range(1, 61):
        freq_a = single_prob[a]
        if freq_a > 0:
            cond_prob.loc[a] = pair_freq.loc[a] / freq_a
        else:
            cond_prob.loc[a] = 0
    cond_prob = cond_prob.clip(lower=1e-6)
    return cond_prob

def check_constraints(numbers):
    odd_count = sum(n % 2 == 1 for n in numbers)
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

def bayesian_number_model(df):
    observations = []
    for nums in df['Numbers']:
        observations.extend(nums)
    observations = np.array(observations) - 1
    with pm.Model() as model:
        alpha = np.ones(60)
        probs = pm.Dirichlet("probs", a=alpha)
        obs = pm.Categorical("obs", p=probs, observed=observations)
        trace = pm.sample(draws=1000, tune=500, chains=2, progressbar=False, random_seed=42)
    bayes_probs = trace.posterior['probs'].mean(dim=["chain", "draw"]).values
    bayes_probs = np.clip(bayes_probs, 1e-6, None)
    return bayes_probs

def markov_chain_model(df):
    transition_counts = np.ones((60, 60))  # smoothing
    numbers_list = df['Numbers'].tolist()
    for i in range(len(numbers_list) - 1):
        current = numbers_list[i]
        next_ = numbers_list[i + 1]
        for c in current:
            for n in next_:
                transition_counts[c - 1, n - 1] += 1
    transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.clip(transition_probs, 1e-6, None)
    return transition_probs

def xgboost_model(df):
    all_numbers = list(range(1, 61))
    encoder = OneHotEncoder(categories=[all_numbers] * 18, sparse=False, handle_unknown='ignore')
    X, y = [], []
    numbers_list = df['Numbers'].tolist()
    for i in range(3, len(numbers_list)):
        past_3 = numbers_list[i-3:i]
        flat = [num for sublist in past_3 for num in sublist]
        X.append(flat)
        y.append(numbers_list[i])
    X = np.array(X)
    y = np.array(y) - 1
    X_enc = encoder.fit_transform(X)
    models = []
    for i in range(6):
        m = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=50)
        y_i = y[:, i]
        X_train, X_val, y_train, y_val = train_test_split(X_enc, y_i, test_size=0.2, random_state=42, stratify=y_i)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        models.append(m)
    return models, encoder

def combined_score(numbers, single_prob, cond_prob, bayes_probs, markov_probs, xgb_models, encoder, prev_numbers):
    prob_cond = 1.0
    for n in numbers:
        prob_cond *= single_prob[n]
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            prob_cond *= cond_prob.at[numbers[i], numbers[j]]
    prob_bayes = np.prod(bayes_probs[numbers])
    prob_markov = 1.0
    for n in numbers:
        prob_markov *= markov_probs[prev_numbers[n] - 1, n]
    flat_input = []
    for i in range(3):  # son 3 çekiliş
        flat_input.extend(prev_numbers)
    X_input = np.array(flat_input).reshape(1, -1)
    X_input_enc = encoder.transform(X_input)
    prob_xgb = 1.0
    for i, model in enumerate(xgb_models):
        preds = model.predict_proba(X_input_enc)[0]
        prob_xgb *= preds[numbers[i]]
    # log topla
    score_log = np.log(prob_cond + 1e-12) + np.log(prob_bayes + 1e-12) + np.log(prob_markov + 1e-12) + np.log(prob_xgb + 1e-12)
    score = np.exp(score_log)
    return score

def generate_predictions(df, n_preds=1, trials=2000):
    single_prob = weighted_single_probabilities(df)
    pair_freq = pair_frequencies(df)
    cond_prob = conditional_probabilities(single_prob, pair_freq)
    bayes_probs = bayesian_number_model(df)
    markov_probs = markov_chain_model(df)
    xgb_models, encoder = xgboost_model(df)
    predictions = []
    prev_numbers = df['Numbers'].iloc[-1]
    for _ in range(n_preds):
        best_combo = None
        best_score = -1
        for __ in range(trials):
            candidate = np.sort(np.random.choice(range(1, 61), size=6, replace=False, p=single_prob.values))
            if not check_constraints(candidate):
                continue
            score = combined_score(candidate - 1, single_prob, cond_prob, bayes_probs, markov_probs, xgb_models, encoder, prev_numbers)
            if score > best_score:
                best_score = score
                best_combo = candidate
        if best_combo is None:
            best_combo = np.sort(np.random.choice(range(1, 61), size=6, replace=False))
            best_score = 0
        predictions.append((best_combo, best_score))
    return predictions

# --- Streamlit UI ---

def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if {'Date', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6'}.issubset(df.columns):
            df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()
            n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1)
            if st.button("Tahminleri Hesapla"):
                with st.spinner('Tahminler hesaplanıyor, lütfen bekleyin...'):
                    preds = generate_predictions(df, n_preds=n_preds, trials=2000)
                for i, (nums, score) in enumerate(preds, 1):
                    st.write(f"Tahmin {i}: {list(nums)} (Skor: {score:.5f})")
        else:
            st.error("CSV dosyasında Date, Num1, Num2, Num3, Num4, Num5, Num6 sütunları olmalı.")

if __name__ == "__main__":
    main()
