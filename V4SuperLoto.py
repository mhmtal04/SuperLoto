import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pymc as pm
from sklearn.preprocessing import OneHotEncoder

# Sabitler
NUMBERS_RANGE = 60
NUMBERS_DRAWN = 6

# --- Yardımcı Fonksiyonlar ---

def load_data(path):
    df = pd.read_csv(path)
    # Çekiliş sayılarının sütun isimleri örneği: Num1, Num2, ..., Num6
    number_cols = [f'Num{i}' for i in range(1, NUMBERS_DRAWN+1)]
    df = df.dropna(subset=number_cols)
    for col in number_cols:
        df[col] = df[col].astype(int)
    return df

def get_weights(dates):
    # Tarihe göre zaman ağırlıklı ağırlık üret (daha yeni tarihe daha fazla ağırlık)
    dates = pd.to_datetime(dates)
    weights = (dates - dates.min()).dt.days + 1
    weights = weights / weights.sum()
    return weights.values

def compute_singular_frequencies(df, weights):
    counts = np.zeros(NUMBERS_RANGE)
    for idx, row in df.iterrows():
        for n in row:
            counts[n-1] += weights[idx]
    return counts / counts.sum()

def compute_pair_frequencies(df, weights):
    pair_counts = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    for idx, row in df.iterrows():
        nums = row.values - 1
        for i in range(NUMBERS_DRAWN):
            for j in range(i+1, NUMBERS_DRAWN):
                pair_counts[nums[i], nums[j]] += weights[idx]
                pair_counts[nums[j], nums[i]] += weights[idx]
    # Normalize
    pair_probs = pair_counts / pair_counts.sum()
    return pair_probs

def check_constraints(nums):
    # En az 2 tek ve 2 çift sayı olmalı
    odds = sum(1 for n in nums if n % 2 == 1)
    evens = len(nums) - odds
    return odds >= 2 and evens >= 2

# --- Modeller ---

def bayesian_model(df, weights):
    singular_freq = compute_singular_frequencies(df, weights)
    with pm.Model() as model:
        probs = pm.Dirichlet('probs', a=singular_freq * 100 + 1)
        trace = pm.sample(500, tune=300, chains=1, progressbar=False, random_seed=42)
    bayes_probs = np.mean(trace['probs'], axis=0)
    return bayes_probs

def markov_model(df, weights):
    pair_probs = compute_pair_frequencies(df, weights)
    # Basit: her sayının çıkma olasılığına göre normalize edildi
    markov_probs = pair_probs.mean(axis=0)
    return markov_probs / markov_probs.sum()

def xgboost_model(df):
    df = df.reset_index(drop=True)
    X = []
    y = []
    number_cols = [f'Num{i}' for i in range(1, NUMBERS_DRAWN+1)]

    for idx in range(len(df) - 1):
        feature = np.zeros(NUMBERS_RANGE)
        current_numbers = df.loc[idx, number_cols].values
        feature[current_numbers - 1] = 1
        X.append(feature)

        next_numbers = df.loc[idx + 1, number_cols].values
        y.append(next_numbers - 1)

    X = np.array(X)
    y = np.array(y)

    models = []
    for i in range(NUMBERS_DRAWN):
        model = xgb.XGBClassifier(eval_metric='mlogloss', verbosity=0)
        target = y[:, i]
        model.fit(X, target)
        models.append(model)
    return models

def predict_xgboost(models, last_draw):
    feature = np.zeros(NUMBERS_RANGE)
    feature[last_draw - 1] = 1
    preds = []
    for m in models:
        preds.append(m.predict_proba(feature.reshape(1, -1))[0])
    return np.array(preds)

# --- Tahmin Birleştirme ve Kısıt Uygulama ---

def generate_combined_predictions(df, n_preds=10):
    weights = get_weights(df['Date'])

    bayes_probs = bayesian_model(df[[f'Num{i}' for i in range(1, NUMBERS_DRAWN+1)]], weights)
    markov_probs = markov_model(df[[f'Num{i}' for i in range(1, NUMBERS_DRAWN+1)]], weights)

    xgb_models = xgboost_model(df[[f'Num{i}' for i in range(1, NUMBERS_DRAWN+1)]])
    last_draw = df.loc[len(df) - 1, [f'Num{i}' for i in range(1, NUMBERS_DRAWN+1)]].values
    xgb_preds = predict_xgboost(xgb_models, last_draw)

    # Ağırlıklı ortalama (örnek ağırlıklar)
    combined_probs = 0.4 * bayes_probs + 0.3 * markov_probs + 0.3 * xgb_preds.mean(axis=0)

    # En iyi n_preds tahmini üret (kısıtlarla)
    predictions = []
    attempts = 0
    while len(predictions) < n_preds and attempts < 1000:
        candidate = np.random.choice(np.arange(1, NUMBERS_RANGE+1), size=NUMBERS_DRAWN, replace=False, p=combined_probs)
        if check_constraints(candidate):
            candidate_sorted = np.sort(candidate)
            if not any(np.array_equal(candidate_sorted, p) for p in predictions):
                predictions.append(candidate_sorted)
        attempts += 1

    return predictions

# --- Streamlit UI ---

def main():
    st.title("Süper Loto Tahmin Sistemi")

    uploaded_file = st.file_uploader("Çekiliş verisi yükle (CSV)", type=["csv"])
    n_preds = st.slider("Kaç tahmin üretilsin?", 1, 20, 10)

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if st.button("Tahminleri Hesapla"):
            with st.spinner("Tahminler hesaplanıyor..."):
                preds = generate_combined_predictions(df, n_preds)
            st.success("Tahminler hazır!")

            for i, p in enumerate(preds, 1):
                st.write(f"{i}. Tahmin: {p}")

if __name__ == "__main__":
    main() 
