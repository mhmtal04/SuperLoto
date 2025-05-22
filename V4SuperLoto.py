import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pymc as pm
import arviz as az
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

NUMBERS_RANGE = 60  # 1-60 arası sayı
NUMBERS_DRAWN = 6   # Süper Loto'da çekilen sayı adedi

def get_weights(dates):
    # Tarihe göre zaman bazlı ağırlık (en yeni tarihe en yüksek ağırlık)
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days
    weights = 1 / (1 + days_diff)
    weights /= weights.sum()
    return weights.values

def bayesian_model(df, weights):
    counts = np.zeros(NUMBERS_RANGE)
    for row, w in zip(df.values, weights):
        for n in row:
            counts[n - 1] += w
    total = counts.sum()
    probs = counts / total
    return probs

def pair_frequency(df, weights):
    pair_counts = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    for row, w in zip(df.values, weights):
        for i, j in combinations(row - 1, 2):
            pair_counts[i, j] += w
            pair_counts[j, i] += w
    pair_probs = pair_counts / pair_counts.sum(axis=1, keepdims=True)
    pair_probs = np.nan_to_num(pair_probs)
    return pair_probs

def conditional_probs(df, weights):
    # Bir sayının çıkması diğerinin çıkma olasılığını hesapla (koşullu olasılık)
    pair_probs = pair_frequency(df, weights)
    single_probs = bayesian_model(df, weights)
    cond_probs = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    for i in range(NUMBERS_RANGE):
        for j in range(NUMBERS_RANGE):
            if single_probs[i] > 0:
                cond_probs[i, j] = pair_probs[i, j] / single_probs[i]
    cond_probs = np.nan_to_num(cond_probs)
    return cond_probs

def xgboost_model(df):
    df = df.reset_index(drop=True)
    X = []
    y = []

    for idx in range(len(df)-1):
        feature = np.zeros(NUMBERS_RANGE)
        feature[df.loc[idx].values - 1] = 1
        X.append(feature)
        y.append(df.loc[idx+1].values - 1)

    X = np.array(X)
    y = np.array(y)

    models = []
    for i in range(NUMBERS_DRAWN):
        m = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        m.fit(X, y[:, i])
        models.append(m)
    return models

def predict_xgboost(models, last_draw):
    feature = np.zeros(NUMBERS_RANGE)
    feature[last_draw - 1] = 1
    preds = []
    for m in models:
        pred = m.predict_proba(feature.reshape(1, -1))[0]
        preds.append(pred)
    return np.array(preds)  # shape (6, 60)

def apply_constraints(numbers):
    # En az 2 tek sayı ve 2 çift sayı kısıtı
    odd_count = sum(n % 2 == 1 for n in numbers)
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

def generate_predictions(df, n_preds=1):
    weights = get_weights(df['Date'])
    numbers_data = df.loc[:, ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']]

    # Bayesian model
    bayes_probs = bayesian_model(numbers_data, weights)

    # Conditional probabilities
    cond_probs = conditional_probs(numbers_data, weights)

    # XGBoost model
    xgb_models = xgboost_model(numbers_data)

    last_draw = numbers_data.iloc[-1].values

    final_predictions = []
    attempts = 0

    while len(final_predictions) < n_preds and attempts < n_preds * 100:
        attempts += 1
        # Bayesian seçimi: olasılık bazlı rastgele seçim
        bayes_choice = np.random.choice(np.arange(1, NUMBERS_RANGE+1), size=NUMBERS_DRAWN, replace=False, p=bayes_probs)

        # Koşullu olasılık ağırlıklı ikinci seçim
        cond_choice = []
        for n in bayes_choice:
            cp_row = cond_probs[n-1]
            cp_row /= cp_row.sum() if cp_row.sum() > 0 else 1
            cond_choice.append(np.random.choice(np.arange(1, NUMBERS_RANGE+1), p=cp_row))

        cond_choice = np.unique(np.array(cond_choice))
        if len(cond_choice) < NUMBERS_DRAWN:
            # Eksik sayıları tamamla Bayesian olasılık ile
            add_nums = np.random.choice(
                np.setdiff1d(np.arange(1, NUMBERS_RANGE+1), cond_choice),
                size=NUMBERS_DRAWN - len(cond_choice),
                replace=False,
                p=bayes_probs[np.setdiff1d(np.arange(NUMBERS_RANGE), cond_choice-1)]
            )
            cond_choice = np.concatenate([cond_choice, add_nums])

        # XGBoost tahmini
        xgb_probs = predict_xgboost(xgb_models, last_draw)

        # XGBoost'tan 6 sayı için en olası sayıları seç
        xgb_choice = []
        for prob in xgb_probs:
            prob = prob / prob.sum()
            xgb_choice.append(np.random.choice(np.arange(1, NUMBERS_RANGE+1), p=prob))
        xgb_choice = np.unique(np.array(xgb_choice))
        if len(xgb_choice) < NUMBERS_DRAWN:
            add_nums = np.random.choice(
                np.setdiff1d(np.arange(1, NUMBERS_RANGE+1), xgb_choice),
                size=NUMBERS_DRAWN - len(xgb_choice),
                replace=False,
                p=bayes_probs[np.setdiff1d(np.arange(NUMBERS_RANGE), xgb_choice-1)]
            )
            xgb_choice = np.concatenate([xgb_choice, add_nums])

        # Son olarak, tüm seçimleri birleştir
        combined = np.unique(np.concatenate([bayes_choice, cond_choice, xgb_choice]))
        if len(combined) < NUMBERS_DRAWN:
            add_nums = np.random.choice(
                np.setdiff1d(np.arange(1, NUMBERS_RANGE+1), combined),
                size=NUMBERS_DRAWN - len(combined),
                replace=False,
                p=bayes_probs[np.setdiff1d(np.arange(NUMBERS_RANGE), combined-1)]
            )
            combined = np.concatenate([combined, add_nums])
        combined = np.sort(combined)[:NUMBERS_DRAWN]

        if apply_constraints(combined):
            final_predictions.append(combined)

    return final_predictions

def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Tarih sütunu kontrolü
        if 'Date' not in df.columns:
            st.error("CSV dosyasında 'Date' sütunu bulunamadı.")
            return
        for c in ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']:
            if c not in df.columns:
                st.error(f"CSV dosyasında '{c}' sütunu bulunamadı.")
                return

        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=20, value=1, step=1)

        with st.spinner("Tahminler hesaplanıyor..."):
            preds = generate_predictions(df, n_preds)

        st.success(f"{len(preds)} adet tahmin üretildi:")
        for idx, p in enumerate(preds, 1):
            st.write(f"Tahmin {idx}: {sorted(p)}")

if __name__ == "__main__":
    main()
