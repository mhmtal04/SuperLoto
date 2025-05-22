import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pymc as pm

NUMBERS_RANGE = 60
NUMBERS_DRAWN = 6

def get_time_weights(dates):
    """Yeni tarihli çekilişlere daha fazla ağırlık verir."""
    dates = pd.to_datetime(dates)
    days = (dates - dates.min()).dt.days
    weights = days / days.max()
    weights = weights + 0.1  # sıfır ağırlık olmasın
    return weights / weights.sum()

def calculate_single_frequencies(df, weights):
    """Tekil sayıların ağırlıklı frekansları."""
    freq = np.zeros(NUMBERS_RANGE)
    for idx, row in df.iterrows():
        freq[row.values - 1] += weights[idx]
    return freq / freq.sum()

def calculate_pair_frequencies(df, weights):
    """İkili sayı frekansları ve koşullu olasılıklar."""
    pair_freq = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    for idx, row in df.iterrows():
        numbers = row.values - 1
        w = weights[idx]
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i != j:
                    pair_freq[numbers[i], numbers[j]] += w
    # Normalize et
    pair_freq = pair_freq / pair_freq.sum(axis=1, keepdims=True)
    pair_freq = np.nan_to_num(pair_freq)
    return pair_freq

def bayesian_model(single_freq, pair_freq):
    """Basit Bayesian olasılık modeli."""
    # Burada PyMC ile gerçek Bayesian model kurulabilir,
    # fakat performans için basitleştirilmiş hali kullanılıyor.
    def calc_prob(numbers):
        prob = 1.0
        for n in numbers:
            prob *= single_freq[n]
        # Koşullu olasılık eklenebilir
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                prob *= pair_freq[numbers[i], numbers[j]] * pair_freq[numbers[j], numbers[i]]
        return prob
    return calc_prob

def xgboost_model(df):
    df = df.reset_index(drop=True)
    X = []
    y = []
    for idx in range(len(df) - 1):
        feature = np.zeros(NUMBERS_RANGE, dtype=int)
        feature[df.loc[idx].values - 1] = 1
        X.append(feature)
        y.append(df.loc[idx + 1].values - 1)
    X = np.array(X)
    y = np.array(y)
    models = []
    for i in range(NUMBERS_DRAWN):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
        target = y[:, i].astype(int)
        model.fit(X, target)
        models.append(model)
    return models

def predict_xgboost(models, last_draw):
    feature = np.zeros(NUMBERS_RANGE, dtype=int)
    feature[last_draw - 1] = 1
    preds = []
    for m in models:
        pred = m.predict_proba(feature.reshape(1, -1))[0]
        preds.append(pred)
    return np.array(preds)

def apply_constraints(numbers):
    """En az 2 tek ve 2 çift sayı kısıtını uygula."""
    odd_count = sum([1 for n in numbers if (n+1) % 2 != 0])
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

def generate_predictions(df, n_preds=5):
    dates = df['Date']
    numbers_data = df.drop(columns=['Date'])
    weights = get_time_weights(dates)
    single_freq = calculate_single_frequencies(numbers_data, weights)
    pair_freq = calculate_pair_frequencies(numbers_data, weights)
    bayes_prob = bayesian_model(single_freq, pair_freq)

    # XGBoost modellerini eğit
    xgb_models = xgboost_model(numbers_data)

    last_draw = numbers_data.iloc[-1].values
    xgb_probs = predict_xgboost(xgb_models, last_draw)

    preds = []
    attempts = 0
    max_attempts = n_preds * 50
    while len(preds) < n_preds and attempts < max_attempts:
        attempts += 1
        # Bayes ve XGBoost olasılıklarını karışık kullanarak rastgele seçim
        candidate = []
        for i in range(NUMBERS_DRAWN):
            prob_dist = bayes_prob(range(NUMBERS_RANGE)) * xgb_probs[i]
            prob_dist = prob_dist / prob_dist.sum()
            number = np.random.choice(range(NUMBERS_RANGE), p=prob_dist)
            candidate.append(number)
        candidate = sorted(set(candidate))
        if len(candidate) == NUMBERS_DRAWN and apply_constraints(candidate):
            preds.append([n+1 for n in candidate])
    return preds

# Streamlit Arayüzü
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")
    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])
    n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=20, value=5)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not {'Date', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6'}.issubset(df.columns):
            st.error("CSV dosyasında 'Date' ve 'Num1'...'Num6' sütunları bulunmalı.")
            return
        numbers_data = df[['Num1','Num2','Num3','Num4','Num5','Num6']]
        df2 = df[['Date']].copy()
        df2 = pd.concat([df2, numbers_data], axis=1)
        with st.spinner('Tahminler hesaplanıyor, lütfen bekleyin...'):
            predictions = generate_predictions(df2, n_preds=n_preds)
        if predictions:
            st.subheader("Tahmin Sonuçları:")
            for i, pred in enumerate(predictions):
                st.write(f"{i+1}. Tahmin: {pred}")
        else:
            st.warning("Tahmin üretilemedi, lütfen veri kalitesini kontrol edin.")

if __name__ == "__main__":
    main()
