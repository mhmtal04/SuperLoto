import streamlit as st
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import pymc as pm
import arviz as az
import itertools
from datetime import datetime

# Sabitler
NUMBERS_RANGE = 60
NUMBERS_DRAWN = 6

# --- Yardımcı Fonksiyonlar ---

def preprocess_draws(df):
    """ Çekiliş verisini 0 tabanlı indexlere çevir """
    numbers = df.values - 1
    return numbers

def get_time_weights(dates):
    """ Tarihlere göre ağırlık hesapla, en yeni tarihe en yüksek ağırlık """
    dates = pd.to_datetime(dates)
    days = (dates - dates.min()).dt.days
    weights = days / days.max()
    weights = 0.5 + 0.5 * weights  # 0.5-1 arası ağırlık
    return weights.values

def calc_single_freq(numbers, weights):
    """ Tekil sayı frekansları (ağırlıklı) """
    freq = np.zeros(NUMBERS_RANGE)
    for draw, w in zip(numbers, weights):
        freq[draw] += w
    freq /= freq.sum()
    return freq

def calc_pair_freq(numbers, weights):
    """ İkili sayı frekansları (ağırlıklı) """
    pair_counts = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    for draw, w in zip(numbers, weights):
        for a, b in itertools.combinations(draw, 2):
            pair_counts[a, b] += w
            pair_counts[b, a] += w
    pair_sums = pair_counts.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        pair_freq = np.divide(pair_counts, pair_sums[:, None], where=pair_sums[:, None]!=0)
    pair_freq = np.nan_to_num(pair_freq)
    return pair_freq

def conditional_prob(pair_freq, selected):
    """ Seçilen sayıların diğer sayılarla koşullu olasılığı """
    cond_probs = np.ones(NUMBERS_RANGE)
    for s in selected:
        cond_probs *= pair_freq[s]
    cond_probs /= cond_probs.sum()
    return cond_probs

def check_constraints(draw):
    """ En az 2 tek ve 2 çift sayı kısıtı """
    odds = np.sum(draw % 2 == 0)
    evens = NUMBERS_DRAWN - odds
    return (odds >= 2) and (evens >= 2)

# --- Modeller ---

def xgboost_model(df):
    numbers = preprocess_draws(df)
    X, y = [], []
    for i in range(len(numbers) -1):
        feature = np.zeros(NUMBERS_RANGE, dtype=int)
        feature[numbers[i]] = 1
        X.append(feature)
        y.append(numbers[i+1])
    X = np.array(X)
    y = np.array(y)

    model = MultiOutputClassifier(
        xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, verbosity=0)
    )
    model.fit(X, y)
    return model

def predict_xgboost(model, last_draw):
    feature = np.zeros(NUMBERS_RANGE, dtype=int)
    feature[last_draw] = 1
    preds_proba = model.predict_proba(feature.reshape(1,-1))
    preds = np.array([p[0] for p in preds_proba])  # Her hedef için olasılık
    return preds

def bayesian_model(df, weights):
    numbers = preprocess_draws(df)
    counts = np.zeros(NUMBERS_RANGE)
    for draw, w in zip(numbers, weights):
        for n in draw:
            counts[n] += w
    with pm.Model() as model:
        alpha = 1 + counts
        p = pm.Dirichlet('p', a=alpha)
        trace = pm.sample(draws=300, tune=200, chains=1, progressbar=False)
    probs = trace.posterior['p'].mean(dim=['chain','draw']).values
    return probs

def markov_model(df):
    numbers = preprocess_draws(df)
    transition_counts = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    for i in range(len(numbers)-1):
        for a, b in zip(numbers[i], numbers[i+1]):
            transition_counts[a,b] += 1
    row_sums = transition_counts.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_probs = np.divide(transition_counts, row_sums[:, None], where=row_sums[:, None]!=0)
    transition_probs = np.nan_to_num(transition_probs)
    return transition_probs

def generate_predictions(df, n_preds=5):
    weights = get_time_weights(df['Date'])
    single_freq = calc_single_freq(preprocess_draws(df), weights)
    pair_freq = calc_pair_freq(preprocess_draws(df), weights)
    markov_probs = markov_model(df)
    bayes_probs = bayesian_model(df, weights)
    xgb_model = xgboost_model(df)

    last_draw = preprocess_draws(df)[-1]

    preds = []
    trials = 0
    while len(preds) < n_preds and trials < 10000:
        trials += 1
        # XGBoost tahmini ile tekil frekans ve bayes ağırlıklı olasılık karışımı
        xgb_probs = predict_xgboost(xgb_model, last_draw)
        combined_probs = (xgb_probs + single_freq + bayes_probs) / 3
        combined_probs /= combined_probs.sum()

        # En yüksek olasılıklı 20 sayıyı seçelim rastgele 6 çekelim
        top_candidates = combined_probs.argsort()[-20:]
        draw = np.random.choice(top_candidates, NUMBERS_DRAWN, replace=False)

        # Koşullu olasılıkları hesaba kat
        cond_probs = conditional_prob(pair_freq, draw)
        cond_probs /= cond_probs.sum()

        # Koşullu olasılık ile seçim yapalım
        weighted_draw = np.random.choice(np.arange(NUMBERS_RANGE), NUMBERS_DRAWN, replace=False, p=cond_probs)

        # Kısıt kontrolü
        if check_constraints(weighted_draw):
            preds.append(np.sort(weighted_draw + 1))  # 1-based sayı olarak kaydet

    return preds

# --- Streamlit Arayüzü ---

st.title("Super Loto Tahmin Sistemi")

uploaded_file = st.file_uploader("Çekiliş verisi yükleyin (Excel veya CSV)", type=["csv","xlsx"])

n_preds = st.number_input("Tahmin sayısı", min_value=1, max_value=20, value=5)

if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    if 'Date' not in df.columns or df.shape[1] < NUMBERS_DRAWN + 1:
        st.error(f"Veri dosyasında 'Date' sütunu ve en az {NUMBERS_DRAWN} sayı sütunu olmalı.")
    else:
        with st.spinner("Tahminler hesaplanıyor..."):
            preds = generate_predictions(df, n_preds=n_preds)
        st.success(f"{len(preds)} tahmin hesaplandı.")
        for i, p in enumerate(preds, 1):
            st.write(f"Tahmin {i}: {p}")
