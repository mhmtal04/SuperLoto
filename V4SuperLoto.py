import streamlit as st
import pandas as pd
import numpy as np
import random
import pymc as pm
from sklearn.model_selection import train_test_split
import xgboost as xgb

# --- Fonksiyonlar ---

def zaman_agirliklari(df):
    n = len(df)
    return np.linspace(1, 2, n)

def tekil_agirlikli_frekans(df, weights):
    freq = np.zeros(60)
    for i, row in df.iterrows():
        for num in row[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']]:
            freq[num-1] += weights[i]
    return freq / np.sum(freq)

def ikili_frekanslar(df, weights):
    pairs = np.zeros((60, 60))
    for i, row in df.iterrows():
        nums = row[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values - 1
        w = weights[i]
        for a in nums:
            for b in nums:
                if a != b:
                    pairs[a, b] += w
    pairs /= pairs.sum(axis=1, keepdims=True) + 1e-8
    return pairs

def bayesian_model(df, weights):
    freqs = tekil_agirlikli_frekans(df, weights)
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(60))
        obs = pm.Multinomial('obs', n=6, p=p, observed=np.random.multinomial(6, freqs))
        trace = pm.sample(500, tune=500, progressbar=False, chains=1)
    bayes_probs = trace.posterior['p'].mean(dim=('chain', 'draw')).values
    return bayes_probs

def markov_tahmin(df, weights, ikili_freq, n=6):
    tekil_probs = tekil_agirlikli_frekans(df, weights)
    seq = []
    first = np.random.choice(np.arange(60), p=tekil_probs)
    seq.append(first)
    for _ in range(n-1):
        last = seq[-1]
        next_prob = ikili_freq[last].copy()
        for s in seq:
            next_prob[s] = 0
        if next_prob.sum() == 0:
            candidates = [x for x in range(60) if x not in seq]
            next_num = random.choice(candidates)
        else:
            next_prob /= next_prob.sum()
            next_num = np.random.choice(np.arange(60), p=next_prob)
        seq.append(next_num)
    return np.array(seq) + 1

def xgboost_model(df):
    X, y = [], []
    for i, row in df.iterrows():
        feature = np.zeros(60)
        for num in row[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']]:
            feature[num-1] = 1
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

def xgboost_tahmin(models, son_cekilis):
    feature = np.zeros(60)
    for num in son_cekilis:
        feature[num-1] = 1
    preds = []
    for model in models:
        pred = model.predict_proba(feature.reshape(1, -1))[0][1]
        preds.append(pred)
    preds = np.array(preds)
    preds /= preds.sum()
    tahmin = preds.argsort()[-6:][::-1] + 1
    return tahmin

def kisitlari_kontrol_et(sayi_listesi):
    tekler = [s for s in sayi_listesi if s % 2 == 1]
    ciftler = [s for s in sayi_listesi if s % 2 == 0]
    return len(tekler) >= 2 and len(ciftler) >= 2

def tahmin_uret(df, n_tahmin=1):
    weights = zaman_agirliklari(df)
    ikili_freq = ikili_frekanslar(df, weights)
    bayes_probs = bayesian_model(df, weights)
    xgb_models = xgboost_model(df)
    son_cekilis = df.iloc[-1][['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values

    tahminler = []
    deneme_limiti = 1000
    for _ in range(n_tahmin):
        for _ in range(deneme_limiti):
            markov_sayi = markov_tahmin(df, weights, ikili_freq)
            xgb_sayi = xgboost_tahmin(xgb_models, son_cekilis)
            bayes_idx = bayes_probs.argsort()[-6:][::-1] + 1
            combined = (np.array(markov_sayi) + np.array(xgb_sayi) + bayes_idx) / 3
            combined = np.round(combined).astype(int)
            combined = np.clip(combined, 1, 60)
            if kisitlari_kontrol_et(combined):
                tahminler.append(np.unique(combined))
                break
        else:
            tahminler.append(np.random.choice(np.arange(1,61), size=6, replace=False))
    return tahminler

# --- Streamlit Arayüzü ---

st.title("Süper Loto Gelişmiş Tahmin Botu")

uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    if set(['Date', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']).issubset(df.columns):
        st.success(f"Veri yüklendi! Toplam çekiliş: {len(df)}")
        n_tahmin = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1)
        if st.button("Tahminleri Üret"):
            with st.spinner("Tahminler hesaplanıyor... Bu biraz zaman alabilir."):
                tahminler = tahmin_uret(df, n_tahmin)
            st.success("Tahminler hazır!")
            for i, t in enumerate(tahminler):
                st.write(f"Tahmin {i+1}: {sorted(t)}")
    else:
        st.error("CSV dosyasında gerekli sütunlar yok! (Date, Num1~Num6)")

else:
    st.info("Lütfen CSV dosyanızı yükleyin.")
