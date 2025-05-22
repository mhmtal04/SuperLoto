import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import xgboost as xgb
import pymc as pm
from sklearn.model_selection import train_test_split

# -- Önceki fonksiyonlar: get_weights, weighted_single_probabilities, pair_frequencies, conditional_probabilities, check_constraints aynı kalabilir --

# XGBoost Model Eğitimi
def train_xgboost(df):
    X, y = [], []
    for _, row in df.iterrows():
        feature = np.zeros(60)
        for num in row['Numbers']:
            feature[num-1] = 1
        X.append(feature)
        y.append(feature)  # Çoklu çıktı için aynı
    X = np.array(X)
    y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    models = []
    for i in range(60):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train[:, i])
        models.append(model)
    return models

def xgboost_predict(models, last_draw):
    feature = np.zeros(60)
    for num in last_draw:
        feature[num-1] = 1
    preds = np.array([model.predict_proba(feature.reshape(1,-1))[0][1] for model in models])
    preds /= preds.sum()
    return preds

# Bayesian Model (basit Dirichlet)
def bayesian_probs(df):
    weights = get_weights(df['Date'])
    freq = np.zeros(60)
    for i, row in df.iterrows():
        for num in row['Numbers']:
            freq[num-1] += weights[i]
    alpha = freq + 1  # Prior ekle
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=alpha)
        trace = pm.sample(1000, tune=500, chains=1, progressbar=False)
    bayes_prob = trace.posterior['p'].mean(dim=['chain','draw']).values
    return bayes_prob

# Markov Zinciri (ikili geçiş olasılıkları)
def markov_chain(df):
    weights = get_weights(df['Date'])
    pair_freq = np.zeros((60,60))
    for i, row in df.iterrows():
        nums = row['Numbers']
        w = weights[i]
        for a, b in combinations(nums, 2):
            pair_freq[a-1, b-1] += w
            pair_freq[b-1, a-1] += w
    pair_prob = pair_freq / (pair_freq.sum(axis=1, keepdims=True) + 1e-9)
    return pair_prob

def markov_predict(pair_prob, last_nums, n=6):
    preds = []
    used = set()
    current = last_nums[-1] - 1
    preds.append(current)
    used.add(current)
    for _ in range(n-1):
        probs = pair_prob[current].copy()
        for u in used:
            probs[u] = 0
        if probs.sum() == 0:
            candidates = [x for x in range(60) if x not in used]
            next_num = np.random.choice(candidates)
        else:
            probs /= probs.sum()
            next_num = np.random.choice(60, p=probs)
        preds.append(next_num)
        used.add(next_num)
        current = next_num
    return np.array(preds)+1

# Tahmin Üret
def generate_combined_predictions(df, n_preds=1):
    weights = get_weights(df['Date'])
    single_prob = weighted_single_probabilities(df)
    cond_prob = conditional_probabilities(single_prob, pair_frequencies(df))

    xgb_models = train_xgboost(df)
    bayes_prob = bayesian_probs(df)
    markov_prob = markov_chain(df)
    last_draw = df.iloc[-1]['Numbers']

    predictions = []

    for _ in range(n_preds):
        xgb_p = xgboost_predict(xgb_models, last_draw)
        markov_p = markov_predict(markov_prob, last_draw)
        # Markov tahminlerini olasılığa çevir (basitleştirilmiş)
        markov_vec = np.zeros(60)
        markov_vec[markov_p - 1] = 1 / len(markov_p)

        # Kombine et (örneğin eşit ağırlık)
        combined_prob = (single_prob.values + bayes_prob + xgb_p + markov_vec) / 4
        combined_prob /= combined_prob.sum()

        # En yüksek 6 sayıyı seç
        chosen = combined_prob.argsort()[-6:][::-1] + 1

        # Kısıt kontrolü
        if not check_constraints(chosen):
            # Eğer kısıt tutmazsa rastgele 2 tek 2 çift ekle
            odds = [n for n in range(1,61) if n%2==1]
            evens = [n for n in range(1,61) if n%2==0]
            chosen = np.array(
                list(np.random.choice(odds, 2, replace=False)) + 
                list(np.random.choice(evens, 4, replace=False))
            )

        predictions.append(np.sort(chosen))

    return predictions

# Streamlit Arayüzü
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu - Entegre Model")
    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1','Num2','Num3','Num4','Num5','Num6']].values.tolist()

        st.success(f"{len(df)} çekiliş yüklendi.")
        n_preds = st.number_input("Kaç tahmin üretilsin?", 1, 10, 1)

        if st.button("Tahmin Üret"):
            with st.spinner("Tahminler hesaplanıyor..."):
                preds = generate_combined_predictions(df, n_preds)
            for i, p in enumerate(preds, 1):
                st.write(f"Tahmin {i}: {', '.join(map(str, p))}")

if __name__=="__main__":
    main()
