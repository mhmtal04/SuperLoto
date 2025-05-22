import streamlit as st
import pandas as pd
import numpy as np
import pymc3 as pm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import random

# --- Kısıt kontrolü ---
def kisitlar_tamammi(nums):
    tek_sayilar = sum(1 for n in nums if n % 2 == 1)
    cift_sayilar = 6 - tek_sayilar
    return tek_sayilar >= 2 and cift_sayilar >= 2

# --- Zaman bazlı ağırlık (yeni çekilişlere daha fazla ağırlık) ---
def zaman_agirliklari(dates):
    dates = pd.to_datetime(dates)
    n = len(dates)
    weights = np.linspace(1, 2, n)  # Yeni tarihe daha yüksek ağırlık
    weights /= weights.sum()
    return weights

# --- Bayesian Model ---
def bayesian_model(df):
    all_numbers = np.arange(1, 61)
    counts = np.zeros(60)
    for col in ['Num1','Num2','Num3','Num4','Num5','Num6']:
        counts += df[col].value_counts().reindex(all_numbers, fill_value=0).values
    probs = counts / counts.sum()

    with pm.Model() as model:
        p = pm.Dirichlet('p', a=probs*100 + 1)
        obs = pm.Categorical('obs', p=p, observed=np.random.choice(all_numbers-1, size=6))
        trace = pm.sample(500, tune=300, chains=1, progressbar=False, random_seed=42, cores=1)
    bayes_probs = np.mean(trace['p'], axis=0)
    return bayes_probs

# --- Markov Zinciri Modeli ---
def markov_model(df):
    all_numbers = np.arange(1, 61)
    size = len(all_numbers)
    mat = np.zeros((size, size))

    # Sıralı çiftler üzerinden geçiş matrisi oluştur
    for _, row in df.iterrows():
        nums = sorted(row[['Num1','Num2','Num3','Num4','Num5','Num6']])
        for i in range(len(nums)-1):
            mat[nums[i]-1][nums[i+1]-1] += 1

    # Normalize et
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.nan_to_num(mat / mat.sum(axis=1, keepdims=True))

    return mat

# --- XGBoost Modeli ---
def xgboost_model(df):
    all_numbers = np.arange(1, 61)
    features = []
    labels = []

    # Her satır için 60 elemanlı one-hot vector, her sayı için 1
    for _, row in df.iterrows():
        row_features = np.zeros(60)
        for num in row[['Num1','Num2','Num3','Num4','Num5','Num6']]:
            row_features[num-1] = 1
        features.append(row_features)
        labels.append(row_features)  # burada kendi kendini tahmin gibi

    X = np.array(features)
    y = np.array(labels)

    models = []
    for i in range(60):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        y_col = y[:, i]
        X_train, X_val, y_train, y_val = train_test_split(X, y_col, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        models.append(model)

    return models

# --- Koşullu Olasılık ---
def conditional_probabilities(df):
    all_numbers = np.arange(1, 61)
    size = 60
    cond_mat = np.zeros((size, size))

    # Koşullu frekans: Sayı A çıktıysa sayı B kaç kere çıkmış
    for _, row in df.iterrows():
        nums = row[['Num1','Num2','Num3','Num4','Num5','Num6']].values - 1
        for a in nums:
            for b in nums:
                if a != b:
                    cond_mat[a][b] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        cond_mat = np.nan_to_num(cond_mat / cond_mat.sum(axis=1, keepdims=True))
    return cond_mat

# --- İkili Frekans ---
def pair_frequencies(df):
    size = 60
    pair_mat = np.zeros((size, size))

    for _, row in df.iterrows():
        nums = sorted(row[['Num1','Num2','Num3','Num4','Num5','Num6']].values) - 1
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pair_mat[nums[i]][nums[j]] += 1
                pair_mat[nums[j]][nums[i]] += 1

    pair_mat /= pair_mat.sum()
    return pair_mat

# --- Tahmin Üretme ---
def generate_combined_predictions(df, n_preds=1):
    all_numbers = np.arange(1, 61)

    bayes_probs = bayesian_model(df)
    markov_mat = markov_model(df)
    xgb_models = xgboost_model(df)
    cond_mat = conditional_probabilities(df)
    pair_mat = pair_frequencies(df)

    predictions = []
    trials = 10000

    for _ in range(n_preds):
        best_score = -1
        best_combination = None

        for __ in range(trials):
            # Bayesian ağırlıklı seçim
            bayes_choice = np.random.choice(all_numbers, 6, replace=False, p=bayes_probs)

            # Markov ile geçiş ağırlığı
            markov_probs = np.mean(markov_mat[bayes_choice - 1], axis=0)

            # XGBoost ile tahmin olasılıkları
            xgb_probs = np.zeros(60)
            for i, model in enumerate(xgb_models):
                xgb_probs[i] = model.predict_proba(np.eye(60)[i].reshape(1, -1))[0][1]
            xgb_probs /= xgb_probs.sum()

            # Koşullu olasılık ve ikili frekansları ortala
            cond_probs = np.mean(cond_mat[bayes_choice - 1], axis=0)
            pair_probs = np.mean(pair_mat[bayes_choice - 1], axis=0)

            # Tüm olasılıkları ortala ve normalize et
            combined_probs = (bayes_probs + markov_probs + xgb_probs + cond_probs + pair_probs) / 5
            combined_probs /= combined_probs.sum()

            # Yeni tahmin kombinasyonu
            try_combination = np.random.choice(all_numbers, 6, replace=False, p=combined_probs)

            if not kisitlar_tamammi(try_combination):
                continue

            # Skor: Bayesian ağırlıklı toplam
            score = sum(bayes_probs[num - 1] for num in try_combination)
            if score > best_score:
                best_score = score
                best_combination = try_combination

        predictions.append(sorted(best_combination))

    return predictions

# --- Streamlit Arayüzü ---
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Veri yüklendi!")

        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1, step=1)

        if st.button("Tahminleri Hesapla"):
            with st.spinner("Tahminler hesaplanıyor, lütfen bekleyiniz..."):
                preds = generate_combined_predictions(df, n_preds=n_preds)
                for i, p in enumerate(preds, 1):
                    st.write(f"Tahmin {i}: {p}")

if __name__ == "__main__":
    main()
