import streamlit as st
import pandas as pd
import numpy as np
import pymc3 as pm
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# --- Zaman Bazlı Ağırlık Hesaplama ---
def zaman_agirliklari(dates):
    dates = pd.to_datetime(dates)
    days_since = (dates.max() - dates).dt.days
    weights = 1 / (1 + days_since)  # Yeni tarih daha yüksek ağırlık alır
    weights /= weights.sum()
    return weights.values


# --- Kısıt kontrolü ---
def kisitlar_tamammi(nums):
    tek_sayilar = sum(1 for n in nums if n % 2 == 1)
    cift_sayilar = 6 - tek_sayilar
    return tek_sayilar >= 2 and cift_sayilar >= 2


# --- Bayesian Model ---
def bayesian_model(df, weights):
    all_numbers = np.arange(1, 61)
    counts = np.zeros(60)

    for i, row in df.iterrows():
        for col in ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']:
            counts[row[col] - 1] += weights[i]

    probs = counts / counts.sum()

    # Basitleştirilmiş Bayesian Dirichlet ile ağırlıklı olasılık
    alpha = probs * 100 + 1

    with pm.Model() as model:
        p = pm.Dirichlet('p', a=alpha)
        trace = pm.sample(300, tune=200, chains=1, progressbar=False, random_seed=42)
    bayes_probs = np.mean(trace['p'], axis=0)
    return bayes_probs


# --- Markov Zinciri Modeli ---
def markov_model(df):
    size = 60
    mat = np.zeros((size, size))

    for _, row in df.iterrows():
        nums = sorted(row[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']])
        for i in range(len(nums) - 1):
            mat[nums[i] - 1, nums[i + 1] - 1] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.nan_to_num(mat / mat.sum(axis=1, keepdims=True))

    return mat


# --- XGBoost Modeli ---
def xgboost_model(df):
    all_numbers = np.arange(1, 61)
    features = []
    labels = []

    for _, row in df.iterrows():
        feature = np.zeros(60)
        for num in row[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']]:
            feature[num - 1] = 1
        features.append(feature)
        labels.append(feature)

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
    size = 60
    cond_mat = np.zeros((size, size))

    for _, row in df.iterrows():
        nums = row[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values - 1
        for a in nums:
            for b in nums:
                if a != b:
                    cond_mat[a, b] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        cond_mat = np.nan_to_num(cond_mat / cond_mat.sum(axis=1, keepdims=True))
    return cond_mat


# --- İkili Frekans ---
def pair_frequencies(df):
    size = 60
    pair_mat = np.zeros((size, size))

    for _, row in df.iterrows():
        nums = sorted(row[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values) - 1
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_mat[nums[i], nums[j]] += 1
                pair_mat[nums[j], nums[i]] += 1

    pair_mat /= pair_mat.sum()
    return pair_mat


# --- Kısıtları Kontrol Et ---
def kisitlar_tamammi(nums):
    tek_sayilar = sum(1 for n in nums if n % 2 == 1)
    cift_sayilar = 6 - tek_sayilar
    return tek_sayilar >= 2 and cift_sayilar >= 2


# --- Tahmin Üretme ---
def generate_combined_predictions(df, n_preds=1):
    all_numbers = np.arange(1, 61)

    weights = zaman_agirliklari(df['Date'])
    bayes_probs = bayesian_model(df, weights)
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
                # Burada model input olarak 6 sayı içeren vektör bekler, basitleştirilmiş versiyon:
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

        if 'Date' not in df.columns or not all(col in df.columns for col in ['Num1','Num2','Num3','Num4','Num5','Num6']):
            st.error("CSV dosyanızda 'Date' ve 'Num1'...'Num6' sütunlarının olduğundan emin olun.")
            return

        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1, step=1)

        if st.button("Tahminleri Hesapla"):
            with st.spinner("Tahminler hesaplanıyor, lütfen bekleyiniz..."):
                preds = generate_combined_predictions(df, n_preds=n_preds)
                for i, p in enumerate(preds, 1):
                    st.write(f"Tahmin {i}: {p}")


if __name__ == "__main__":
    main()
