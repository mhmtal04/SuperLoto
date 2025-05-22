import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Süper Loto Gelişmiş Tahmin Botu", layout="centered")

def zaman_agirliklari(dates):
    # Tarihler güncel ise daha yüksek ağırlık verir, lineer artış
    n = len(dates)
    return np.linspace(1, 2, n)

def tekil_agirliklar(df):
    # Her sayının frekansı, toplam frekans üzerinden normalize
    all_numbers = np.arange(1, 61)
    counts = np.zeros(60)
    for col in ['Num1','Num2','Num3','Num4','Num5','Num6']:
        counts += df[col].value_counts().reindex(all_numbers, fill_value=0).values
    weights = counts / counts.sum()
    return weights

def ikili_frekans(df):
    # İkili sayıların beraber çıkma frekansı matrisi
    freq_matrix = np.zeros((60,60))
    for _, row in df.iterrows():
        numbers = sorted(row[['Num1','Num2','Num3','Num4','Num5','Num6']].values)
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                freq_matrix[numbers[i]-1, numbers[j]-1] +=1
                freq_matrix[numbers[j]-1, numbers[i]-1] +=1
    freq_matrix /= freq_matrix.sum()
    return freq_matrix

def kosullu_olasilik(df):
    # P(A|B) = birlikte çıkma / B'nin çıkma frekansı
    freq_matrix = ikili_frekans(df)
    single_freq = np.sum(freq_matrix, axis=1)
    cond_probs = np.divide(freq_matrix.T, single_freq, out=np.zeros_like(freq_matrix.T), where=single_freq!=0).T
    return cond_probs

def kisitlar_tamammi(tahmin):
    # En az 2 tek sayı ve 2 çift sayı kısıtı
    tek_sayilar = sum([1 for n in tahmin if n%2==1])
    cift_sayilar = sum([1 for n in tahmin if n%2==0])
    return tek_sayilar >= 2 and cift_sayilar >= 2

def bayesian_model(df, weights):
    all_numbers = np.arange(1, 61)
    counts = np.zeros(60)
    for col in ['Num1','Num2','Num3','Num4','Num5','Num6']:
        counts += df[col].value_counts().reindex(all_numbers, fill_value=0).values

    # Ağırlıklı frekans (zaman + tekil)
    weighted_counts = counts * weights
    probs = weighted_counts / weighted_counts.sum()

    # Basit örnek: Kategorik dağılım
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=probs*100 + 1)
        obs = pm.Categorical('obs', p=p, observed=np.random.choice(all_numbers-1, size=6))
        trace = pm.sample(500, tune=300, chains=1, progressbar=False, random_seed=42, cores=1)
    bayes_probs = np.mean(trace['p'], axis=0)
    return bayes_probs

def markov_model(df):
    # Markov geçiş matrisi oluştur (basit, ardışık çekilişler)
    states = np.zeros((60, 60))
    prev_draws = df[['Num1','Num2','Num3','Num4','Num5','Num6']].values[:-1]
    next_draws = df[['Num1','Num2','Num3','Num4','Num5','Num6']].values[1:]

    for prev, nxt in zip(prev_draws, next_draws):
        for p in prev:
            for n in nxt:
                states[p-1, n-1] += 1

    # Normalize
    row_sums = states.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    trans_matrix = states / row_sums
    return trans_matrix

def xgboost_model(df):
    # Her çekiliş için 60 sayıdan one-hot encoding özellik vektörü (0/1)
    X = []
    y = []
    for idx in range(len(df)-1):
        current_nums = df.iloc[idx][['Num1','Num2','Num3','Num4','Num5','Num6']].values
        next_nums = df.iloc[idx+1][['Num1','Num2','Num3','Num4','Num5','Num6']].values

        features = np.zeros(60)
        features[current_nums - 1] = 1
        X.append(features)

        label = np.zeros(60)
        label[next_nums - 1] = 1
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    models = []
    for i in range(60):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y[:,i])
        models.append(model)
    return models

def generate_combined_predictions(df, n_preds=1):
    # Zaman bazlı ağırlıklar
    weights = zaman_agirliklari(pd.to_datetime(df['Date']))
    # Bayesian
    bayes_probs = bayesian_model(df, weights)
    # Markov
    markov_mat = markov_model(df)
    # XGBoost
    xgb_models = xgboost_model(df)

    all_numbers = np.arange(1, 61)

    predictions = []
    trials = 10000  # Deneme sayısı tahmin seçimi için
    for _ in range(n_preds):
        best_score = -1
        best_combination = None
        for __ in range(trials):
            # Bayesian ve Markov ağırlıklı seçim
            bayes_choice = np.random.choice(all_numbers, 6, replace=False, p=bayes_probs)
            # Markov'dan geçiş olasılıkları
            markov_probs = np.mean(markov_mat[bayes_choice-1], axis=0)
            # XGBoost tahminleriyle ağırlık çarpımı
            xgb_probs = np.array([model.predict_proba(np.eye(60)[i].reshape(1,-1))[0][1] for i, model in enumerate(xgb_models)])
            combined_probs = (bayes_probs + markov_probs + xgb_probs) / 3
            combined_probs /= combined_probs.sum()

            try_combination = np.random.choice(all_numbers, 6, replace=False, p=combined_probs)
            if not kisitlar_tamammi(try_combination):
                continue
            # Koşullu olasılık kısıtları eklenebilir burada...

            # Skor: Basit frekans toplamı (daha iyi skor mantığı geliştirilebilir)
            score = sum(weights[num-1] for num in try_combination)
            if score > best_score:
                best_score = score
                best_combination = try_combination
        predictions.append(sorted(best_combination))
    return predictions

def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu (Bayesian - Markov - XGBoost - Koşullu Olasılık)")
    st.write("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları olmalı)")

    uploaded_file = st.file_uploader("CSV dosyanızı seçin", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in ['Date','Num1','Num2','Num3','Num4','Num5','Num6']):
            st.error("CSV dosyanızda 'Date' ve 'Num1'...'Num6' sütunları bulunmalı!")
            return

        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1, step=1)

        with st.spinner("Tahminler hesaplanıyor, lütfen bekleyiniz..."):
            preds = generate_combined_predictions(df, n_preds=n_preds)

        st.success("Tahminler hazır!")
        for i, p in enumerate(preds, 1):
            st.write(f"{i}. Tahmin: {p}")

if __name__ == "__main__":
    main()
