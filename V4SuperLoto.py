import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb

NUMBERS_RANGE = 60
NUMBERS_DRAWN = 6

# --- Yardımcı fonksiyonlar ---

def get_time_weights(dates):
    # Tarihlere göre güncel tarih referansı, normalize ağırlıklar
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    weights = 1 / (days_ago + 1)  # Yeni çekilişlere yüksek ağırlık
    weights /= weights.sum()
    return weights.values

def calculate_single_weights(df, weights):
    counts = np.zeros(NUMBERS_RANGE)
    for idx, row in df.iterrows():
        for num in row:
            counts[num - 1] += weights[idx]
    return counts / counts.sum()

def calculate_pair_weights(df, weights):
    pair_counts = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    for idx, row in df.iterrows():
        numbers = sorted(row)
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                pair_counts[numbers[i]-1][numbers[j]-1] += weights[idx]
                pair_counts[numbers[j]-1][numbers[i]-1] += weights[idx]
    return pair_counts

def calculate_conditional_probs(pair_weights, single_weights):
    conditional = np.zeros_like(pair_weights)
    for i in range(NUMBERS_RANGE):
        if single_weights[i] > 0:
            conditional[i] = pair_weights[i] / single_weights[i]
    return conditional

def passes_constraints(numbers):
    even_count = sum(1 for n in numbers if n % 2 == 0)
    odd_count = NUMBERS_DRAWN - even_count
    return even_count >= 2 and odd_count >= 2

# --- Modeller ---

def bayesian_model(df, weights):
    single_weights = calculate_single_weights(df, weights)
    return single_weights

def markov_model(df, weights):
    pair_weights = calculate_pair_weights(df, weights)
    single_weights = calculate_single_weights(df, weights)
    conditional_probs = calculate_conditional_probs(pair_weights, single_weights)
    return conditional_probs

def xgboost_model(df):
    df = df.reset_index(drop=True)
    X = []
    y = []

    for idx in range(len(df)-1):
        feature = np.zeros(NUMBERS_RANGE, dtype=int)
        feature[df.loc[idx].values - 1] = 1
        X.append(feature)
        y.append(df.loc[idx+1].values - 1)

    X = np.array(X)
    y = np.array(y)

    models = []
    for i in range(NUMBERS_DRAWN):
        model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, verbosity=0)
        model.fit(X, y[:, i])
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

# --- Tahmin üretme ---

def generate_combined_predictions(df, n_preds=5):
    weights = get_time_weights(df['Date'])

    # Bayesian tekil ağırlıklı olasılık
    bayes_probs = bayesian_model(df[ [f'Num{i+1}' for i in range(NUMBERS_DRAWN)] ], weights)

    # Markov koşullu olasılıklar
    markov_probs = markov_model(df[ [f'Num{i+1}' for i in range(NUMBERS_DRAWN)] ], weights)

    # XGBoost modelleri
    xgb_models = xgboost_model(df[ [f'Num{i+1}' for i in range(NUMBERS_DRAWN)] ])

    last_draw = df.iloc[-1][ [f'Num{i+1}' for i in range(NUMBERS_DRAWN)] ].values

    # Birleştirilmiş skor: ağırlıklı ortalama
    combined_scores = (bayes_probs + markov_probs.mean(axis=0)) / 2

    preds = []
    tries = 0
    max_tries = 10000

    while len(preds) < n_preds and tries < max_tries:
        tries += 1
        # XGBoost tahmin dağılımı üzerinden rastgele seçim
        xgb_preds = predict_xgboost(xgb_models, last_draw)

        # XGBoost’tan yüksek ihtimalli sayıları seç
        xgb_choice = []
        for p in xgb_preds:
            sorted_idx = np.argsort(p)[::-1]
            xgb_choice.append(sorted_idx[0]+1)  # En yüksek olasılık

        # Birleşik skorlar ile ağırlıklı seçim
        combined_probs = combined_scores + xgb_preds.mean(axis=0)
        combined_probs /= combined_probs.sum()

        # 6 sayı seçimi (ilk önce xgb, sonra olasılıkları karıştırarak)
        chosen = set(xgb_choice)
        while len(chosen) < NUMBERS_DRAWN:
            chosen.add(np.random.choice(range(1, NUMBERS_RANGE+1), p=combined_probs))

        chosen = sorted(chosen)

        if passes_constraints(chosen) and chosen not in preds:
            preds.append(chosen)

    return preds

# --- Streamlit arayüzü ---

def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        expected_cols = ['Date'] + [f'Num{i+1}' for i in range(NUMBERS_DRAWN)]
        if not all(col in df.columns for col in expected_cols):
            st.error(f"CSV dosyası şu sütunları içermelidir: {expected_cols}")
            return

        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=20, value=5, step=1)

        if st.button("Tahminleri Hesapla"):
            with st.spinner("Tahminler hesaplanıyor..."):
                preds = generate_combined_predictions(df, n_preds)
            st.success(f"{len(preds)} tahmin hazır.")
            for i, p in enumerate(preds, 1):
                st.write(f"Tahmin {i}: {p}")

if __name__ == "__main__":
    main() 
