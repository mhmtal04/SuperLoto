import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pymc as pm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

NUMBERS_RANGE = 60
NUMBERS_DRAWN = 6

st.title("Süper Loto Tahmin Sistemi")

@st.cache_data
def preprocess_draws(df):
    # Sadece sayısal çekiliş sütunlarını numpy array haline getir
    draws = df.iloc[:, 1:].values  # ilk sütun tarih olabilir
    return draws.astype(int)

def calc_time_weights(dates):
    # Tarihlere göre ağırlık, en yeni tarihe 1, en eskiye 0.1 civarı
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days
    weights = 1 - days_diff / days_diff.max()
    weights = 0.1 + 0.9 * weights  # min 0.1, max 1
    return weights.values

def calc_single_freq(draws, weights):
    freq = np.zeros(NUMBERS_RANGE)
    total_weight = 0
    for draw, w in zip(draws, weights):
        for num in draw:
            freq[num - 1] += w
        total_weight += w * NUMBERS_DRAWN
    return freq / total_weight

def calc_pair_freq(draws, weights):
    pair_freq = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    total_weight = 0
    for draw, w in zip(draws, weights):
        for i in range(NUMBERS_DRAWN):
            for j in range(i+1, NUMBERS_DRAWN):
                n1, n2 = draw[i] - 1, draw[j] - 1
                pair_freq[n1, n2] += w
                pair_freq[n2, n1] += w
        total_weight += w * (NUMBERS_DRAWN*(NUMBERS_DRAWN-1)/2)
    return pair_freq / total_weight

def apply_constraints(comb):
    evens = sum(1 for n in comb if n % 2 == 0)
    odds = len(comb) - evens
    return evens >= 2 and odds >= 2

def bayesian_model(draws, weights):
    # Basit bayes örneği: tekil frekansları probabilite olarak kullanıyoruz
    freq = calc_single_freq(draws, weights)
    probs = freq / freq.sum()
    return probs

def markov_model(draws, weights):
    # 1. dereceden Markov geçiş matrisi (basit haliyle)
    pair_freq = calc_pair_freq(draws, weights)
    row_sums = pair_freq.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_probs = np.divide(pair_freq, row_sums, out=np.zeros_like(pair_freq), where=row_sums!=0)
    return transition_probs

def xgboost_model(draws):
    X = []
    y = []
    mlb = MultiLabelBinarizer(classes=range(1, NUMBERS_RANGE+1))
    y_binarized = mlb.fit_transform(draws)
    for i in range(len(draws) - 1):
        X.append(y_binarized[i])
        y.append(draws[i+1])
    X = np.array(X)
    y = np.array(y)
    
    # Çoklu çıktı için RandomForestClassifier alternatif olarak kullandım (daha stabil)
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X, y)
    return clf

def predict_xgboost(model, last_draw):
    mlb = MultiLabelBinarizer(classes=range(1, NUMBERS_RANGE+1))
    input_vec = mlb.fit_transform([last_draw])[0].reshape(1, -1)
    preds_proba = model.predict_proba(input_vec)
    # Her model için 60 olasılık, 6 tane çıktı var (6 pozisyon)
    combined_probs = np.zeros(NUMBERS_RANGE)
    for pos_probs in preds_proba:
        for num, prob in enumerate(pos_probs[0]):
            combined_probs[num] += prob
    return combined_probs / NUMBERS_DRAWN

def generate_predictions(df, n_preds):
    draws = preprocess_draws(df)
    weights = calc_time_weights(df.iloc[:,0])

    # Modeller
    bayes_probs = bayesian_model(draws, weights)
    markov_probs = markov_model(draws, weights)
    rf_model = xgboost_model(draws)

    last_draw = draws[-1]

    xgb_probs = predict_xgboost(rf_model, last_draw)

    combined_probs = (bayes_probs + markov_probs.mean(axis=1) + xgb_probs) / 3

    # Olasılık sırasına göre tahmin oluştur
    candidates = np.argsort(combined_probs)[::-1] + 1

    final_predictions = []
    i = 0
    while len(final_predictions) < n_preds and i < len(candidates):
        comb = candidates[i:i+NUMBERS_DRAWN]
        if len(comb) == NUMBERS_DRAWN and apply_constraints(comb):
            final_predictions.append(sorted(comb))
        i += 1

    return final_predictions

def main():
    uploaded_file = st.file_uploader("Çekiliş verisi içeren CSV dosyasını yükleyin (Tarih, Num1, Num2, ...)", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Yüklenen Veri")
        st.dataframe(df, height=300)

        n_preds = st.number_input("Kaç tahmin yapılsın?", min_value=1, max_value=20, value=5, step=1)

        if st.button("Tahminleri Hesapla"):
            with st.spinner("Tahminler hesaplanıyor..."):
                predictions = generate_predictions(df, n_preds)
            st.success("Tahminler hazır!")
            for i, comb in enumerate(predictions):
                st.write(f"{i+1}. Tahmin: {comb}")

if __name__ == "__main__":
    main()
