import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer

# --- 1. Veri ön işleme ve ağırlık hesaplama (zaman bazlı) ---
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    weights = (max_days - days_ago) / max_days
    return weights

def preprocess_df(df):
    # 'Numbers' sütunu: [Num1, Num2, ..., Num6] listesi
    df['Date'] = pd.to_datetime(df['Date'])
    df['Numbers'] = df[['Num1','Num2','Num3','Num4','Num5','Num6']].values.tolist()
    return df

# --- 2. Tekil sayıların ağırlıklı frekansı ---
def weighted_single_probabilities(df):
    weights = get_weights(df['Date'])
    total_weight = weights.sum()

    freq = pd.Series(0, index=range(1, 61), dtype=float)
    for idx, row in df.iterrows():
        numbers = row['Numbers']
        w = weights[idx]
        for n in numbers:
            freq[n] += w

    prob = freq / total_weight
    return prob

# --- 3. İkili sayı frekansları ---
def pair_frequencies(df):
    pair_freq = pd.DataFrame(0, index=range(1,61), columns=range(1,61), dtype=float)
    weights = get_weights(df['Date'])
    for idx, row in df.iterrows():
        numbers = row['Numbers']
        w = weights[idx]
        for a,b in combinations(numbers,2):
            pair_freq.at[a,b] += w
            pair_freq.at[b,a] += w
    return pair_freq

# --- 4. Koşullu olasılıklar ---
def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1,61), columns=range(1,61), dtype=float)
    for a in range(1,61):
        freq_a = single_prob[a]
        if freq_a > 0:
            cond_prob.loc[a] = pair_freq.loc[a] / freq_a
        else:
            cond_prob.loc[a] = 0
    return cond_prob

# --- 5. Kısıt: en az 2 tek ve 2 çift sayı ---
def check_constraints(numbers):
    odd_count = sum(n%2==1 for n in numbers)
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

# --- 6. Kombinasyon olasılığı ---
def combo_probability(numbers, single_prob, cond_prob):
    prob = 1.0
    for n in numbers:
        prob *= single_prob[n]
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            prob *= cond_prob.at[numbers[i], numbers[j]]
    return prob

# --- 7. Bayesian Model (MultinomialNB) Eğitimi ---
def bayesian_model(df):
    mlb = MultiLabelBinarizer(classes=range(1,61))
    X = mlb.fit_transform(df['Numbers'])

    # Hedef olarak aynı veride bir sonraki çekilişi kullanalım (shift -1)
    y = df['Numbers'].shift(-1)[:-1]
    X = X[:-1]

    y_bin = mlb.transform(y)

    model = MultinomialNB()
    model.fit(X, y_bin)
    return model, mlb

# --- 8. Markov Zinciri Modeli ---
def markov_model(df):
    transition = pd.DataFrame(0, index=range(1,61), columns=range(1,61), dtype=float)
    counts = pd.Series(0, index=range(1,61), dtype=float)
    for i in range(len(df)-1):
        curr = df.loc[i,'Numbers']
        nxt = df.loc[i+1,'Numbers']
        for c in curr:
            counts[c] += 1
            for n in nxt:
                transition.at[c,n] += 1
    # Normalize et
    for c in range(1,61):
        if counts[c] > 0:
            transition.loc[c] /= counts[c]
    return transition

# --- 9. XGBoost Modelleri Eğitimi ---
def xgboost_models(df):
    NUM_NUMBERS = 6
    X = []
    y = []

    mlb = MultiLabelBinarizer(classes=range(1,61))
    X_bin = mlb.fit_transform(df['Numbers'])
    for i in range(len(df)-1):
        X.append(X_bin[i])
        y.append(df.loc[i+1,'Numbers'])

    X = np.array(X)
    y = np.array(y)

    models = []
    for i in range(NUM_NUMBERS):
        target = y[:, i] - 1
        unique_classes = np.unique(target)
        if len(unique_classes) < 2:
            # Yetersiz sınıf sayısı, model eğitilmez
            models.append(None)
            continue
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
        model.fit(X, target)
        models.append(model)
    return models, mlb

# --- 10. Tahmin Üretimi (Monte Carlo + Modellerle Kombinasyon) ---
def generate_predictions(df, single_prob, cond_prob, bayes_model, bayes_mlb, markov_trans, xgb_models, xgb_mlb, n_preds=1, n_numbers=6, trials=10000):
    numbers_list = list(range(1,61))
    single_probs = single_prob.values
    predictions = []

    for _ in range(n_preds):
        best_combo = None
        best_score = -1
        for __ in range(trials):
            # Monte Carlo seçim (tekil olasılık ile)
            chosen = np.random.choice(numbers_list, size=n_numbers, replace=False, p=single_probs/single_probs.sum())
            chosen = np.sort(chosen)

            if not check_constraints(chosen):
                continue

            # Kombinasyon olasılığı
            p = combo_probability(chosen, single_prob, cond_prob)

            # Bayesian skoru
            X_test = bayes_mlb.transform([chosen])
            bayes_prob = bayes_model.predict_proba(X_test)[0].mean()

            # Markov skoru: seçilen sayılar için ortalama geçiş olasılığı
            markov_score = 0
            for i in chosen:
                markov_score += markov_trans.loc[i].mean()
            markov_score /= n_numbers

            # XGBoost skoru (varsa)
            xgb_score = 0
            valid_models = 0
            X_xgb = xgb_mlb.transform([chosen])
            for mdl in xgb_models:
                if mdl is None:
                    continue
                proba = mdl.predict_proba(X_xgb)[0].mean()
                xgb_score += proba
                valid_models += 1
            if valid_models > 0:
                xgb_score /= valid_models
            else:
                xgb_score = 0.001  # Çok düşük ama sıfır değil

            # Kombine skor (ağırlıklı toplam)
            combined_score = (0.4 * p) + (0.2 * bayes_prob) + (0.2 * markov_score) + (0.2 * xgb_score)

            if combined_score > best_score:
                best_score = combined_score
                best_combo = chosen

        if best_combo is not None:
            predictions.append((best_combo, best_score))
        else:
            chosen = np.random.choice(numbers_list, size=n_numbers, replace=False)
            predictions.append((np.sort(chosen), 0))

    return predictions

# --- 11. Streamlit Arayüzü ---
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu")
    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess_df(df)

        st.success(f"Veriler yüklendi. Toplam satır sayısı: {len(df)}")
        st.write(df)  # Tüm satırları göster

        with st.spinner("Olasılıklar hesaplanıyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)

        st.success("Olasılıklar hesaplandı!")

        with st.spinner("Modeller eğitiliyor... Bu biraz zaman alabilir..."):
            bayes_model, bayes_mlb = bayesian_model(df
