import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
import pymc as pm
import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- 1. Zaman bazlı ağırlık ---
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    weights = (max_days - days_ago) / max_days
    return weights

# --- 2. Tekil sayıların ağırlıklı olasılıkları ---
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

# --- 3. İkili sayıların ağırlıklı frekansları ---
def pair_frequencies(df):
    pair_freq = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    weights = get_weights(df['Date'])
    for idx, row in df.iterrows():
        numbers = row['Numbers']
        w = weights[idx]
        for a, b in combinations(numbers, 2):
            pair_freq.at[a, b] += w
            pair_freq.at[b, a] += w
    return pair_freq

# --- 4. Koşullu olasılıklar ---
def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    for a in range(1, 61):
        freq_a = single_prob[a]
        if freq_a > 0:
            for b in range(1, 61):
                cond_prob.at[a, b] = pair_freq.at[a, b] / freq_a
        else:
            cond_prob.loc[a, :] = 0
    return cond_prob

# --- 5. Kısıtlar ---
def check_constraints(numbers):
    odd_count = sum(n % 2 == 1 for n in numbers)
    even_count = len(numbers) - odd_count
    return odd_count >= 2 and even_count >= 2

# --- 6. Bayesian modelleme (tekil sayı olasılıklarına Bayesian yaklaşımı) ---
def bayesian_number_model(df):
    # Her sayı için toplam çıkış sayısı
    counts = np.zeros(60)
    for row in df['Numbers']:
        for n in row:
            counts[n - 1] += 1
    # Beta ön bilgisi (tüm sayılar eşit ön bilgi)
    alpha_prior, beta_prior = 1, 1
    with pm.Model() as model:
        p = pm.Beta("p", alpha=alpha_prior, beta=beta_prior, shape=60)
        obs = pm.Binomial("obs", n=len(df), p=p, observed=counts)
        trace = pm.sample(draws=1000, tune=1000, chains=2, progressbar=False, random_seed=42)
    # Posterior ortalamalar
    posterior_means = trace.posterior["p"].mean(dim=["chain", "draw"]).values
    return pd.Series(posterior_means, index=range(1, 61))

# --- 7. Markov zinciri (ardışık çekilişler arasındaki geçiş olasılıkları) ---
def markov_chain_probabilities(df):
    transition_counts = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    total_transitions = pd.Series(0, index=range(1, 61), dtype=float)
    numbers_list = df['Numbers'].tolist()
    for i in range(len(numbers_list) - 1):
        current_draw = numbers_list[i]
        next_draw = numbers_list[i + 1]
        for c in current_draw:
            total_transitions[c] += 1
            for n in next_draw:
                transition_counts.at[c, n] += 1
    # Normalize et
    markov_probs = transition_counts.div(total_transitions, axis=0).fillna(0)
    return markov_probs

# --- 8. XGBoost tabanlı tahmin modeli ---
def xgboost_model(df):
    # Özellik oluşturma: her satırda 60 sütun, sayı çıkışları 0/1 olarak
    X = []
    y = []
    numbers_list = df['Numbers'].tolist()
    for i in range(len(numbers_list) - 1):
        current_draw = numbers_list[i]
        next_draw = numbers_list[i + 1]
        x_row = np.zeros(60)
        y_row = np.zeros(60)
        for c in current_draw:
            x_row[c - 1] = 1
        for n in next_draw:
            y_row[n - 1] = 1
        X.append(x_row)
        y.append(y_row)
    X = np.array(X)
    y = np.array(y)
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    # Çoklu çıktı için her sayı ayrı sınıf gibi ayrı model kurulabilir, burada basitçe her sayı için binary sınıflandırma yapıyoruz
    models = []
    for i in range(60):
        m = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        m.fit(X_train, y_train[:, i])
        models.append(m)
    return models, X_test, y_test

# --- 9. XGBoost tahmini ---
def xgboost_predict(models, last_draw):
    x_row = np.zeros(60)
    for n in last_draw:
        x_row[n - 1] = 1
    probs = np.zeros(60)
    for i, m in enumerate(models):
        proba = m.predict_proba(x_row.reshape(1, -1))[0][1]  # Pozitif sınıf olasılığı
        probs[i] = proba
    return pd.Series(probs, index=range(1, 61))

# --- 10. Tahmin üret ---
def generate_predictions(df, n_preds=1, trials=10000):
    # 1) Ağırlıklı olasılıklar
    single_prob = weighted_single_probabilities(df)
    pair_freq = pair_frequencies(df)
    cond_prob = conditional_probabilities(single_prob, pair_freq)

    # 2) Bayesian
    bayes_probs = bayesian_number_model(df)

    # 3) Markov
    markov_probs = markov_chain_probabilities(df)

    # 4) XGBoost
    xgb_models, _, _ = xgboost_model(df)
    last_draw = df['Numbers'].iloc[-1]
    xgb_probs = xgboost_predict(xgb_models, last_draw)

    # 5) Bütün olasılıkları ağırlıklı birleştir (normalize etmeden önce)
    combined_score = (single_prob + bayes_probs + markov_probs.mean(axis=0) + xgb_probs) / 4

    numbers_list = list(range(1, 61))
    predictions = []

    for _ in range(n_preds):
        best_combo = None
        best_score = 0
        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=6, replace=False, p=combined_score.values/combined_score.values.sum())
            if not check_constraints(chosen):
                continue
            # Kombinasyon olasılığı
            prob = 1.0
            for n in chosen:
                prob *= combined_score[n]
            # Koşullu olasılıkları dahil et
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    prob *= cond_prob.at[chosen[i], chosen[j]]
            if prob > best_score:
                best_score = prob
                best_combo = chosen
        if best_combo is None:
            # Kısıt uygulanmıyorsa rastgele seçim
            best_combo = np.random.choice(numbers_list, size=6, replace=False)
            best_score = 0
        predictions.append((np.sort(best_combo), best_score))

    return predictions

# --- 11. Streamlit Arayüzü ---
def main():
    st.title("Süper Loto Gelişmiş Tahmin Botu (Bayesian - Markov - XGBoost - Koşullu Olasılık)")
    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1','Num2','Num3','Num4','Num5','Num6']].values.tolist()

        st.success("Veri yüklendi!")
        st.write(df)

        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1)
        if st.button("Tahmin Üret"):
            with st.spinner("Tahminler hesaplanıyor..."):
                preds = generate_predictions(df, n_preds=n_preds, trials=5000)
            st.success("Tahminler hazır!")
            for i, (combo, prob) in enumerate(preds, 1):
                st.write(f"Tahmin {i}: {', '.join(map(str, combo))}  (Skor: {prob:.6e})")

if __name__ == "__main__":
    main()
