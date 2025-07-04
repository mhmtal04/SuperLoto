import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB

# --- Yardımcı Fonksiyonlar ---
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    return (max_days - days_ago) / max_days

def weighted_single_probabilities(df):
    weights = get_weights(df['Date'])
    total_weight = weights.sum()
    freq = pd.Series(0, index=range(1, 61), dtype=float)
    for idx, row in df.iterrows():
        for n in row['Numbers']:
            freq[n] += weights[idx]
    return freq / total_weight

def pair_frequencies(df):
    weights = get_weights(df['Date'])
    pair_freq = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    for idx, row in df.iterrows():
        for a, b in combinations(row['Numbers'], 2):
            pair_freq.at[a, b] += weights[idx]
            pair_freq.at[b, a] += weights[idx]
    return pair_freq

def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    for a in range(1, 61):
        if single_prob[a] > 0:
            cond_prob.loc[a] = pair_freq.loc[a] / single_prob[a]
    return cond_prob

def model_pattern_score(combo):
    ranges = {"0s": 0, "10s": 0, "20s": 0, "30s": 0, "40s": 0, "50s": 0}
    for n in combo:
        if n < 10: ranges["0s"] += 1
        elif n < 20: ranges["10s"] += 1
        elif n < 30: ranges["20s"] += 1
        elif n < 40: ranges["30s"] += 1
        elif n < 50: ranges["40s"] += 1
        else: ranges["50s"] += 1
    pattern = [ranges[k] for k in ["0s", "10s", "20s", "30s", "40s", "50s"]]
    return 1.0 if pattern == [1, 1, 1, 2, 1, 0] else 0.1

# --- Yeni: Vektörleştirilmiş Tahmin Fonksiyonu ---
def generate_predictions_vectorized(df, single_prob, cond_prob, nb_model, gb_model, markov_probs, pair_freq, n_preds=1, trials=500000):
    predictions = []
    numbers = np.arange(1, 61)
    single_probs = single_prob.values
    single_probs /= single_probs.sum()  # Normalize

    all_combos = np.array([
        np.sort(np.random.choice(numbers, size=6, replace=False, p=single_probs))
        for _ in range(trials)
    ])

    single_scores = np.prod(single_prob[all_combos], axis=1)

    pair_scores = np.ones(trials)
    for i in range(6):
        for j in range(i+1, 6):
            pair_vals = np.array([cond_prob.at[a, b] if cond_prob.at[a, b] > 0 else 1e-6
                                  for a, b in zip(all_combos[:, i], all_combos[:, j])])
            pair_scores *= pair_vals

    X_test = np.array([[len(df) + 1]])
    nb_probs = nb_model.predict_proba(X_test)[0]
    nb_classes = nb_model.classes_

    nb_scores = np.mean([
        [nb_probs[np.where(nb_classes == n)[0][0]] if n in nb_classes else 0 for n in combo]
        for combo in all_combos
    ], axis=1)

    gb_pred = gb_model.predict(X_test)[0]
    markov_scores = np.array([np.mean(markov_probs[combo]) for combo in all_combos])

    pattern_scores = []
    for combo in all_combos:
        model_score = model_pattern_score(combo)
        pair_product = 1.0
        for a, b in combinations(combo, 2):
            f = pair_freq.at[a, b]
            pair_product *= f if f > 0 else 1e-6
        pattern_scores.append(model_score * pair_product)
    pattern_scores = np.array(pattern_scores)

    final_scores = single_scores * (1 + nb_scores) * (1 + gb_pred / 60.0) * (1 + markov_scores) * (1 + pattern_scores)

    top_indices = np.argsort(final_scores)[-n_preds:][::-1]
    for idx in top_indices:
        predictions.append((all_combos[idx], final_scores[idx]))

    return predictions

# --- Makine Öğrenmesi Eğitimleri ---
def train_naive_bayes(df):
    X = np.repeat(df.index.values.reshape(-1, 1), 6, axis=0)
    y = np.array([n for row in df['Numbers'] for n in row])
    model = GaussianNB()
    model.fit(X, y)
    return model

def train_gradient_boost(df):
    X = np.repeat(df.index.values.reshape(-1, 1), 6, axis=0)
    y = np.array([n for row in df['Numbers'] for n in row])
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

def markov_chain(df):
    transitions = np.zeros((61, 61))
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]['Numbers']
        curr = df.iloc[i]['Numbers']
        for a in prev:
            for b in curr:
                transitions[a][b] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    return np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums != 0)

# --- Arayüz ---
def main():
    st.title("🧠 Süper Loto | Hızlı Vektörleştirilmiş Tahmin Botu (v9)")

    uploaded_file = st.file_uploader("📁 Çekiliş CSV dosyasını yükle (Date, Num1~Num6)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

        st.success(f"✅ Veri yüklendi. Toplam çekiliş sayısı: {len(df)}")

        with st.spinner("⏳ Model eğitiliyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)
            nb_model = train_naive_bayes(df)
            gb_model = train_gradient_boost(df)
            markov_probs = markov_chain(df)

        n_preds = st.number_input("🎲 Kaç tahmin üretilsin?", 1, 10, 3)
        trials = st.number_input("🌀 Kaç kombinasyon denensin? (varsayılan 500,000)", 10000, 5000000, 500000)

        if st.button("🚀 Tahminleri Başlat"):
            with st.spinner("🔍 En iyi kombinasyonlar aranıyor..."):
                results = generate_predictions_vectorized(df, single_prob, cond_prob, nb_model, gb_model,
                                                          markov_probs, pair_freq, n_preds=n_preds, trials=trials)
            st.success("🎉 Tahminler tamamlandı!")
            for i, (combo, score) in enumerate(results):
                st.write(f"{i+1}. Tahmin: {', '.join(map(str, combo))} | Skor: {score:.2e}")

if __name__ == "__main__":
    main()
