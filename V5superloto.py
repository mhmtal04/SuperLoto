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
        w = weights[idx]
        for n in row['Numbers']:
            freq[n] += w
    return freq / total_weight

def pair_frequencies(df):
    pair_freq = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float)
    weights = get_weights(df['Date'])
    for idx, row in df.iterrows():
        w = weights[idx]
        for a, b in combinations(row['Numbers'], 2):
            pair_freq.at[a, b] += w
            pair_freq.at[b, a] += w
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
        if n < 10:
            ranges["0s"] += 1
        elif n < 20:
            ranges["10s"] += 1
        elif n < 30:
            ranges["20s"] += 1
        elif n < 40:
            ranges["30s"] += 1
        elif n < 50:
            ranges["40s"] += 1
        else:
            ranges["50s"] += 1
    pattern = [ranges[k] for k in ["0s", "10s", "20s", "30s", "40s", "50s"]]
    return 1.0 if pattern == [1, 1, 1, 2, 1, 0] else 0.1

def structured_pattern_score(combo, single_prob, pair_freq):
    model_score = model_pattern_score(combo)
    single_product = np.prod([single_prob[n] for n in combo])
    pair_product = 1.0
    for a, b in combinations(combo, 2):
        f = pair_freq.at[a, b]
        pair_product *= f if f > 0 else 1e-6
    return model_score * single_product * pair_product

def markov_chain(df):
    transitions = np.zeros((61, 61))
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]['Numbers']
        curr = df.iloc[i]['Numbers']
        for a in prev:
            for b in curr:
                transitions[a][b] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    return np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums!=0)

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

def generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_probs, pair_freq, n_preds=1, trials=20000):
    predictions = []
    numbers_list = list(range(1, 61))
    single_probs_list = single_prob.values

    for _ in range(n_preds):
        best_combo = None
        best_score = -1
        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=6, replace=False, p=single_probs_list / single_probs_list.sum())
            chosen = np.sort(chosen)

            combo_score = 1.0
            for i in range(6):
                combo_score *= single_prob[chosen[i]]
                for j in range(i+1, 6):
                    combo_score *= cond_prob.at[chosen[i], chosen[j]]

            X_test = np.array([[len(df) + 1]])
            classes = nb_model.classes_
            probs = nb_model.predict_proba(X_test)[0]
            nb_score = np.mean([probs[np.where(classes == n)[0][0]] if n in classes else 0 for n in chosen])

            gb_pred = gb_model.predict(X_test)[0]
            markov_score = np.mean([markov_probs[a].mean() if a < markov_probs.shape[0] else 0 for a in chosen])
            red_score = structured_pattern_score(chosen, single_prob, pair_freq)

            final_score = combo_score * (1 + nb_score) * (1 + gb_pred / 60.0) * (1 + markov_score) * (1 + red_score)

            if final_score > best_score:
                best_score = final_score
                best_combo = chosen

        if best_combo is not None:
            predictions.append((best_combo, best_score))

    return predictions

# --- Streamlit Arayüz ---
def main():
    st.title("Süper Loto | Gelişmiş Tahmin Botu v7 (Kısıtsız + Karşılaştırmalı)")

    uploaded_file = st.file_uploader("CSV 1: Geçmiş çekiliş verisi (Date, Num1~Num6)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

        st.success(f"Veriler yüklendi. Toplam satır: {len(df)}")
        st.write(df)

        with st.spinner("Model eğitiliyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)
            nb_model = train_naive_bayes(df)
            gb_model = train_gradient_boost(df)
            markov_probs = markov_chain(df)

        n_preds = st.number_input("Kaç tahmin üretmek istiyorsunuz?", min_value=1, max_value=10, value=3)
        if st.button("Tahminleri Hesapla"):
            with st.spinner("Tahminler üretiliyor..."):
                preds = generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_probs, pair_freq, n_preds=n_preds)
            st.success("Tahminler hazır!")
            for i, (comb, _) in enumerate(preds):
                st.write(f"{i+1}. Tahmin: {', '.join(map(str, comb))}")

    # --- Tahmin Karşılaştırma ---
    st.subheader("Tahmin – Gerçek Karşılaştırması")
    uploaded_compare = st.file_uploader("CSV 2: Tahmin/gerçek karşılaştırma (Tahmin, Gerçek Sonuç)", type=["csv"], key="compare")

    if uploaded_compare is not None:
        df_compare = pd.read_csv(uploaded_compare)
        results = []

        for idx, row in df_compare.iterrows():
            try:
                prediction = list(map(int, str(row['Prediction']).split(',')))
                real = list(map(int, str(row['Real Result']).split(',')))
                known = sorted(set(prediction) & set(real))
                hits = len(known)
                results.append({
                    "Prediction Number": idx + 1,
                    "Prediction": prediction,
                    "Real Result": real,
                    "Numbers known": known,
                    "Number of Hits": hits
                })
            except Exception as e:
                st.error(f"{idx+1}. satırda hata: {e}")

        if results:
            df_results = pd.DataFrame(results)
            st.success("Karşılaştırma tamamlandı.")
            st.write(df_results)

            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("Sonuçları CSV olarak indir", csv_data, "karsilastirma_sonuclari.csv", "text/csv")

if __name__ == "__main__":
    main()
