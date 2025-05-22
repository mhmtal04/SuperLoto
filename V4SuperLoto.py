import streamlit as st import pandas as pd import numpy as np from itertools import combinations from datetime import datetime from sklearn.ensemble import GradientBoostingRegressor from sklearn.naive_bayes import GaussianNB from sklearn.preprocessing import MultiLabelBinarizer

------------------------------ Yardımcı Fonksiyonlar ------------------------------

def get_weights(dates): dates = pd.to_datetime(dates) days_ago = (dates.max() - dates).dt.days max_days = days_ago.max() + 1 weights = (max_days - days_ago) / max_days return weights

def check_constraints(numbers): odd_count = sum(n % 2 == 1 for n in numbers) even_count = len(numbers) - odd_count return odd_count >= 2 and even_count >= 2

------------------------------ Olasılık Hesaplamaları ------------------------------

def weighted_single_probabilities(df): weights = get_weights(df['Date']) total_weight = weights.sum() freq = pd.Series(0, index=range(1, 61), dtype=float) for idx, row in df.iterrows(): numbers = row['Numbers'] w = weights[idx] for n in numbers: freq[n] += w prob = freq / total_weight return prob

def pair_frequencies(df): pair_freq = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float) weights = get_weights(df['Date']) for idx, row in df.iterrows(): numbers = row['Numbers'] w = weights[idx] for a, b in combinations(numbers, 2): pair_freq.at[a, b] += w pair_freq.at[b, a] += w return pair_freq

def conditional_probabilities(single_prob, pair_freq): cond_prob = pd.DataFrame(0, index=range(1, 61), columns=range(1, 61), dtype=float) for a in range(1, 61): freq_a = single_prob[a] if freq_a > 0: for b in range(1, 61): cond_prob.at[a, b] = pair_freq.at[a, b] / freq_a return cond_prob

------------------------------ Makine Öğrenimi Modelleri ------------------------------

def train_naive_bayes(df): mlb = MultiLabelBinarizer(classes=range(1, 61)) X = df.index.values.reshape(-1, 1) Y = mlb.fit_transform(df['Numbers']) model = GaussianNB() model.fit(X, Y) return model, mlb

def train_gradient_boost(df): X = df.index.values.reshape(-1, 1) y = [int(n) for row in df['Numbers'] for n in row] X_boost = np.repeat(X, 6, axis=0) model = GradientBoostingRegressor() model.fit(X_boost, y) return model

def markov_chain(df): transitions = np.zeros((61, 61)) for i in range(1, len(df)): prev = df.iloc[i - 1]['Numbers'] curr = df.iloc[i]['Numbers'] for a in prev: for b in curr: transitions[a][b] += 1 transition_probs = transitions / transitions.sum(axis=1, keepdims=True) return transition_probs

------------------------------ Tahmin Üret ------------------------------

def generate_predictions(df, single_prob, cond_prob, nb_model, mlb, gb_model, markov_probs, n_preds=1, trials=5000): predictions = [] numbers_list = list(range(1, 61)) single_probs_list = single_prob.values for _ in range(n_preds): best_combo = None best_score = -1 for _ in range(trials): chosen = np.random.choice(numbers_list, size=6, replace=False, p=single_probs_list / single_probs_list.sum()) chosen = np.sort(chosen) if not check_constraints(chosen): continue # Kombine skorlama combo_score = 1.0 for i in range(6): combo_score *= single_prob[chosen[i]] for j in range(i + 1, 6): combo_score *= cond_prob.at[chosen[i], chosen[j]] X_test = np.array([[len(df) + 1]]) nb_pred = nb_model.predict_proba(X_test)[0] gb_pred = gb_model.predict(X_test)[0] markov_score = np.mean([markov_probs[a].mean() for a in chosen if a < len(markov_probs)]) final_score = combo_score * (1 + nb_pred.mean()) * (1 + gb_pred / 60.0) * (1 + markov_score) if final_score > best_score: best_score = final_score best_combo = chosen if best_combo is not None: predictions.append((best_combo, best_score)) return predictions

------------------------------ Streamlit Arayüz ------------------------------

def main(): st.title("Süper Loto | Gelişmiş Tahmin Botu v4") uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()

    st.success(f"Veriler yüklendi. Toplam satır: {len(df)}")
    st.write(df)

    with st.spinner("Model eğitiliyor ve olasılıklar hesaplanıyor..."):
        single_prob = weighted_single_probabilities(df)
        pair_freq = pair_frequencies(df)
        cond_prob = conditional_probabilities(single_prob, pair_freq)
        nb_model, mlb = train_naive_bayes(df)
        gb_model = train_gradient_boost(df)
        markov_probs = markov_chain(df)

    n_preds = st.slider("Kaç tahmin üretilsin?", min_value=1, max_value=10, value=3)

    if st.button("Tahminleri Hesapla"):
        with st.spinner("Tahminler hesaplanıyor..."):
            preds = generate_predictions(df, single_prob, cond_prob, nb_model, mlb, gb_model, markov_probs, n_preds=n_preds)
        st.success("Tahminler hazır!")
        for i, (comb, score) in enumerate(preds):
            st.write(f"{i+1}. Tahmin: {', '.join(map(str, comb))} | Skor: {score:.4e}")

if name == "main": main()

