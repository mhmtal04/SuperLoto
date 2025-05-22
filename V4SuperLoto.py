import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import xgboost as xgb
from sklearn.model_selection import train_test_split

st.title("Süper Loto Gelişmiş Tahmin Botu (Bayesian - Markov - XGBoost - Koşullu Olasılık)")

uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num6 sütunları)", type="csv")

def preprocess(df):
    df = df.dropna()
    df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.tolist()
    return df

def bayesian_number_model(df):
    observations = np.array([num for sublist in df['Numbers'].tolist() for num in sublist]) - 1
    with pm.Model() as model:
        alpha = np.ones(60)
        p = pm.Dirichlet("p", a=alpha)
        pm.Categorical("obs", p=p, observed=observations)
        trace = pm.sample(1000, tune=1000, cores=1, progressbar=False)
    probs = trace.posterior['p'].mean(dim=["chain", "draw"]).values
    return probs

def markov_chain_model(df):
    transition_matrix = np.ones((60, 60))
    numbers = df['Numbers'].tolist()
    for seq in numbers:
        for i in range(len(seq) - 1):
            transition_matrix[seq[i] - 1, seq[i + 1] - 1] += 1
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    return transition_matrix

def xgboost_model(df):
    numbers_list = df['Numbers'].tolist()
    X, y = [], []
    for i in range(3, len(numbers_list)):
        past_3 = numbers_list[i-3:i]
        flat = [num for sublist in past_3 for num in sublist]
        X.append(flat)
        y.append(numbers_list[i])
    X = np.array(X)
    y = np.array(y) - 1
    models = []
    for i in range(6):
        m = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=50)
        y_i = y[:, i]
        X_train, X_val, y_train, y_val = train_test_split(X, y_i, test_size=0.2, random_state=42, stratify=y_i)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        models.append(m)
    return models

def conditional_probabilities(df):
    counts = pd.DataFrame(0, index=range(1,61), columns=range(1,61))
    singles = pd.Series(0, index=range(1,61))
    for numbers in df['Numbers']:
        for n in numbers:
            singles[n] += 1
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                counts.at[numbers[i], numbers[j]] += 1
                counts.at[numbers[j], numbers[i]] += 1
    singles = singles.replace(0, 1)
    cond_probs = counts.div(singles, axis=0)
    return cond_probs, singles / singles.sum()

def combined_score(numbers, single_prob, cond_prob, bayes_probs, markov_probs, xgb_models, prev_numbers):
    prob_cond = 1.0
    for n in numbers:
        prob_cond *= single_prob[n]
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            prob_cond *= cond_prob.at[numbers[i], numbers[j]]
    prob_bayes = np.prod(bayes_probs[numbers])
    prob_markov = 1.0
    for n in numbers:
        prob_markov *= markov_probs[prev_numbers[n] - 1, n]
    flat_input = []
    for _ in range(3):  # son 3 çekiliş için aynı prev_numbers kullanıldı
        flat_input.extend(prev_numbers)
    X_input = np.array(flat_input).reshape(1, -1)
    prob_xgb = 1.0
    for i, model in enumerate(xgb_models):
        preds = model.predict_proba(X_input)[0]
        prob_xgb *= preds[numbers[i]]
    score_log = np.log(prob_cond + 1e-12) + np.log(prob_bayes + 1e-12) + np.log(prob_markov + 1e-12) + np.log(prob_xgb + 1e-12)
    score = np.exp(score_log)
    return score

def generate_predictions(df, n_preds=1, trials=10000):
    bayes_probs = bayesian_number_model(df)
    markov_probs = markov_chain_model(df)
    xgb_models = xgboost_model(df)
    cond_prob, single_prob = conditional_probabilities(df)
    prev_numbers = df['Numbers'].iloc[-1]

    candidates = []
    while len(candidates) < trials:
        candidate = np.random.choice(range(1, 61), size=6, replace=False)
        # Kısıtlar: en az 2 tek, 2 çift sayı
        if (sum(n % 2 == 0 for n in candidate) < 2) or (sum(n % 2 == 1 for n in candidate) < 2):
            continue
        score = combined_score(candidate, single_prob, cond_prob, bayes_probs, markov_probs, xgb_models, prev_numbers)
        candidates.append((score, candidate))
    candidates.sort(reverse=True, key=lambda x: x[0])
    return [list(c[1]) for c in candidates[:n_preds]]

def main():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess(df)
        st.success("Veri yüklendi!")
        n_preds = st.number_input("Kaç tahmin istersiniz?", min_value=1, max_value=10, value=1, step=1)
        if st.button("Tahminleri Hesapla"):
            with st.spinner("Tahminler hesaplanıyor, lütfen bekleyin..."):
                preds = generate_predictions(df, n_preds=n_preds, trials=5000)
            for i, p in enumerate(preds, 1):
                st.write(f"Tahmin {i}: {sorted(p)}")
    else:
        st.info("Lütfen CSV dosyasını yükleyin.")

if __name__ == "__main__":
    main()
