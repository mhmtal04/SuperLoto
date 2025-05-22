import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
import pymc as pm

NUMBERS_RANGE = 60
NUMBERS_DRAWN = 6

# --- Veri ön işleme ---
def preprocess_draws(df):
    # df: çekilişler dataframe, sütunlar sayılar (num1, num2, ...)
    # numpy array döner, her satırda 6 sayı
    numbers = df.iloc[:, 1:].values.astype(int)
    return numbers

# --- Zaman ağırlıklı ağırlık hesaplama ---
def get_weights(dates):
    # dates: çekiliş tarihleri pandas Series
    # Yeni tarihlere daha fazla ağırlık verir
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    weights = (dates - dates.min()) / (max_date - dates.min())
    weights = weights.fillna(0.5)  # NaN varsa ortalama ver
    return weights.values

# --- Tekil frekans hesaplama ---
def calc_single_freq(numbers, weights=None):
    counts = np.zeros(NUMBERS_RANGE)
    for i, draw in enumerate(numbers):
        for num in draw:
            counts[num-1] += weights[i] if weights is not None else 1
    freq = counts / counts.sum()
    return freq

# --- İkili frekans hesaplama ---
def calc_pair_freq(numbers, weights=None):
    counts = np.zeros((NUMBERS_RANGE, NUMBERS_RANGE))
    for i, draw in enumerate(numbers):
        for x in draw:
            for y in draw:
                if x != y:
                    counts[x-1, y-1] += weights[i] if weights is not None else 1
    # Normalize et
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cond_prob = counts / row_sums
    return cond_prob

# --- Basit koşullu olasılık ---
def calc_conditional_prob(pair_freq, single_freq):
    # P(A|B) = P(A,B)/P(B) yaklaşık olarak
    cond_prob = pair_freq / (single_freq + 1e-6)
    cond_prob[np.isnan(cond_prob)] = 0
    cond_prob[np.isinf(cond_prob)] = 0
    return cond_prob

# --- Markov geçiş matrisi ---
def markov_transition_matrix(numbers):
    n_states = NUMBERS_RANGE
    trans_counts = np.zeros((n_states, n_states))

    for i in range(len(numbers) - 1):
        current_draw = numbers[i] - 1
        next_draw = numbers[i+1] - 1
        for cur_num in current_draw:
            for next_num in next_draw:
                trans_counts[cur_num, next_num] += 1

    trans_probs = np.zeros_like(trans_counts)
    for i in range(n_states):
        row_sum = trans_counts[i].sum()
        if row_sum > 0:
            trans_probs[i] = trans_counts[i] / row_sum
        else:
            trans_probs[i] = np.ones(n_states) / n_states
    return trans_probs

def markov_predict_probs(trans_probs, last_draw):
    n_states = NUMBERS_RANGE
    markov_scores = np.zeros(n_states)
    for num in last_draw:
        markov_scores += trans_probs[num-1]
    if markov_scores.sum() > 0:
        markov_scores /= markov_scores.sum()
    else:
        markov_scores = np.ones(n_states) / n_states
    return markov_scores

# --- XGBoost modeli ---
def xgboost_model(numbers):
    X = []
    y = []
    for i in range(len(numbers)-1):
        feature = np.zeros(NUMBERS_RANGE, dtype=int)
        feature[numbers[i] - 1] = 1
        X.append(feature)
        y.append(numbers[i+1] - 1)

    X = np.array(X)
    y = np.array(y)

    model = MultiOutputClassifier(xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, verbosity=0))
    model.fit(X, y)
    return model

def predict_xgboost(model, last_draw):
    feature = np.zeros(NUMBERS_RANGE, dtype=int)
    feature[last_draw - 1] = 1
    preds = model.predict_proba(feature.reshape(1, -1))
    avg_probs = np.mean(np.array([p[0] for p in preds]), axis=0)
    return avg_probs

# --- Basit Bayesian model ---
def bayesian_model(single_freq, cond_prob, numbers):
    with pm.Model() as model:
        p = pm.Dirichlet("p", a=single_freq * 100 + 1)
        observed = pm.Categorical("obs", p=p, observed=numbers.flatten()-1)
        trace = pm.sample(200, tune=200, cores=1, chains=1, progressbar=False)
    bayes_probs = np.mean(trace["p"], axis=0)
    return bayes_probs

# --- Kısıtlar ---
def check_constraints(selected_nums):
    evens = sum(1 for n in selected_nums if n % 2 == 0)
    odds = len(selected_nums) - evens
    if evens < 2 or odds < 2:
        return False
    return True

# --- Tahminleri birleştir ---
def generate_predictions(df, n_preds=5):
    numbers = preprocess_draws(df)
    weights = get_weights(df.iloc[:, 0])

    single_freq = calc_single_freq(numbers, weights)
    pair_freq = calc_pair_freq(numbers, weights)
    cond_prob = calc_conditional_prob(pair_freq, single_freq)

    xgb_mod = xgboost_model(numbers)
    last_draw = numbers[-1]

    xgb_probs = predict_xgboost(xgb_mod, last_draw)
    bayes_probs = bayesian_model(single_freq, cond_prob, numbers)
    trans_probs = markov_transition_matrix(numbers)
    markov_probs = markov_predict_probs(trans_probs, last_draw)

    combined_probs = (single_freq + xgb_probs + bayes_probs + markov_probs) / 4

    # En yüksek olasılıkları seç
    sorted_nums = np.argsort(combined_probs)[::-1] + 1

    final_preds = []
    for num in sorted_nums:
        if len(final_preds) < NUMBERS_DRAWN:
            final_preds.append(num)
        if len(final_preds) == NUMBERS_DRAWN and check_constraints(final_preds):
            break

    return final_preds[:NUMBERS_DRAWN]

# --- Streamlit arayüzü ---
def main():
    st.title("Super Loto Tahmin Sistemi")

    uploaded_file = st.file_uploader("Çekiliş CSV dosyasını yükleyin", type=["csv"])
    n_preds = st.slider("Kaç tahmin yapılacak?", 1, 10, 5)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Veri önizlemesi:")
        st.dataframe(df.head())

        if st.button("Tahminleri hesapla"):
            with st.spinner("Tahminler hesaplanıyor..."):
                preds = generate_predictions(df, n_preds)
                st.success(f"Tahmin edilen sayılar: {preds}")

if __name__ == "__main__":
    main()
