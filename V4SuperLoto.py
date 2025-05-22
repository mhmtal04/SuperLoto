import csv
import numpy as np
from collections import Counter, defaultdict
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# -------------------
# 1. CSV'den çekiliş verisi yükleme
# -------------------
def load_draws_from_csv(uploaded_file):
    draws = []
    decoded = uploaded_file.read().decode('utf-8').splitlines()
    reader = csv.reader(decoded)
    for row in reader:
        try:
            draw = list(map(int, row))
            if len(draw) == 6:
                draws.append(draw)
        except ValueError:
            continue
    return draws

# -------------------
# 2. Zaman Bazlı Ağırlıklı Frekans Hesaplama
# -------------------
def weighted_frequency(draws, max_number=60):
    weights = np.linspace(1, 2, len(draws))
    freq = Counter()
    for w, draw in zip(weights, draws):
        for num in draw:
            freq[num] += w
    total = sum(freq.values())
    return {num: freq.get(num, 0) / total for num in range(1, max_number + 1)}

# -------------------
# 3. İkili sayı frekansları ve koşullu olasılıklar
# -------------------
def pair_frequencies(draws):
    pair_counts = defaultdict(int)
    single_counts = Counter()
    for draw in draws:
        for num in draw:
            single_counts[num] += 1
        for i in range(len(draw)):
            for j in range(i+1, len(draw)):
                pair = tuple(sorted([draw[i], draw[j]]))
                pair_counts[pair] += 1
    return pair_counts, single_counts

def conditional_probabilities(pair_counts, single_counts):
    cond_probs = {}
    for (a,b), count in pair_counts.items():
        cond_probs[(a,b)] = count / single_counts[a] if single_counts[a] > 0 else 0
        cond_probs[(b,a)] = count / single_counts[b] if single_counts[b] > 0 else 0
    return cond_probs

# -------------------
# 4. Markov Zinciri Geçiş Olasılıkları
# -------------------
def compute_markov_probs(draws):
    transitions = defaultdict(list)
    for draw in draws:
        for i in range(len(draw)-1):
            curr_num = draw[i]
            next_num = draw[i+1]
            transitions[curr_num].append(next_num)

    markov_probs = {}
    for curr_num, next_nums in transitions.items():
        count = Counter(next_nums)
        total = sum(count.values())
        markov_probs[curr_num] = {num: cnt / total for num, cnt in count.items()}
    return markov_probs

# -------------------
# 5. Bayesian Güncelleme
# -------------------
def bayesian_update(prior_probs, observed_counts, total_observations):
    posterior_probs = {}
    for num in prior_probs.keys():
        likelihood = observed_counts.get(num, 0) / total_observations if total_observations > 0 else 0
        posterior_probs[num] = likelihood * prior_probs[num]
    total = sum(posterior_probs.values())
    if total == 0:
        return prior_probs
    return {k: v / total for k, v in posterior_probs.items()}

# -------------------
# 6. XGBoost Hazırlık ve Eğitim
# -------------------
def prepare_features(draws, max_number=60):
    X = []
    y = []
    for i in range(len(draws) - 1):
        current_draw = draws[i]
        next_draw = draws[i + 1]
        features = [1 if num in current_draw else 0 for num in range(1, max_number + 1)]
        for num in range(1, max_number + 1):
            label = 1 if num in next_draw else 0
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

def train_xgboost(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return model, acc

def predict_xgboost(model, last_draw, max_number=60):
    features = [1 if num in last_draw else 0 for num in range(1, max_number + 1)]
    proba = model.predict_proba([features])[0]
    return {num+1: proba[num] for num in range(max_number)}

# -------------------
# 7. Tahmin oluşturma ve kısıtlar
# -------------------
def is_valid_combination(numbers):
    evens = sum(1 for n in numbers if n % 2 == 0)
    odds = len(numbers) - evens
    return odds >= 2 and evens >= 2

def combined_prediction(draws, model_xgb, alpha=0.4, beta=0.3, gamma=0.3):
    max_number = 60
    last_draw = draws[-1]
    prior_probs = weighted_frequency(draws, max_number)
    pair_counts, single_counts = pair_frequencies(draws)
    cond_probs = conditional_probabilities(pair_counts, single_counts)
    observed_counts = Counter(last_draw)
    total_obs = len(last_draw)
    bayes_probs = bayesian_update(prior_probs, observed_counts, total_obs)
    markov_probs = compute_markov_probs(draws)
    xgb_probs = predict_xgboost(model_xgb, last_draw, max_number)
    combined_scores = {}
    for num in range(1, max_number + 1):
        markov_prob = np.mean(list(markov_probs.get(num, {}).values())) if markov_probs.get(num) else 0
        cond_prob = sum(cond_probs.get((ld_num, num), 0) for ld_num in last_draw) / len(last_draw)
        combined_scores[num] = (alpha * xgb_probs.get(num, 0) +
                                beta * bayes_probs.get(num, 0) +
                                gamma * (0.5*markov_prob + 0.5*cond_prob))
    from itertools import combinations
    top_candidates = sorted(combined_scores, key=combined_scores.get, reverse=True)[:20]
    valid_combos = [(combo, sum(combined_scores[n] for n in combo)) for combo in combinations(top_candidates, 6) if is_valid_combination(combo)]
    if not valid_combos:
        return sorted(combined_scores, key=combined_scores.get, reverse=True)[:6], combined_scores
    best_combo = max(valid_combos, key=lambda x: x[1])[0]
    return best_combo, combined_scores

# -------------------
# Streamlit Uygulaması
# -------------------
st.title("Süper Loto V4 Tahmin Botu")

uploaded_file = st.file_uploader("CSV formatında geçmiş çekiliş verisini yükleyin.", type="csv")

if uploaded_file is not None:
    draws = load_draws_from_csv(uploaded_file)
    if len(draws) < 2:
        st.error("Geçerli en az 2 çekiliş verisi gereklidir.")
    else:
        X, y = prepare_features(draws)
        model_xgb, acc = train_xgboost(X, y)
        st.success(f"XGBoost doğruluk: {acc:.4f}")

        predicted_numbers, combined_scores = combined_prediction(draws, model_xgb)
        st.subheader("Tahmin Edilen 6 Sayı")
        st.write(sorted(predicted_numbers))
        st.subheader("Olasılık Skorları (İlk 10)")
        top_scores = dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10])
        st.write(top_scores)
