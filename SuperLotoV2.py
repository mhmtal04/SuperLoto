import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import random

st.title("Süper Loto Matematiksel Tahmin Botu")

@st.cache_data
def load_data_from_github(url):
    df = pd.read_csv(url)
    # Sayıları listeye çevir (boşluk veya virgül ayracına göre)
    if 'numbers' in df.columns:
        if isinstance(df['numbers'].iloc[0], str):
            if ',' in df['numbers'].iloc[0]:
                df['numbers'] = df['numbers'].apply(lambda x: list(map(int, x.split(','))))
            else:
                df['numbers'] = df['numbers'].apply(lambda x: list(map(int, x.split())))
    return df

def calculate_weighted_frequencies(draws):
    N = len(draws)
    weights = np.array([1/(N - i) for i in range(N)])
    weights /= weights.sum()
    
    freq = {num:0 for num in range(1,61)}

    for i, row in enumerate(draws.itertuples()):
        nums = row.numbers
        for n in nums:
            freq[n] += weights[i]
    return freq

def calculate_pair_frequencies(draws):
    pair_freq = {}
    for row in draws.itertuples():
        nums = row.numbers
        for a,b in combinations(sorted(nums),2):
            pair_freq[(a,b)] = pair_freq.get((a,b),0) + 1
    return pair_freq

def conditional_probabilities(freq, pair_freq):
    cond_probs = {}
    for (a,b), val in pair_freq.items():
        cond_probs[(a,b)] = val / freq[a] if freq[a] > 0 else 0
        cond_probs[(b,a)] = val / freq[b] if freq[b] > 0 else 0
    return cond_probs

def check_constraints(numbers):
    evens = sum(1 for n in numbers if n%2==0)
    avg = sum(numbers)/len(numbers)
    cats = ['low' if n<=20 else 'mid' if n<=40 else 'high' for n in numbers]
    return (
        2 <= evens <= 4 and
        25 <= avg <= 35 and
        cats.count('low') == 2 and
        cats.count('mid') == 2 and
        cats.count('high') == 2
    )

def calculate_set_probability(numbers, freq, cond_probs):
    prob = 1.0
    for n in numbers:
        prob *= freq.get(n, 0.0001)
    for a,b in combinations(numbers, 2):
        prob *= cond_probs.get((a,b), 0.0001)
    return prob

def monte_carlo_sampling(freq, cond_probs, trials=10000):
    best_sets = []
    for _ in range(trials):
        candidate = sorted(random.sample(range(1,61), 6))
        if not check_constraints(candidate):
            continue
        p = calculate_set_probability(candidate, freq, cond_probs)
        best_sets.append((candidate, p))
    best_sets.sort(key=lambda x: x[1], reverse=True)
    return best_sets[:5]

# ----------------------- Kullanıcıdan Github URL alma -----------------------

csv_url = st.text_input("GitHub CSV dosyasının raw URL'sini girin:", "")

if csv_url:
    try:
        data = load_data_from_github(csv_url)
        st.success(f"{len(data)} çekiliş yüklendi.")
        
        freq = calculate_weighted_frequencies(data)
        pair_freq = calculate_pair_frequencies(data)
        cond_probs = conditional_probabilities(freq, pair_freq)
        
        if st.button("Tahmin Üret"):
            with st.spinner("Tahminler hesaplanıyor..."):
                predictions = monte_carlo_sampling(freq, cond_probs, trials=15000)
            st.subheader("En yüksek olasılıklı 5 tahmin:")
            for i, (numbers, prob) in enumerate(predictions, 1):
                st.write(f"Tahmin {i}: {', '.join(map(str, numbers))} (Olasılık: {prob:.6e})")
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
else:
    st.info("Lütfen geçerli bir GitHub raw CSV URL'si girin.")
