import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import random

@st.cache_data(ttl=3600)
def fetch_super_loto_results():
    url = 'https://www.millipiyangoonline.com/super-loto/cekilis-sonuclari' 
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table')
    if not table:
        return None

    headers = [th.text.strip() for th in table.find_all('th')]
    rows = []
    for tr in table.find_all('tr')[1:]:
        cols = [td.text.strip() for td in tr.find_all('td')]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)
    return df

def extract_numbers(df):
    all_numbers = []
    if df is None:
        return all_numbers
    for index, row in df.iterrows():
        # Çekiliş sonuçları genellikle 6 sayı olur
        # Sütun isimleri farklı olabilir ama genellikle numaralar 2. veya 3. sütundan başlar
        # Burada "Çekiliş Sonuçları" sütununda sayılar varsa kullanabiliriz
        # Alternatif olarak sayıları ayıracağız
        for col in df.columns:
            if 'sonuç' in col.lower() or 'numara' in col.lower():
                numbers_str = row[col]
                # Sayıları ayır, boşluk ve - işaretine göre
                numbers = [int(n) for n in numbers_str.replace('-', ' ').split() if n.isdigit()]
                all_numbers.extend(numbers)
                break
    return all_numbers

def main():
    st.title("Milli Piyango Süper Loto Sonuçları ve Tahmin Botu")

    st.markdown("Milli Piyango Süper Loto geçmiş çekiliş sonuçlarını çekip gösterir ve basit analiz yapar.")

    df = fetch_super_loto_results()

    if df is None or df.empty:
        st.error("Çekiliş sonuçları bulunamadı. Lütfen daha sonra tekrar deneyin.")
        return

    st.subheader("Geçmiş Süper Loto Çekiliş Sonuçları")
    st.dataframe(df)

    numbers = extract_numbers(df)
    if not numbers:
        st.warning("Çekiliş sonuçlarından sayı bilgisi çıkarılamadı.")
        return

    counts = Counter(numbers)
    most_common = counts.most_common(10)

    st.subheader("En Çok Çıkan Sayılar")
    most_common_df = pd.DataFrame(most_common, columns=['Sayı', 'Çıkış Sayısı'])
    st.table(most_common_df)

    st.subheader("Tahmin Önerisi")
    # En çok çıkan sayılar arasından rastgele 6 sayı seçelim
    top_numbers = [num for num, count in most_common]
    if len(top_numbers) < 6:
        st.warning("Yeterli sayı bilgisi yok, tahmin yapılamıyor.")
        return

    tahmin = sorted(random.sample(top_numbers, 6))
    st.write("Bu çekiliş için tahmin ettiğimiz sayılar:")
    st.write(tahmin)

if __name__ == "__main__":
    main()
