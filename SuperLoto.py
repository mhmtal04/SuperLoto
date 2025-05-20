import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter

@st.cache_data(ttl=3600)
def fetch_super_loto_results():
    url = "https://www.millipiyangoonline.com/super-loto/cekilis-sonuclari"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        st.error(f"Veri çekilemedi, status code: {res.status_code}")
        return None
    
    soup = BeautifulSoup(res.text, "html.parser")
    # Tabloyu bul (ilk çekiliş sonuçları tablosu)
    table = soup.find("table", class_="tbl-cekilis-sonuclari")
    if not table:
        st.error("Sonuç tablosu bulunamadı.")
        return None
    
    # Tablo başlıklarını al
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]

    # Satırları işle
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        rows.append(cells)

    df = pd.DataFrame(rows, columns=headers)
    return df

def analyze_numbers(df):
    # Süper Loto sayıları sütun adlarına göre değişir,
    # Örnek olarak 'Çekiliş' ve 6 sayı sütunu varsayalım
    number_columns = [col for col in df.columns if col.lower().startswith("sayı") or col.lower().startswith("numara")]
    if not number_columns:
        # Eğer sütun isimleri farklıysa elle belirle
        number_columns = df.columns[1:]  # 1. sütun çekiliş no veya tarih olabilir

    all_numbers = []
    for _, row in df.iterrows():
        for col in number_columns:
            try:
                num = int(row[col])
                all_numbers.append(num)
            except:
                continue
    counts = Counter(all_numbers)
    most_common = counts.most_common(10)
    return most_common

def main():
    st.title("Süper Loto Otomatik Veri Çekme ve Tahmin")

    df = fetch_super_loto_results()
    if df is None:
        st.stop()

    st.subheader("Çekiliş Sonuçları")
    st.dataframe(df)

    st.subheader("En Çok Çıkan Sayılar")
    most_common = analyze_numbers(df)
    if most_common:
        result_df = pd.DataFrame(most_common, columns=["Sayı", "Çıkış Sayısı"])
        st.table(result_df)
    else:
        st.info("Sayı analizi için uygun veri bulunamadı.")

if __name__ == "__main__":
    main()
