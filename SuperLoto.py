import streamlit as st
from bs4 import BeautifulSoup
import pandas as pd
import random

# HTML'den veri çekme
def cekilis_verilerini_al(dosya_adi):
    with open(dosya_adi, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    tarih_tags = soup.find_all("span", class_="draw-date ng-star-inserted")
    cekilis_kutulari = soup.find_all("div", class_="result-group ng-star-inserted")

    veriler = []
    for tarih, kutu in zip(tarih_tags, cekilis_kutulari):
        sayilar = [int(span.text) for span in kutu.find_all("span", class_="ball ng-star-inserted")]
        if len(sayilar) == 6:
            veriler.append({
                "Tarih": tarih.text.strip(),
                "Sayılar": sayilar
            })
    return veriler

# Tahmin üretme (basit rastgele)
def tahmin_uret():
    return sorted(random.sample(range(1, 61), 6))

# Streamlit Arayüzü
st.title("Süper Loto Tahmin Botu")
st.markdown("Milli Piyango verileriyle çalışır.")

veriler = cekilis_verilerini_al("superloto.html")

if veriler:
    st.subheader("Geçmiş Çekilişler")
    for v in veriler:
        st.write(f"{v['Tarih']}: {', '.join(map(str, v['Sayılar']))}")
else:
    st.error("Geçerli çekiliş verisi bulunamadı.")

st.subheader("Tahmin")
if st.button("Tahmin Üret"):
    tahmin = tahmin_uret()
    st.success("Tahmin Edilen Sayılar: " + ", ".join(map(str, tahmin)))
