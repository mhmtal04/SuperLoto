import streamlit as st
from bs4 import BeautifulSoup
from collections import Counter

st.set_page_config(page_title="Süper Loto Tahmin Botu", layout="centered")
st.title("Süper Loto Tahmin Botu")

try:
    with open("superloto.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Çekiliş kartlarını bulma (çoklu class kontrolü)
    draw_cards = soup.find_all("div", class_=lambda x: x and "draw-result-card" in x.split())
    st.write(f"Bulunan çekiliş kartı sayısı: {len(draw_cards)}")

    veriler = []

    for i, card in enumerate(draw_cards):
        spans = card.find_all("span", class_=lambda x: x and "number" in x.split())
        sayilar = []
        for span in spans:
            try:
                sayi = int(span.text.strip())
                sayilar.append(sayi)
            except ValueError:
                continue

        st.write(f"Çekiliş {i+1} sayıları: {sayilar}")

        if len(sayilar) == 6:
            veriler.append(sayilar)

    if not veriler:
        st.error("Geçerli çekiliş verisi bulunamadı.")
    else:
        tum_sayilar = [sayi for cekilis in veriler for sayi in cekilis]
        sayi_sayaci = Counter(tum_sayilar)
        tahmin = sayi_sayaci.most_common(6)

        st.success(f"Toplam {len(veriler)} çekiliş analiz edildi.")
        st.subheader("Tahmin Edilen Sayılar (En Çok Çıkan 6 Sayı):")
        st.write([sayi for sayi, _ in tahmin])

        with st.expander("Tüm Sayıların Çıkma Sıklığı"):
            st.dataframe(dict(sayi_sayaci.most_common()))

except FileNotFoundError:
    st.error("HTML dosyası bulunamadı. 'superloto.html' dosyasını ekleyin.")
except Exception as e:
    st.error(f"Bir hata oluştu: {str(e)}")
