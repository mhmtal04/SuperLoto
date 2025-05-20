import streamlit as st
from bs4 import BeautifulSoup
import pandas as pd

def parse_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    result_blocks = soup.find_all("div", class_="game-result-card")

    data = []
    for block in result_blocks:
        tarih_tag = block.find("div", class_="game-date")
        sayi_tags = block.find_all("div", class_="lottery-number")

        if tarih_tag and len(sayi_tags) >= 6:
            tarih = tarih_tag.get_text(strip=True)
            sayilar = [int(tag.get_text(strip=True)) for tag in sayi_tags[:6]]
            data.append({"Tarih": tarih, "Sayilar": sayilar})

    return data

def tahmin_uret(cekilisler):
    sayilar = []
    for cekilis in cekilisler:
        sayilar.extend(cekilis["Sayilar"])

    df = pd.Series(sayilar).value_counts().reset_index()
    df.columns = ["Sayi", "Frekans"]
    tahmin = sorted(df.head(6)["Sayi"].tolist())
    return tahmin, df

def main():
    st.title("Süper Loto Tahmin ve Sonuç Analizi")

    html_dosyasi = "superloto.html"
    cekilisler = parse_html(html_dosyasi)

    if not cekilisler:
        st.error("Geçerli çekiliş verisi bulunamadı.")
        return

    st.subheader("Son 8 Süper Loto Çekilişi")
    for cekilis in cekilisler:
        st.write(f"{cekilis['Tarih']}: {', '.join(map(str, cekilis['Sayilar']))}")

    tahmin, frekans_df = tahmin_uret(cekilisler)

    st.subheader("Tahmin Edilen Sayılar (En Sık Çıkan 6)")
    st.success(", ".join(map(str, tahmin)))

    st.subheader("Sayı Frekansları")
    st.dataframe(frekans_df.sort_values("Frekans", ascending=False))

if __name__ == "__main__":
    main()
