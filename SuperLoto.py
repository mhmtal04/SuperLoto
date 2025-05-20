import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime
import os

st.set_page_config(page_title="Hisse Tahmin Uygulaması", layout="centered")
st.title("📈 Yarınki Fiyat ve Yüzde Tahmini (BIST Destekli)")

symbol_input = st.text_input("Hisse kodunu girin (örnek: THYAO veya ALTINS1)", "")
symbol_raw = symbol_input.strip().upper()

# Tarih aralığı seçimi
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Başlangıç tarihi", datetime.date.today() - datetime.timedelta(days=180))
with col2:
    end_date = st.date_input("Bitiş tarihi", datetime.date.today() + datetime.timedelta(days=1))

if symbol_raw:
    symbol = symbol_raw + ".IS"
    st.write(f"Veri yükleniyor: **{symbol_raw}**")

    # 1. Öncelikle Yahoo Finance'ten veri çekmeye çalış
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("Yahoo verisi boş")

        ticker = yf.Ticker(symbol)
        current_price = float(ticker.info.get("currentPrice", data["Close"].dropna().iloc[-1]))
        st.info(f"Gerçek Zamanlı Fiyat (Yahoo): {current_price:.2f} TL")

    except:
        st.warning("Yahoo Finance verisi bulunamadı. Yerel CSV dosyasına geçiliyor...")

        # 2. Yerel CSV dosyasından veriyi oku
        csv_path = "bist_ornek.csv"
        if not os.path.exists(csv_path):
            st.error("Yerel CSV dosyası bulunamadı: bist_ornek.csv")
            st.stop()

        try:
            df = pd.read_csv(csv_path, encoding="ISO-8859-9", sep=";")
            df_filtered = df[df["MENKUL KIYMET"] == symbol_raw]
            if df_filtered.empty:
                st.error(f"{symbol_raw} için CSV'de veri bulunamadı.")
                st.stop()

            # Basit dataframe oluştur (örnek kolonlara göre)
            df_filtered["Close"] = df_filtered["KAPANIŞ"].str.replace(",", ".").astype(float)
            df_filtered["TARIH"] = pd.to_datetime(df_filtered["TARIH"])
            df_filtered = df_filtered.sort_values("TARIH")
            df_filtered.set_index("TARIH", inplace=True)

            current_price = df_filtered["Close"].iloc[-1]
            st.info(f"Kapanış Fiyatı (CSV): {current_price:.2f} TL")

            data = df_filtered[["Close"]].copy()

        except Exception as e:
            st.error(f"CSV verisi okunamadı: {e}")
            st.stop()

    # Grafik çiz
    st.line_chart(data["Close"], use_container_width=True)

    # Özellikleri hazırla
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["Target"] = data["Close"].shift(-1)  # Yarınki kapanış tahmini için
    data = data.dropna()

    if data.shape[0] < 20:
        st.warning("Yeterli veri yok. Daha uzun zaman dilimi seçin.")
    else:
        features = ["Close", "MA5", "MA10"]
        X = data[features]
        y = data["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        st.success(f"Model Ortalama Hata: ±{mae:.2f} TL")

        latest_data = X.tail(1)
        prediction_raw = model.predict(latest_data)[0]

        # BIST limitleri: %10 yukarı/aşağı sınır
        upper_limit = current_price * 1.10
        lower_limit = current_price * 0.90
        predicted_price = max(min(prediction_raw, upper_limit), lower_limit)

        percent_change = ((predicted_price - current_price) / current_price) * 100
        percent_change = max(min(percent_change, 10), -10)

        st.subheader("Tahmin Sonucu:")
        st.write(f"Yarınki tahmini kapanış fiyatı: **{predicted_price:.2f} TL**")
        if abs(percent_change) >= 9.9:
            st.warning(f"Tahmin %10 BIST sınırına ulaştı.")
        st.write(f"Beklenen değişim: **{percent_change:+.2f}%**")