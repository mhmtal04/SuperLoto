import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st
import datetime

st.set_page_config(page_title="Hisse Tahmin Uygulaması", layout="centered")
st.title("📈 Hisse Yüzde Değişim Tahmini")

symbol = st.text_input("Hisse kodunu girin (örnek: THYAO)", "")

# Tarih aralığı seçimi
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Başlangıç tarihi", datetime.date.today() - datetime.timedelta(days=180))
with col2:
    end_date = st.date_input("Bitiş tarihi", datetime.date.today())

if symbol:
    symbol = symbol.upper() + ".IS"
    st.write(f"**{symbol}** verisi indiriliyor...")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.warning("Veri indirilemedi. Lütfen geçerli bir hisse kodu veya tarih aralığı girin.")
    else:
        # Anlık fiyatı göster
        current_price = data["Close"].iloc[-1]
        st.info(f"Anlık Fiyat: {current_price:.2f} TL")

        # Kapanış fiyatı grafiği
        st.line_chart(data["Close"], use_container_width=True)

        # Özellikleri hazırla
        data["Return"] = data["Close"].pct_change()
        data["Target"] = data["Return"].shift(-1) * 100  # yüzdesel değişim
        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
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
            st.success(f"Model Ortalama Hata: ±{mae:.2f}%")

            latest_data = X.tail(1)
            prediction = model.predict(latest_data)[0]

            st.subheader("Tahmin Sonucu:")
            if prediction > 0:
                st.write(f"Hisse yarın yaklaşık **%+{prediction:.2f}** artış gösterebilir.")
            else:
                st.write(f"Hisse yarın yaklaşık **{prediction:.2f}%** düşüş gösterebilir.")