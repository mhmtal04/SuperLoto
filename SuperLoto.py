from bs4 import BeautifulSoup
from collections import Counter

# HTML dosyasını oku
with open("superloto.html", "r", encoding="utf-8") as file:
    html = file.read()

# BeautifulSoup ile ayrıştır
soup = BeautifulSoup(html, "html.parser")

# Çekiliş kartlarını bul (sınıf isimleri HTML’e göre düzenlenebilir)
draw_cards = soup.find_all("div", class_="draw-result-card")

# Geçmiş çekiliş sayıları burada toplanacak
veriler = []

for card in draw_cards:
    # Sayı toplama
    sayilar = [int(span.text.strip()) for span in card.find_all("span", class_="number")]
    
    if len(sayilar) == 6:
        veriler.append(sayilar)

# Eğer veri alınamadıysa uyarı ver
if not veriler:
    print("Veri alınamadı. HTML sınıf adları değişmiş olabilir.")
    exit()

# Tüm sayıları tek listede topla
tum_sayilar = [sayi for cekilis in veriler for sayi in cekilis]

# En sık çıkan 6 sayıyı bul
sayi_sayaci = Counter(tum_sayilar)
tahmin = sayi_sayaci.most_common(6)

# Sonuçları yazdır
print("Toplam Çekiliş Sayısı:", len(veriler))
print("Tahmin Edilen Sayılar (En Çok Çıkan 6 Sayı):")
print([sayi for sayi, adet in tahmin])
