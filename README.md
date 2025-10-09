# 🚀 JetX Predictor

AI destekli JetX tahmin sistemi - Para kazandırmak için tasarlandı!

## 📋 Proje Hakkında

Bu proje, JetX çarpan oyununda bir sonraki değeri tahmin etmeye çalışan yapay zeka destekli deneysel bir sistemdir. 7000+ geçmiş oyun verisindeki pattern'leri analiz ederek gelecek tahmininde bulunur.

### 🎯 Kritik Bilgi

**1.5x eşik değerdir!** 
- 1.5x altı = Kayıp 💰❌
- 1.5x üstü = Kazanç 💰✅
- Bu 0.01'lik fark kritiktir!

## 🏗️ Proje Yapısı

```
jetxpredictor/
├── app.py                      # Ana Streamlit uygulaması
├── category_definitions.py     # Kategori tanımları (ortak)
├── requirements.txt            # Python paketleri
├── config/
│   └── config.yaml            # Konfigürasyon
├── data/
│   └── jetx_data.db           # SQLite veritabanı
├── models/                     # Eğitilmiş modeller (Colab'dan)
│   ├── jetx_model.h5          # Ana model
│   └── scaler.pkl             # Scaler
├── notebooks/                  # Google Colab notebooks
│   └── jetx_training.ipynb    # Model eğitim notebook'u
├── pages/                      # Streamlit sayfaları
│   └── 1_📊_Analiz.py        # Veri analiz sayfası
└── utils/                      # Yardımcı modüller
    ├── database.py            # Veritabanı yönetimi
    ├── predictor.py           # Tahmin motoru
    └── risk_manager.py        # Risk yönetimi
```

## 🚀 Hızlı Başlangıç

### 1️⃣ Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2️⃣ Model Eğitimi (Google Colab + GitHub)

1. Bu repository'yi fork edin veya klonlayın
2. `notebooks/jetx_model_training.ipynb` dosyasını Google Colab'da açın
3. Colab'da GitHub repository'sini klonlayın:
   ```python
   !git clone https://github.com/onndd/jetxpredictor.git
   %cd jetxpredictor
   ```
4. Notebook'u çalıştırarak modeli eğitin
5. Eğitilmiş modelleri (`jetx_model.h5` ve `scaler.pkl`) GitHub Release olarak yükleyin veya direkt `models/` klasörüne commit edin

### 3️⃣ Uygulamayı Çalıştırma (Lokal)

```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde açılacak.

## 🎮 Kullanım

### Tahmin Modları

1. **🛡️ Rolling Mod (Konservatif)**
   - %80+ güven seviyesi gerekir
   - Sermaye koruma odaklı
   - En güvenli mod
   - Önerilen çıkış: 1.5x

2. **🎯 Normal Mod (Dengeli)**
   - %65+ güven seviyesi gerekir
   - Dengeli risk/getiri
   - Standart kullanım için ideal

3. **⚡ Agresif Mod (Riskli)**
   - %50+ güven seviyesi gerekir
   - Yüksek risk, yüksek getiri
   - Sadece deneyimli kullanıcılar için

### Ana Özellikler

- ✅ Gerçek zamanlı tahmin
- ✅ 1.5x eşik analizi (kritik!)
- ✅ Güven skoru göstergesi
- ✅ Risk seviyesi uyarıları
- ✅ Manuel veri girişi
- ✅ Detaylı veri analizi
- ✅ Görsel grafikler

## 🔧 Teknik Detaylar

### Model Mimarisi

- **Tahmin Motoru:** Deep Learning (LSTM/GRU veya XGBoost)
- **Özellik Sayısı:** 30+ özellik
- **Pencere Boyutu:** 50-500 geçmiş oyun
- **Hedef:** 1.5x eşik tahmini (%75+ doğruluk)

### Özellik Mühendisliği

- Hareketli ortalamalar (5, 10, 20, 50 pencere)
- 1.5x eşik özellikleri
- Büyük çarpan mesafeleri (10x, 20x, 50x, 100x)
- Ardışık pattern'ler
- Volatilite metrikleri

### Kategori Sistemi

3 ana kategori:
- 🔴 Kayıp Bölgesi (< 1.5x)
- 🟢 Güvenli Bölge (1.5x - 3.0x)
- 🔵 Yüksek Çarpan (> 3.0x)

## 📊 Veritabanı

SQLite veritabanı 2 tablo içerir:

1. **jetx_results**
   - Geçmiş oyun sonuçları
   - 6000+ kayıt

2. **predictions**
   - Tahmin geçmişi
   - Performans metrikleri

## ⚠️ ÖNEMLİ UYARILAR

- 🚨 **Bu sistem %100 doğru DEĞİLDİR**
- 💰 **Para kaybedebilirsiniz**
- 🎯 **1.5x kritik eşiktir**
- 🛡️ **Rolling modu tercih edin**
- 📊 **Düşük güvende oynamayın**
- ⚡ **Ardışık kayıplara dikkat edin**

### Risk Yönetimi Kuralları

1. **Rolling modda %80 güven altında OYNAMA**
2. **Normal modda %65 güven altında OYNAMA**
3. **3 ardışık yanlış tahminden sonra DUR**
4. **1.45-1.55 kritik bölgesinde KEsinLİKLE OYNAMA**
5. **Sermayenin maksimum %5'ini riske AT**

## 🔄 Geliştirme Süreci

### Google Colab'da (Model Eğitimi)

```python
# 1. Veriyi yükle
data = load_data_from_sqlite()

# 2. Özellikleri çıkar
features = extract_features(data)

# 3. Modeli eğit
model = train_model(features)

# 4. Değerlendir
accuracy = evaluate_model(model, test_data)

# 5. Kaydet
model.save('jetx_model.h5')
```

### Lokalde (Tahmin)

```python
# 1. Modeli yükle
predictor = JetXPredictor()

# 2. Geçmiş verileri al
history = db.get_recent_results(500)

# 3. Tahmin yap
prediction = predictor.predict(history, mode='rolling')

# 4. Karar ver
if prediction['confidence'] > 0.80 and prediction['above_threshold']:
    print("OYNA!")
else:
    print("BEKLE!")
```

## 📈 Performans Hedefleri

- ✅ 1.5x eşik doğruluğu: **%75+**
- ✅ Tahmin hızı: **<1 saniye**
- ✅ Rolling mod doğruluğu: **%85+**
- ✅ Ardışık yanlış maksimum: **5**

## 🛠️ Bağımlılıklar

### Ana Paketler

- `streamlit` - Web arayüzü
- `tensorflow` veya `torch` - Model
- `pandas`, `numpy` - Veri işleme
- `plotly` - Görselleştirme
- `scikit-learn` - ML araçları

## 📝 TODO

- [ ] Model eğitimini tamamla (Colab)
- [ ] Eğitilmiş modeli lokale aktar
- [ ] Gerçek verilerle test et
- [ ] Performans optimizasyonu
- [ ] Otomatik veri çekme (gelecek)
- [ ] API entegrasyonu (gelecek)

## 🤝 Katkıda Bulunma

Bu deneysel bir projedir. Geliştirmeler için:

1. Model doğruluğunu artırma
2. Yeni özellik ekleme
3. Risk yönetimi iyileştirmeleri
4. UI/UX geliştirmeleri

## 📜 Lisans

Bu proje eğitim ve araştırma amaçlıdır.

## ⚖️ Feragatname

**BU YAZILIM "OLDUĞU GİBİ" SUNULMUŞTUR. HİÇBİR GARANTİ VERİLMEZ.**

- Kumar bağımlılığı ciddi bir sorundur
- Sorumlu oynamak sizin sorumluluğunuzdur
- Kaybetmeyi göze alamayacağınız parayla OYNAMAYIN
- Bu sistem akademik/eğitim amaçlıdır

## 📞 Destek

Sorularınız için:
- Kod inceleyin: Tüm kod açık ve anlaşılır
- Dokümantasyonu okuyun: `PROJE_PLANI.md`
- Test edin: Önce küçük miktarlarla

---

**Başarılar! 🚀 Ama dikkatli olun! ⚠️**

*Son Güncelleme: 08.10.2025*
