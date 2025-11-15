# JetX Predictor - Proje Brief

## Proje Amacı

JetX Predictor, JetX çarpma oyunu için geliştirilmiş ileri düzey bir makine öğrenmesi tahmin sistemidir. Temel amaç, oyunun çarpanlarını yüksek doğrulukla tahmin ederek kullanıcıların para kaybı riskini azaltmak ve kâr potansiyelini artırmaktır.

## Temel Hedefler

### Birincil Hedefler
- **Doğruluk Oranı**: Minimum %70 tahmin doğruluğu hedefi
- **Risk Yönetimi**: Para kaybı riskini %30'un altına indirmek
- **Kâr Optimizasyonu**: Sharpe ratio > 1.0 hedefi
- **Gerçek Zamanlı Tahmin**: <1 saniye içinde tahmin üretimi

### İkincil Hedefler
- **Model Sağlamlığı**: Farklı piyasa koşullarında tutarlı performans
- **Kullanıcı Deneyimi**: Anlaşılır ve kullanışlı arayüz
- **Ölçeklenebilirlik**: Çoklu kullanıcı desteği
- **Güvenilirlik**: %99+ sistem uptime

## Proje Kapsamı

### Dahil Olan Özellikler
1. **Multi-Model Ensemble Sistemi**
   - Neural Network (TensorFlow/Keras)
   - CatBoost (Regressor + Classifier)
   - AutoGluon (AutoML)
   - TabNet (Attention-based)
   - LightGBM (CPU optimized)

2. **Multi-Scale Time Window Analysis**
   - 5 farklı pencere boyutu: 500, 250, 100, 50, 20
   - Her pencere için ayrı model eğitimi
   - Weighted ensemble birleştirme

3. **Gelişmiş Risk Yönetimi**
   - 3 farklı bankroll sistemi
   - Kelly Criterion bahis optimizasyonu
   - Stop-loss ve take-profit mekanizmaları
   - Psychological pattern analizi

4. **Comprehensive Feature Engineering**
   - 150+ istatistiksel özellik
   - Volatilite ve streak analizi
   - Wavelet ve Fourier dönüşümleri
   - Psychological pattern tespiti

### Hariç Tutulanlar
- Gerçek para ile bahis işlemleri
- Yasal olmayan veri toplama yöntemleri
- %100 doğruluk garantisi
- Diğer crash oyunları için destek

## Başarı Kriterleri

### Teknik Metrikler
- **Doğruluk**: >%70 genel tahmin doğruluğu
- **ROI**: >%20 yıllık yatırım getirisi
- **Sharpe Ratio**: >1.0 risk-ayarlı getiri
- **Maximum Drawdown**: <%30 maksimum düşüş

### Kullanıcı Metrikleri
- **Yanıt Süresi**: <1 saniye tahmin süresi
- **Sistem Kullanılabilirliği**: >%99 uptime
- **Kullanıcı Memnuniyeti**: >%4.0/5.0 puan

### İş Metrikleri
- **Model Eğitim Süresi**: <24 saat multi-model eğitim
- **Bakım Kolaylığı**: Otomatik model güncelleme
- **Maliyet**: <%100 TL/ay işlem maliyeti

## Kısıtlamalar ve Varsayımlar

### Teknik Kısıtlamalar
- Maksimum 10000 veri noktası işleme
- Minimum 50 geçmiş veri gereksinimi
- GPU bellek limiti: 8GB
- SQLite veritabanı boyut limiti: 1GB

### İş Kısıtlamaları
- Sadece Türkçe dil desteği
- Sadece lokal deployment
- Sadece tek kullanıcı modu
- Haftalık 7 gün destek

### Varsayımlar
- Kullanıcıların temel ML konseptlerini anladığı
- İnternet bağlantısının stabil olduğu
- Donanımın minimum gereksinimleri karşıladığı
- Verinin gerçek ve zamanında girildiği

## Risk Değerlendirmesi

### Teknik Riskler
- **Overfitting**: Modelin eğitim verisine ezberlemesi
- **Data Leakage**: Gelecek bilgisinin geçmişe sızması
- **Model Drift**: Piyasa koşullarının değişmesi
- **Computational Limits**: Kaynak yetersizliği

### İş Riskleri
- **Yasal Riskler**: Bahis yasal düzenlemeleri
- **Piyasa Riskleri**: JetX oyununun kapanması
- **Kullanıcı Riskleri**: Sistemin kötüye kullanılması
- **Rekabet Riskleri**: Benzer sistemlerin ortaya çıkması

### Risk Azaltma Stratejileri
- **Validation**: Katmanlı doğrulama süreçleri
- **Monitoring**: Gerçek zamanlı performans izleme
- **Backup**: Veri ve model yedekleme
- **Documentation**: Kapsamlı dokümantasyon

## Proje Takvimi

### Aşama 1: Temel Sistem (Tamamlandı)
- [x] Temel tahmin motoru
- [x] Streamlit arayüzü
- [x] Veritabanı yönetimi
- [x] Risk yönetimi

### Aşama 2: Gelişmiş Özellikler (Devam Ediyor)
- [x] Multi-model ensemble
- [x] Multi-scale analysis
- [x] Advanced bankroll systems
- [ ] Model seçim optimizasyonu
- [ ] Lazy learning çözümleri

### Aşama 3: Production Hazırlığı (Planlandı)
- [ ] Model monitoring sistemi
- [ ] A/B testing framework
- [ ] Otomatik model güncelleme
- [ ] Çoklu kullanıcı desteği

## Başlangıç Tarihi ve Versiyon

- **Başlangıç**: 2024-Q4
- **Mevcut Versiyon**: v2.0
- **Son Güncelleme**: 2025-01-XX
- **Sonraki Versiyon**: v2.1 (Planlandı)

---

*Bu belge projenin temel hedeflerini, kapsamını ve başarı kriterlerini tanımlar. Tüm geliştirme kararları bu brief'e uygun olmalıdır.*
