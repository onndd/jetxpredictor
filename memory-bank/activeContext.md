# JetX Predictor - Active Context

## Güncel Çalışma Alanı

Bu belge mevcut çalışma alanını, son değişiklikleri ve bir sonraki adımları belgelemek için kullanılır.

## Son Değişiklikler

### 2025-11-21 - Threshold Management Sistemi

**Ana Değişiklik:**
- "Raporlama vs. Eylem" tutarsızlıklarını kökten çözen merkezi threshold yönetimi sistemi oluşturuldu
- Tüm hardcoded threshold değerleri config'den dinamik olarak alınır hale getirildi

**Yapılan İşlemler:**
1. **Config Yapılandırması:** `config/config.yaml`'a yeni threshold ve loss penalty ayarları eklendi
2. **Threshold Manager:** `utils/threshold_manager.py` oluşturuldu - merkezi yönetim
3. **Callback Güncellemeleri:** `utils/virtual_bankroll_callback.py` config'den threshold alıyor
4. **Loss Fonksiyonları:** `utils/custom_losses.py` gömülü sabitler parametrik hale getirildi
5. **Test Framework:** `tests/test_threshold_consistency.py` tutarlılık testleri oluşturuldu

**Çözülen Sorunlar:**
- VirtualBankrollCallback'de %50 hardcoded threshold → %70 config'den
- CatBoost training'de %65 hardcoded threshold → %70 config'den  
- Loss fonksiyonlarında gömülü sabitler → parametrik config'den
- Tüm threshold'ların tek yerden yönetimi

## Mevcut Durum

### Aktif Development
- **Threshold Management:** Sistemin tamamlandı ve test ediliyor
- **Keskin Nişancı Stratejisi:** Tüm callback'ler %70 threshold kullanıyor
- **Config Integration:** Merkezi yönetim aktif

### Kritik Dosyalar
- `utils/threshold_manager.py` - Merkezi threshold yönetimi
- `config/config.yaml` - Tüm threshold ayarları
- `utils/virtual_bankroll_callback.py` - Eğitim raporları
- `utils/custom_losses.py` - Parametrik loss fonksiyonları
- `tests/test_threshold_consistency.py` - Tutarlılık testleri

## Bir Sonraki Adımlar

### Kısa Vadeli (Bu Oturum)
- [x] Memory Bank güncellemelerini tamamla
- [ ] GitHub'a tüm değişiklikleri push yap
- [ ] Testleri çalıştır ve sonuçları doğrula

### Orta Vadeli (1-2 Hafta)
- [ ] Training script'lerinde AdaptiveWeightScheduler tutarlılığı
- [ ] Production ve training threshold'larının senkronizasyonu
- [ ] Dokümantasyon güncellemeleri

### Uzun Vadeli (1 Ay+)
- [ ] Threshold otomasyonu (auto-tuning)
- [ ] Performance monitoring ve alert sistemi
- [ ] Advanced threshold optimizasyon algoritmaları

## Teknik Kararlar

### Threshold Hiyerarşisi
1. **Production Default (0.80):** En yüksek güven, gerçek para için
2. **Model Checkpoint (0.70):** Model kaydetme için "Keskin Nişancı"
3. **Virtual Bankroll (0.70):** Eğitim raporları için
4. **CatBoost Evaluation (0.70):** Test sonuçları için
5. **Conservative Mode (0.65):** Düşük risk testleri için

### Loss Penalty Stratejisi
- **False Positive (5.0x):** Para kaybı en riskli
- **Critical Zone (4.0x):** 1.4-1.6 arası hassas bölge
- **False Negative (3.0x):** Fırsat kaçırma

## Bilinmeyenler ve Açık Sorular

### Technical Debt
- [ ] Eski training script'lerindeki hardcoded değerler
- [ ] AdaptiveWeightScheduler başlangıç değerlerinin tutarlılığı
- [ ] CatBoost'un internal threshold'ları

### Feature Requests
- [ ] Threshold auto-tuning algoritmaları
- [ ] Performance dashboard
- [ ] Threshold optimization için A/B test framework

## Notlar

### Öğrenilenler
- Merkezi config yönetimi maintenance'ı büyük ölçüde azaltır
- Threshold tutarlılığı model performansını doğrudan etkiler
- Test-driven development kritik sistemlerde zorunludur

### İpuçları
- Her threshold değişikliğinde mutlaka test çalıştır
- Production'dan önce mutlaka validation yap
- Config değişikliklerini versiyonla

---
*Son Güncelleme: 2025-11-21*
*Status: Active Development - Threshold Management*
