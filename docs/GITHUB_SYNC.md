# GitHub Senkronizasyon Rehberi

## Mevcut Durum

- **Repository**: https://github.com/onndd/jetxpredictor.git
- **Branch**: main
- **Remote**: origin (GitHub)

## Yapılacaklar

### 1. Yeni Dosyaları Ekle

```bash
# Yeni oluşturulan dosyalar
git add docs/
git add utils/ab_testing.py
git add utils/model_loader.py
git add utils/model_versioning.py
```

### 2. Değiştirilen Dosyaları Ekle

```bash
# Güncellenen dosyalar
git add README.md
git add app.py
git add app_cpu_models.py
git add notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py
git add notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py
git add pages/cpu/*.py
```

### 3. Commit

```bash
git commit -m "feat: Model versiyonlama, A/B testing ve gelişmiş model yükleme sistemi

- Model versiyonlama sistemi eklendi (utils/model_versioning.py)
- A/B testing sistemi eklendi (utils/ab_testing.py)
- Model loader eklendi (utils/model_loader.py) - Colab → Lokal döngüsü için optimize
- app.py güncellendi - Tüm modellerin çıktılarını gösterme özelliği
- Sanal kasa sistemi dokümantasyonu eklendi
- Detaylı dokümantasyon dosyaları eklendi (docs/)
- Model seçim mekanizması iyileştirildi"
```

### 4. GitHub'a Push

```bash
git push origin main
```

## .gitignore Kontrolü

Aşağıdaki dosyalar GitHub'a yüklenmez (büyük dosyalar):
- `models/*.h5` (Model dosyaları)
- `models/*.pkl` (Scaler dosyaları)
- `models/*.cbm` (CatBoost modelleri)
- `*.db` (Veritabanı dosyaları)
- `data/cache/` (Cache dosyaları)

## Önemli Notlar

1. **Model Dosyaları**: Model dosyaları `.gitignore`'da olduğu için GitHub'a yüklenmez
   - Model dosyalarını manuel olarak indirip `models/` klasörüne koymanız gerekir
   - Veya GitHub Releases kullanarak model dosyalarını paylaşabilirsiniz

2. **Veritabanı**: `jetx_data.db` dosyası da `.gitignore`'da
   - Veritabanı dosyası GitHub'a yüklenmez
   - Her kullanıcı kendi veritabanını oluşturur

3. **Dokümantasyon**: Tüm dokümantasyon dosyaları (`docs/`) GitHub'a yüklenecek

## Otomatik Senkronizasyon

Gelecekte otomatik senkronizasyon için:

```bash
# Tüm değişiklikleri ekle
git add .

# Commit
git commit -m "Update: [açıklama]"

# Push
git push origin main
```

## Sorun Giderme

### Push Hatası
```bash
# Önce pull yap
git pull origin main

# Sonra push
git push origin main
```

### Conflict Çözümü
```bash
# Conflict'leri çöz
git merge origin/main

# Veya rebase
git rebase origin/main
```

