# GitHub Senkronizasyon Talimatları

## Hızlı Başlangıç

### 1. Tüm Değişiklikleri Ekle

```bash
# Yeni dosyalar ve değişiklikleri ekle
git add .

# Durumu kontrol et
git status
```

### 2. Commit Yap

```bash
git commit -m "feat: Model versiyonlama, A/B testing ve gelişmiş özellikler

✨ Yeni Özellikler:
- Model versiyonlama sistemi (utils/model_versioning.py)
- A/B testing sistemi (utils/ab_testing.py)
- Model loader - Colab → Lokal döngüsü için optimize (utils/model_loader.py)
- Tüm modellerin çıktılarını gösterme özelliği (app.py)
- Gelişmiş model durumu kontrolü

📚 Dokümantasyon:
- WORKFLOW_AND_SYSTEMS.md - İş akışı ve sistemler
- QUICK_START_GUIDE.md - Hızlı başlangıç rehberi
- ARCHITECTURE_AND_TRAINING.md - Mimari ve eğitim detayları

🔧 İyileştirmeler:
- Model seçim mekanizması iyileştirildi
- Sanal kasa sistemi dokümante edildi
- Model yükleme otomatikleştirildi"
```

### 3. GitHub'a Push

```bash
git push origin main
```

## Detaylı Adımlar

### Adım 1: Değişiklikleri İncele

```bash
# Değişen dosyaları görüntüle
git status

# Değişiklikleri önizle
git diff
```

### Adım 2: Dosyaları Ekle

```bash
# Tüm değişiklikleri ekle
git add .

# Veya seçici olarak
git add docs/
git add utils/ab_testing.py
git add utils/model_loader.py
git add utils/model_versioning.py
git add app.py
git add README.md
```

### Adım 3: Commit

```bash
git commit -m "feat: Model versiyonlama, A/B testing ve gelişmiş özellikler"
```

### Adım 4: Push

```bash
# Ana branch'e push
git push origin main

# Veya ilk kez push ediyorsanız
git push -u origin main
```

## Önemli Notlar

### .gitignore Kuralları

Aşağıdaki dosyalar GitHub'a yüklenmez:
- ✅ Model dosyaları (`.h5`, `.cbm`, `.pkl`) - Büyük dosyalar
- ✅ Veritabanı dosyaları (`.db`, `.sqlite`)
- ✅ Cache dosyaları
- ✅ Log dosyaları
- ✅ Virtual environment (`venv/`)

**Yüklenen Dosyalar**:
- ✅ Tüm kaynak kodlar (`.py`)
- ✅ Dokümantasyon (`docs/`)
- ✅ Config dosyaları (`config/`)
- ✅ Model info dosyaları (`model_info.json`)
- ✅ README ve diğer dokümantasyon

### Model Dosyaları

Model dosyaları GitHub'a yüklenmez (çok büyük). Bunun yerine:

1. **Google Colab'da eğitilen modeller** ZIP olarak indirilir
2. **Lokal projeye kopyalanır** (`models/` klasörüne)
3. **GitHub Releases** kullanarak model dosyalarını paylaşabilirsiniz

### Veritabanı

`jetx_data.db` dosyası da GitHub'a yüklenmez. Her kullanıcı:
1. Uygulamayı ilk çalıştırdığında otomatik oluşturulur
2. Veya manuel olarak veri ekler

## Sorun Giderme

### Push Reddedildi

```bash
# Önce remote'daki değişiklikleri çek
git pull origin main

# Conflict varsa çöz
# Sonra tekrar push
git push origin main
```

### Büyük Dosya Hatası

Eğer yanlışlıkla büyük dosya eklediyseniz:

```bash
# Son commit'i geri al
git reset HEAD~1

# .gitignore'u kontrol et
# Dosyayı .gitignore'a ekle
# Tekrar commit yap
```

### Branch Yönetimi

```bash
# Yeni branch oluştur
git checkout -b feature/yeni-ozellik

# Değişiklikleri commit et
git commit -m "feat: Yeni özellik"

# Branch'i push et
git push origin feature/yeni-ozellik

# GitHub'da Pull Request oluştur
```

## Otomatik Senkronizasyon

Gelecekte değişiklikleri senkronize etmek için:

```bash
# 1. Değişiklikleri kontrol et
git status

# 2. Değişiklikleri ekle
git add .

# 3. Commit yap
git commit -m "Update: [açıklama]"

# 4. Push et
git push origin main
```

## GitHub Repository Bilgileri

- **Repository**: https://github.com/onndd/jetxpredictor
- **Branch**: main
- **Remote**: origin

---

**Not**: Bu dosya commit edilmeden önce silinebilir veya `docs/` klasörüne taşınabilir.

