# 🚀 Projeyi GitHub'a Yükleme Talimatları

## Adım 1: GitHub'da Yeni Repository Oluştur

1. [GitHub](https://github.com) sitesine git ve giriş yap
2. Sağ üstteki **"+"** butonuna tıkla ve **"New repository"** seç
3. Repository bilgilerini gir:
   - **Repository name:** `jetxpredictor`
   - **Description:** "AI destekli JetX tahmin sistemi - Para kazandırmak için tasarlandı"
   - **Public** veya **Private** seç (tercihen Public yapabilirsin)
   - ⚠️ **"Add a README file", ".gitignore", "license" seçeneklerini EKLEME** (zaten var)
4. **"Create repository"** butonuna tıkla

## Adım 2: Lokal Repository'yi GitHub'a Bağla

GitHub'da repository oluşturulduktan sonra, gösterilen komutları kullan:

```bash
# GitHub repository'nin URL'ini ekle (kendi kullanıcı adınla değiştir)
git remote add origin https://github.com/KULLANICI_ADINIZ/jetxpredictor.git

# Ana branch'i main olarak ayarla (varsayılan zaten main olmalı)
git branch -M main

# İlk push'u yap
git push -u origin main
```

## Adım 3: Başarıyı Doğrula

Push işlemi tamamlandıktan sonra:
1. GitHub repository sayfasını yenile
2. Tüm dosyaların yüklendiğini kontrol et
3. README.md'nin düzgün görüntülendiğini kontrol et

## 🔄 Gelecekte Değişiklik Yapmak İçin

Projeye yeni değişiklikler eklediğinde:

```bash
# Değişiklikleri ekle
git add .

# Commit yap
git commit -m "Değişiklik açıklaması"

# GitHub'a push et
git push
```

## 📦 Model Dosyalarını Yükleme

Model dosyaları (`.h5`, `.pkl`) .gitignore'da olduğu için otomatik yüklenmez. Bunları yüklemek için iki seçenek:

### Seçenek 1: GitHub Release Kullan (Önerilen)
1. GitHub repository sayfasında **"Releases"** > **"Create a new release"** tıkla
2. Tag version: `v1.0.0`
3. Release title: `İlk Model Release`
4. Model dosyalarını sürükle-bırak ile yükle
5. **"Publish release"** tıkla

### Seçenek 2: Git LFS Kullan (Büyük dosyalar için)
```bash
# Git LFS yükle (macOS)
brew install git-lfs

# Git LFS başlat
git lfs install

# Model dosyalarını track et
git lfs track "*.h5"
git lfs track "*.pkl"

# .gitattributes dosyasını commit et
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push

# Model dosyalarını ekle
git add models/*.h5 models/*.pkl
git commit -m "Add trained models"
git push
```

## ✅ İşlem Tamamlandı!

Projeniz artık GitHub'da! 🎉

Repository URL'niz: `https://github.com/KULLANICI_ADINIZ/jetxpredictor`
