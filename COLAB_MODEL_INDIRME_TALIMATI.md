# 📥 COLAB MODELLERINI İNDİRME TALİMATI

## 🎯 Gerekli Dosyalar

### CatBoost Modelleri
```
jetx_models_catboost_v2.0.zip
```

İçindekiler:
- `catboost_regressor.cbm`
- `catboost_classifier.cbm`
- `catboost_scaler.pkl`
- `catboost_model_info.json`

### Progressive NN Modelleri (Opsiyonel)
```
jetx_models_progressive_v2.0.zip
```

İçindekiler:
- `jetx_progressive_transformer.h5`
- `scaler_progressive_transformer.pkl`
- `model_info.json`
- `stage1_best.h5`
- `stage2_best.h5`
- `stage3_best.h5`

---

## 📋 ADIM ADIM İNDİRME

### Colab Notebook'ta

1. **Dosya Panelini Aç**
   - Colab sol tarafta "📁 Files" ikonuna tıklayın

2. **Model Dosyalarını Bul**
   - `jetx_models_catboost_v2.0.zip` dosyasını bulun
   - `jetx_models_progressive_v2.0.zip` dosyasını bulun (opsiyonel)

3. **İndir**
   - Dosyaya sağ tıklayın
   - "Download" seçeneğine tıklayın
   - Dosyalar bilgisayarınıza indirilecek

### VEYA Manuel Olarak Topla

Eğer ZIP dosyaları yoksa, tek tek indirin:

```
CatBoost için:
- catboost_regressor.cbm
- catboost_classifier.cbm
- catboost_scaler.pkl
- catboost_model_info.json

Progressive NN için:
- jetx_progressive_transformer.h5
- scaler_progressive_transformer.pkl
- model_info.json
```

---

## 📂 LOKAL PROJEYE KOPYALAMA

### 1. ZIP Dosyasını Çıkartın
```bash
# İndirilenler klasöründe
unzip jetx_models_catboost_v2.0.zip -d catboost_models
unzip jetx_models_progressive_v2.0.zip -d progressive_models
```

### 2. Dosyaları Projeye Kopyalayın

**CatBoost Modelleri:**
```bash
cd /Users/numanondes/Desktop/jetxpredictor

# CatBoost dosyalarını kopyala
cp ~/Downloads/catboost_models/catboost_regressor.cbm models/
cp ~/Downloads/catboost_models/catboost_classifier.cbm models/
cp ~/Downloads/catboost_models/catboost_scaler.pkl models/
cp ~/Downloads/catboost_models/catboost_model_info.json models/
```

**Progressive NN Modelleri:**
```bash
# NN dosyalarını kopyala
cp ~/Downloads/progressive_models/jetx_progressive_transformer.h5 models/
cp ~/Downloads/progressive_models/scaler_progressive_transformer.pkl models/
cp ~/Downloads/progressive_models/model_info.json models/model_info_nn.json
```

### 3. Dosyaları Doğrulayın
```bash
ls -lh models/
```

Şu dosyaları görmelisiniz:
```
catboost_classifier.cbm
catboost_regressor.cbm
catboost_scaler.pkl
catboost_model_info.json
jetx_progressive_transformer.h5 (opsiyonel)
scaler_progressive_transformer.pkl (opsiyonel)
model_info_nn.json (opsiyonel)
```

---

## ✅ DOĞRULAMA

Dosyalar kopyalandıktan sonra bana haber verin, doğrulama yapacağım:

```bash
# Dosya boyutlarını kontrol et
du -h models/*

# CatBoost modelleri test et
python3 -c "from catboost import CatBoost; model = CatBoost(); model.load_model('models/catboost_regressor.cbm'); print('✅ CatBoost Regressor OK')"
```

---

## 🚨 SORUN GİDERME

### "Dosya bulunamadı" Hatası

**Colab'da:**
```python
import os
print("Mevcut dosyalar:")
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'catboost' in file or 'jetx' in file or '.zip' in file:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath) / (1024*1024)
            print(f"{filepath} ({size:.2f} MB)")
```

### ZIP Dosyası Oluşturmadıysa

**Colab'da tekrar çalıştırın:**
```python
import zipfile
import os

# CatBoost ZIP
with zipfile.ZipFile('jetx_models_catboost_v2.0.zip', 'w') as zipf:
    zipf.write('catboost_regressor.cbm')
    zipf.write('catboost_classifier.cbm')
    zipf.write('catboost_scaler.pkl')
    zipf.write('catboost_model_info.json')

print("✅ ZIP oluşturuldu:", os.path.getsize('jetx_models_catboost_v2.0.zip') / (1024*1024), "MB")
```

---

## 📞 YARDIM

Dosyaları indirdikten ve kopyaladıktan sonra:

```
Bana şunu yazın: "Modeller kopyalandı"
```

Ben de:
1. Dosyaları doğrulayacağım
2. CatBoost entegrasyonunu tamamlayacağım
3. Ensemble predictor'ı yazacağım
4. Dinamik threshold sistemini ekleyeceğim

---

**Hazırlayan:** Roo  
**Tarih:** 2025-10-12