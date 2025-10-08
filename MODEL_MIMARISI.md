# JetX Predictor - Model Mimarisi ve Çalışma Mantığı

Bu dokümantasyon, JetX Predictor projesinde kullanılan hibrit N-BEATS + TCN modelinin detaylı mimari açıklamasını içerir.

## 📋 İçindekiler

1. [Veri Hazırlık Katmanı](#1-veri-hazirlik-katmani)
2. [N-BEATS Modülü](#2-n-beats-modülü)
3. [TCN Modülü](#3-tcn-modülü)
4. [Attention Mekanizması](#4-attention-mekanizmasi)
5. [Psikolojik Analiz Motoru](#5-psikolojik-analiz-motoru)
6. [Ensemble Fusion Katmanı](#6-ensemble-fusion-katmani)
7. [Training Loop](#7-training-loop-mantigi)
8. [Inference Aşaması](#8-inference-tahmin-aşamasi)
9. [Model Optimizasyon](#9-model-optimizasyon-teknikleri)
10. [Rolling Mod Adaptasyonu](#10-rolling-mod-için-özel-adaptasyon)

---

## 1. VERİ HAZIRLIK KATMANI

### Input Pipeline Yapısı

Veri hazırlık aşaması üç paralel akıştan oluşur:

#### 1.1 Ham Veri Akışı

- SQLite'tan çekilen 7000 değer bir array'e yüklenir
- Her tahmin için son 1000 değer "sliding window" mantığıyla alınır
- Örneğin 3500. değeri tahmin ederken, 2500-3499 arası değerler input olur

#### 1.2 Kategori Dönüşüm Akışı

- Her değer 15 farklı kategori setinde encode edilir
- 1.67x değeri → Set1'de kategori-4, Set2'de kategori-3, Set3'de kategori-2... şeklinde
- Bu 15 farklı encoding paralel olarak saklanır
- Her değer için 15 boyutlu bir kategori vektörü oluşur

#### 1.3 Özellik Türetme Akışı

- Her pozisyon için dinamik özellikler hesaplanır
- "Son 10x'ten beri kaç el geçti" gibi özellikler her tahmin anında güncellenir
- Pencere içindeki istatistikler (değer dağılımı, streak'ler) hesaplanır

---

## 2. N-BEATS MODÜLÜ DETAYI

### Üç Farklı N-BEATS Bloğu

#### 2.1 Kısa Pencere Bloğu (50 el)

- **Giriş:** Son 50 değer + bu değerlerin 15 kategori encoding'i
- **İlk katman:** Basis expansion (temel fonksiyonlara ayırma)
- **İşlem:** Trend ve sezonellik ayrıştırması yapılır
- **Çıkış:** 64 boyutlu özellik vektörü + gelecek tahmini

#### 2.2 Orta Pencere Bloğu (200 el)

- **Giriş:** Son 200 değer + kategori encoding'leri
- **Mimari:** Daha derin stack (4-5 katman)
- **İşlem:** Hem backward (geçmişi açıklama) hem forward (gelecek tahmini) çıkışı
- **Çıkış:** 128 boyutlu özellik vektörü + tahmin

#### 2.3 Uzun Pencere Bloğu (500 el)

- **Giriş:** Son 500 değer + özellikler
- **Mimari:** En derin mimari (6-7 katman)
- **İşlem:** Uzun vadeli pattern'leri ve döngüleri yakalar
- **Çıkış:** 256 boyutlu özellik vektörü + tahmin

### N-BEATS Blokların Birleşimi

Her blok hem kendi tahminini hem de öğrendiği özellikleri verir. Birleşim:

- **Tahminler:** Ağırlıklı ortalama ile
  - `(0.5 × kısa + 0.3 × orta + 0.2 × uzun)`
- **Özellik vektörleri:** Concatenate edilir
  - `[64 + 128 + 256] = 448 boyutlu vektör`

---

## 3. TCN (TEMPORAL CONVOLUTIONAL NETWORK) MODÜLÜ

### TCN'in Katmanlı Yapısı

#### Dilated Convolution Katmanları

```
1. Katman: Dilation=1,  son 2 değere bakar
2. Katman: Dilation=2,  4 değer aralığına bakar
3. Katman: Dilation=4,  8 değer aralığına bakar
...
N. Katman: Dilation=32, 64 değer aralığına bakar
```

Bu sayede toplam 1000 değerlik pencereyi verimli şekilde tarar.

#### Residual Bağlantılar

Her katmanda input direkt olarak output'a eklenir (skip connection). Bu:
- Gradient vanishing problemini önler
- Derin ağ eğitimini kolaylaştırır

#### Masking Mekanizması

- TCN sadece geçmiş değerlere bakar (causal convolution)
- Gelecek değerleri görmez, bu da overfitting'i önler

### TCN'in Çıktısı

TCN, her zaman adımı için bir hidden state üretir. Son hidden state, tüm 1000 değerlik bilgiyi özetleyen **512 boyutlu** bir vektördür.

---

## 4. ATTENTION MEKANİZMASI

Model içinde implicit (örtük) attention mekanizmaları bulunur:

### N-BEATS İçindeki Attention

- Hangi geçmiş değerlerin daha önemli olduğunu otomatik öğrenir
- Basis fonksiyonları aslında bir tür attention görevi görür

### TCN İçindeki Attention

- Dilated convolution'lar farklı mesafelere farklı ağırlıklar verir
- Önemli pattern'ler daha yüksek aktivasyon üretir

---

## 5. PSİKOLOJİK ANALİZ MOTORU

### Pattern Detection Modülleri

#### 5.1 Trap Detector (Tuzak Tespit Edici)

- Son 10 eldeki değer dizilimini kontrol eder
- Bilinen tuzak pattern'lerini arar:
  - 2-3 orta kazanç → büyük kayıp
  - Recovery trap (toparlanma tuzağı)
  - False momentum (sahte ivme)
- Sinir ağı yerine kural tabanlı veya küçük LSTM
- **Çıktı:** Tuzak olasılığı (0-1 arası)

#### 5.2 Soğuma Dönemi Detector

- Model kendisi öğrenir, sabit kural yok
- Son büyük çarpandan beri geçen süreyi input alır
- O süre içindeki değer dağılımına bakar
- Soğuma bitişi sinyallerini arar (volatilite artışı gibi)
- **Çıktı:** Soğuma durumu skoru

#### 5.3 Momentum Analyzer

- Ardışık değer değişimlerini analiz eder
- Hızlanma/yavaşlama pattern'lerini tespit eder
- **Çıktı:** Momentum vektörü (yön + güç)

---

## 6. ENSEMBLE FUSION KATMANI

### Üç Ana Bileşenin Birleşimi

#### 6.1 N-BEATS + TCN Füzyonu (%60 ağırlık)

```
N-BEATS → 448 boyutlu vektör
TCN     → 512 boyutlu vektör
        ↓ concatenate
        960 boyut
        ↓ 2 fully connected katman
        256 boyut
```

#### 6.2 Psikolojik Motor (%30 ağırlık)

```
Tuzak + Soğuma + Momentum skorları
        ↓ encoding
        32 boyutlu vektör
        ↓ neural network
        Psikolojik özellikler
```

#### 6.3 İstatistiksel Baseline (%10 ağırlık)

```
Moving average + Volatilite + Histogram + Markov chain
        ↓ encoding
        16 boyutlu vektör
```

### Final Tahmin Katmanı

Üç bileşen birleşir: `256 + 32 + 16 = 304 boyut`

#### Çıktılar:

**1. 15 Kategori Tahmini:**
- Her kategori seti için ayrı softmax çıktısı
- Örnek: Set1 için `[0.1, 0.3, 0.4, 0.2...]` olasılık dağılımı
- Toplam: `15 set × ~15 kategori = 225 çıktı nöronu`

**2. Regresyon Tahmini:**
- Direkt değer tahmini (örn: 1.67x)
- Min-max aralık tahmini (örn: 1.5x-2000.0x)

**3. Güven Skoru:**
- Tahmin güvenilirliği (0-100%)
- Modelin kendi tahmininden ne kadar emin olduğu

---

## 7. TRAINING LOOP MANTIĞI

### Loss Fonksiyonu Tasarımı

**Multi-Task Learning Loss:**

```
Total Loss = α₁ × Kategori Loss + α₂ × Regresyon Loss + α₃ × 1.5x Eşik Loss
```

#### 7.1 Kategori Loss

- 15 kategori setinin her biri için cross-entropy
- Ağırlıklı ortalama alınır

#### 7.2 Regresyon Loss

- Mean Absolute Error (MAE) veya Huber Loss
- Büyük hatalara daha az duyarlı

#### 7.3 1.5x Eşik Loss (Özel)

- 1.5 altı/üstü binary classification loss
- Yanlış taraf tahminine ekstra ceza
- Özellikle 1.45-1.55 aralığında hassas

### Training Stratejisi

#### Curriculum Learning

1. İlk epochlarda basit pattern'ler (1.5 altı/üstü)
2. Sonra kategori tahmini
3. En son exact değer tahmini

#### Data Augmentation

- Pencereyi kaydırma (sliding window)
- Gürültü ekleme (robustness için)
- Synthetic pattern üretimi (bilinen pattern'leri varyasyonlarla)

---

## 8. INFERENCE (TAHMİN) AŞAMASI

### Gerçek Zamanlı Tahmin Akışı

```
1. Veri Hazırlık (5ms)
   - Son 1000 değer alınır
   - 15 kategori encoding'i paralel yapılır
   - Özellikler hesaplanır

2. Model Forward Pass (50ms)
   - N-BEATS paralel çalışır (3 pencere)
   - TCN sequential işler
   - Psikolojik motor analiz yapar

3. Ensemble Fusion (10ms)
   - Üç bileşen birleştirilir
   - Final tahmin hesaplanır

4. Post-Processing (5ms)
   - Güven skoru kalibrasyonu
   - Risk uyarıları eklenir
   - Sonuç formatlanır

Toplam: <100ms tahmin süresi
```

---

## 9. MODEL OPTİMİZASYON TEKNİKLERİ

### Quantization (Modeli Küçültme)

- Float32 → Int8 dönüşümü
- Model boyutu 4'te 1'e düşer
- Hız 2-3x artar

### Pruning (Gereksiz Bağlantıları Kesme)

- Düşük ağırlıklı connection'lar silinir
- Model %50-70 küçülür
- Accuracy kaybı minimal

### Knowledge Distillation

- Büyük model (teacher) küçük modele (student) öğretir
- Student model 10x daha küçük olabilir
- Performance kaybı %5'ten az

---

## 10. ROLLING MOD İÇİN ÖZEL ADAPTASYON

Rolling modda model davranışı şöyle değişir:

### Güven Eşiği Yükseltme

```
Normal mod:  %65+ güvende tahmin göster
Rolling mod: %80+ güvende tahmin göster
```

### Conservative Bias Ekleme

- Tahmin aralığını daralt
- Riskli kategorilere düşük olasılık ver
- 1.5x altı olasılığını %10 artır (güvenlik için)

### Sermaye Koruma Filtresi

- Ardışık 3 kazançtan sonra "DUR" sinyali
- Soğuma döneminde kesinlikle oynama
- Volatilite yüksekse pas geç

---

## 📊 Model Performans Hedefleri

| Metrik | Hedef | Açıklama |
|--------|-------|----------|
| 1.5x Eşik Doğruluğu | %75+ | En kritik metrik |
| Kategori Doğruluğu | %60+ | Genel pattern tanıma |
| Tahmin Hızı | <100ms | Gerçek zamanlı kullanım |
| Rolling Mod Doğruluğu | %85+ | Yüksek güven gerektiren mod |
| Ardışık Yanlış Max | 5 | Risk yönetimi için |

---

## 🔧 Teknik Stack

- **Deep Learning Framework:** TensorFlow/Keras veya PyTorch
- **N-BEATS:** Custom implementation
- **TCN:** Temporal Convolutional Network
- **Optimization:** Quantization, Pruning, Knowledge Distillation
- **Inference:** CPU optimized (lokal kullanım için)

---

## 📚 Referanslar

- N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
- TCN: Temporal Convolutional Networks for Sequence Modeling
- Ensemble Learning: Combining multiple models for better predictions

---

**Son Güncelleme:** 08.10.2025

Bu dokümantasyon, JetX Predictor projesinin model mimarisini açıklar. Uygulama detayları için `notebooks/jetx_model_training.ipynb` dosyasına bakın.
