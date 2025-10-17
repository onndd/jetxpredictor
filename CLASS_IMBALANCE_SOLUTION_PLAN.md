# Model Performansını Artırma Planı: Dengesiz Veri Sorunu Çözümü

## 1. Mevcut Durum ve Problem

- **Veri Dengesizliği:** JetX veri seti, doğası gereği dengesizdir. Yaklaşık %65'i 1.5x ve üzeri (çoğunluk sınıfı), %35'i ise 1.5x altı (azınlık sınıfı) sonuçlardan oluşmaktadır.
- **Performans Sınırı:** Bu dengesizlik nedeniyle, mevcut modeller "tembel öğrenme" eğilimi göstermekte ve doğruluk oranı ~%64 seviyesini aşamamaktadır. Model, riskli olan azınlık sınıfını (1.5x altı) etkin bir şekilde öğrenememektedir.
- **Mevcut Çözüm:** `auto_class_weights='Balanced'` parametresi kullanılmaktadır ancak bu, problemi çözmek için yetersiz kalmaktadır.

## 2. Hedef

Modelin, özellikle kritik ve azınlıkta olan **1.5x altı** sınıfını doğru tahmin etme yeteneğini önemli ölçüde artırmak ve genel doğruluk oranını %70-%75'in üzerine çıkarmak.

## 3. Önerilen Çözüm Stratejileri (Deneyler)

Bu sorunu çözmek için aşağıdaki deneyleri sırasıyla uygulamayı öneriyorum. Her deneyin sonucu bir sonrakinin stratejisini etkileyebilir.

### Deney 1: Focal Loss Implementasyonu

- **Açıklama:** Projede zaten var olan `utils/focal_loss.py` dosyasındaki mantığı kullanarak `CatBoostClassifier` için özel bir kayıp fonksiyonu (objective) ve değerlendirme metriği (eval_metric) oluşturacağız. Focal Loss, modelin zorlandığı örneklere odaklanmasını sağlayarak azınlık sınıfının öğrenilmesini güçlendirir.
- **Uygulama Adımları:**
    1. `FocalLoss` sınıfını CatBoost'un beklediği formata (değer ve gradyanları döndüren) uygun hale getirmek.
    2. `jetx_CATBOOST_TRAINING.py` betiğinde `loss_function` ve `eval_metric` parametrelerini bu özel sınıfla değiştirmek.
    3. Modeli yeniden eğitip performansı (özellikle 1.5 altı doğruluk ve confusion matrix) standart `Logloss` ile karşılaştırmak.

### Deney 2: Manuel ve Agresif Sınıf Ağırlıklandırması

- **Açıklama:** `auto_class_weights` yerine, `class_weights` parametresini manuel olarak ayarlayarak azınlık sınıfına (class 0) çok daha agresif bir ağırlık vereceğiz. Bu, modelin azınlık sınıfındaki bir hatayı yapmaktan daha fazla "korkmasını" sağlar.
- **Uygulama Adımları:**
    1. `class_weights` parametresini `[çoğunluk_sayısı / azınlık_sayısı]` oranıyla veya daha yüksek (örneğin 20, 30 gibi) değerlerle denemek.
    2. Farklı ağırlıkların modelin 1.5 altı ve 1.5 üstü doğrulukları üzerindeki etkisini analiz etmek.

## 4. Sonraki Adım

Bu planın onaylanmasının ardından, "Code" moduna geçilerek **Deney 1 (Focal Loss Implementasyonu)** ile işe başlanacaktır.