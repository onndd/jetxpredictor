# CatBoost Eğitiminde GPU Kullanımını Aktifleştirme Planı

## 1. Sorunun Tespiti

Mevcut `notebooks/jetx_CATBOOST_TRAINING.py` betiği, Google Colab'da A100 gibi güçlü bir GPU üzerinde çalışmasına rağmen model eğitimini CPU üzerinde yapmaktadır. Bu durum, eğitim sürelerini ciddi şekilde uzatmaktadır.

Yaptığımız analiz sonucunda sorunun kaynağının, `utils/virtual_bankroll_callback.py` dosyasından gelen `CatBoostBankrollCallback` adlı özel bir callback fonksiyonu olduğu tespit edilmiştir.

- **Neden:** Bu callback, CatBoost'un eğitim döngüsüyle uyumlu değildir ve kod içindeki yorumlara göre hem CPU'da hem de GPU'da hatalara neden olmaktadır (`# callbacks kaldırıldı - CPU'da bile hata veriyordu`).
- **Sonuç:** Geliştirici, bu sorunu aşmak için callback'i kullanımdan kaldırmış ve bir önlem olarak eğitimi `task_type='CPU'` olarak sabitlemiştir.

## 2. Çözüm Stratejisi

İyi haber şu ki, betik zaten eğitimin sonunda çok daha kapsamlı bir sanal kasa simülasyonu yapmaktadır. Bu nedenle, hatalı olan ve artık kullanılmayan callback fonksiyonuna **ihtiyaç yoktur**.

Çözüm, iki basit adımdan oluşmaktadır:

1.  **Gereksiz Callback Kodunu Kaldırmak:**
    - `from utils.virtual_bankroll_callback import CatBoostBankrollCallback` import satırı silinecek.
    - `virtual_bankroll_reg` ve `virtual_bankroll_cls` nesnelerinin oluşturulduğu kod blokları kaldırılacak.
    - `fit` metodundaki `# callbacks kaldırıldı...` yorumları temizlenecek.

2.  **GPU Eğitimini Etkinleştirmek:**
    - `CatBoostRegressor` ve `CatBoostClassifier` tanımlamalarında bulunan `task_type='CPU'` parametresi `task_type='GPU'` olarak değiştirilecektir.

## 3. Beklenen Sonuç

Bu değişiklikler sonucunda:
- Model eğitimi, Google Colab ortamındaki A100 GPU'nun tüm gücünü kullanarak gerçekleşecektir.
- Eğitim süresi **önemli ölçüde kısalacaktır** (saatlerden dakikalara inmesi beklenir).
- Betiğin mevcut işlevselliğinde (sonuç raporlama, sanal kasa simülasyonu vb.) **hiçbir kayıp olmayacaktır**.

## 4. Sonraki Adım

Bu planın onaylanmasının ardından, "Code" moduna geçilerek önerilen değişiklikler `notebooks/jetx_CATBOOST_TRAINING.py` dosyasına uygulanacaktır.