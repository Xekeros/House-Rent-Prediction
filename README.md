

# 🏠 House Rent Prediction

Bu proje, Türkiye'deki ev kiraları üzerine yapılan bir veri bilimi çalışmasıdır. Amacımız, emlak verilerine dayalı olarak kira bedellerini tahmin eden bir regresyon modeli geliştirmek ve bu tahminleri kategorilere ayırarak sınıflandırma analizi yapmaktır.

---

## 🎯 Proje Amacı

Ev kiraları; lokasyon, oda sayısı, metrekare gibi birçok faktörden etkilenir. Bu projede:
- Ev kiralarını tahmin eden bir regresyon modeli geliştirildi.
- Tahminler kategorilere ayrılarak (Low, Medium, High) sınıflandırma sonuçları da analiz edildi.

---

## 📁 Proje Yapısı

```
project/
│
├── model_finisher_personB.py     # Model eğitimi, tahmin ve değerlendirme
├── X_train.csv                   # Eğitim verisi (özellikler)
├── X_test.csv                    # Test verisi (özellikler)
├── y_train.csv                   # Eğitim verisi (kira)
├── y_test.csv                    # Test verisi (kira)
└── README.md                     # Açıklayıcı dosya
```

---

## ⚙️ Kullanılan Teknolojiler

- Python 3
- [scikit-learn](https://scikit-learn.org)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

---

## 🧠 Model Bilgisi

Model: `RandomForestRegressor`

Kullanılan parametreler:
- `n_estimators=200`
- `max_depth=10`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `bootstrap=True`
- `random_state=42`

---

## 🚀 Nasıl Çalıştırılır?

1. Gerekli bağımlılıkları kur:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Dosyaların bulunduğu dizinde çalıştır:

```bash
python model_finisher_personB.py
```

---

## 📊 Çıktılar

### 📈 Regresyon Metrikleri
- MAE (Ortalama Mutlak Hata)
- MSE (Ortalama Kare Hata)
- RMSE (Karekök Ortalama Hata)
- R² Score (Doğruluk)

### 🧮 Sınıflandırma
- Kiralar şu sınıflara bölünür:
  - **Low**: 0 - 10,000
  - **Medium**: 10,000 - 30,000
  - **High**: 30,000+

- Sınıflandırma performansı `classification_report` ile değerlendirilir.

- Ayrıca **confusion matrix** (karmaşıklık matrisi) aşağıdaki gibi görselleştirilir:

### 🔍 Örnek Confusion Matrix

> Bu görsel çalışma tamamlandığında oluşur:

![Confusion Matrix](confusion_matrix_example.png)

> Not: Yukarıdaki görselin oluşturulabilmesi için betik çalıştırıldıktan sonra `.png` olarak kaydedilmesi gerekir.

---

## 📬 Katkı ve İletişim

Bu proje, **CMPE 442 - Machine Learning** dersi kapsamında geliştirilmiştir. Geliştiriciye ulaşmak için GitHub üzerinden iletişime geçebilirsiniz.