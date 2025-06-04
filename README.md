# 🏠 House Rent Prediction

This project is a data science study on house rents in Turkey. Our goal is to develop a regression model that predicts rental prices based on real estate data and to analyze these predictions by categorizing them for classification analysis.

---

## 🎯 Project Objective

House rents are affected by many factors such as location, number of rooms, and square meters. In this project:
- A regression model was developed to predict house rents.
- Predictions were categorized (Low, Medium, High) and classification results were also analyzed.

---

## 📁 Project Structure

```
project/
│
├── model_finisher_personB.py     # Model training, prediction, and evaluation
├── X_train.csv                   # Training data (features)
├── X_test.csv                    # Test data (features)
├── y_train.csv                   # Training data (rent)
├── y_test.csv                    # Test data (rent)
└── README.md                     # Documentation
```

---

## ⚙️ Technologies Used

- Python 3
- [scikit-learn](https://scikit-learn.org)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

---

## 🧠 Model Information

Model: `RandomForestRegressor`

Parameters used:
- `n_estimators=200`
- `max_depth=10`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `bootstrap=True`
- `random_state=42`

---

## 🚀 How to Run

1. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Run in the directory containing the files:

```bash
python model_finisher_personB.py
```

---

## 📊 Outputs

### 📈 Regression Metrics
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

### 🧮 Classification
- Rents are divided into the following classes:
  - **Low**: 0 - 10,000
  - **Medium**: 10,000 - 30,000
  - **High**: 30,000+

- Classification performance is evaluated with `classification_report`.

- Additionally, the **confusion matrix** is visualized as follows:

### 🔍 Example Confusion Matrix

> This image is generated after running the script:

![Confusion Matrix](confusion_matrix_example.png)

> Note: To generate the above image, the script must be run and the plot saved as `.png`.

---

## 📬 Contribution & Contact

This project was developed for the **CMPE 442 - Machine Learning** course. You can contact the developer via GitHub.
