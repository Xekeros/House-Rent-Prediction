import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Step 1: Load your data
# -------------------------------
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")

# -------------------------------
# Step 2: Keep only numeric columns
# -------------------------------
X_train = X_train.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

# 🔁 Make sure y_train matches the new X_train index
y_train = y_train.loc[X_train.index]

# -------------------------------
# Step 3: Train a model
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# -------------------------------
# Step 4: Pick a few test examples for explanation
# -------------------------------
sample = X_test.sample(3, random_state=1)

# -------------------------------
# Step 5: Generate SHAP values
# -------------------------------
shap.initjs()
explainer = shap.Explainer(model, X_train)
shap_values = explainer(sample)

# -------------------------------
# Step 6: Save SHAP force plots as HTML
# -------------------------------
for i in range(len(sample)):
    shap_html = shap.force_plot(
        explainer.expected_value,
        shap_values[i].values,
        sample.iloc[i],
        feature_names=sample.columns,
        matplotlib=False
    )
    filename = f"force_plot_{i+1}.html"
    shap.save_html(filename, shap_html)
    print(f"✅ Saved {filename}")