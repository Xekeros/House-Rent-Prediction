# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import warnings

warnings.filterwarnings('ignore')

# Step 2: Load Dataset
file_path = 'House_Rent_Dataset.csv'  # Make sure the CSV is in your PyCharm project folder
data = pd.read_csv(file_path)

print("Before Cleaning:")
print(data.info())
print(data.head())

# Step 3: Dropping Unnecessary Columns
data = data.drop(columns=['Posted On', 'Point of Contact'])
# Dropping Area Locality (because it's text and not encoded)
data = data.drop(columns=['Area Locality'])

# Step 4: Handling Missing Values
if data['Size'].isnull().sum() > 0:
    data['Size'].fillna(data['Size'].median(), inplace=True)

# Step 5: Feature Engineering - Parse Floor
def split_floor(floor):
    if pd.isna(floor):
        return (np.nan, np.nan)

    floor = str(floor)  # Force to string in case it's a number

    if "Ground" in floor:
        if "out of" in floor:
            try:
                total_floors = int(floor.split('out of')[-1].strip())
            except:
                total_floors = np.nan
            return (0, total_floors)
        else:
            return (0, np.nan)  # Only "Ground" no total floors

    if "Upper Basement" in floor or "Lower Basement" in floor:
        if "out of" in floor:
            try:
                total_floors = int(floor.split('out of')[-1].strip())
            except:
                total_floors = np.nan
            return (-1, total_floors)
        else:
            return (-1, np.nan)

    if "out of" in floor:
        try:
            parts = floor.split('out of')
            current_floor = int(parts[0].strip())
            total_floors = int(parts[1].strip())
            return (current_floor, total_floors)
        except:
            return (np.nan, np.nan)

    return (np.nan, np.nan)  # If floor value is totally strange


# Apply parsing
data[['Current_Floor', 'Total_Floors']] = data['Floor'].apply(lambda x: pd.Series(split_floor(x)))

# Drop original Floor column
data.drop('Floor', axis=1, inplace=True)

# Fill missing values after parsing
data['Current_Floor'].fillna(data['Current_Floor'].median(), inplace=True)
data['Total_Floors'].fillna(data['Total_Floors'].median(), inplace=True)

# Step 6: Encode Categorical Variables
categorical_cols = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Step 7: Normalize Numerical Features
scaler = StandardScaler()
numerical_cols = ['Size', 'Current_Floor', 'Total_Floors']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Step 8: Train-Test Split
X = data.drop('Rent', axis=1)
y = data['Rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Step 9: Model Training and Cross-Validation
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor()
}

results = []

for name, model in models.items():
    mae = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
    mse = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
    results.append([name, mae, mse, r2])

# Step 10: Display Results
results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'MSE', 'R2'])
print("\nModel Performance Summary:")
print(results_df)

# Step 11: Save Train-Test Splits for Person B
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nCleaned datasets saved! Ready for fine-tuning (Person B).")
