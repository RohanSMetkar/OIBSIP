"""
Car Price Prediction with Machine Learning
============================================
This script predicts the selling price of used cars using the user's
own dataset (car_data.xls, which is actually a CSV of car listings).

Unlike the original notebook -- which downloaded a different dataset
from a GitHub URL -- this script loads the given car_data.xls file
directly, so all cleaning, EDA and modeling reflect the real data.

Steps:
1. Load data
2. Clean data (duplicates, column names, missing values)
3. Clean categorical variables
4. Feature engineering (car age, brand)
5. Exploratory Data Analysis (plots saved to ./plots)
6. Prepare features/target + one-hot encoding
7. Train/test split
8. Train Linear Regression, Random Forest, Gradient Boosting
9. Evaluate & pick the best model
10. Feature importance
11. Predict price for a sample car
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CURRENT_YEAR = 2026
DATA_PATH = "car_data.xls"          # your uploaded file (CSV format despite the extension)
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved plot: {path}")


# ---------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------------------
# 2. Initial inspection
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. INITIAL INSPECTION")
print("=" * 60)

print(df.info())
print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------------
# 3. Data cleaning
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DATA CLEANING")
print("=" * 60)

df = df.drop_duplicates().reset_index(drop=True)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
print("Cleaned column names:", df.columns.tolist())

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Remaining missing values:", df.isnull().sum().sum())

# ---------------------------------------------------------------------
# 4. Clean categorical variables
# ---------------------------------------------------------------------
for col in ["fuel_type", "seller_type", "transmission"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

print("\nFuel type counts:")
print(df["fuel_type"].value_counts())

# ---------------------------------------------------------------------
# 5. Feature engineering: car age & brand
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. FEATURE ENGINEERING")
print("=" * 60)

df["car_age"] = CURRENT_YEAR - df["year"]
df = df[df["car_age"] >= 0].copy()

df["brand"] = (
    df["car_name"]
    .astype(str)
    .str.split()
    .str[0]
    .str.lower()
    .str.strip()
)

print(df[["car_name", "brand", "year", "car_age"]].head())

# ---------------------------------------------------------------------
# 6. Exploratory Data Analysis
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig = plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"], kde=True)
plt.title("Distribution of Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
save_plot(fig, "01_selling_price_distribution.png")

fig = plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="fuel_type", y="selling_price")
plt.title("Selling Price vs Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
save_plot(fig, "02_price_vs_fuel_type.png")

fig = plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="car_age", y="selling_price")
plt.title("Selling Price vs Car Age")
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
save_plot(fig, "03_price_vs_car_age.png")

fig = plt.figure(figsize=(10, 7))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
save_plot(fig, "04_correlation_heatmap.png")

# ---------------------------------------------------------------------
# 7. Prepare features and target
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. FEATURES AND TARGET")
print("=" * 60)

X = df.drop(columns=["selling_price", "car_name", "year"], errors="ignore")
y = df["selling_price"]

categorical_features = X.select_dtypes(include="object").columns.tolist()
numerical_features = X.select_dtypes(exclude="object").columns.tolist()

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# ---------------------------------------------------------------------
# 8. Train/test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ---------------------------------------------------------------------
# 9. Train models
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("9. TRAINING MODELS")
print("=" * 60)

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
print("Trained: Linear Regression")

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Trained: Random Forest")

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
])
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("Trained: Gradient Boosting")

# ---------------------------------------------------------------------
# 10. Model evaluation
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("10. MODEL EVALUATION")
print("=" * 60)


def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return [name, mae, rmse, r2]


results = pd.DataFrame([
    evaluate_model("Linear Regression", y_test, linear_pred),
    evaluate_model("Random Forest", y_test, rf_pred),
    evaluate_model("Gradient Boosting", y_test, gb_pred)
], columns=["Model", "MAE", "RMSE", "R2 Score"])

print(results)

best_model_row = results.loc[results["R2 Score"].idxmax()]
print("\nBest Model:", best_model_row["Model"])
print("MAE:", best_model_row["MAE"])
print("RMSE:", best_model_row["RMSE"])
print("R2 Score:", best_model_row["R2 Score"])

fig = plt.figure(figsize=(8, 5))
sns.barplot(data=results, x="Model", y="R2 Score")
plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
save_plot(fig, "05_r2_score_comparison.png")

fig = plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices (Random Forest)")
save_plot(fig, "06_actual_vs_predicted.png")

# ---------------------------------------------------------------------
# 11. Feature importance
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("11. FEATURE IMPORTANCE")
print("=" * 60)

rf_preprocessor = rf_model.named_steps["preprocessor"]
rf_regressor = rf_model.named_steps["model"]
feature_names = rf_preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_regressor.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance.head(15))

fig = plt.figure(figsize=(10, 6))
sns.barplot(data=importance.head(15), x="Importance", y="Feature")
plt.title("Top 15 Feature Importances")
save_plot(fig, "07_feature_importance.png")

# ---------------------------------------------------------------------
# 12. Predict price for a new/sample car
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("12. SAMPLE PREDICTION")
print("=" * 60)

sample_car = pd.DataFrame({
    "present_price": [5.5],
    "kms_driven": [30000],
    "fuel_type": ["petrol"],
    "seller_type": ["individual"],
    "transmission": ["manual"],
    "owner": [0],
    "car_age": [5],
    "brand": ["maruti"]
})

predicted_price = rf_model.predict(sample_car)
print("Predicted Selling Price:", predicted_price[0])

print("\n" + "=" * 60)
print("DONE. All plots saved to the 'plots' folder.")
print("=" * 60)