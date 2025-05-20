
# Regression Models Comparison: Linear Regression, Decision Tree, Random Forest, XGBoost, SVR

## Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## Generate synthetic regression data
X, y = make_regression(n_samples=300, n_features=1, noise=20, random_state=42)

## Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Standardize for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
    'SVR': SVR()
}

## Train, predict, and evaluate
for name, model in models.items():
    print(f"\n=== {name} ===")
    if name == 'SVR':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}")

    ## Plot predictions
    plt.scatter(X_test, y_test, color='gray', label='True')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.title(f"{name} - Predictions")
    plt.legend()
    plt.show()
