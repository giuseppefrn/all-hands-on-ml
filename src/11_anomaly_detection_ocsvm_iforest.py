
# Anomaly Detection with One-Class SVM and Isolation Forest

## Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

## Generate synthetic data with outliers
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=42)
rng = np.random.RandomState(42)
outliers = rng.uniform(low=-4, high=4, size=(30, 2))
X = np.vstack([X, outliers])

## Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Define models
models = {
    "One-Class SVM": OneClassSVM(kernel='rbf', nu=0.05, gamma='auto'),
    "Isolation Forest": IsolationForest(contamination=0.1, random_state=42)
}

## Train and plot results
for name, model in models.items():
    model.fit(X_scaled)
    y_pred = model.predict(X_scaled)

    plt.figure()
    plt.title(f"{name} - Anomaly Detection")
    plt.scatter(X_scaled[y_pred == 1][:, 0], X_scaled[y_pred == 1][:, 1], 
                c='b', label='Normal')
    plt.scatter(X_scaled[y_pred == -1][:, 0], X_scaled[y_pred == -1][:, 1], 
                c='r', label='Anomaly')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
