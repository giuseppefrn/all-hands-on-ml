
# K-Means Clustering Example

## Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

## Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

## Visualize raw data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Raw Data")
plt.show()

## Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)

## Plot clustered data
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title("K-Means Clustering Result")
plt.legend()
plt.show()

## Optional: Elbow Method to choose K
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(1, 10), inertias, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


## DBSCAN Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Standardize the data before DBSCAN
X_scaled = StandardScaler().fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

## Plot DBSCAN clustering result
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, s=50, cmap='plasma')
plt.title("DBSCAN Clustering Result")
plt.show()

## Summary:
# - K-Means requires K and assumes spherical clusters
# - DBSCAN finds arbitrarily shaped clusters and detects noise (label = -1)
