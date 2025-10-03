# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Table of Contents
1. Introduction and Intuition
2. Key Parameters: eps and min_samples
3. Core, Border, and Noise Points
4. Algorithm Steps
5. Strengths and Limitations
6. Choosing eps and min_samples
7. Comparisons: DBSCAN vs K-means vs Hierarchical
8. Practical Tips and Common Pitfalls
9. Code Examples (scikit-learn)

---

## 1. Introduction and Intuition
DBSCAN groups together points that are closely packed (high density regions) and marks points that lie alone in low-density regions as outliers. It can discover clusters of arbitrary shape and is robust to noise, unlike K-means which assumes spherical clusters.

## 2. Key Parameters: eps and min_samples
- eps (ε): Neighborhood radius around a point. Smaller ε means stricter neighborhood.
- min_samples: Minimum number of points required to form a dense region (including the point itself).
  - Rule of thumb: min_samples ≈ dimensionality + 1 (often 4 for 2D), but higher values are more robust to noise.

## 3. Core, Border, and Noise Points
- Core point: Has at least min_samples points within ε radius
- Border point: Not core, but within ε of a core point
- Noise point (outlier): Neither core nor border

Density reachability and connectivity define how clusters expand from core points to neighbors, chaining through density-connected points.

## 4. Algorithm Steps
1. For each unvisited point p:
   - If p is a core point, create a new cluster and grow it by recursively adding density-reachable points within ε
   - If p is not a core point and not within ε of a core point, mark it as noise (may later become border)
2. Continue until all points are visited

Time complexity depends on neighborhood queries (can be O(n log n) with spatial index like k-d tree; otherwise O(n^2)).

## 5. Strengths and Limitations
Strengths:
- Finds arbitrarily shaped clusters
- Identifies noise explicitly
- No need to pre-specify number of clusters
Limitations:
- Single global ε struggles with varying densities
- Sensitive to parameter choice (ε, min_samples)
- High-dimensional data can make distance less meaningful (curse of dimensionality)

## 6. Choosing eps and min_samples
- k-distance plot: For each point, compute distance to its k-th nearest neighbor (k = min_samples). Sort descending and look for an “elbow.” The y-value at the elbow is a good ε.
- Domain knowledge: Use expected neighborhood scale
- Scale your features: Standardize or use domain-appropriate metrics

## 7. Comparisons: DBSCAN vs K-means vs Hierarchical
- vs K-means: DBSCAN handles noise, doesn’t require k, and finds non-spherical clusters; K-means is faster and better for large, well-separated spherical clusters.
- vs Hierarchical: DBSCAN gives a flat clustering with explicit noise handling; hierarchical provides full multi-scale structure but no explicit noise concept (unless adapted).

## 8. Practical Tips and Common Pitfalls
- Always standardize continuous features (z-score) before Euclidean-based DBSCAN
- Consider alternative metrics (cosine for text embeddings, haversine for geo)
- Use a spatial index (ball tree, k-d tree) for performance on large datasets
- For varying-density data, consider HDBSCAN (hierarchical DBSCAN) which handles variable density better and removes the need to pick ε
- In high dimensions, consider dimensionality reduction (PCA/UMAP) before DBSCAN

## 9. Code Examples (scikit-learn)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Synthetic data with non-spherical shape
X, y_true = make_moons(n_samples=600, noise=0.06, random_state=42)
X = StandardScaler().fit_transform(X)

# Grid search a few eps values (illustrative)
for eps in [0.1, 0.2, 0.3, 0.4]:
    model = DBSCAN(eps=eps, min_samples=5, metric='euclidean')
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.mean(labels == -1)
    if n_clusters > 1:
        sil = silhouette_score(X, labels, metric='euclidean')
    else:
        sil = float('nan')
    print(f"eps={eps:.2f} | clusters={n_clusters} | noise={noise_ratio:.2f} | silhouette={sil:.3f}")

# Visualize a chosen configuration
model = DBSCAN(eps=0.3, min_samples=5)
labels = model.fit_predict(X)
plt.figure(figsize=(8, 6))
unique_labels = np.unique(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for lab, col in zip(unique_labels, colors):
    mask = labels == lab
    if lab == -1:
        plt.scatter(X[mask, 0], X[mask, 1], c=[col], s=10, label='noise')
    else:
        plt.scatter(X[mask, 0], X[mask, 1], c=[col], s=10, label=f'cluster {lab}')
plt.legend()
plt.title('DBSCAN clustering with noise')
plt.show()
```

---

Next: Cluster Evaluation → See cluster_evaluation.md
