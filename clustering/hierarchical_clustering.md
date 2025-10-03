# Hierarchical Clustering

## Table of Contents
1. Introduction
2. Types: Agglomerative vs Divisive
3. Linkage Criteria
4. Dendrograms and Cut Thresholds
5. Distance Metrics
6. Step-by-step Example (Agglomerative)
7. Pros and Cons
8. Practical Tips and Pitfalls
9. Comparing to K-means and DBSCAN
10. Code Examples (Scikit-learn & SciPy)

---

## 1. Introduction
Hierarchical clustering builds a hierarchy of clusters without requiring a pre-specified number of clusters. The result is commonly visualized as a dendrogram, which shows how clusters merge or split at different distance thresholds.

Use cases include taxonomy, customer segmentation, and exploratory data analysis where the cluster structure at multiple granularities is valuable.

## 2. Types: Agglomerative vs Divisive
- Agglomerative (bottom-up):
  - Start with each data point as a singleton cluster
  - Iteratively merge the two closest clusters
  - Most common in practice
- Divisive (top-down):
  - Start with all points in one cluster
  - Recursively split clusters until singletons or stopping criteria
  - Less common due to computational cost

## 3. Linkage Criteria
How distance between clusters is computed greatly affects the shape of clusters.
- Single linkage (nearest neighbor):
  - Distance between clusters = minimum pairwise distance
  - Tends to form “chains”; good for elongated clusters; sensitive to noise
- Complete linkage (farthest neighbor):
  - Distance = maximum pairwise distance
  - Produces compact clusters; robust to chaining; sensitive to outliers
- Average linkage (UPGMA):
  - Distance = average pairwise distance between all points across clusters
  - Balanced behavior between single and complete
- Ward’s linkage:
  - Merges clusters that result in the minimum increase in total within-cluster variance (inertia)
  - Favors spherical clusters; often performs well on standardized data

## 4. Dendrograms and Cut Thresholds
- Dendrogram encodes the sequence of merges (or splits) and the distance at which they occur.
- To obtain a flat clustering:
  - Choose a distance threshold (horizontal cut) to cut the dendrogram
  - Or choose a target number of clusters and cut accordingly
- Interpretation tips:
  - Large vertical jumps before merges indicate well-separated clusters
  - Horizontal cuts above those large jumps separate meaningful clusters

## 5. Distance Metrics
Common choices: Euclidean, Manhattan (L1), Cosine, Correlation. The appropriate metric depends on scale and data type; always standardize when using Euclidean/Ward.

## 6. Step-by-step Example (Agglomerative)
Given 5 points A–E with pairwise distances, agglomerative clustering proceeds:
1) Initialize: {A}, {B}, {C}, {D}, {E}
2) Find closest pair (by chosen linkage), merge → e.g., {A,B}
3) Recompute distances between new cluster and others (by linkage rule)
4) Repeat until one cluster remains
5) To get k clusters, stop when there are k clusters or cut dendrogram at threshold

Illustrative matrix update (average linkage):
- d({A,B}, C) = (d(A,C) + d(B,C)) / 2
- d({A,B}, {D,E}) = average of all 4 pairwise distances

## 7. Pros and Cons
Pros:
- No need to pre-specify k
- Rich multi-scale view of structure via dendrogram
- Works with different linkage and distance metrics
Cons:
- O(n^2) memory and typically O(n^2 log n) or worse time; struggles with very large n
- Sensitive to noise and outliers (varies by linkage)
- Early merge decisions are irrevocable (greedy)

## 8. Practical Tips and Pitfalls
- Always scale features; Ward requires Euclidean and benefits from standardization
- Try multiple linkages; inspect dendrogram stability and cluster silhouettes
- Prune small/noisy branches; consider robust metrics or preprocessing outliers
- Use truncated dendrograms for large n to keep plots readable
- For very large datasets, consider approximate methods or K-means/GMM first to prototype

## 9. Comparing to K-means and DBSCAN
- vs K-means:
  - No k required upfront; non-spherical shapes handled better (depending on linkage)
  - Slower; can be more sensitive to noise than K-means++ on standardized data
- vs DBSCAN:
  - DBSCAN handles noise explicitly and finds arbitrary shapes without k
  - Hierarchical yields full hierarchy; DBSCAN requires epsilon/min_samples tuning
  - Hierarchical can give structure at many levels; DBSCAN gives a flat clustering

## 10. Code Examples

### Scikit-learn: Agglomerative Clustering
```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Data
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=1.2, random_state=42)
X = StandardScaler().fit_transform(X)

# Model
model = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = model.fit_predict(X)

print('Silhouette:', silhouette_score(X, labels))

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
plt.title('Agglomerative (Ward) Clustering')
plt.show()
```

### SciPy: Linkage + Dendrogram
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

# X is (n_samples, n_features)
X_scaled = StandardScaler().fit_transform(X)
Z = linkage(X_scaled, method='average', metric='euclidean')

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=5)  # truncate for readability
plt.title('Dendrogram (Average Linkage)')
plt.xlabel('Samples or (merged clusters)')
plt.ylabel('Distance')
plt.show()

# Cut at distance threshold t to obtain flat clusters
threshold = 5.0
flat_labels = fcluster(Z, t=threshold, criterion='distance')
```

---

Next: DBSCAN → See dbscan.md
