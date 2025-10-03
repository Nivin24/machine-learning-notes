# Cluster Evaluation

## Table of Contents
1. Why Evaluate Clusters?
2. Internal Metrics
   - Silhouette Score
   - Davies–Bouldin Index (DBI)
   - Calinski–Harabasz Index (CH)
3. External Metrics (when ground truth available)
   - Adjusted Rand Index (ARI)
   - Normalized Mutual Information (NMI)
4. Heuristics for Selecting K
   - Elbow Method
   - Silhouette Analysis
   - Gap Statistic (concept)
5. Visualization Guidelines
6. Practical Workflow and Caveats
7. Code Examples (scikit-learn)

---

## 1. Why Evaluate Clusters?
Clustering is unsupervised—there is no single “right” answer. Evaluation checks cohesion (points close within clusters) and separation (clusters far apart), balances model selection (e.g., choose k), and warns about artifacts like chaining or density issues.

## 2. Internal Metrics
These use only the data and cluster assignments.

### Silhouette Score (range: [-1, 1], higher is better)
- For each point i:
  - a(i) = average distance to points in same cluster
  - b(i) = minimum average distance to points in any other cluster
  - silhouette s(i) = (b(i) - a(i)) / max{a(i), b(i)}
- Interpretations:
  - ~0.5–1.0: dense, well-separated clusters
  - ~0.2–0.5: overlapping clusters; acceptable
  - < 0: likely misassignment or too many/few clusters

### Davies–Bouldin Index (lower is better)
- DBI = average, over clusters, of similarity to the most similar other cluster
- Similarity uses within-cluster scatter and centroid distances
- Lower DBI indicates compact, well-separated clusters

### Calinski–Harabasz Index (higher is better)
- Ratio of between-cluster dispersion to within-cluster dispersion
- Favors well-separated clusters; computationally efficient

## 3. External Metrics (need ground truth labels)
If you have known labels (e.g., synthetic data), compare clustering to truth.

### Adjusted Rand Index (ARI)
- Measures pairwise agreement of assignments, adjusted for chance
- Range: [-1, 1]; 0 ~ random, 1 = perfect match

### Normalized Mutual Information (NMI)
- Information-theoretic similarity between predicted clusters and labels
- Range: [0, 1]; 1 = perfect agreement

## 4. Heuristics for Selecting K

### Elbow Method
- Plot WCSS (inertia) versus k
- Choose the “elbow” where marginal gain diminishes
- Works best for roughly spherical clusters

### Silhouette Analysis
- Compute mean silhouette for each k
- Choose k maximizing mean silhouette (subject to domain sense)

### Gap Statistic (concept)
- Compares within-cluster dispersion to expected dispersion under a null reference
- Choose k with largest gap (or first local maximum)

## 5. Visualization Guidelines
- 2D/3D scatter plots with colors for clusters; overlay centroids
- Use dimensionality reduction (PCA/UMAP/t-SNE) for high-dimensional data
- Dendrograms for hierarchical clustering; choose cut based on large linkage jumps
- k-distance plots for DBSCAN to select eps (look for elbow)
- Pair plots or parallel coordinates for multi-feature inspection

## 6. Practical Workflow and Caveats
- Standardize features before distance-based algorithms and metrics
- Combine metrics; no single metric suffices in all cases
- Beware of “good-looking” metrics on meaningless projections
- Validate stability: rerun with different seeds/subsamples
- Consider domain utility: cluster interpretability and actionability

## 7. Code Examples (scikit-learn)
```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

# Data (with optional ground truth)
X, y_true = make_blobs(n_samples=800, centers=4, cluster_std=1.2, random_state=42)
X = StandardScaler().fit_transform(X)

k_range = range(2, 10)
mean_sil = []
dbi = []
chi = []
inertia = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)
    mean_sil.append(silhouette_score(X, labels))
    dbi.append(davies_bouldin_score(X, labels))
    chi.append(calinski_harabasz_score(X, labels))

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
axs[0,0].plot(k_range, inertia, 'o-'); axs[0,0].set_title('Elbow (Inertia)'); axs[0,0].set_xlabel('k'); axs[0,0].set_ylabel('WCSS')
axs[0,1].plot(k_range, mean_sil, 'o-'); axs[0,1].set_title('Mean Silhouette'); axs[0,1].set_xlabel('k'); axs[0,1].set_ylabel('Score')
axs[1,0].plot(k_range, dbi, 'o-'); axs[1,0].set_title('Davies–Bouldin (lower)'); axs[1,0].set_xlabel('k'); axs[1,0].set_ylabel('DBI')
axs[1,1].plot(k_range, chi, 'o-'); axs[1,1].set_title('Calinski–Harabasz (higher)'); axs[1,1].set_xlabel('k'); axs[1,1].set_ylabel('CH')
plt.tight_layout(); plt.show()

# External metrics (if y_true known)
km = KMeans(n_clusters=4, n_init='auto', random_state=42)
pred = km.fit_predict(X)
print('ARI:', adjusted_rand_score(y_true, pred))
print('NMI:', normalized_mutual_info_score(y_true, pred))
```

---

Next: See notes.md for quick references and interview questions.
