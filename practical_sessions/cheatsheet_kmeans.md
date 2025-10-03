# K-Means Clustering Cheatsheet

## Core Scikit-Learn Classes
```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
```

## Common Parameters

### KMeans
- `n_clusters`: Number of clusters (default=8). MUST specify!
- `init`: Initialization method ('k-means++' default, 'random', or array)
- `n_init`: Number of times algorithm runs with different seeds (default=10)
- `max_iter`: Maximum iterations per run (default=300)
- `random_state`: Seed for reproducibility
- `algorithm`: 'lloyd' (default), 'elkan' (faster for dense data)
- `n_jobs`: Number of parallel jobs (-1 = use all cores)

### MiniBatchKMeans (for large datasets)
- `batch_size`: Size of mini-batches (default=1024)
- Other parameters same as KMeans

## Code Pattern

### Basic K-Means
```python
# Scale features recommended for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Basic clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Get cluster labels
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_

# Predict clusters for new data
X_new_scaled = scaler.transform(X_new)
new_labels = kmeans.predict(X_new_scaled)
```

### With Elbow Method to Find K
```python
import matplotlib.pyplot as plt

inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method')
plt.show()
```

### With Silhouette Analysis
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}: silhouette score={score:.3f}")

# Higher is better (range: -1 to 1)
optimal_k = K_range[np.argmax(silhouette_scores)]
```

## Workflow Tips

1. **Scale features**: K-Means uses Euclidean distance - scaling helps
2. **Use k-means++ initialization**: Default and usually best
3. **Run multiple times**: `n_init=10` or more for stability
4. **Find optimal k**: Use elbow method, silhouette score, or domain knowledge
5. **Check convergence**: `kmeans.n_iter_` shows iterations taken
6. **For large data**: Use MiniBatchKMeans for speed

## Scaling Notes

✅ **Recommended but not always required**

```python
# StandardScaler most common
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler alternative
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**Why scaling matters**: Features with larger ranges dominate distance calculations. Scale when features have different units or ranges.

## Finding Optimal K

### 1. Elbow Method
```python
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Look for "elbow" in the plot
plt.plot(range(1, 11), inertias)
```

### 2. Silhouette Score
```python
# Range: -1 (bad) to 1 (good), aim for > 0.5
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, labels)
```

### 3. Davies-Bouldin Index
```python
# Lower is better (minimum is 0)
from sklearn.metrics import davies_bouldin_score
score = davies_bouldin_score(X_scaled, labels)
```

### 4. Calinski-Harabasz Index
```python
# Higher is better
from sklearn.metrics import calinski_harabasz_score
score = calinski_harabasz_score(X_scaled, labels)
```

## Practical Guidance

### When to Use K-Means
✅ Spherical/globular clusters expected
✅ Clusters roughly equal size
✅ Fast clustering needed
✅ Large datasets
✅ Numeric features only
✅ Need simple, interpretable results

### When NOT to Use K-Means
❌ Non-spherical clusters (elongated, irregular shapes)
❌ Very different cluster sizes
❌ Outliers present (sensitive to outliers)
❌ Number of clusters unknown and hard to determine
❌ Categorical features
❌ Need hierarchical structure

### Common Pitfalls
- Not scaling features → biased results
- Wrong k chosen → poor clustering
- Random initialization → inconsistent results (use k-means++)
- Ignoring outliers → distorted centroids
- Assuming equal cluster sizes

## Advanced Usage

### MiniBatch K-Means (Large Datasets)
```python
from sklearn.cluster import MiniBatchKMeans

# 10-100x faster than regular K-Means
mbkmeans = MiniBatchKMeans(
    n_clusters=5,
    batch_size=1000,
    random_state=42
)
mbkmeans.fit(X_scaled)
labels = mbkmeans.labels_
```

### Custom Initialization
```python
# Provide your own initial centers
initial_centers = np.array([[0, 0], [1, 1], [2, 2]])
kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1)
```

### Get Distance to Centroids
```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

# Distance from each point to each centroid
distances = kmeans.transform(X_scaled)

# Distance to assigned centroid
min_distances = distances.min(axis=1)
```

### Cluster Quality Metrics
```python
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Inertia (within-cluster sum of squares) - lower is better
print(f"Inertia: {kmeans.inertia_}")

# Silhouette score - higher is better
from sklearn.metrics import silhouette_score
print(f"Silhouette: {silhouette_score(X_scaled, labels)}")

# Davies-Bouldin - lower is better
from sklearn.metrics import davies_bouldin_score
print(f"Davies-Bouldin: {davies_bouldin_score(X_scaled, labels)}")
```

### Handle Outliers
```python
# Option 1: Remove outliers before clustering
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1)
outlier_mask = iso.fit_predict(X_scaled) == 1
X_clean = X_scaled[outlier_mask]

# Option 2: Increase number of clusters to isolate outliers
kmeans = KMeans(n_clusters=k+2)  # Extra clusters for outliers
```

### Visualize Clusters (2D)
```python
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_

# For 2D data
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title('K-Means Clustering')
plt.show()

# For higher dimensions, use PCA first
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
```

## Performance Tips

1. **Use k-means++ initialization**: Faster convergence
2. **Reduce n_init for speed**: Trade stability for speed
3. **Use MiniBatchKMeans**: For datasets > 10K samples
4. **Parallel processing**: Set `n_jobs=-1`
5. **Elkan algorithm**: Faster for dense data
   ```python
   kmeans = KMeans(algorithm='elkan')
   ```

## K Selection Guidelines

| Dataset Size | Suggested K Range | Method |
|-------------|------------------|--------|
| < 100 | 2-5 | Domain knowledge |
| 100-1000 | 2-10 | Elbow + Silhouette |
| > 1000 | 3-20 | Silhouette analysis |

**Rule of thumb**: k ≈ √(n/2) where n is number of samples

## Quick Reference

| Task | Model | Key Params |
|------|-------|------------|
| Basic Clustering | KMeans | n_clusters, random_state |
| Large Dataset (>10K) | MiniBatchKMeans | n_clusters, batch_size |
| High Dimensions | KMeans + PCA | n_clusters, preprocessing |
| Unknown K | Multiple runs | Elbow + Silhouette |
| Need Speed | MiniBatchKMeans | n_clusters, n_init=3 |
| Dense Data | KMeans | algorithm='elkan' |
| Reproducible Results | KMeans | random_state=42 |

## Complete Workflow Example

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Find optimal k
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal k: {optimal_k}")

# 3. Train final model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 4. Analyze results
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Iterations: {kmeans.n_iter_}")

# 5. Use for prediction
X_new_scaled = scaler.transform(X_new)
predicted_labels = kmeans.predict(X_new_scaled)
```
