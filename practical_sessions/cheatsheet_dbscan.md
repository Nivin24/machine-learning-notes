# DBSCAN Cheatsheet

## Class
```python
from sklearn.cluster import DBSCAN
```

## Main Parameters
- **eps**: Maximum distance between two samples to be considered neighbors (default: 0.5)
- **min_samples**: Minimum number of samples in a neighborhood for a point to be core (default: 5)
- **metric**: Distance metric to use (default: 'euclidean')
  - Options: 'euclidean', 'manhattan', 'cosine', etc.
- **algorithm**: Algorithm to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')

## Workflow
1. Choose eps and min_samples parameters
2. Scale/normalize your data
3. Fit the model
4. Examine labels (noise points labeled as -1)

## Minimal Code
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
labels = dbscan.fit_predict(X_scaled)

# Get core samples and number of clusters
core_samples = dbscan.core_sample_indices_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
```

## Advantages
- No need to specify number of clusters beforehand
- Can find arbitrarily shaped clusters
- Robust to outliers (labels them as noise)
- Only 2 main parameters to tune

## Pitfalls & Tips
- **Sensitive to eps and min_samples**: Use domain knowledge or elbow method on k-distance graph
- **Scaling is crucial**: Always standardize features before DBSCAN
- **Struggles with varying densities**: May not work well if clusters have different densities
- **High-dimensional curse**: Performance degrades in high dimensions
- **Parameter selection**: 
  - Start with min_samples = 2 * dimensions as rule of thumb
  - Use k-distance plot to choose eps (look for "elbow")

## Parameter Tuning Tips
```python
# K-distance plot for eps selection
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

distances = np.sort(distances[:, -1], axis=0)
plt.plot(distances)
plt.ylabel('k-NN distance')
plt.xlabel('Sorted observations')
plt.title('K-distance Graph')
plt.show()
# Look for "elbow" in the plot to choose eps
```

## Visualization Example
```python
import matplotlib.pyplot as plt

# Plot clusters
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'black'  # Noise points in black
    
    class_mask = (labels == label)
    plt.scatter(X_scaled[class_mask, 0], X_scaled[class_mask, 1],
                c=[color], label=f'Cluster {label}', alpha=0.7)

plt.title('DBSCAN Clustering')
plt.legend()
plt.show()
```
