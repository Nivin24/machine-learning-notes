# K-Means Clustering

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Algorithm Steps](#algorithm-steps)
4. [Convergence Criteria](#convergence-criteria)
5. [Choosing K](#choosing-k)
6. [Practical Considerations](#practical-considerations)
7. [Advantages and Disadvantages](#advantages-and-disadvantages)
8. [Code Implementation](#code-implementation)
9. [Real-world Applications](#real-world-applications)

## Introduction

K-means clustering is one of the most popular unsupervised learning algorithms used for partitioning data into k distinct, non-overlapping clusters. It aims to group similar data points together while keeping dissimilar points in different clusters.

### Key Concepts:
- **Centroid**: The center point of a cluster
- **Cluster**: A group of similar data points
- **Inertia**: Sum of squared distances from each point to its cluster centroid
- **Lloyd's Algorithm**: The standard algorithm for k-means

## Mathematical Foundation

### Objective Function
K-means minimizes the within-cluster sum of squares (WCSS):

```
J = Σ(i=1 to k) Σ(x in Ci) ||x - μi||²
```

Where:
- `J` = objective function to minimize
- `k` = number of clusters
- `Ci` = set of points in cluster i
- `μi` = centroid of cluster i
- `||x - μi||²` = squared Euclidean distance

### Distance Calculation
Euclidean distance between point x and centroid μ:

```
d(x, μ) = √(Σ(j=1 to n) (xj - μj)²)
```

### Centroid Update Formula
New centroid calculation:

```
μi = (1/|Ci|) Σ(x in Ci) x
```

Where `|Ci|` is the number of points in cluster i.

## Algorithm Steps

### Step-by-Step Process:

1. **Initialize**: Randomly place k centroids in the data space
2. **Assignment**: Assign each data point to the nearest centroid
3. **Update**: Calculate new centroid positions as the mean of assigned points
4. **Repeat**: Continue steps 2-3 until convergence

### Detailed Algorithm:

```python
# Pseudocode for K-means
function kmeans(X, k, max_iterations=100):
    # Step 1: Initialize centroids randomly
    centroids = randomly_select_k_points(X, k)
    
    for iteration in range(max_iterations):
        # Step 2: Assign points to nearest centroid
        clusters = []
        for point in X:
            nearest_centroid = find_nearest_centroid(point, centroids)
            assign_to_cluster(point, nearest_centroid, clusters)
        
        # Step 3: Update centroids
        new_centroids = []
        for cluster in clusters:
            new_centroid = calculate_mean(cluster)
            new_centroids.append(new_centroid)
        
        # Step 4: Check convergence
        if centroids_converged(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    return centroids, clusters
```

## Convergence Criteria

### Common Stopping Conditions:

1. **Centroid Stability**: Centroids don't move significantly
   ```
   ||μi(t+1) - μi(t)|| < ε
   ```

2. **Assignment Stability**: No points change clusters

3. **Maximum Iterations**: Prevent infinite loops

4. **Objective Function**: WCSS improvement falls below threshold
   ```
   |J(t) - J(t+1)| < ε
   ```

### Typical Convergence Behavior:
- Usually converges within 10-20 iterations
- May get stuck in local minima
- Different initializations can yield different results

## Choosing K

### 1. Elbow Method
- Plot WCSS vs number of clusters (k)
- Look for "elbow" point where improvement slows
- Mathematical indicator: rate of change in WCSS decreases significantly

```python
# Calculate WCSS for different k values
wcss_values = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss_values.append(kmeans.inertia_)
```

### 2. Silhouette Analysis
- Measures how similar points are within clusters vs other clusters
- Silhouette coefficient ranges from -1 to 1
- Higher average silhouette score indicates better clustering

### 3. Gap Statistic
- Compares within-cluster dispersion to expected dispersion
- Choose k where gap statistic is maximized

## Practical Considerations

### Initialization Strategies

1. **Random Initialization**: Simple but can lead to poor results
2. **K-means++**: Smart initialization that spreads initial centroids
   - Choose first centroid randomly
   - Choose subsequent centroids with probability proportional to squared distance

### Data Preprocessing

1. **Scaling**: Normalize features to prevent dominance by large-scale features
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   scaled_data = scaler.fit_transform(data)
   ```

2. **Handling Outliers**: Remove or treat outliers as they can skew centroids

3. **Feature Selection**: Include only relevant features

### Common Issues and Solutions

| Issue | Problem | Solution |
|-------|---------|----------|
| Empty Clusters | Centroid has no assigned points | Re-initialize empty centroids |
| Local Minima | Algorithm stuck in suboptimal solution | Multiple random initializations |
| Unequal Cluster Sizes | Natural clusters have different sizes | Consider alternatives like GMM |
| Non-spherical Clusters | K-means assumes spherical clusters | Use spectral clustering or DBSCAN |

## Advantages and Disadvantages

### Advantages ✅
- Simple and intuitive algorithm
- Computationally efficient: O(tkn) where t=iterations, k=clusters, n=points
- Works well with spherical, well-separated clusters
- Guaranteed to converge
- Scales well to large datasets

### Disadvantages ❌
- Must specify k in advance
- Sensitive to initialization
- Assumes spherical clusters of similar size
- Sensitive to outliers
- Struggles with non-linear cluster boundaries
- Can be affected by curse of dimensionality

## Code Implementation

### Using Scikit-learn

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Get centroids
centroids = kmeans.cluster_centers_

# Plot results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red')
plt.title('K-means Clustering Results')
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')
plt.colorbar(scatter)
plt.show()

print(f'Inertia (WCSS): {kmeans.inertia_:.2f}')
print(f'Number of iterations: {kmeans.n_iter_}')
```

### Manual Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist

class KMeansManual:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for i in range(self.max_iters):
            # Assign points to nearest centroid
            distances = cdist(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([
                X[self.labels == j].mean(axis=0) if len(X[self.labels == j]) > 0 
                else self.centroids[j] for j in range(self.k)
            ])
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                print(f'Converged after {i+1} iterations')
                break
                
            self.centroids = new_centroids
        
        # Calculate inertia
        self.inertia_ = sum(
            np.sum((X[self.labels == j] - self.centroids[j])**2)
            for j in range(self.k)
        )
        
        return self
    
    def predict(self, X):
        distances = cdist(X, self.centroids)
        return np.argmin(distances, axis=1)
```

### Finding Optimal K with Elbow Method

```python
def find_optimal_k(X, max_k=10):
    wcss = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    return wcss

# Usage
wcss_values = find_optimal_k(X_scaled)
```

## Real-world Applications

### 1. Customer Segmentation
- **Use Case**: Group customers based on purchasing behavior
- **Features**: Purchase frequency, amount spent, product categories
- **Business Value**: Targeted marketing campaigns

### 2. Image Compression
- **Use Case**: Reduce color palette in images
- **Process**: Cluster pixel colors, replace with centroid colors
- **Result**: Smaller file sizes with acceptable quality loss

### 3. Market Research
- **Use Case**: Segment survey responses
- **Features**: Demographic data, preferences, behaviors
- **Output**: Distinct customer personas

### 4. Gene Expression Analysis
- **Use Case**: Group genes with similar expression patterns
- **Features**: Expression levels across different conditions
- **Application**: Drug discovery and disease research

### 5. Anomaly Detection
- **Use Case**: Identify outliers in network traffic
- **Method**: Points far from all centroids may be anomalous
- **Application**: Cybersecurity and fraud detection

## Performance Tips

1. **Use K-means++** initialization for better results
2. **Run multiple times** with different random seeds
3. **Preprocess data** appropriately (scaling, outlier removal)
4. **Consider mini-batch K-means** for large datasets
5. **Use parallel processing** when available
6. **Monitor convergence** to avoid unnecessary iterations

## When NOT to Use K-means

- Non-spherical cluster shapes
- Clusters of very different sizes
- Clusters of very different densities
- Presence of noise and outliers
- Unknown or varying number of clusters
- High-dimensional data with curse of dimensionality

## Summary

K-means is a fundamental clustering algorithm that works well for many applications, particularly when:
- Clusters are roughly spherical
- Clusters are of similar size
- The number of clusters is known or can be estimated
- Data is properly preprocessed

Despite its limitations, k-means remains popular due to its simplicity, efficiency, and interpretability. Understanding its mathematical foundation and practical considerations is essential for successful application in real-world scenarios.

---

*Next: [Hierarchical Clustering](hierarchical_clustering.md)*
