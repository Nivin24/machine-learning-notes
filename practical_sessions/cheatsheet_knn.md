# K-Nearest Neighbors (KNN) Cheatsheet

## Core Scikit-Learn Classes
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.preprocessing import StandardScaler
```

## Common Parameters

### KNeighborsClassifier / KNeighborsRegressor
- `n_neighbors`: Number of neighbors (default=5). Must be tuned!
- `weights`: 'uniform' or 'distance' (weight by inverse distance)
- `algorithm`: 'auto', 'ball_tree', 'kd_tree', 'brute'
- `metric`: Distance metric ('euclidean', 'manhattan', 'minkowski', etc.)
- `p`: Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
- `n_jobs`: Number of parallel jobs (-1 = use all cores)

## Code Pattern

### Classification
```python
# Scale features - KNN is distance-based!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic KNN classifier
model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# With distance weighting
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
model.fit(X_train_scaled, y_train)

# Get probabilities
proba = model.predict_proba(X_test_scaled)
```

### Regression
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsRegressor(n_neighbors=5, weights='distance')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

## Workflow Tips

1. **Always scale features**: KNN uses distances - unscaled features dominate
2. **Tune n_neighbors**: Use cross-validation (typically odd numbers for binary classification)
3. **Try distance weighting**: `weights='distance'` often improves performance
4. **Start with k=5**: Good default, then tune
5. **Consider odd k for binary**: Avoids ties in voting
6. **Use all cores**: Set `n_jobs=-1` for faster predictions

## Scaling Notes

⚠️ **CRITICAL**: Feature scaling is mandatory for KNN!

```python
# StandardScaler recommended
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# MinMaxScaler also works well
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Always fit on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why scaling matters**: A feature with range [0, 1000] will dominate over a feature with range [0, 1] when calculating distances.

## Finding Optimal K

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

print(f"Best k: {grid.best_params_['n_neighbors']}")
print(f"Best score: {grid.best_score_}")
```

### Elbow Method
```python
from sklearn.model_selection import cross_val_score
import numpy as np

k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    scores.append(score.mean())

# Plot and find elbow
import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('K')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

optimal_k = k_range[np.argmax(scores)]
```

## Practical Guidance

### When to Use KNN
✅ Small to medium datasets (KNN stores all training data)
✅ Low-dimensional data (< 20 features)
✅ Need simple, interpretable model
✅ Non-linear decision boundaries
✅ Real-time learning (no training phase)
✅ Multi-class classification

### When NOT to Use KNN
❌ Very large datasets (slow predictions)
❌ High-dimensional data (curse of dimensionality)
❌ Need fast prediction time
❌ Memory constraints (stores all training data)
❌ Data has many irrelevant features

### Common Pitfalls
- Forgetting to scale features → poor performance
- Using k=1 → overfitting, sensitive to noise
- Using too large k → underfitting
- Not handling imbalanced data
- Ignoring computational cost for large datasets

## Distance Metrics

```python
# Euclidean (default, L2)
model = KNeighborsClassifier(metric='euclidean')  # or p=2

# Manhattan (L1)
model = KNeighborsClassifier(metric='manhattan')  # or p=1

# Minkowski (generalized)
model = KNeighborsClassifier(metric='minkowski', p=3)

# Chebyshev
model = KNeighborsClassifier(metric='chebyshev')

# Custom distance function
def custom_distance(x, y):
    return np.sum(np.abs(x - y))

model = KNeighborsClassifier(metric=custom_distance)
```

## Advanced Usage

### Get Nearest Neighbors
```python
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Find k nearest neighbors
distances, indices = model.kneighbors(X_test_scaled)

print(f"Distances to 5 nearest neighbors: {distances[0]}")
print(f"Indices of 5 nearest neighbors: {indices[0]}")

# Get neighbors for specific k
distances, indices = model.kneighbors(X_test_scaled, n_neighbors=3)
```

### Radius-Based Neighbors
```python
# Instead of k neighbors, use all within radius
from sklearn.neighbors import RadiusNeighborsClassifier

model = RadiusNeighborsClassifier(radius=1.0, weights='distance')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

### Handle Imbalanced Data
```python
# Use distance weighting
model = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Or combine with resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
model.fit(X_resampled, y_resampled)
```

### Efficient Algorithms
```python
# Ball Tree: efficient for high-dimensional data
model = KNeighborsClassifier(algorithm='ball_tree', leaf_size=30)

# KD Tree: efficient for low-dimensional data
model = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30)

# Brute force: simple, good for small datasets
model = KNeighborsClassifier(algorithm='brute')

# Auto: automatically chooses best algorithm
model = KNeighborsClassifier(algorithm='auto')  # default
```

## Performance Tips

1. **Reduce dimensionality**: Use PCA before KNN
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=10)
   X_train_pca = pca.fit_transform(X_train_scaled)
   X_test_pca = pca.transform(X_test_scaled)
   ```

2. **Use appropriate algorithm**: KD-tree for low dimensions, Ball-tree for high
3. **Parallel processing**: Set `n_jobs=-1`
4. **Distance weighting**: Usually better than uniform
5. **Feature selection**: Remove irrelevant features

## K Selection Guidelines

| Dataset Size | Suggested K Range | Notes |
|-------------|------------------|-------|
| < 100 | 3-7 | Small k, avoid overfitting |
| 100-1000 | 5-15 | Standard range |
| > 1000 | 10-50 | Larger k for stability |

**Rule of thumb**: Start with k = √(n) where n is number of training samples

## Quick Reference

| Task | Model | Key Params |
|------|-------|------------|
| Binary Classification | KNeighborsClassifier | n_neighbors=5, weights='distance' |
| Multi-class Classification | KNeighborsClassifier | n_neighbors=5, weights='uniform' |
| Regression | KNeighborsRegressor | n_neighbors=5, weights='distance' |
| Fast Prediction (small data) | KNeighborsClassifier | algorithm='brute', n_jobs=-1 |
| High Dimensions | KNeighborsClassifier | algorithm='ball_tree' |
| Low Dimensions | KNeighborsClassifier | algorithm='kd_tree' |
| Imbalanced Data | KNeighborsClassifier | weights='distance' |
| Variable Density | RadiusNeighborsClassifier | radius=1.0 |

## Decision Flow

```
1. Scale features → ALWAYS
2. Start with k=5, weights='distance'
3. Cross-validate different k values (odd for binary)
4. Try different distance metrics if needed
5. For large data: reduce dimensions or use approximate methods
6. For real-time: consider faster alternatives
```
