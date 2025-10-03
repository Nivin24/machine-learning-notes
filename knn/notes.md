# K-Nearest Neighbors (KNN) - Complete Review Notes

## Core Principles

### What is KNN?
- **Instance-based learning algorithm** (lazy learner)
- Non-parametric: Makes no assumptions about data distribution
- Stores all training data and makes predictions based on similarity
- Classification: Majority vote of k nearest neighbors
- Regression: Mean/median of k nearest neighbors

### How KNN Works
1. Choose the number k of neighbors
2. Calculate distance between query point and all training samples
3. Sort distances and select k nearest neighbors
4. For classification: Use majority voting
5. For regression: Use average of neighbors

## Distance Metrics

### Common Distance Measures
```python
# Euclidean Distance (L2)
import numpy as np
euclidean = np.sqrt(np.sum((x1 - x2)**2))

# Manhattan Distance (L1)
manhattan = np.sum(np.abs(x1 - x2))

# Minkowski Distance (generalized)
minkowski = np.sum(np.abs(x1 - x2)**p)**(1/p)

# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(x1.reshape(1,-1), x2.reshape(1,-1))
```

### When to Use Which?
- **Euclidean**: Continuous features, assumes equal importance
- **Manhattan**: High-dimensional data, less sensitive to outliers
- **Cosine**: Text data, direction matters more than magnitude
- **Hamming**: Categorical features

## Feature Scaling - CRITICAL!

### Why Feature Scaling is Essential
- KNN is **distance-based** - features on larger scales dominate
- Without scaling: Feature with range [0-1000] overpowers [0-1]
- **Always scale features** before applying KNN

### Scaling Techniques
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

⚠️ **Important**: Fit scaler on training data only, then transform both train and test!

## Choosing K

### Effect of K Value
- **Small k (k=1)**: 
  - High variance, low bias
  - Overfitting, sensitive to noise
  - Complex decision boundaries
  
- **Large k**:
  - Low variance, high bias
  - Underfitting, smoother boundaries
  - More computational cost

### Optimal K Selection
```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Test different k values
k_range = range(1, 31)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    scores.append(score.mean())

# Plot and select k with best cross-validation score
import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()
```

**Rule of thumb**: k = sqrt(n), but always validate with cross-validation

## Bias-Variance Tradeoff

- **High Bias (Large k)**: Underfits, misses patterns
- **High Variance (Small k)**: Overfits, captures noise
- **Sweet Spot**: Use cross-validation to find optimal k
- KNN generally has **low bias, high variance**

## Complete Implementation

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CRUCIAL: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
knn.fit(X_train_scaled, y_train)

# Predict
y_pred = knn.predict(X_test_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Get probabilities
proba = knn.predict_proba(X_test_scaled)
```

### Weighted KNN
```python
# Distance-weighted voting (closer neighbors have more influence)
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_weighted.fit(X_train_scaled, y_train)
```

## Pros and Cons

### Advantages ✓
- Simple to understand and implement
- No training phase (lazy learning)
- Naturally handles multi-class problems
- Works well with small datasets
- Non-parametric (no distribution assumptions)
- Can capture complex decision boundaries

### Disadvantages ✗
- **Computationally expensive** at prediction time: O(nd) for n samples, d dimensions
- **Memory intensive**: Stores entire training dataset
- **Curse of dimensionality**: Performance degrades in high dimensions
- **Requires feature scaling**: Essential preprocessing step
- **Sensitive to irrelevant features**: All features affect distance equally
- **Imbalanced data issues**: Majority class dominates
- **No interpretability**: Black box model

## Handling Common Issues

### Curse of Dimensionality
```python
from sklearn.decomposition import PCA

# Dimensionality reduction before KNN
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_train_scaled)
```

### Imbalanced Classes
```python
# Use weighted KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Or use class_weight in preprocessing
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### Speed Optimization
```python
# Use Ball Tree or KD Tree for faster neighbor search
knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
# Options: 'auto', 'ball_tree', 'kd_tree', 'brute'
```

## Interview Questions

### Conceptual Questions
1. **Why is KNN called a lazy learner?**
   - No explicit training phase; stores all data and computes at prediction time

2. **Why must we scale features for KNN?**
   - KNN uses distance metrics; unscaled features with larger ranges dominate distance calculations

3. **What happens with k=1 vs k=n?**
   - k=1: High variance, overfits, decision boundary follows training points
   - k=n: Predicts majority class for all points, extreme underfitting

4. **How does KNN handle multi-class classification?**
   - Naturally multi-class through majority voting of k neighbors

5. **What is the curse of dimensionality in KNN?**
   - In high dimensions, all points become equidistant, making nearest neighbors meaningless

6. **Time complexity of KNN?**
   - Training: O(1) - no training
   - Prediction: O(nd) - compute distance to all n points in d dimensions
   - With KD-tree: O(d log n) average case

### Practical Questions
7. **How to choose optimal k?**
   - Use cross-validation, plot accuracy vs k, choose k with best CV score
   - Start with k = sqrt(n), use odd k for binary classification

8. **When to use uniform vs distance weights?**
   - Uniform: All neighbors vote equally
   - Distance: Closer neighbors have more influence; better for varying density data

9. **When NOT to use KNN?**
   - Very large datasets (slow prediction)
   - High-dimensional data (curse of dimensionality)
   - Real-time predictions needed
   - When model interpretability is required

10. **How does KNN compare to Naive Bayes?**
    - KNN: Non-parametric, instance-based, slow prediction, no assumptions
    - NB: Parametric, probability-based, fast, assumes feature independence

## Exam Quick Tips

### Must Remember
- ✓ **Feature scaling is mandatory**
- ✓ Small k → high variance, large k → high bias
- ✓ O(1) training, O(nd) prediction complexity
- ✓ Distance metrics: Euclidean most common, Manhattan for high-dim
- ✓ Odd k for binary classification (avoids ties)
- ✓ Instance-based, lazy learner, non-parametric

### Common Mistakes to Avoid
- ✗ Forgetting to scale features
- ✗ Using same scaler fit on test data
- ✗ Not handling imbalanced classes
- ✗ Choosing k without cross-validation
- ✗ Using KNN on high-dimensional data without dimensionality reduction

## Practical Usage Tips

### When to Use KNN
- Small to medium-sized datasets
- Low-dimensional feature space (< 20 features)
- Complex decision boundaries needed
- No time constraints on prediction
- Baseline model for comparison

### Production Considerations
```python
# Save trained model (though it just stores data)
import joblib
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # SAVE SCALER TOO!

# Load and predict
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
X_new_scaled = scaler.transform(X_new)
predictions = knn.predict(X_new_scaled)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

## Summary Cheat Sheet

| Aspect | Details |
|--------|--------|
| **Type** | Supervised, instance-based, non-parametric |
| **Use Cases** | Classification, regression |
| **Training Time** | O(1) - instant |
| **Prediction Time** | O(nd) - slow |
| **Key Hyperparameter** | k (number of neighbors) |
| **Preprocessing** | **Feature scaling required** |
| **Strengths** | Simple, no assumptions, multi-class |
| **Weaknesses** | Slow, memory-intensive, curse of dimensionality |
| **Best For** | Small datasets, low dimensions |
| **Avoid For** | Large datasets, high dimensions, real-time |

---

**Last Updated**: October 2025  
**Recommended Review Time**: 30-45 minutes before exam/interview
