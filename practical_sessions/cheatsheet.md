# Scikit-Learn Cheat Sheet

A comprehensive reference guide for commonly used machine learning algorithms in scikit-learn.

## Table of Contents
1. [Linear Regression](#linear-regression)
2. [Logistic Regression](#logistic-regression)
3. [Decision Trees](#decision-trees)
4. [Support Vector Machines (SVM)](#support-vector-machines-svm)
5. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
6. [Clustering](#clustering)
7. [General Workflow](#general-workflow)
8. [Useful Links](#useful-links)

---

## Linear Regression

### Import
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### Basic Usage
```python
# Create model
model = LinearRegression()

# Fit model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Key Parameters
- `fit_intercept=True`: Whether to calculate intercept
- `normalize=False`: Whether to normalize features
- `copy_X=True`: Whether to copy X

### Practical Tips
- Scale features for better performance
- Check for multicollinearity
- Use regularization (Ridge/Lasso) for high-dimensional data

---

## Logistic Regression

### Import
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Basic Usage
```python
# Create model
model = LogisticRegression()

# Fit model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Key Parameters
- `C=1.0`: Regularization strength (smaller = more regularization)
- `penalty='l2'`: Regularization type ('l1', 'l2', 'elasticnet')
- `solver='lbfgs'`: Algorithm to use
- `max_iter=100`: Maximum iterations
- `random_state=None`: Random seed

### Practical Tips
- Use `class_weight='balanced'` for imbalanced datasets
- Try different solvers for different penalty types
- Scale features for better convergence

---

## Decision Trees

### Import
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
```

### Basic Usage
```python
# Create model
model = DecisionTreeClassifier()

# Fit model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Visualize tree
plot_tree(model, filled=True, feature_names=feature_names)
```

### Key Parameters
- `criterion='gini'`: Split quality measure ('gini', 'entropy')
- `max_depth=None`: Maximum tree depth
- `min_samples_split=2`: Minimum samples to split
- `min_samples_leaf=1`: Minimum samples in leaf
- `max_features=None`: Number of features to consider
- `random_state=None`: Random seed

### Practical Tips
- Prune trees to avoid overfitting
- Use ensemble methods (Random Forest, Gradient Boosting)
- Feature importance: `model.feature_importances_`

---

## Support Vector Machines (SVM)

### Import
```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
```

### Basic Usage
```python
# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
model = SVC()

# Fit model
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
```

### Key Parameters
- `C=1.0`: Regularization parameter
- `kernel='rbf'`: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
- `gamma='scale'`: Kernel coefficient
- `degree=3`: Polynomial kernel degree
- `probability=False`: Enable probability estimates

### Practical Tips
- Always scale features
- Use GridSearchCV for hyperparameter tuning
- Linear kernel for high-dimensional data
- RBF kernel for non-linear problems

---

## K-Nearest Neighbors (KNN)

### Import
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
```

### Basic Usage
```python
# Scale features (recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
model = KNeighborsClassifier(n_neighbors=5)

# Fit model
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
```

### Key Parameters
- `n_neighbors=5`: Number of neighbors
- `weights='uniform'`: Weight function ('uniform', 'distance')
- `algorithm='auto'`: Algorithm to compute neighbors
- `metric='minkowski'`: Distance metric
- `p=2`: Power parameter for Minkowski metric

### Practical Tips
- Choose odd k to avoid ties in classification
- Use cross-validation to find optimal k
- Scale features for distance-based algorithms
- Consider curse of dimensionality

---

## Clustering

### K-Means
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create model
model = KMeans(n_clusters=3, random_state=42)

# Fit and predict
cluster_labels = model.fit_predict(X)

# Evaluate
silhouette_avg = silhouette_score(X, cluster_labels)
```

### DBSCAN
```python
from sklearn.cluster import DBSCAN

# Create model
model = DBSCAN(eps=0.5, min_samples=5)

# Fit and predict
cluster_labels = model.fit_predict(X)
```

### Hierarchical Clustering
```python
from sklearn.cluster import AgglomerativeClustering

# Create model
model = AgglomerativeClustering(n_clusters=3)

# Fit and predict
cluster_labels = model.fit_predict(X)
```

### Key Parameters
**K-Means:**
- `n_clusters=8`: Number of clusters
- `init='k-means++'`: Initialization method
- `n_init=10`: Number of random initializations
- `max_iter=300`: Maximum iterations

**DBSCAN:**
- `eps=0.5`: Maximum distance between samples
- `min_samples=5`: Minimum samples in neighborhood

### Practical Tips
- Use elbow method or silhouette score to choose k
- Scale features before clustering
- DBSCAN finds arbitrary shaped clusters
- Hierarchical clustering good for dendrograms

---

## General Workflow

### 1. Data Preparation
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Model Training and Evaluation
```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### 3. Model Persistence
```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
loaded_model = joblib.load('model.pkl')
```

---

## Common Evaluation Metrics

### Classification
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Regression
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
```

---

## Useful Links

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Algorithm Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- [User Guide](https://scikit-learn.org/stable/user_guide.html)
- [API Reference](https://scikit-learn.org/stable/modules/classes.html)
- [Examples Gallery](https://scikit-learn.org/stable/auto_examples/index.html)

---

## Quick Tips

1. **Always split your data** before any preprocessing
2. **Scale features** for distance-based algorithms (KNN, SVM, clustering)
3. **Use cross-validation** for model evaluation
4. **Handle missing values** before training
5. **Check for data leakage** in preprocessing steps
6. **Set random_state** for reproducible results
7. **Use appropriate metrics** for your problem type
8. **Validate on unseen data** to assess generalization

---

*Happy Machine Learning! 🚀*
