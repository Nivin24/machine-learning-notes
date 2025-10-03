# SVM (Support Vector Machine) Cheatsheet

## Core Scikit-Learn Classes
```python
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.preprocessing import StandardScaler
```

## Common Parameters

### SVC (Classification)
- `C`: Regularization parameter (default=1.0). Smaller = more regularization
- `kernel`: 'linear', 'poly', 'rbf' (default), 'sigmoid', or custom
- `gamma`: Kernel coefficient for 'rbf', 'poly', 'sigmoid' ('scale' or 'auto')
- `degree`: Degree for polynomial kernel (default=3)
- `class_weight`: Handle imbalanced data ('balanced' or dict)
- `probability`: Enable probability estimates (default=False)

### SVR (Regression)
- `epsilon`: Epsilon-tube width (default=0.1)
- `C`, `kernel`, `gamma`: Same as SVC

## Code Pattern

### Classification
```python
# Always scale features for SVM!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic SVM classifier
model = SVC(C=1.0, kernel='rbf', gamma='scale')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# With probability estimates
model = SVC(kernel='rbf', probability=True)
model.fit(X_train_scaled, y_train)
proba = model.predict_proba(X_test_scaled)
```

### Regression
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

## Kernel Selection Guide

- **Linear**: Data is linearly separable, high dimensions, text classification
- **RBF (Radial)**: Default choice, general purpose, non-linear patterns
- **Polynomial**: Specific non-linear relationships, lower degree (2-3) usually better
- **Sigmoid**: Neural network-like behavior, rarely used

## Workflow Tips

1. **Always scale features**: SVM is highly sensitive to feature scales
2. **Start with RBF kernel**: Good default for most problems
3. **Grid search C and gamma**: Use GridSearchCV or RandomizedSearchCV
4. **LinearSVC for large datasets**: Faster than SVC with linear kernel
5. **Check support vectors**: `model.n_support_` shows complexity

## Scaling Notes

⚠️ **CRITICAL**: Feature scaling is mandatory for SVM!

```python
# StandardScaler is most common
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# MinMaxScaler alternative
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Always fit on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform!
```

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")
best_model = grid.best_estimator_
```

## Practical Guidance

### When to Use SVM
✅ Small to medium datasets (< 10,000 samples)
✅ High-dimensional data
✅ Clear margin of separation exists
✅ Binary or multi-class classification
✅ Need robust model with good generalization

### When NOT to Use SVM
❌ Very large datasets (slow training)
❌ Data has lots of noise
❌ Need probability estimates (requires extra computation)
❌ Interpretability is critical

### Common Pitfalls
- Forgetting to scale features → poor performance
- Using RBF with high C and gamma → overfitting
- Not setting `probability=True` when needed
- Using SVC instead of LinearSVC for linear problems

## Advanced Usage

### Handle Imbalanced Data
```python
model = SVC(class_weight='balanced')  # Auto-adjust weights
# Or manually:
model = SVC(class_weight={0: 1, 1: 10})  # Weight class 1 more
```

### Custom Kernel
```python
from sklearn.metrics.pairwise import rbf_kernel

def custom_kernel(X, Y):
    return rbf_kernel(X, Y, gamma=0.1)

model = SVC(kernel=custom_kernel)
```

### Multi-class Strategy
```python
# One-vs-Rest (default)
model = SVC(decision_function_shape='ovr')

# One-vs-One
model = SVC(decision_function_shape='ovo')
```

### Get Support Vectors
```python
model.fit(X_train_scaled, y_train)
print(f"Number of support vectors: {model.n_support_}")
print(f"Support vectors: {model.support_vectors_}")
print(f"Support vector indices: {model.support_}")
```

### Decision Function
```python
# Get decision scores instead of labels
scores = model.decision_function(X_test_scaled)
# Higher absolute value = more confident
```

## Performance Tips

1. **Use LinearSVC for linear kernels**: 10-100x faster
   ```python
   from sklearn.svm import LinearSVC
   model = LinearSVC(C=1.0, max_iter=1000)
   ```

2. **Reduce training size**: SVM doesn't scale well
3. **Cache kernel computations**: `cache_size` parameter (MB)
4. **Shrinking heuristic**: Enabled by default, speeds up training
5. **Early stopping**: Set `max_iter` to limit iterations

## Quick Reference

| Task | Model | Key Params |
|------|-------|------------|
| Binary Classification | SVC | C, kernel='rbf', gamma |
| Multi-class Classification | SVC | C, kernel='rbf', decision_function_shape |
| Linear Classification (fast) | LinearSVC | C, max_iter |
| Regression | SVR | C, epsilon, kernel |
| Linear Regression (fast) | LinearSVR | C, epsilon, max_iter |
| Imbalanced Data | SVC | class_weight='balanced' |
| Probability Estimates | SVC | probability=True |
