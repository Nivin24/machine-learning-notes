# Logistic Regression Cheatsheet

## Core Scikit-Learn Class
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
```

## Common Parameters

- `penalty`: Regularization type ('l1', 'l2' default, 'elasticnet', 'none')
- `C`: Inverse regularization strength (default=1.0). Smaller = stronger regularization
- `solver`: Algorithm ('lbfgs' default, 'liblinear', 'newton-cg', 'sag', 'saga')
- `max_iter`: Maximum iterations (default=100, often needs increase)
- `multi_class`: 'auto', 'ovr' (one-vs-rest), 'multinomial'
- `class_weight`: Handle imbalanced data ('balanced' or dict)
- `random_state`: Seed for reproducibility
- `n_jobs`: Parallel processing (-1 = all cores)

## Code Pattern

### Binary Classification
```python
# Scaling recommended but not mandatory
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Probability estimates
proba = model.predict_proba(X_test_scaled)
print(f"Class 0 prob: {proba[0][0]}, Class 1 prob: {proba[0][1]}")

# Decision function (log-odds)
scores = model.decision_function(X_test_scaled)
```

### Multi-class Classification
```python
# Multinomial logistic regression
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)
model.fit(X_train_scaled, y_train)

# One-vs-Rest (faster for many classes)
model = LogisticRegression(multi_class='ovr', solver='liblinear')
```

## Workflow Tips

1. **Increase max_iter**: Default 100 often insufficient (use 1000+)
2. **Scale features**: Improves convergence, though not strictly required
3. **Choose solver wisely**: 
   - `lbfgs`: Good default for most cases
   - `liblinear`: Good for small datasets, binary problems
   - `saga`: Best for large datasets, supports all penalties
4. **Regularization**: Start with C=1.0, tune via cross-validation
5. **Check convergence**: Watch for convergence warnings

## Scaling Notes

✅ **Recommended for better convergence**

```python
# StandardScaler improves convergence speed
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why scaling helps**: Faster convergence and more stable coefficient estimates, especially with regularization.

## Solver Selection Guide

| Solver | Use Case | Penalties Supported | Multi-class |
|--------|----------|-------------------|-------------|
| `lbfgs` | General purpose, default | L2, None | Multinomial |
| `liblinear` | Small datasets, binary | L1, L2 | OVR only |
| `newton-cg` | Large datasets | L2, None | Multinomial |
| `sag` | Large datasets | L2, None | Multinomial |
| `saga` | Large datasets, L1 penalty | L1, L2, Elasticnet, None | Multinomial |

```python
# L1 regularization (feature selection)
model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)

# L2 regularization (default, prevents overfitting)
model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)

# ElasticNet (L1 + L2)
model = LogisticRegression(penalty='elasticnet', solver='saga', 
                           l1_ratio=0.5, max_iter=1000)
```

## Practical Guidance

### When to Use Logistic Regression
✅ Binary or multi-class classification
✅ Need probability estimates
✅ Need interpretable model (coefficients)
✅ Linear decision boundary expected
✅ Fast training and prediction
✅ Baseline model for comparison

### When NOT to Use
❌ Complex non-linear relationships
❌ Very high-dimensional sparse data (use specialized methods)
❌ Perfect class separation (use regularization or SVM)
❌ Need complex decision boundaries

### Common Pitfalls
- Not increasing max_iter → convergence warnings
- Wrong solver for penalty type → error
- Not scaling features → slow convergence
- Using default C without tuning → suboptimal performance
- Ignoring class imbalance

## Regularization & C Parameter

```python
# C is inverse of regularization strength
# Smaller C = Stronger regularization = Simpler model
# Larger C = Weaker regularization = More complex model

# Strong regularization (simple model)
model = LogisticRegression(C=0.01, max_iter=1000)

# Weak regularization (complex model)
model = LogisticRegression(C=100, max_iter=1000)

# No regularization
model = LogisticRegression(penalty='none', max_iter=1000)
```

### Grid Search for C
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['saga']  # Supports both L1 and L2
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), 
                    param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print(f"Best C: {grid.best_params_['C']}")
print(f"Best penalty: {grid.best_params_['penalty']}")
print(f"Best score: {grid.best_score_}")
```

## Advanced Usage

### Handle Imbalanced Data
```python
# Auto-adjust weights inversely proportional to class frequencies
model = LogisticRegression(class_weight='balanced', max_iter=1000)

# Custom weights
model = LogisticRegression(class_weight={0: 1, 1: 3}, max_iter=1000)
```

### Feature Selection with L1
```python
# L1 penalty drives some coefficients to zero
model = LogisticRegression(penalty='l1', solver='saga', 
                           C=0.1, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Get non-zero coefficients (selected features)
selected_features = X_train.columns[model.coef_[0] != 0]
print(f"Selected features: {selected_features.tolist()}")
```

### Get Coefficients & Intercept
```python
model.fit(X_train_scaled, y_train)

# Coefficients (weights)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Feature importance (absolute coefficients)
import numpy as np
feature_importance = np.abs(model.coef_[0])
top_features = np.argsort(feature_importance)[-5:]  # Top 5
print(f"Top features: {X_train.columns[top_features]}")
```

### Calibrate Probabilities
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate probability estimates
model = LogisticRegression(max_iter=1000)
calibrated = CalibratedClassifierCV(model, cv=5, method='sigmoid')
calibrated.fit(X_train_scaled, y_train)

# More reliable probabilities
proba = calibrated.predict_proba(X_test_scaled)
```

### Warm Start (Incremental Learning)
```python
# Continue training from previous fit
model = LogisticRegression(warm_start=True, max_iter=100)
model.fit(X_train_scaled, y_train)

# Continue training
model.max_iter = 200
model.fit(X_train_scaled, y_train)
```

## Performance Tips

1. **Scale features**: Faster convergence
2. **Use appropriate solver**:
   - `saga` for large datasets
   - `liblinear` for small datasets
3. **Increase max_iter**: Avoid convergence warnings
4. **Parallel processing**: Set `n_jobs=-1`
5. **Warm start**: For incremental learning

## Interpreting Results

### Coefficients
```python
model.fit(X_train_scaled, y_train)

# Positive coefficient = increases probability of class 1
# Negative coefficient = decreases probability of class 1
for feature, coef in zip(X_train.columns, model.coef_[0]):
    print(f"{feature}: {coef:.3f}")

# Odds ratios (exp(coefficient))
import numpy as np
odds_ratios = np.exp(model.coef_[0])
for feature, odds in zip(X_train.columns, odds_ratios):
    print(f"{feature}: {odds:.3f}x odds")
```

## Quick Reference

| Task | Solver | Penalty | Key Params |
|------|--------|---------|------------|
| Binary Classification | lbfgs | L2 | C=1.0, max_iter=1000 |
| Multi-class (small data) | liblinear | L2 | multi_class='ovr' |
| Multi-class (large data) | lbfgs | L2 | multi_class='multinomial' |
| Feature Selection | saga | L1 | C=0.1, max_iter=1000 |
| Imbalanced Data | lbfgs | L2 | class_weight='balanced' |
| Large Dataset | saga | L2 | max_iter=100 |
| No Regularization | lbfgs | none | max_iter=1000 |
| ElasticNet | saga | elasticnet | l1_ratio=0.5 |

## Common Issues & Solutions

### Convergence Warning
```python
# Problem: ConvergenceWarning: lbfgs failed to converge
# Solution 1: Increase iterations
model = LogisticRegression(max_iter=5000)

# Solution 2: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Solution 3: Increase regularization
model = LogisticRegression(C=0.1, max_iter=1000)
```

### Solver Incompatibility
```python
# Problem: liblinear doesn't support multinomial
# Solution: Use lbfgs or saga
model = LogisticRegression(solver='lbfgs', multi_class='multinomial')

# Problem: lbfgs doesn't support L1
# Solution: Use saga or liblinear
model = LogisticRegression(penalty='l1', solver='saga')
```

## Complete Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train model
model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 3. Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# 4. Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 5. Feature importance
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.3f}")
```
