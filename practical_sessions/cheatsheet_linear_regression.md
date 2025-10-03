# Linear Regression Cheatsheet

## Core Scikit-Learn Classes
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
```

## Common Parameters

### LinearRegression
- `fit_intercept`: Whether to calculate intercept (default=True)
- `n_jobs`: Parallel processing (-1 = all cores)

### Ridge (L2 Regularization)
- `alpha`: Regularization strength (default=1.0). Higher = more regularization
- `fit_intercept`: Calculate intercept (default=True)
- `solver`: 'auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'

### Lasso (L1 Regularization)
- `alpha`: Regularization strength (default=1.0)
- `max_iter`: Maximum iterations (default=1000)
- `selection`: 'cyclic' or 'random' for coordinate descent

### ElasticNet (L1 + L2)
- `alpha`: Overall regularization strength
- `l1_ratio`: Mix of L1 and L2 (0=Ridge, 1=Lasso, 0.5=equal mix)

## Code Pattern

### Basic Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# R² score
from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(y_test, y_pred)}")
```

### Ridge Regression (L2)
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Lasso Regression (L1 - Feature Selection)
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1, max_iter=10000)
model.fit(X_train, y_train)

# Get selected features (non-zero coefficients)
selected = X_train.columns[model.coef_ != 0]
print(f"Selected features: {selected.tolist()}")
```

### ElasticNet
```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
model.fit(X_train, y_train)
```

## Workflow Tips

1. **Start with basic LinearRegression**: Baseline model
2. **Check for overfitting**: If train R² >> test R², add regularization
3. **Use Ridge for correlated features**: Handles multicollinearity
4. **Use Lasso for feature selection**: Drives coefficients to zero
5. **Scale features for regularization**: Ridge/Lasso sensitive to scale
6. **Polynomial features**: For non-linear relationships

## Scaling Notes

⚠️ **Required for regularized models (Ridge, Lasso, ElasticNet)**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Then fit regularized model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)
```

**Why scaling matters**: Regularization penalizes large coefficients. Without scaling, features with larger ranges get unfairly penalized.

## Regularization Selection

### When to Use Each

**LinearRegression (No Regularization)**
- Small datasets
- Low-dimensional data
- No multicollinearity
- All features are important

**Ridge (L2)**
- Multicollinearity present
- Want to keep all features
- Prevent overfitting
- Coefficients shrink toward zero but never exactly zero

**Lasso (L1)**
- Feature selection needed
- Many irrelevant features
- Sparse solution desired
- Some coefficients become exactly zero

**ElasticNet (L1 + L2)**
- Multicollinearity + feature selection
- Groups of correlated features
- Best of both worlds

## Tuning Alpha (Regularization Strength)

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(Ridge(), param_grid, cv=5, 
                    scoring='neg_mean_squared_error')
grid.fit(X_train_scaled, y_train)

print(f"Best alpha: {grid.best_params_['alpha']}")
print(f"Best score: {-grid.best_score_}")
```

## Practical Guidance

### When to Use Linear Regression
✅ Linear relationship between features and target
✅ Continuous target variable
✅ Need interpretable model
✅ Fast training and prediction
✅ Baseline model
✅ Feature importance analysis

### When NOT to Use
❌ Non-linear relationships (use polynomial or other models)
❌ Outliers heavily present (consider robust regression)
❌ Categorical target (use classification)
❌ Perfect multicollinearity

### Common Pitfalls
- Not scaling features with regularization → biased coefficients
- Using LinearRegression with multicollinearity → unstable coefficients
- Too high alpha → underfitting
- Too low alpha → overfitting
- Ignoring feature scaling differences

## Advanced Usage

### Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create pipeline
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Get Feature Importance
```python
import numpy as np
import pandas as pd

model.fit(X_train, y_train)

# Absolute coefficients as importance
feature_importance = np.abs(model.coef_)

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

model = Ridge(alpha=1.0)
scores = cross_val_score(model, X_train_scaled, y_train, 
                        cv=5, scoring='r2')

print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Residual Analysis
```python
import matplotlib.pyplot as plt

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Check for patterns - ideally random scatter around 0
```

### Handle Outliers with Robust Regression
```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor

# Huber Regressor
huber = HuberRegressor(epsilon=1.35)
huber.fit(X_train, y_train)

# RANSAC (Random Sample Consensus)
ransac = RANSACRegressor(random_state=42)
ransac.fit(X_train, y_train)
```

### Regularization Path
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

alphas = np.logspace(-6, 6, 200)
coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)

plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge Regularization Path')
plt.legend(X_train.columns)
plt.show()
```

## Model Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test_scaled)

# R² Score (coefficient of determination)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.3f}")  # 1 = perfect, 0 = baseline

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.3f}")

# Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.3f}")

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.3f}")

# Adjusted R² (for multiple features)
n = len(y_test)
k = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print(f"Adjusted R²: {adj_r2:.3f}")
```

## Quick Reference

| Model | Use Case | Alpha | Scaling |
|-------|----------|-------|----------|
| LinearRegression | Baseline, no regularization | N/A | Optional |
| Ridge | Multicollinearity | 0.1-100 | Required |
| Lasso | Feature selection | 0.01-10 | Required |
| ElasticNet | Both above | 0.1-10, l1_ratio=0.5 | Required |
| HuberRegressor | Outliers present | N/A | Optional |
| RANSACRegressor | Heavy outliers | N/A | Optional |

## Alpha Guidelines

| Alpha Value | Effect | Use When |
|------------|--------|----------|
| 0 | No regularization | Same as LinearRegression |
| 0.001-0.1 | Light regularization | Slight overfitting |
| 1-10 | Moderate regularization | Moderate overfitting |
| 100-1000 | Strong regularization | Heavy overfitting |

## Complete Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Tune alpha with cross-validation
param_grid = {'alpha': np.logspace(-3, 3, 20)}
grid = GridSearchCV(Ridge(), param_grid, cv=5, 
                    scoring='neg_mean_squared_error')
grid.fit(X_train_scaled, y_train)

print(f"Best alpha: {grid.best_params_['alpha']:.4f}")

# 4. Train final model
model = Ridge(alpha=grid.best_params_['alpha'])
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print(f"Train R²: {r2_score(y_train, y_pred_train):.3f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")

# 6. Feature importance
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.3f}")
```
