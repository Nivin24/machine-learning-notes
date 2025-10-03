# Overfitting and Pruning in Decision Trees

## Overfitting in Trees: Why it Happens
- Trees greedily split to maximize impurity reduction, which can chase noise.
- Deep trees partition data into tiny regions, memorizing idiosyncrasies → high variance.
- Signals of overfitting: training accuracy ≫ validation accuracy; many leaves with very few samples; complex, deep paths.

Bias–Variance view:
- Shallow tree: high bias, low variance.
- Deep tree: low bias, high variance.
- Goal: find a sweet spot (optimal complexity).

## Pre-pruning (Early Stopping) Strategies
Stop tree growth before it perfectly fits training data.
- max_depth: Limit tree depth.
- min_samples_split: Minimum samples to split an internal node.
- min_samples_leaf: Minimum samples required at a leaf (strong regularizer).
- max_leaf_nodes: Cap number of leaves.
- min_impurity_decrease: Require a minimum impurity reduction for a split.
- max_features: Randomly restrict features considered per split (adds stochasticity).

Practical guidance:
- Always set min_samples_leaf > 1 to combat overfitting (e.g., 5–20 for medium datasets).
- Use cross-validation to choose hyperparameters.
- For imbalanced data, prefer min_samples_leaf over max_depth.

## Post-pruning (Prune After Full Growth)
Grow a large tree, then prune back based on validation performance or a complexity penalty.

### Cost-Complexity Pruning (CART)
Define for a tree T with leaves |T|:
R(T) = empirical error of T (e.g., misclassification rate or MSE)
|T| = number of terminal nodes
Cost(T; α) = R(T) + α |T|

- For a given α ≥ 0, find subtree T_α that minimizes Cost(T; α).
- As α increases, optimal subtree has fewer leaves (simpler model).
- The algorithm computes an “optimal pruning sequence” of subtrees by successively collapsing “weakest links.”

In scikit-learn:
- ccp_alpha controls pruning (Cost-Complexity Pruning α). Larger ccp_alpha → more pruning.
- Use DecisionTreeClassifier(..., ccp_alpha=α) and tune α via cross-validation.

### Reduced-Error Pruning (REP)
- Split data into train/validation.
- Iteratively prune subtrees if pruning does not decrease validation accuracy.
- Simple, but requires holdout set and can be greedy.

### Minimal Cost-Complexity Pruning Details
For each internal node t with subtree T_t, define:
α_t = (R(t) − R(T_t)) / (|T_t| − 1)
Collapse the subtree with the smallest α_t; repeat to produce sequence of subtrees with nondecreasing α.

## Visual Examples (ASCII)
Before pruning (overfit):
```
Root
├─ Feature A ≤ 3.2
│  ├─ Feature B ≤ 10.5 → Class 0 (3 samples)
│  └─ Feature B > 10.5
│     ├─ Feature C ∈ {x,y} → Class 1 (1 sample)
│     └─ Feature C ∈ {z}   → Class 0 (2 samples)
└─ Feature A > 3.2 → Class 1 (20 samples)
```
After pruning:
```
Root
├─ Feature A ≤ 3.2 → Class 0
└─ Feature A > 3.2 → Class 1
```

## Practical Playbook
- Start with moderate constraints: max_depth=6–12, min_samples_leaf=5–20.
- Use cross-validation grid over: max_depth, min_samples_leaf, min_samples_split, ccp_alpha.
- Monitor validation curves vs tree size (leaves/depth) to detect overfitting.
- Prefer simpler trees when performances are statistically similar (Occam’s razor).
- For small datasets, use stronger regularization and pruning.
- For large noisy datasets, combine trees into ensembles (RF, GBM) rather than a single deep tree.

## scikit-learn Examples
Classification:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'max_depth': [None, 4, 6, 8, 12],
    'min_samples_leaf': [1, 5, 10, 20],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.05]
}
clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.best_estimator_.predict(X_val)
print('Best params:', clf.best_params_)
print('Val accuracy:', accuracy_score(y_val, y_pred))
```

Regression with pruning:
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=False)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

for alpha in [0.0, 1e-4, 1e-3, 1e-2]:
    reg = DecisionTreeRegressor(random_state=0, ccp_alpha=alpha, min_samples_leaf=5)
    reg.fit(X_tr, y_tr)
    print(alpha, mean_squared_error(y_te, reg.predict(X_te))**0.5)
```

## Summary
- Overfitting is common in decision trees; control complexity via pre- and post-pruning.
- Cost-complexity pruning offers a principled trade-off using ccp_alpha.
- Always validate with cross-validation and prefer simpler models when accuracy is similar.
