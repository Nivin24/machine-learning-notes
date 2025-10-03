# Gini Impurity and CART Splitting

## Intuition
Gini impurity measures how often a randomly chosen sample from a node would be incorrectly labeled if it were randomly labeled according to the node’s class distribution. Lower Gini means purer node.

## Definition
For class probabilities p1, p2, ..., pk in a node:
Gini(p) = 1 - Σ_i p_i^2 = Σ_i Σ_{j≠i} p_i p_j

Properties:
- 0 ≤ Gini ≤ 1 - 1/k. For binary classes, 0 ≤ Gini ≤ 0.5.
- Gini = 0 for a pure node.
- Smooth, differentiable (unlike entropy’s log) and cheaper to compute.

## Comparison with Entropy
- Both are minimum at purity and maximum near uniform distribution.
- Numerically similar; Gini often prefers larger partitions with a dominant class, entropy can be more sensitive near 0/1 boundaries.
- In practice, tree accuracy is usually similar; Gini is default in CART/scikit-learn for speed.

## CART Splitting Criterion
CART uses impurity decrease (a.k.a. Gini gain):
ΔGini(X, split) = Gini(parent) - [w_L Gini(left) + w_R Gini(right)]
where w_L, w_R are child proportions. Choose the split with maximum ΔGini.

For multi-way splits, CART typically uses binary splits, recursively partitioning.

## Worked Example
At a node: 10 samples, 6 of class A, 4 of class B.
- Gini(parent) = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48
Consider a binary split producing:
- Left: 5 samples, (4 A, 1 B) → Gini_L = 1 - (0.8^2 + 0.2^2) = 0.32
- Right: 5 samples, (2 A, 3 B) → Gini_R = 1 - (0.4^2 + 0.6^2) = 0.48
Weighted child impurity = (5/10)*0.32 + (5/10)*0.48 = 0.40
ΔGini = 0.48 - 0.40 = 0.08

## Handling Numeric Features
- Sort unique values.
- Sweep thresholds; maintain class counts on left/right to update Gini in O(1) per step.
- Choose threshold maximizing ΔGini.

## Multiclass Case
Gini extends directly: sum over all k classes. Fast and stable for many classes.

## Practical Tips
- Use class_weight to mitigate imbalance.
- Combine with regularization: min_samples_leaf, max_depth, min_impurity_decrease.
- Evaluate stability with cross-validation; small data can cause high variance.

## When to Prefer Gini vs Entropy
- Prefer Gini for speed and when you expect one class to dominate after good splits.
- Prefer Entropy if you want a slightly more “information-theoretic” criterion; differences are usually minor.

## References
- Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). Classification and Regression Trees.
- scikit-learn: DecisionTreeClassifier (criterion='gini').
