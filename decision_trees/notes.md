# Decision Trees — Quick Reference Notes

## Cheat Sheet
- Algorithms: ID3 (Entropy/IG), C4.5 (Gain Ratio), CART (Gini/MSE, binary splits)
- Criteria: Entropy, Gini, MSE (regression)
- Regularization: max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, ccp_alpha
- Handling numeric features: threshold search via sort + sweep
- Missing values: surrogate splits (CART) or separate category

## Pros
- Interpretable rules and easy visualization
- Handles numerical and categorical data
- Little preprocessing (no scaling needed)
- Captures non-linear relationships and feature interactions
- Fast inference; provides feature importance

## Cons
- Prone to overfitting and high variance
- Unstable to small data changes
- Can create biased splits on imbalanced data
- Many splits needed for linear boundaries

## Tips
- Use min_samples_leaf to reduce overfitting
- Tune max_depth and ccp_alpha via CV
- Balance classes (class_weight, resampling)
- Prefer binary splits for numeric features
- Use ensembles (RandomForest, Gradient Boosting, XGBoost) for performance

## Interview Qs
- Difference between Gini and Entropy? When choose each?
- How does cost-complexity pruning work? Role of ccp_alpha?
- How to handle continuous features and missing values?
- Why are trees high-variance models? How to mitigate?
- Explain Gain Ratio and why IG can be biased.

## Key Libraries
- scikit-learn: DecisionTreeClassifier/Regressor, plot_tree, export_text, export_graphviz
- graphviz: render exported DOT graphs
- dtreeviz: rich tree visualizations

## Mini Tables
Criterion comparison:
- Entropy: info-theoretic, slower (logs), slightly more sensitive to rare classes
- Gini: faster, similar results, default in CART/Sklearn

Regularization knobs:
- Depth control: max_depth, max_leaf_nodes
- Leaf size: min_samples_leaf, min_samples_split
- Split quality: min_impurity_decrease
- Post-pruning: ccp_alpha

## Common Pitfalls
- Overfitting with deep trees and tiny leaves
- Ignoring class imbalance → biased predictions
- Using misclassification error to split (too coarse)
- Not validating pruning strength (α) with CV

## Quick Code Snippets
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = DecisionTreeClassifier(max_depth=None, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
plot_tree(clf, feature_names=features, class_names=classes, filled=True)
```

## Summary
- Start simple, regularize, and visualize.
- Validate with cross-validation and prune.
- For best accuracy and stability, prefer tree ensembles.
