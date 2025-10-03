# Decision Tree Cheatsheet

## Class
- `DecisionTreeClassifier` - for classification problems
- `DecisionTreeRegressor` - for regression problems

## Main Parameters
- `criterion`: Split quality measure
  - Classification: 'gini' (default), 'entropy', 'log_loss'
  - Regression: 'squared_error' (default), 'friedman_mse', 'absolute_error'
- `max_depth`: Maximum tree depth (None = unlimited)
- `min_samples_split`: Minimum samples required to split (2 default)
- `min_samples_leaf`: Minimum samples required at leaf node (1 default)
- `max_features`: Features to consider for best split
  - 'sqrt', 'log2', int, float, or None
- `random_state`: Controls randomness for reproducible results

## Code Example
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import tree

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")

# Feature importance
print("Feature importance:", clf.feature_importances_)
```

## Typical Workflow
1. **Data Preparation**: Clean and split data
2. **Model Creation**: Set hyperparameters
3. **Training**: Fit model to training data
4. **Evaluation**: Test on validation/test set
5. **Hyperparameter Tuning**: Use GridSearchCV/RandomizedSearchCV
6. **Final Model**: Train on full dataset if satisfied

## Visual Tips
```python
# Visualize the tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'])
plt.show()

# Export tree as text
from sklearn.tree import export_text
tree_rules = export_text(clf, feature_names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'])
print(tree_rules)
```

## Overfitting Handling
- **Pre-pruning**: Set max_depth, min_samples_split, min_samples_leaf
- **Post-pruning**: Use cost_complexity_pruning_path and ccp_alpha
- **Cross-validation**: Find optimal hyperparameters
- **Ensemble methods**: Use Random Forest or Gradient Boosting

```python
# Cost complexity pruning example
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Find optimal alpha using cross-validation
from sklearn.model_selection import cross_val_score
scores = []
for alpha in ccp_alphas:
    clf_pruned = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    score = cross_val_score(clf_pruned, X_train, y_train, cv=5, scoring='accuracy')
    scores.append(score.mean())

best_alpha = ccp_alphas[scores.index(max(scores))]
```

## Scikit-learn Specifics
- **No preprocessing needed**: Can handle mixed data types
- **Feature scaling not required**: Tree-based splits are not affected by scale
- **Handles missing values**: Use SimpleImputer if needed
- **Feature importance**: Available via `feature_importances_` attribute
- **Tree structure**: Access via `tree_` attribute for detailed analysis
- **Prediction path**: Use `decision_path()` to see which nodes were used

## Common Gotchas
- Trees can easily overfit - always validate performance
- Single trees are unstable - small data changes can create very different trees
- Biased toward features with more levels
- No built-in feature scaling - categorical encoding may be needed
