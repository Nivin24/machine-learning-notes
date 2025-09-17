# Logistic Regression - Quick Reference Notes

A comprehensive guide to logistic regression for binary classification, covering theory, implementation, and practical insights.

## # Core Concepts

### What is Logistic Regression?
- **Purpose**: Binary classification algorithm that predicts probabilities
- **Output**: Continuous values between 0 and 1 (interpreted as probabilities)
- **Decision Rule**: Classify as class 1 if P(y=1|x) > 0.5, otherwise class 0
- **Key Difference from Linear Regression**: Uses sigmoid function instead of linear mapping

### Why Use Logistic Regression?
- Simple and interpretable
- No assumptions about feature distributions
- Fast training and prediction
- Provides probabilistic output
- Good baseline for classification problems

## # Key Formulas

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```
- Maps any real number to (0, 1)
- S-shaped curve with smooth transitions
- Derivative: σ(z) * (1 - σ(z))

### Logistic Regression Model
```
P(y=1|x) = σ(w^T * x + b)
z = w^T * x + b  (linear combination)
```
- w: weight vector
- b: bias term
- x: feature vector

### Cost Function (Binary Cross-Entropy)
```
J(w,b) = -1/N * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
```
- Convex function (guaranteed global minimum)
- Penalizes confident wrong predictions heavily

### Gradient Descent Updates
```
dJ/dw = 1/N * X^T * (ŷ - y)
dJ/db = 1/N * Σ(ŷ - y)
w = w - α * dJ/dw
b = b - α * dJ/db
```
- α: learning rate
- Update weights in direction of steepest descent

## # Implementation Steps

### 1. Data Preparation
```python
# Feature scaling (recommended)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 2. Model Implementation (From Scratch)
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Prevent overflow

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            
            # Compute cost (optional, for monitoring)
            cost = self.compute_cost(y, predictions)
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return [0 if y <= 0.5 else 1 for y in y_pred]
```

### 3. Using Scikit-learn
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
```

## # Tips & Best Practices

### Data Preprocessing
- **Feature Scaling**: Recommended for gradient descent convergence
- **Handle Missing Values**: Impute or remove missing data
- **Feature Engineering**: Create polynomial features if needed
- **Outlier Treatment**: Logistic regression is somewhat robust to outliers

### Hyperparameter Tuning
- **Learning Rate**: Start with 0.01-0.1, adjust based on convergence
- **Iterations**: Monitor cost function to determine when to stop
- **Regularization**: Use L1 (Lasso) or L2 (Ridge) to prevent overfitting

### Model Evaluation
```python
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# ROC Curve and AUC
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### Common Issues & Solutions
- **Convergence Problems**: Reduce learning rate or scale features
- **Overfitting**: Use regularization (C parameter in sklearn)
- **Class Imbalance**: Use class_weight='balanced' in sklearn
- **Perfect Separation**: Add regularization or remove redundant features

## # Interview Q&A

### Theoretical Questions

**Q: Why do we use the sigmoid function in logistic regression?**
A: The sigmoid function maps any real-valued input to a value between 0 and 1, making it perfect for probability estimation. It's also differentiable everywhere, which is essential for gradient-based optimization.

**Q: What's the difference between linear and logistic regression?**
A: Linear regression predicts continuous values and uses MSE loss, while logistic regression predicts probabilities using the sigmoid function and binary cross-entropy loss. Linear regression assumes linear relationship; logistic regression models log-odds linearly.

**Q: Why can't we use MSE as the cost function for logistic regression?**
A: MSE would create a non-convex cost function with multiple local minima, making optimization difficult. Binary cross-entropy is convex, guaranteeing a global minimum.

**Q: How do you interpret logistic regression coefficients?**
A: The coefficient represents the change in log-odds for a unit increase in the feature. exp(coefficient) gives the odds ratio - how much the odds of the positive class multiply by for a unit increase in the feature.

### Practical Questions

**Q: How do you handle multiclass problems with logistic regression?**
A: Use multinomial logistic regression (softmax) or one-vs-rest (OvR) approach where you train multiple binary classifiers.

**Q: What assumptions does logistic regression make?**
A: 
- Linear relationship between features and log-odds
- Independence of observations
- No severe multicollinearity
- Large sample size (at least 10 cases per feature)

**Q: How do you detect and handle overfitting?**
A: Use cross-validation, regularization (L1/L2), feature selection, and monitor training vs validation performance. Add regularization parameter (C in sklearn) to penalize large weights.

**Q: What's the difference between L1 and L2 regularization?**
A: L1 (Lasso) adds absolute value of coefficients to cost function and can drive some coefficients to zero (feature selection). L2 (Ridge) adds squared coefficients and shrinks all coefficients but doesn't eliminate them.

### Coding Questions

**Q: Implement sigmoid function that handles overflow**
```python
def safe_sigmoid(z):
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))
```

**Q: Calculate accuracy manually**
```python
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
```

## # External Resources

### Essential Reading
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Andrew Ng's Machine Learning Course - Logistic Regression](https://www.coursera.org/learn/machine-learning)
- [Introduction to Statistical Learning (ISLR) - Chapter 4](https://www.statlearning.com/)

### Advanced Topics
- [Regularized Logistic Regression](https://en.wikipedia.org/wiki/Regularized_logistic_regression)
- [Multinomial Logistic Regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

### Tools & Libraries
- [Scikit-learn](https://scikit-learn.org/) - Primary ML library for Python
- [Statsmodels](https://www.statsmodels.org/) - Statistical modeling with detailed outputs
- [MLxtend](https://rasbt.github.io/mlxtend/) - Additional ML utilities
- [Yellowbrick](https://www.scikit-yb.org/) - ML visualization

### Practice Datasets
- [Titanic Dataset](https://www.kaggle.com/c/titanic) - Classic binary classification
- [Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) - Medical diagnosis
- [Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/Adult) - Socioeconomic prediction

---

*Last updated: September 2025*
*For questions or contributions, please refer to the main repository.*
