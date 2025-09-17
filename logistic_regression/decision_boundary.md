# Decision Boundaries in Logistic Regression

Decision boundaries are fundamental concepts in machine learning classification tasks. In logistic regression, the decision boundary represents the threshold that separates different classes in the feature space. Understanding decision boundaries is crucial for interpreting how logistic regression models make predictions and for visualizing the model's behavior.

## Mathematical Foundation

In logistic regression, the decision boundary is mathematically defined by the point where the predicted probability equals 0.5. Given the logistic regression model:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

Where:
- $\mathbf{x}$ is the input feature vector
- $\mathbf{w}$ is the weight vector
- $b$ is the bias term
- $\sigma$ is the sigmoid function

The decision boundary occurs when:

$$P(y=1|\mathbf{x}) = 0.5$$

This happens when the linear combination equals zero:

$$\mathbf{w}^T\mathbf{x} + b = 0$$

For a 2D case with features $x_1$ and $x_2$:

$$w_1x_1 + w_2x_2 + b = 0$$

Solving for $x_2$:

$$x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}$$

This represents a straight line with slope $-\frac{w_1}{w_2}$ and y-intercept $-\frac{b}{w_2}$.

## 2D Visualization Implementation

Here's a Python implementation to visualize decision boundaries in a 2D feature space:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """
    Plot the decision boundary for a 2D dataset
    
    Parameters:
    X: Feature matrix (n_samples, 2)
    y: Target labels (n_samples,)
    model: Trained logistic regression model
    title: Plot title
    """
    # Create a mesh grid
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the contour and training examples
    colors = ['lightcoral', 'lightblue']
    cmap = ListedColormap(colors)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue']), 
                         edgecolors='black', alpha=0.7, s=50)
    
    plt.colorbar(label='Predicted Probability')
    plt.xlabel('Feature 1 ($x_1$)')
    plt.ylabel('Feature 2 ($x_2$)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(['Decision Boundary (P=0.5)', 'Class 0', 'Class 1'], 
              loc='upper right')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                              n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Plot the decision boundary
    plot_decision_boundary(X, y, model, "Logistic Regression Decision Boundary")
    
    # Print the mathematical equation of the decision boundary
    w1, w2 = model.coef_[0]
    b = model.intercept_[0]
    
    print(f"Decision Boundary Equation: {w1:.3f}x₁ + {w2:.3f}x₂ + {b:.3f} = 0")
    print(f"Simplified: x₂ = {-w1/w2:.3f}x₁ + {-b/w2:.3f}")
```

## Key Properties of Decision Boundaries

### 1. Linear Separation
- **Characteristic**: Logistic regression creates linear decision boundaries
- **Implication**: Can only separate linearly separable classes effectively
- **Mathematical basis**: The decision boundary is defined by a linear equation

### 2. Probability Interpretation
- **At the boundary**: Predicted probability = 0.5
- **Above the boundary**: P(y=1) > 0.5 (typically classified as class 1)
- **Below the boundary**: P(y=1) < 0.5 (typically classified as class 0)

### 3. Geometric Properties
- **Shape**: Always a hyperplane (line in 2D, plane in 3D, etc.)
- **Orientation**: Determined by the weight vector $\mathbf{w}$
- **Position**: Controlled by the bias term $b$

### 4. Sensitivity to Outliers
- **Moderate robustness**: Less sensitive than perceptron but more than SVM
- **Impact**: Outliers can shift the boundary but not dramatically
- **Regularization**: L1/L2 regularization can reduce outlier sensitivity

### 5. Feature Scaling Effects
- **Importance**: Feature scaling affects the boundary orientation
- **Without scaling**: Features with larger scales dominate the boundary
- **Best practice**: Standardize features before training

## Practical Examples

### Example 1: Email Spam Classification

Consider a simplified email spam classifier with two features:
- $x_1$: Number of exclamation marks
- $x_2$: Frequency of promotional words

If the trained model yields:
$$2.5x_1 + 1.8x_2 - 3.2 = 0$$

**Interpretation**:
- Decision boundary: $x_2 = -1.39x_1 + 1.78$
- Emails above this line: Likely spam (P > 0.5)
- Emails below this line: Likely legitimate (P < 0.5)

### Example 2: Medical Diagnosis

For a binary medical diagnosis with:
- $x_1$: Patient age (normalized)
- $x_2$: Biomarker level (normalized)

Decision boundary equation: $0.8x_1 + 2.1x_2 - 1.5 = 0$

**Clinical interpretation**:
- Boundary represents the diagnostic threshold
- Patients above the line have higher probability of positive diagnosis
- The steeper slope (2.1 vs 0.8) indicates biomarker level is more influential

### Example 3: Customer Churn Prediction

Predicting customer churn with:
- $x_1$: Customer satisfaction score
- $x_2$: Usage frequency

Decision boundary: $-1.2x_1 + 0.6x_2 + 2.0 = 0$

**Business insights**:
- Negative coefficient for satisfaction: Higher satisfaction reduces churn probability
- Positive coefficient for usage: Paradoxically, very high usage might indicate dissatisfaction
- The boundary helps identify at-risk customers

## Advanced Considerations

### Multiclass Extensions

For multiclass logistic regression (softmax), decision boundaries become more complex:
- **One-vs-Rest**: Multiple binary boundaries
- **Multinomial**: Boundaries are intersections of hyperplanes
- **Visualization**: Requires techniques like dimensionality reduction

### Non-linear Boundaries

To create non-linear decision boundaries with logistic regression:
- **Polynomial features**: Add $x_1^2$, $x_2^2$, $x_1x_2$ terms
- **Feature engineering**: Create interaction terms
- **Kernel trick**: Not directly applicable to logistic regression

### Evaluation Metrics

The quality of decision boundaries can be assessed through:
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Class-specific performance
- **ROC Curve**: Boundary performance across different thresholds
- **Visualization**: Direct inspection of the boundary

## Summary

Decision boundaries in logistic regression represent the fundamental mechanism by which the algorithm separates classes. Understanding their mathematical foundation, visualization techniques, and practical implications is essential for:

1. **Model Interpretation**: Understanding how predictions are made
2. **Feature Engineering**: Identifying when non-linear transformations are needed
3. **Debugging**: Diagnosing model performance issues
4. **Communication**: Explaining model behavior to stakeholders

The linear nature of logistic regression decision boundaries makes them interpretable and computationally efficient, while their probabilistic foundation provides meaningful confidence estimates for predictions.
