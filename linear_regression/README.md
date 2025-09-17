# Linear Regression

## Overview

Linear regression is one of the fundamental algorithms in machine learning, used for predicting continuous numerical values. It attempts to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. This technique forms the foundation for many advanced machine learning algorithms and provides excellent interpretability of results.

## Contents

1. [Introduction](#introduction)
2. [Cost Function](#cost-function)
3. [Gradient Descent](#gradient-descent)
4. [Python Implementation](#python-implementation)
5. [Further Reading](#further-reading)

## Introduction

Linear regression assumes a linear relationship between the input features and the target variable. The goal is to find the best-fitting line (or hyperplane in higher dimensions) that minimizes the difference between predicted and actual values.

### Simple Linear Regression
- One independent variable
- Equation: `y = mx + b`
- Finding the best fit line

### Multiple Linear Regression
- Multiple independent variables
- Equation: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`
- Matrix representation: `y = Xβ + ε`

## Cost Function

The cost function measures how well our model fits the data. For linear regression, we use the Mean Squared Error (MSE):

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $J(\theta)$ is the cost function
- $m$ is the number of training examples
- $h_\theta(x^{(i)}) = \theta_0 + \theta_1 x^{(i)}$ is our hypothesis function
- $y^{(i)}$ is the actual value for the i-th training example

The factor of $\frac{1}{2}$ is used to simplify the derivative calculation.

## Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating the parameters:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

For linear regression, the partial derivatives are:

$$\frac{\partial}{\partial \theta_0} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$

$$\frac{\partial}{\partial \theta_1} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

The update rules become:

$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$

$$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

Where $\alpha$ is the learning rate.

## Python Implementation

For detailed Python implementations and examples, see our comprehensive Jupyter notebook:

📓 **[Linear Regression Implementation Notebook](./linear_regression_implementation.ipynb)**

The notebook covers:
- Implementation from scratch using NumPy
- Using scikit-learn
- Data preprocessing and feature scaling
- Model evaluation metrics
- Regularization techniques (Ridge, Lasso)
- Real-world examples and datasets

### Key Libraries Used
- NumPy for numerical computations
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization
- Scikit-learn for built-in implementations

## Further Reading

### Books
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop
- **"Hands-On Machine Learning"** by Aurélien Géron

### Online Resources
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning) - Excellent foundation
- [Khan Academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra) - Mathematical prerequisites
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html) - Implementation details

### Research Papers
- Legendre, A. M. (1805). *Nouvelles méthodes pour la détermination des orbites des comètes* - Original least squares method
- Gauss, C. F. (1809). *Theoria motus corporum coelestium* - Gaussian method of least squares

### Prerequisites
Before diving into linear regression, ensure you understand:
- Linear algebra (vectors, matrices, matrix operations)
- Basic calculus (derivatives, partial derivatives)
- Statistics and probability
- Python programming fundamentals

---

**Next Steps:** After mastering linear regression, consider exploring:
- Logistic Regression
- Polynomial Regression
- Regularization Techniques
- Support Vector Machines
