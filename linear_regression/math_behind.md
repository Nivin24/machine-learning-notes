# Mathematics Behind Linear Regression

## Overview

This document provides a concise mathematical foundation for linear regression, covering the core equations, matrix formulation, and analytical solution derivation.

## 1. Model Equation

### Simple Linear Regression

For a single feature, the linear regression model is:

$$y = \beta_0 + \beta_1 x + \epsilon$$

Where:
- $y$ is the dependent variable (target)
- $x$ is the independent variable (feature)
- $\beta_0$ is the y-intercept (bias term)
- $\beta_1$ is the slope (weight)
- $\epsilon$ is the error term

### Multiple Linear Regression

For multiple features, the model generalizes to:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p + \epsilon$$

## 2. Matrix Formulation

### Design Matrix

For $n$ observations and $p$ features, we construct the design matrix $\mathbf{X}$:

$$\mathbf{X} = \begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}$$

### Vector Form

The model can be written compactly as:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Where:
- $\mathbf{y} = [y_1, y_2, \ldots, y_n]^T$ is the target vector
- $\boldsymbol{\beta} = [\beta_0, \beta_1, \ldots, \beta_p]^T$ is the parameter vector
- $\boldsymbol{\epsilon} = [\epsilon_1, \epsilon_2, \ldots, \epsilon_n]^T$ is the error vector

## 3. Cost Function (Mean Squared Error)

The objective is to minimize the sum of squared residuals:

$$J(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

In matrix form:

$$J(\boldsymbol{\beta}) = \frac{1}{2n} (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

## 4. Derivation of Least Squares Solution

### Step 1: Expand the Cost Function

$$J(\boldsymbol{\beta}) = \frac{1}{2n} [\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}]$$

### Step 2: Take the Gradient

To find the minimum, we take the partial derivative with respect to $\boldsymbol{\beta}$ and set it to zero:

$$\frac{\partial J(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = \frac{1}{n} [-\mathbf{X}^T\mathbf{y} + \mathbf{X}^T\mathbf{X}\boldsymbol{\beta}] = \mathbf{0}$$

### Step 3: Solve for β

Rearranging the equation:

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

## 5. Normal Equation (Analytical Solution)

The closed-form solution for the optimal parameters is:

$$\boldsymbol{\hat{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### Conditions for Existence

The normal equation solution exists when:
1. $\mathbf{X}^T\mathbf{X}$ is invertible (non-singular)
2. $n \geq p + 1$ (more observations than parameters)
3. Features are linearly independent

### Alternative Forms

Using the Moore-Penrose pseudoinverse:

$$\boldsymbol{\hat{\beta}} = \mathbf{X}^{+}\mathbf{y}$$

Where $\mathbf{X}^{+} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is the pseudoinverse.

## 6. Predictions and Residuals

### Predicted Values

$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\hat{\beta}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{H}\mathbf{y}$$

Where $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is the hat matrix (projection matrix).

### Residuals

$$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$$

## 7. Statistical Properties

### Assumptions

1. **Linearity**: $E[\mathbf{y}|\mathbf{X}] = \mathbf{X}\boldsymbol{\beta}$
2. **Independence**: Observations are independent
3. **Homoscedasticity**: $\text{Var}(\epsilon_i) = \sigma^2$ for all $i$
4. **Normality**: $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$

### Unbiasedness

Under the above assumptions:

$$E[\boldsymbol{\hat{\beta}}] = \boldsymbol{\beta}$$

### Variance-Covariance Matrix

$$\text{Var}(\boldsymbol{\hat{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$$

## 8. Geometric Interpretation

- The predicted values $\hat{\mathbf{y}}$ lie in the column space of $\mathbf{X}$
- The least squares solution finds the orthogonal projection of $\mathbf{y}$ onto this column space
- The residual vector $\mathbf{e}$ is orthogonal to the column space: $\mathbf{X}^T\mathbf{e} = \mathbf{0}$

## Summary

Linear regression provides a mathematically elegant framework for modeling linear relationships. The normal equation $\boldsymbol{\hat{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ gives the analytical solution that minimizes the mean squared error, offering both computational efficiency for small problems and theoretical insights into the geometry of least squares fitting.
