# Mathematical Foundations of Logistic Regression

## 1. Introduction

Logistic regression is a classification algorithm rooted in probability theory and statistics. It models the probability that a given input belongs to a particular class using a logistic (sigmoid) function applied to a linear combination of features.

---

## 2. The Logit and Sigmoid Connection

Given a feature vector \( x \) and parameter vector \( w \) with bias \( b \):

\[
z = w^T x + b
\]

The **logit function** (log-odds) is the natural log of the odds:

\[
\text{logit}(P) = \ln\left(\frac{P}{1-P}\right)
\]

The **sigmoid function** maps real values to (0, 1):

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Thus, logistic regression models the conditional probability:

\[
P(y = 1|x) = \frac{1}{1 + e^{-(w^T x + b)}}
\]

Taking the inverse (logit) gives the linear relationship:

\[
\ln\left(\frac{P(y=1|x)}{1 - P(y=1|x)}\right) = w^T x + b
\]

---

## 3. The Cost Function (Binary Cross-Entropy)

For \( N \) data points, the **log-loss** or **binary cross-entropy** measures the model's performance:

\[
J(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \bigg[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \bigg]
\]

where  
- \( y^{(i)} \): true label (0 or 1)  
- \( \hat{y}^{(i)} \): predicted probability

---

## 4. Gradient Derivation (Optimization)

To find optimal \( w, b \), use **gradient descent** by computing the derivatives of the cost function.

### Derivative of the sigmoid

\[
\frac{d}{dz} \sigma(z) = \sigma(z) \big(1 - \sigma(z)\big)
\]

### Gradient with respect to weights \( w \):

\[
\frac{\partial J}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)}
\]

### Gradient with respect to bias \( b \):

\[
\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{y}^{(i)} - y^{(i)}\right)
\]

---

## 5. Gradient Descent Update Rule

Parameters are updated iteratively as:

\[
w := w - \alpha \frac{\partial J}{\partial w}
\]
\[
b := b - \alpha \frac{\partial J}{\partial b}
\]

where \( \alpha \) is the learning rate.

---

## 6. Summary

- Logistic regression uses the sigmoid to model probabilities.
- Logit transforms probabilities to linear combinations of features.
- Binary cross-entropy measures model fit.
- Gradients enable optimization using gradient descent.
- The process is fully differentiable and suitable for binary classification tasks.

