# Hyperplanes and Margins in SVM

## Table of Contents
1. [Introduction](#introduction)
2. [What is a Hyperplane?](#what-is-a-hyperplane)
3. [Mathematical Definition](#mathematical-definition)
4. [Margins in SVM](#margins-in-svm)
5. [Support Vectors](#support-vectors)
6. [Graphical Intuition](#graphical-intuition)
7. [Key Properties](#key-properties)

---

## Introduction

Hyperplanes and margins are fundamental concepts in Support Vector Machines (SVMs). Understanding these concepts is crucial for grasping how SVMs work for classification tasks.

**Key Insight**: SVMs aim to find the optimal hyperplane that separates different classes while maximizing the margin between them.

---

## What is a Hyperplane?

A **hyperplane** is a decision boundary that separates data points of different classes in a feature space.

### Dimension-Based Definition:
- **1D space**: A hyperplane is a point
- **2D space**: A hyperplane is a line
- **3D space**: A hyperplane is a plane
- **nD space**: A hyperplane is an (n-1)-dimensional subspace

### Intuitive Understanding:
Think of a hyperplane as a "flat" affine subspace that divides the feature space into two half-spaces, each corresponding to a different class label.

---

## Mathematical Definition

### General Form

A hyperplane in n-dimensional space can be represented as:

```
w₁x₁ + w₂x₂ + ... + wₙxₙ + b = 0
```

Or in vector notation:

```
w^T·x + b = 0
```

Where:
- **w** = (w₁, w₂, ..., wₙ) is the weight vector (normal to the hyperplane)
- **x** = (x₁, x₂, ..., xₙ) is the input feature vector
- **b** is the bias term (intercept)
- **w^T** denotes the transpose of w

### Decision Function

For classification, we use the sign of the decision function:

```
f(x) = sign(w^T·x + b)
```

- If f(x) > 0: Point belongs to class +1
- If f(x) < 0: Point belongs to class -1
- If f(x) = 0: Point lies on the hyperplane

### Distance from Point to Hyperplane

The perpendicular distance from a point x to the hyperplane is:

```
distance = |w^T·x + b| / ||w||
```

Where ||w|| is the Euclidean norm (magnitude) of w.

---

## Margins in SVM

### What is a Margin?

The **margin** is the distance between the decision boundary (hyperplane) and the nearest data points from either class.

### Types of Margins:

#### 1. **Hard Margin**
- Requires complete linear separability
- No data points allowed within the margin
- All training points correctly classified
- Used when data is perfectly separable

#### 2. **Soft Margin**
- Allows some misclassifications
- Tolerates points within or on wrong side of margin
- More robust to outliers and noise
- Used in real-world scenarios (C parameter controls trade-off)

### Margin Width Calculation

The margin width (M) between the two margin hyperplanes is:

```
M = 2 / ||w||
```

**Maximizing the margin** is equivalent to **minimizing ||w||**.

### Margin Hyperplanes

For binary classification with labels y ∈ {-1, +1}:

- **Positive margin hyperplane**: w^T·x + b = +1
- **Negative margin hyperplane**: w^T·x + b = -1
- **Decision hyperplane**: w^T·x + b = 0

The distance between margin hyperplanes is 2/||w||.

---

## Support Vectors

### Definition

**Support vectors** are the data points that lie closest to the decision boundary—specifically, those that lie exactly on the margin hyperplanes.

### Key Characteristics:

1. **Critical Points**: Only support vectors determine the hyperplane position
2. **Constraint Satisfaction**: For support vectors: |w^T·x + b| = 1
3. **Lagrange Multipliers**: Support vectors have non-zero αᵢ > 0
4. **Minimal Set**: Removing non-support vectors doesn't change the model
5. **Sparse Solution**: Usually only a small fraction of training points are support vectors

### Mathematical Constraint

For all training points (xᵢ, yᵢ), the constraint is:

```
yᵢ(w^T·xᵢ + b) ≥ 1
```

Support vectors satisfy this with equality:

```
yᵢ(w^T·xᵢ + b) = 1
```

### Why They Matter:

- **Model Efficiency**: Only support vectors are needed for prediction
- **Memory Efficient**: No need to store all training data
- **Kernel Methods**: Only support vectors participate in kernel computations
- **Generalization**: Focusing on boundary points improves generalization

---

## Graphical Intuition

### 2D Example Visualization

```
                    + (class +1)
                 +
              +           
    ┌────────────────────────────┐
    │         +                  │ ← Positive margin (w^T·x + b = +1)
    │    +                       │
    ├────────────────────────────┤ ← Decision boundary (w^T·x + b = 0)
    │                   -        │
    │              -             │ ← Negative margin (w^T·x + b = -1)
    └────────────────────────────┘
           -              -
                 - (class -1)

    ★ = Support Vectors (points on margin)
    ←─→ = Margin width (2/||w||)
```

### What to Observe:

1. **Decision Boundary**: Solid line in the middle
2. **Margin Boundaries**: Dashed lines parallel to decision boundary
3. **Support Vectors**: Points touching the margin boundaries
4. **Maximum Margin**: The widest possible separation
5. **Symmetry**: Margin is symmetric around the decision boundary

### 3D Intuition

In 3D:
- Decision boundary becomes a plane
- Margin boundaries are parallel planes
- Support vectors are points touching these planes
- Margin is the 3D "slab" between the two planes

---

## Key Properties

### 1. **Uniqueness**
- For linearly separable data, the maximum-margin hyperplane is unique
- There exists only one optimal solution

### 2. **Margin Maximization Objective**

SVM optimization problem:

```
Minimize: (1/2)||w||²
Subject to: yᵢ(w^T·xᵢ + b) ≥ 1, for all i
```

This is a convex quadratic programming problem with linear constraints.

### 3. **Invariance to Scaling**
- Hyperplane defined by (w, b) and (kw, kb) for k > 0 are identical
- Normalization is needed: we set the functional margin to 1

### 4. **Geometric vs Functional Margin**

**Functional Margin**:
```
γ̂ᵢ = yᵢ(w^T·xᵢ + b)
```
- Not scale-invariant
- Can be made arbitrarily large by scaling w and b

**Geometric Margin**:
```
γᵢ = yᵢ(w^T·xᵢ + b) / ||w||
```
- Scale-invariant
- Actual perpendicular distance to hyperplane
- What SVM maximizes

### 5. **Role in Generalization**

**Large Margin Theory**:
- Larger margins tend to better generalization
- VC dimension bounds are tighter with larger margins
- Reduces overfitting by focusing on robust boundaries

### 6. **Sensitivity**
- Only support vectors affect the hyperplane
- Adding/removing points far from the margin has no effect
- Robust to outliers (especially with soft margin)

---

## Summary

| Concept | Definition | Mathematical Form |
|---------|------------|-------------------|
| **Hyperplane** | Decision boundary | w^T·x + b = 0 |
| **Margin Width** | Distance between margin boundaries | 2/‖w‖ |
| **Support Vectors** | Points on margin boundaries | yᵢ(w^T·xᵢ + b) = 1 |
| **Geometric Margin** | Perpendicular distance to hyperplane | yᵢ(w^T·xᵢ + b)/‖w‖ |
| **Optimization Goal** | Maximize margin | Minimize ‖w‖² |

### Key Takeaways:

✓ Hyperplanes are decision boundaries that separate classes

✓ Margins represent the confidence region around the decision boundary

✓ Support vectors are the critical points that define the optimal hyperplane

✓ SVM finds the hyperplane with maximum margin for better generalization

✓ Only support vectors matter—other points don't affect the solution

✓ Geometric margin is scale-invariant and what SVM actually maximizes

---

## Additional Resources

- See `intro.md` for SVM overview
- See `kernel_trick.md` for non-linear boundaries
- See `math_behind.md` for detailed optimization formulation
- See `svm.ipynb` for visualizations and code examples
