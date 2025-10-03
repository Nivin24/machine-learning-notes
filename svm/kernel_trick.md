# Kernel Trick in SVM

## Table of Contents
1. Motivation: Why kernels?
2. Feature maps and implicit high-dimensional spaces
3. Kernel function definition and properties (Mercer)
4. Common kernels: linear, polynomial, RBF, sigmoid
5. Mathematical formulation of the kernelized SVM (dual)
6. Intuition and analogies
7. Choosing and tuning kernels
8. Visual aids and diagram ideas
9. Practical notes and pitfalls

---

## 1. Motivation: Why kernels?
Many real-world datasets are not linearly separable in their original feature space. A classic approach is to map data to a higher-dimensional feature space where linear separation is possible. Directly computing this mapping can be expensive or intractable. The kernel trick lets us compute dot products in that high-dimensional space without explicitly computing the mapping, enabling efficient learning of nonlinear decision boundaries.

Key idea: Replace x·x' with K(x, x'), where K is a kernel that equals φ(x)·φ(x') for some feature map φ.

---

## 2. Feature maps and implicit high-dimensional spaces
- Explicit mapping: φ: R^d → H (possibly very high or infinite dimensional). Learn a linear separator w in H.
- Implicit mapping: Avoid constructing φ(x). Use a kernel K(x, x') = ⟨φ(x), φ(x')⟩ to access inner products in H.
- Benefit: Complexity depends on evaluating K, not the dimensionality of H.

Example: In 2D, adding quadratic features [x1^2, x2^2, √2 x1x2, √2 x1, √2 x2, 1] can separate concentric patterns. A polynomial kernel achieves this implicitly.

---

## 3. Kernel function definition and properties (Mercer)
A kernel K: X×X → R is a symmetric, positive semidefinite (PSD) function that corresponds to an inner product in some feature space.

Properties:
- Symmetry: K(x, x') = K(x', x)
- PSD: For any {x1, ..., xm}, the Gram matrix G with G_ij = K(xi, xj) is PSD
- By Mercer’s theorem, such K corresponds to an inner product in a (possibly infinite-dimensional) Hilbert space.

Checking validity: If K is PSD, it can be used as a kernel. Sums, products, and positive scalings of valid kernels are also valid.

---

## 4. Common kernels

### Linear kernel
K(x, x') = x^T x'
- Equivalent to standard linear SVM
- Fast, strong baseline; good for very high-dimensional sparse data (e.g., text)

### Polynomial kernel
K(x, x') = (γ x^T x' + r)^d, with γ > 0, degree d, coef0 r
- Captures interactions up to degree d
- Be careful with large d; can overfit and be expensive

### RBF (Gaussian) kernel
K(x, x') = exp(-γ ||x - x'||^2), γ > 0
- Infinite-dimensional feature space
- Local influence; popular default for nonlinear problems
- γ controls the width: small γ → smoother, large γ → more complex decision boundary

### Sigmoid (tanh) kernel
K(x, x') = tanh(γ x^T x' + r)
- Related to neural nets; not always PSD for all parameters
- Use with care; RBF or poly often preferred

---

## 5. Kernelized SVM: the dual problem
Starting from the soft-margin dual (see math_behind.md), kernelize by replacing inner products with K.

Dual optimization:
maximize over α ∈ R^m
    W(α) = Σ_i α_i − 1/2 Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
subject to 0 ≤ α_i ≤ C, and Σ_i α_i y_i = 0

Decision function:
    f(x) = sign( Σ_i α_i y_i K(x_i, x) + b )

Only support vectors (α_i > 0) contribute. No explicit w in input space; instead, w exists implicitly in feature space H.

---

## 6. Intuition and analogies
- Lifting the space: Imagine unfolding a crumpled sheet so that points become linearly separable.
- Similarity measure: K(x, x') measures similarity; RBF measures proximity; polynomial captures feature interactions.
- Convolutional view: With appropriate K, decision boundary is a weighted combination of similarities to support vectors.

---

## 7. Choosing and tuning kernels
- Start simple: linear for high-dimensional sparse features; RBF for general nonlinear tasks
- Grid search or Bayesian optimization on C and kernel params (γ, degree, coef0)
- Data scaling matters: standardize features before using RBF/poly
- Heuristics for RBF γ: 1/(median pairwise distance)^2 as a starting point
- Regularization C: larger C reduces margin violations but risks overfitting; smaller C increases margin and tolerance to errors

---

## 8. Visual aids and diagram ideas
- 2D toy datasets: moons, circles; show how linear fails and RBF succeeds
- Feature space sketch: depict φ(x) mapping to a higher-dimensional space with a separating hyperplane
- Similarity heatmap: Gram matrix visualization for different kernels

ASCII sketch (conceptual):

Input space (nonlinear):
  o o     x x
 o   o   x   x
  o o     x x

Feature space via φ (linear separation):
  o o o | x x x
  o o o | x x x
  -------+------  ← linear hyperplane in feature space

---

## 9. Practical notes and pitfalls
- Watch out for non-PSD kernels (e.g., improper sigmoid params)
- RBF often a strong default; polynomial can explode with degree
- Normalize/standardize features; handle categorical variables appropriately
- With many samples, kernel methods scale as O(n^2) memory and O(n^3) training; consider linear SVM or approximate kernels (Nyström, random features)
- Interpretability: kernel SVMs are less interpretable; use support vector counts and prototypes to aid insight

References:
- See math_behind.md for derivations
- See svm.ipynb for visual demos and comparisons
- Bishop, PRML; Schölkopf & Smola; Cortes & Vapnik (1995)
