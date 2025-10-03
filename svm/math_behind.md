# The Math Behind SVM

## Table of Contents
1. Problem setup and notation
2. Hard-margin SVM: primal and geometric interpretation
3. Soft-margin SVM: hinge loss and regularization
4. Lagrangian, KKT conditions, and the dual
5. Dual solution and support vectors
6. Kernelization of the dual
7. Summary tables and annotated derivations

---

## 1. Problem setup and notation
- Data: {(x_i, y_i)}_{i=1}^m, with x_i ∈ R^d and y_i ∈ {−1, +1}
- Classifier: f(x) = sign(w^T x + b)
- Margin: geometric margin γ = min_i y_i (w^T x_i + b) / ||w||

Goal: maximize margin (prefer large separation) while controlling training errors.

---

## 2. Hard-margin SVM (linearly separable case)

Primal optimization:
minimize over w, b:   (1/2) ||w||^2
subject to:            y_i (w^T x_i + b) ≥ 1,   for i = 1..m

Interpretation:
- Minimizing ||w||^2 maximizes the margin 2/||w||
- Constraints enforce correct classification with margin ≥ 1

Distance and margin:
- Functional margin: y_i (w^T x_i + b)
- Geometric margin:  y_i (w^T x_i + b) / ||w||
- Margin width: 2 / ||w|| between the two supporting hyperplanes

---

## 3. Soft-margin SVM (nonseparable case)
Introduce slack variables ξ_i ≥ 0 to allow violations.

Primal optimization:
minimize over w, b, ξ:   (1/2) ||w||^2 + C Σ_i ξ_i
subject to:               y_i (w^T x_i + b) ≥ 1 − ξ_i,   ξ_i ≥ 0

- C > 0 trades off margin maximization and training error (hinge loss)
- Larger C penalizes violations more → narrower margin, lower bias, higher variance
- Smaller C allows more violations → wider margin, higher bias, lower variance

Hinge loss form (unconstrained view):
min_w,b  (1/2)||w||^2 + C Σ_i max(0, 1 − y_i (w^T x_i + b))

---

## 4. Lagrangian, KKT conditions, and the dual

Lagrangian (soft-margin):
L(w, b, ξ, α, μ) = (1/2)||w||^2 + C Σ_i ξ_i − Σ_i α_i [y_i (w^T x_i + b) − 1 + ξ_i] − Σ_i μ_i ξ_i
with α_i ≥ 0, μ_i ≥ 0

Stationarity conditions:
- ∂L/∂w = 0 → w = Σ_i α_i y_i x_i
- ∂L/∂b = 0 → Σ_i α_i y_i = 0
- ∂L/∂ξ_i = 0 → α_i + μ_i = C ⇒ 0 ≤ α_i ≤ C

Complementary slackness:
- α_i [y_i (w^T x_i + b) − 1 + ξ_i] = 0
- μ_i ξ_i = 0

Primal feasibility:
- y_i (w^T x_i + b) ≥ 1 − ξ_i,  ξ_i ≥ 0

Dual feasibility:
- α_i ≥ 0, μ_i ≥ 0, and α_i ≤ C

Dual problem (soft-margin):
maximize_α  W(α) = Σ_i α_i − (1/2) Σ_i Σ_j α_i α_j y_i y_j x_i^T x_j
subject to:  0 ≤ α_i ≤ C,  and  Σ_i α_i y_i = 0

---

## 5. Dual solution and support vectors
- Only points with α_i > 0 are support vectors; they lie on or inside the margin
- If 0 < α_i < C → point lies exactly on the margin (y_i (w^T x_i + b) = 1)
- If α_i = C → point is a margin violator or misclassified
- If α_i = 0 → point is irrelevant to the decision boundary

Recover w and b:
- w = Σ_i α_i y_i x_i
- For any support vector with 0 < α_i < C, compute b = y_i − Σ_j α_j y_j x_j^T x_i and average over such SVs

Decision function:
- f(x) = sign(w^T x + b) = sign(Σ_i α_i y_i x_i^T x + b)

---

## 6. Kernelization of the dual
Replace inner products with kernels K(x_i, x_j) = φ(x_i)^T φ(x_j):

Dual (kernel SVM):
maximize_α  W(α) = Σ_i α_i − (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
subject to:  0 ≤ α_i ≤ C,  and  Σ_i α_i y_i = 0

Decision function:
- f(x) = sign(Σ_i α_i y_i K(x_i, x) + b)

No need to form φ explicitly; K gives access to the inner product in feature space.

---

## 7. Summary tables and annotated derivations

Key formulations:

| Formulation | Objective | Constraints |
|---|---|---|
| Hard-margin primal | min (1/2)||w||^2 | y_i (w^T x_i + b) ≥ 1 |
| Soft-margin primal | min (1/2)||w||^2 + C Σ ξ_i | y_i (w^T x_i + b) ≥ 1 − ξ_i, ξ_i ≥ 0 |
| Soft-margin dual | max Σ α_i − (1/2)ΣΣ α_i α_j y_i y_j x_i^T x_j | 0 ≤ α_i ≤ C, Σ α_i y_i = 0 |
| Kernel dual | same as soft-margin dual with K(x_i, x_j) | replace x_i^T x_j with K |

Notes:
- Convex QP: global optimum
- Role of C: regularization vs hinge loss trade-off
- KKT tells which points are support vectors and how to compute b
- Kernel choice transforms linear separators in feature space to nonlinear boundaries in input space

Pointers:
- See hyperplanes_margins.md for geometry
- See kernel_trick.md for kernel choices
- See svm.ipynb for practical implementations and tuning
