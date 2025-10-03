# SVM Notes and Cheat Sheet

## 1) Quick Pros/Cons
Pros:
- Effective in high-dimensional spaces, especially with clear margin
- Works well with kernels for nonlinear boundaries
- Sparse solution via support vectors
- Convex optimization → global optimum

Cons:
- Kernel SVMs scale poorly with large datasets (O(n^2)/O(n^3))
- Choice and tuning of kernel/parameters can be tricky
- Less interpretable than linear models/trees
- Sensitive to feature scaling (especially RBF/poly)

---

## 2) Types of SVM
- Linear SVM (hard/soft margin)
- Kernel SVM (RBF, polynomial, sigmoid, custom PSD kernels)
- One-vs-Rest or One-vs-One for multiclass
- Support Vector Regression (SVR: ε-insensitive loss)
- One-Class SVM (novelty detection)

---

## 3) Common Kernels
- Linear: K(x, x') = x^T x'
- Polynomial: K(x, x') = (γ x^T x' + r)^d
- RBF: K(x, x') = exp(−γ ||x − x'||^2)
- Sigmoid: K(x, x') = tanh(γ x^T x' + r) [not always PSD]

Parameter notes:
- C controls regularization (error tolerance)
- γ (RBF) controls radius of influence; degree d controls complexity (poly)
- r (coef0) shifts polynomial/sigmoid

---

## 4) Key Formulas
- Decision function: f(x) = sign(Σ_i α_i y_i K(x_i, x) + b)
- Hard-margin primal: min (1/2)||w||^2 s.t. y_i(w^T x_i + b) ≥ 1
- Soft-margin primal: min (1/2)||w||^2 + C Σ ξ_i s.t. y_i(w^T x_i + b) ≥ 1 − ξ_i, ξ_i ≥ 0
- Dual: max Σ α_i − (1/2)ΣΣ α_i α_j y_i y_j K(x_i, x_j), 0 ≤ α_i ≤ C, Σ α_i y_i = 0
- Margin width: 2/||w||

---

## 5) Practical Tips
- Always scale/standardize numeric features
- Start with linear (for sparse/high-dim) or RBF (general nonlinear)
- Use cross-validation to tune C, γ, and degree
- Monitor support vector count; too many SVs may indicate overfitting or need for more data/regularization
- For very large n: consider LinearSVC or SGD with hinge loss; or kernel approximations (Nyström, Random Fourier Features)
- Handle class imbalance with class_weight='balanced' or custom weights

---

## 6) Interview Questions
- Intuition of margin maximization and generalization
- Difference between hard vs soft margin; role of C
- Explain kernel trick and Mercer’s condition
- What does γ do in RBF? Interplay with C?
- Primal vs dual; why solve dual? Where do support vectors come from?
- When prefer linear SVM vs RBF? Complexity considerations
- How to compute b from support vectors?

---

## 7) Quick Workflow
1. Preprocess: impute, scale, encode categorical (one-hot)
2. Choose kernel (linear/RBF)
3. Hyperparameters: C, γ (RBF), degree/coef0 (poly)
4. Cross-validate via GridSearchCV/RandomizedSearchCV
5. Inspect: accuracy/F1, confusion matrix, #SVs
6. Calibrate probabilities if needed (CalibratedClassifierCV)

---

## 8) References
- See intro.md for overview and history
- See hyperplanes_margins.md for geometry
- See kernel_trick.md for kernels
- See math_behind.md for derivations
- See svm.ipynb for code demos and visualizations
