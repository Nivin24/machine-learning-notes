# Math Behind Split Selection in Decision Trees

## Objectives
- Formalize impurity measures (Entropy, Gini, MSE for regression)
- Derive information gain and Gini decrease
- Show step-by-step split computation with examples
- Cover threshold search efficiency for continuous features

## Impurity Measures
Classification (k classes, probabilities p_i at a node):
- Entropy: H = -Σ p_i log2 p_i
- Gini: G = 1 - Σ p_i^2
- Misclassification rate: 1 - max_i p_i (rarely used for splitting)

Regression (responses y in node):
- MSE impurity: I = (1/n) Σ (y_j - ȳ)^2 = Var(y)

## Information Gain (ID3/C4.5)
For parent node S and split creating children S_j with weights w_j = |S_j|/|S|:
IG = Impurity(S) - Σ w_j Impurity(S_j)
- With Entropy → “Information Gain”
- With Gini → “Gini decrease” (CART)

Gain Ratio (C4.5): IG / SplitInfo, where SplitInfo = -Σ w_j log2 w_j

## Binary Threshold Splits for Numeric Features
Consider feature x and threshold t. Partition:
- Left L = {i: x_i ≤ t}, Right R = {i: x_i > t}
Compute class counts on L and R, get p_i^L and p_i^R, then
Score(t) = Impurity(S) - [ (|L|/|S|) Imp(L) + (|R|/|S|) Imp(R) ]
Choose t that maximizes Score(t).

### Efficient Sweep Algorithm (O(n log n))
1) Sort samples by x. 2) Initialize all counts on right. 3) Move samples one-by-one from right to left, updating counts and evaluating candidate thresholds between distinct x values.

Pseudocode:
- Sort pairs (x_i, y_i) by x_i
- Maintain class counts C_L and C_R; initially C_R = counts over S, C_L = 0
- For i from 1 to n-1:
  - Move sample i from R to L (update counts)
  - If x_i != x_{i+1}: evaluate threshold t = (x_i + x_{i+1})/2
  - Compute Gini/Entropy on L and R and update best score

## Worked Binary Classification Example
Data (x, y):
- (2.0, A), (2.5, A), (3.0, B), (4.5, B), (5.0, B)
Sorted by x already. Parent counts: A=2, B=3
Parent Gini: 1 - (2/5)^2 - (3/5)^2 = 1 - (0.16 + 0.36) = 0.48

Sweep thresholds:
- Move (2.0,A) to L: L A=1,B=0; R A=1,B=3
  Next x=2.5 → t=2.25
  Gini_L=0, Gini_R=1-(1/4)^2-(3/4)^2=0.375; weighted= (1/5)*0 + (4/5)*0.375=0.300
  ΔGini=0.48-0.300=0.180
- Move (2.5,A): L A=2,B=0; R A=0,B=3
  Next x=3.0 → t=2.75
  Gini_L=0, Gini_R=0; weighted=0; ΔGini=0.48 → best so far
- Move (3.0,B): L A=2,B=1; R A=0,B=2
  Next x=4.5 → t=3.75
  Gini_L=1-(2/3)^2-(1/3)^2=0.444, Gini_R=0; weighted=(3/5)*0.444=0.266
  ΔGini=0.214
- Move (4.5,B): L A=2,B=2; R A=0,B=1
  Next x=5.0 → t=4.75
  Gini_L=1-2*(0.5^2)=0.5, Gini_R=0; weighted=(4/5)*0.5=0.4
  ΔGini=0.08
Best threshold t=2.75 with ΔGini=0.48 (perfect separation).

## Multiclass Example Sketch
Compute counts for k classes; formulas extend directly by summing over k.

## Regression Split Criterion (CART)
Use MSE decrease:
Δ = Var_parent - [ (n_L/n) Var_L + (n_R/n) Var_R ]
Efficient updates:
- Maintain n, sum y, sum y^2; Var = (sum y^2)/n - (ȳ)^2
- As you move samples across threshold, update these in O(1)

Example:
Parent y: [3, 4, 7, 8]
Var = mean(y^2) - mean(y)^2 = (9+16+49+64)/4 - (5.5)^2 = 34.5 - 30.25 = 4.25
Try split [3,4] | [7,8]: Var_L = 0.25, Var_R = 0.25, weighted = 0.25
Δ = 4.25 - 0.25 = 4.0

## Tie-breaking and Practicalities
- When multiple thresholds tie, prefer larger child sizes for stability
- Enforce min_samples_leaf to avoid small leaves
- Randomly subsample features (max_features) to reduce variance
- Use class_weight or sample_weight in impurity computations when needed

## Complexity Summary
- Categorical feature with C categories: try partitions; practical approach uses one-vs-rest or sorted by target statistics for binary trees
- Continuous features: O(n log n) per feature with sweep
- Total training: O(F n log n) typically (F features)

## References
- Breiman et al., CART (1984)
- Quinlan, C4.5 (1993)
- scikit-learn User Guide: Decision Trees
