# Entropy and Information Gain in Decision Trees

## Intuition: Measuring Uncertainty and Purity
- A split is good if it creates child nodes that are purer than the parent node.
- Purity means the node contains mostly one class; impurity means classes are mixed.
- Entropy quantifies uncertainty/impurity; Information Gain measures impurity reduction after a split.

## Entropy
Entropy (base-2) of a discrete distribution p1, p2, ..., pk is:

H(Y) = - Σ_i p_i log2 p_i, with the convention 0 log 0 = 0

Properties:
- 0 ≤ H(Y) ≤ log2(k). For binary classification, 0 ≤ H(Y) ≤ 1.
- H(Y) = 0 when the node is pure (all one class).
- H(Y) is maximal when classes are uniformly distributed.

Examples (binary):
- p = [1, 0] → H = 0
- p = [0.5, 0.5] → H = 1
- p = [0.8, 0.2] → H = -(0.8 log2 0.8 + 0.2 log2 0.2) ≈ 0.722

## Conditional Entropy and Information Gain
Given a split on feature X into partitions {S1, S2, ..., Sm}, with proportions w_j = |S_j| / |S| and entropies H(Y|S_j), the conditional entropy is:

H(Y | split on X) = Σ_j w_j H(Y | S_j)

Information Gain (IG) from splitting on X is the reduction in entropy:

IG(Y; X) = H(Y) - H(Y | split on X)

We choose the split with the highest IG (ID3), or the highest Gain Ratio (C4.5).

## Derivation Sketch: Why IG equals mutual information
Let Y be the label and X the splitting attribute. Mutual information I(Y; X) is defined as:

I(Y; X) = H(Y) - H(Y | X)

When we split dataset S by X into partitions S_j corresponding to X values/thresholds, the empirical estimate of H(Y | X) equals Σ_j w_j H(Y | S_j). Therefore IG as used in decision trees is the empirical mutual information between Y and X.

## Worked Example: Weather (Play Tennis)
Dataset (binary label: Play = Yes/No). Suppose at some node we have 14 samples: 9 Yes, 5 No.

- Parent entropy:
  H_parent = - (9/14) log2(9/14) - (5/14) log2(5/14)
           ≈ -0.6439 - 0.5305 ≈ 1.1744 ≈ 0.940
  (Note: numerically 0.940 due to base-2 logs.)

Split by Outlook with 3 values: Sunny (5), Overcast (4), Rainy (5)
- Sunny: 2 Yes, 3 No → H_sunny = - (2/5) log2(2/5) - (3/5) log2(3/5) ≈ 0.971
- Overcast: 4 Yes, 0 No → H_overcast = 0
- Rainy: 3 Yes, 2 No → H_rainy ≈ 0.971

Conditional entropy:
H(Y | Outlook) = (5/14)*0.971 + (4/14)*0 + (5/14)*0.971 ≈ 0.694

Information Gain:
IG(Y; Outlook) = 0.940 - 0.694 = 0.246

You would compare IG for other features (Humidity, Wind, Temperature) and pick the highest.

## Handling Continuous Features: Threshold Splits
For a numeric feature x, we consider binary splits of the form x ≤ t vs x > t.
Procedure:
1. Sort unique values of x.
2. Consider midpoints between consecutive sorted values as candidate thresholds.
3. For each threshold t, partition the set, compute H(Y | t) and IG(t).
4. Choose t with maximum IG.

Time complexity for one feature: O(n log n) for sorting + O(n) to sweep thresholds with running counts.

## Gain Ratio (C4.5)
Information Gain can favor attributes with many distinct values (e.g., ID-like). C4.5 normalizes IG by the intrinsic information (split info):

SplitInfo(X) = - Σ_j w_j log2 w_j

GainRatio(X) = IG(Y; X) / SplitInfo(X)

Choose the split with the highest GainRatio (subject to IG being above average to avoid zero-denominator artifacts). This penalizes overly fragmented splits.

## Relation to Gini (CART)
- Entropy and Gini are both impurity measures; both are concave and minimized at pure nodes.
- Gini(p) = 1 - Σ_i p_i^2; tends to be slightly faster to compute (no logs).
- Empirically, both yield similar trees; choice often has minimal impact on accuracy.

## Practical Tips
- For class imbalance, consider class weights when evaluating splits, or use stratified sampling.
- Use min_samples_leaf to avoid tiny leaves that artificially inflate IG.
- Prefer binary splits for continuous variables for simpler trees and better generalization.
- When many missing values, consider surrogate splits (CART) or treat missing as a separate category.

## Pseudocode
Compute information gain for candidate split:

function information_gain(S, partitions):
    H_parent = entropy(S.labels)
    H_cond = 0
    for part in partitions:
        w = len(part)/len(S)
        H_cond += w * entropy(part.labels)
    return H_parent - H_cond

function best_split(S, feature):
    parts = partition_by_feature(S, feature)  # categorical: by value; numeric: by threshold
    return argmax_t information_gain(S, parts(t))

## Visual Intuition
- Entropy measures how “mixed” a node is. Uniform mix → high entropy. Pure node → entropy 0.
- A good split creates child nodes with lower entropy than the parent; the bigger the drop, the better the split.

## References and Further Reading
- Quinlan, J. R. (1986). Induction of decision trees (ID3)
- Quinlan, J. R. (1993). C4.5: Programs for Machine Learning
- Breiman et al. (1984). Classification and Regression Trees (CART)
- scikit-learn user guide: Tree-based models
