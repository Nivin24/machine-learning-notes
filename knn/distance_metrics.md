# Distance Metrics in K-Nearest Neighbors

## Introduction

In KNN, **distance metrics** are the mathematical functions that determine how "close" or "similar" two data points are. The choice of distance metric can dramatically affect your model's performance, as it fundamentally changes which points are considered "neighbors."

**Think of it this way**: If you're looking for restaurants similar to your favorite one, do you care more about:
- Physical distance (Euclidean)?
- Travel time through city blocks (Manhattan)?
- Overall style and cuisine similarity (Cosine)?

Different metrics capture different notions of similarity!

## 1. Euclidean Distance 📟

### Formula
**For 2D**: `d = √[(x₂-x₁)² + (y₂-y₁)²]`

**General Formula**: 
```
d(p,q) = √[Σᵢ(pᵢ - qᵢ)²]
```
where p and q are two points, and i goes over all dimensions.

### Intuition
- **"Straight-line distance"** or "as the crow flies"
- Measures shortest path between two points
- Most natural and commonly used metric

### Example Calculation
```
Point A: (1, 2, 3)
Point B: (4, 6, 7)

Euclidean Distance:
d = √[(4-1)² + (6-2)² + (7-3)²]
d = √[3² + 4² + 4²]
d = √[9 + 16 + 16]
d = √41 ≈ 6.40
```

### When to Use ✅
- **Continuous features** with similar scales
- **Geometric/spatial data** (coordinates, images)
- When **magnitude matters** (e.g., house prices, temperatures)
- **Physical measurements** (height, weight, distance)

### Limitations ❌
- **Sensitive to feature scaling** (large-scale features dominate)
- **Curse of dimensionality** (becomes less meaningful in high dimensions)
- **Assumes all features equally important**

---

## 2. Manhattan Distance (L1 Distance) 🏢

### Formula
```
d(p,q) = Σᵢ|pᵢ - qᵢ|
```

### Intuition
- **"City block distance"** or "taxi cab distance"
- Sum of absolute differences
- Like navigating city streets (can't cut through buildings!)

### Example Calculation
```
Point A: (1, 2, 3)
Point B: (4, 6, 7)

Manhattan Distance:
d = |4-1| + |6-2| + |7-3|
d = |3| + |4| + |4|
d = 3 + 4 + 4 = 11
```

### Visual Comparison
```
Euclidean vs Manhattan paths from A to B:

A -------- B    (Euclidean: straight line)
|          |
A          B    (Manhattan: along grid lines)
├─→─→─→─→─→┤
```

### When to Use ✅
- **High-dimensional sparse data** (e.g., text data, word counts)
- When **outliers are problematic** (more robust than Euclidean)
- **Mixed data types** or when features have different units
- **Grid-like data** or when movement is constrained

### Advantages ✅
- **More robust to outliers** than Euclidean
- **Faster to compute** (no square root)
- **Less affected by curse of dimensionality**

---

## 3. Minkowski Distance (Generalized) 🔄

### Formula
```
d(p,q) = [Σᵢ|pᵢ - qᵢ|ᵖ]^(1/p)
```

### Special Cases
- **p = 1**: Manhattan Distance
- **p = 2**: Euclidean Distance
- **p = ∞**: Chebyshev Distance (maximum difference)

### Example with p=3
```
Point A: (1, 2)
Point B: (4, 6)

Minkowski Distance (p=3):
d = [|4-1|³ + |6-2|³]^(1/3)
d = [3³ + 4³]^(1/3)
d = [27 + 64]^(1/3)
d = [91]^(1/3) ≈ 4.50
```

### Parameter Selection
- **Lower p values (1-2)**: More robust to outliers
- **Higher p values (>2)**: More sensitive to large differences
- **p → ∞**: Only largest difference matters

---

## 4. Cosine Distance 📐

### Formula
```
Cosine Similarity = (A·B) / (||A|| × ||B||)
Cosine Distance = 1 - Cosine Similarity
```

Where:
- A·B = dot product of vectors A and B
- ||A|| = magnitude (length) of vector A

### Intuition
- Measures **angle between vectors**, not magnitude
- **Direction matters, magnitude doesn't**
- Perfect for when you care about **proportional relationships**

### Example Calculation
```
Point A: (3, 4)
Point B: (6, 8)

Dot product: A·B = (3×6) + (4×8) = 18 + 32 = 50
Magnitude A: ||A|| = √(3² + 4²) = √25 = 5
Magnitude B: ||B|| = √(6² + 8²) = √100 = 10

Cosine Similarity = 50 / (5 × 10) = 50/50 = 1.0
Cosine Distance = 1 - 1.0 = 0

# Note: B is exactly 2×A, so they have same direction (distance = 0)
```

### When to Use ✅
- **Text analysis** (document similarity, word embeddings)
- **Recommendation systems** (user preferences)
- **High-dimensional data** where magnitude varies widely
- When you care about **patterns, not absolute values**

### Example Use Case
```
Document A: [10, 5, 2] (word frequencies for "machine", "learning", "data")
Document B: [20, 10, 4] (exactly double the frequencies)

→ Euclidean Distance: Large (different magnitudes)
→ Cosine Distance: 0 (same proportions/topic)
```

---

## 5. Hamming Distance 🔢

### Formula
```
d(p,q) = Number of positions where pᵢ ≠ qᵢ
```

### Intuition
- **Counts number of differences** between strings/sequences
- Perfect for **categorical/binary data**
- Used in **genetics, error correction, text comparison**

### Example Calculation
```
String A: "HELLO"
String B: "HALLO"

Position: H E L L O
          H A L L O
          ✓ ✗ ✓ ✓ ✓

Hamming Distance = 1 (only position 2 differs)

Binary Example:
A: [1, 0, 1, 1, 0]
B: [1, 1, 1, 0, 0]
   ✓ ✗ ✓ ✗ ✓

Hamming Distance = 2
```

### When to Use ✅
- **Categorical features** (color, brand, category)
- **Binary data** (yes/no, on/off)
- **Text comparison** (DNA sequences, error detection)
- **One-hot encoded data**

---

## 6. Jaccard Distance 🧩

### Formula
```
Jaccard Similarity = |A ∩ B| / |A ∪ B|
Jaccard Distance = 1 - Jaccard Similarity
```

### Intuition
- Measures **overlap between sets**
- **Size of intersection / Size of union**
- Perfect for **binary/categorical features**

### Example Calculation
```
Set A: {apple, banana, orange}
Set B: {banana, orange, grape}

Intersection: {banana, orange} → size = 2
Union: {apple, banana, orange, grape} → size = 4

Jaccard Similarity = 2/4 = 0.5
Jaccard Distance = 1 - 0.5 = 0.5
```

### When to Use ✅
- **Set-based data** (tags, categories, preferences)
- **Market basket analysis** (product purchases)
- **Text analysis** (unique words in documents)
- **Binary features** where presence/absence matters

---

## Distance Metrics Comparison Table

| Metric | Best For | Sensitive to Scale? | Robust to Outliers? | Computation Cost |
|--------|----------|-------------------|-------------------|------------------|
| **Euclidean** | Continuous, spatial data | ❌ Very | ❌ No | Medium |
| **Manhattan** | High-dim, mixed data | ✅ Less | ✅ More | Low |
| **Cosine** | Text, proportional data | ✅ No (normalized) | ✅ Yes | Medium |
| **Hamming** | Categorical, binary | ✅ N/A | ✅ Yes | Very Low |
| **Jaccard** | Sets, binary features | ✅ N/A | ✅ Yes | Low |

## Practical Guidelines

### 🎯 **Feature Type Based Selection**
```
📊 Continuous Features:
   ├── Similar scales → Euclidean
   ├── Different scales → Manhattan or normalize + Euclidean
   └── High dimensions → Manhattan or Cosine

🏷️ Categorical Features:
   ├── Binary → Hamming or Jaccard
   ├── Nominal → Hamming
   └── Text/Sets → Jaccard or Cosine

📝 Text Data:
   ├── Document similarity → Cosine
   ├── String comparison → Hamming
   └── Keyword overlap → Jaccard
```

### ⚖️ **Feature Scaling Impact**

**Without Scaling:**
```python
Feature 1: Age (20-80)          # Range: 60
Feature 2: Income (20k-200k)    # Range: 180,000

# Income dominates distance calculation!
```

**Solutions:**
1. **Use Manhattan/Cosine** (less sensitive)
2. **Normalize features** before using Euclidean
3. **Weight features** differently

### 🧪 **Experimental Approach**

```python
# Try multiple metrics and compare
metrics = ['euclidean', 'manhattan', 'cosine']
for metric in metrics:
    knn = KNeighborsClassifier(metric=metric)
    score = cross_val_score(knn, X, y, cv=5)
    print(f"{metric}: {score.mean():.3f}")
```

## Common Mistakes to Avoid ⚠️

1. **Using Euclidean without scaling** mixed-scale features
2. **Using Cosine for negative values** (can give counterintuitive results)
3. **Using Hamming for continuous** data
4. **Ignoring domain knowledge** when choosing metrics
5. **Not testing multiple metrics** to find the best one

## Summary & Next Steps

**Key Takeaways:**
- Distance metric choice is **crucial** for KNN performance
- **Euclidean**: Default for similar-scale continuous data
- **Manhattan**: Better for high-dimensional/mixed data
- **Cosine**: Perfect for text and proportional data
- **Hamming/Jaccard**: Essential for categorical data
- **Always experiment** with different metrics!

**Next Steps:**
- Learn how to [choose the right K value](choosing_k.md)
- Explore [real-world applications](applications.md)
- Practice with [hands-on implementation](knn.ipynb)

---

**Pro Tip**: In practice, start with Euclidean (with proper scaling), then try Manhattan and Cosine. The best metric often depends on your specific dataset and problem domain!
