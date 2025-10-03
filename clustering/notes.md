# Clustering Notes – Quick Reference

## Key Concepts
- Similarity: distance metrics (Euclidean, Manhattan, Cosine)
- Cohesion vs Separation: within-cluster compactness vs between-cluster distance
- Scaling: standardize features for distance-based methods
- Outliers: can distort centroids and distances

## Algorithms at a Glance
- K-means: fast, spherical clusters, needs k, sensitive to init/outliers
- Hierarchical (Agglomerative): linkage choices, dendrogram, no k upfront, O(n^2)
- DBSCAN: density-based, finds arbitrary shapes, handles noise, sensitive to eps/min_samples

## Choosing an Algorithm
- Spherical, balanced clusters, large n: K-means / MiniBatchKMeans
- Arbitrary shapes, noise present: DBSCAN (or HDBSCAN for varying densities)
- Need multi-scale structure or small n: Hierarchical (Ward/average)

## Distance Metrics
- Euclidean: continuous, after scaling
- Manhattan: robust to outliers, L1 geometry
- Cosine: orientation similarity (text, embeddings)
- Haversine: geospatial lat/long

## Feature Engineering Tips
- Scale numeric, one-hot encode categoricals (or use appropriate distances)
- Dimensionality reduction (PCA/UMAP/t-SNE) for visualization and sometimes performance
- Remove/clip extreme outliers or use robust scalers

## Practical Workflow
1. EDA and scaling
2. Try multiple algorithms and parameters
3. Evaluate with internal metrics and domain sense
4. Visualize clusters; iterate
5. Validate stability across seeds/subsamples

## Interview Questions
1. Why does K-means prefer spherical clusters? What does inertia measure?
2. Compare single vs complete vs average vs Ward linkage
3. How do you pick eps and min_samples for DBSCAN? What’s the k-distance plot?
4. What are silhouette and Davies–Bouldin indices? Interpret ranges.
5. How to handle high-dimensional data in clustering?
6. Differences between K-means and GMM for clustering?
7. When would hierarchical clusterings be preferable to K-means?

## Limitations Table (High-level)
- K-means: needs k, poor on non-spherical, sensitive to outliers
- Hierarchical: computationally heavy, irreversibility, sensitive to noise
- DBSCAN: single global eps struggles with varying density, parameter sensitive

## Use Cases
- Customer segmentation
- Anomaly detection (DBSCAN noise label)
- Document/topic grouping (cosine distance)
- Image color quantization (K-means)
- Taxonomy/phylogeny (hierarchical)

See detailed guides: kmeans.md, hierarchical_clustering.md, dbscan.md, cluster_evaluation.md
