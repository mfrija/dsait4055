# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 5
- Largest cluster share: 0.325
- Smallest cluster share: 0.124
- Largest/smallest ratio: 2.62

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.1193

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 9194
- Dominant category: news (0.378 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, newspolitics, football_nfl

### Cluster 1

- Users: 9984
- Dominant category: news (0.305 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, football_nfl, tv-celebrity

### Cluster 2

- Users: 15956
- Dominant category: news (0.376 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, newspolitics, football_nfl

### Cluster 3

- Users: 6084
- Dominant category: news (0.401 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newsus, newscrime, tv-celebrity

### Cluster 4

- Users: 7890
- Dominant category: news (0.311 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, football_nfl, newspolitics

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.9970
- Max off-diagonal cosine similarity: 0.9986
- Min off-diagonal cosine similarity: 0.9918

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
