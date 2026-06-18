# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 3
- Largest cluster share: 0.366
- Smallest cluster share: 0.310
- Largest/smallest ratio: 1.18

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.3487

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 17991
- Dominant category: news (0.397 of listed category clicks)
- Top categories: news, sports, finance
- Top subcategories: football_nfl, newspolitics, newsus

### Cluster 1

- Users: 15903
- Dominant category: news (0.354 of listed category clicks)
- Top categories: news, lifestyle, finance
- Top subcategories: newsus, newscrime, newsworld

### Cluster 2

- Users: 15214
- Dominant category: news (0.283 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: tv-celebrity, newsus, newscrime

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.8245
- Max off-diagonal cosine similarity: 0.8770
- Min off-diagonal cosine similarity: 0.7378

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
