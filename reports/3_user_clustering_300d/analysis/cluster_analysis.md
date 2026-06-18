# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 3
- Largest cluster share: 0.400
- Smallest cluster share: 0.268
- Largest/smallest ratio: 1.49

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.4187

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 13150
- Dominant category: sports (0.392 of listed category clicks)
- Top categories: sports, news, tv
- Top subcategories: football_nfl, newsus, newspolitics

### Cluster 1

- Users: 19621
- Dominant category: news (0.305 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newsus, tv-celebrity, newscrime

### Cluster 2

- Users: 16337
- Dominant category: news (0.469 of listed category clicks)
- Top categories: news, finance, lifestyle
- Top subcategories: newsus, newspolitics, newsworld

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.5645
- Max off-diagonal cosine similarity: 0.6530
- Min off-diagonal cosine similarity: 0.5051

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
