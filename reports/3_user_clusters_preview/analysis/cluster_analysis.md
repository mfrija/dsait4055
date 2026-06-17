# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 3
- Largest cluster share: 0.357
- Smallest cluster share: 0.295
- Largest/smallest ratio: 1.21

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.2655

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 17099
- Dominant category: news (0.332 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, football_nfl, newspolitics

### Cluster 1

- Users: 17509
- Dominant category: news (0.346 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, newspolitics, football_nfl

### Cluster 2

- Users: 14500
- Dominant category: news (0.405 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, newscrime, tv-celebrity

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.9979
- Max off-diagonal cosine similarity: 0.9983
- Min off-diagonal cosine similarity: 0.9971

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
