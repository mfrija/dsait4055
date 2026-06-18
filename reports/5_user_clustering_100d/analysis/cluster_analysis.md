# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 5
- Largest cluster share: 0.211
- Smallest cluster share: 0.177
- Largest/smallest ratio: 1.19

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.1489

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 9783
- Dominant category: sports (0.439 of listed category clicks)
- Top categories: sports, news, finance
- Top subcategories: football_nfl, newsus, football_ncaa

### Cluster 1

- Users: 10163
- Dominant category: lifestyle (0.238 of listed category clicks)
- Top categories: lifestyle, news, tv
- Top subcategories: tv-celebrity, lifestyleroyals, newsus

### Cluster 2

- Users: 8671
- Dominant category: news (0.559 of listed category clicks)
- Top categories: news, sports, finance
- Top subcategories: newspolitics, newsus, newsworld

### Cluster 3

- Users: 10132
- Dominant category: news (0.452 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, newscrime, tv-celebrity

### Cluster 4

- Users: 10359
- Dominant category: news (0.279 of listed category clicks)
- Top categories: news, lifestyle, finance
- Top subcategories: newsus, lifestylebuzz, newsworld

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.8069
- Max off-diagonal cosine similarity: 0.9007
- Min off-diagonal cosine similarity: 0.6410

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
