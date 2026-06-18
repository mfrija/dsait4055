# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 5
- Largest cluster share: 0.249
- Smallest cluster share: 0.162
- Largest/smallest ratio: 1.53

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.1040

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 8503
- Dominant category: news (0.546 of listed category clicks)
- Top categories: news, sports, finance
- Top subcategories: newspolitics, newsus, newsworld

### Cluster 1

- Users: 10467
- Dominant category: sports (0.424 of listed category clicks)
- Top categories: sports, news, tv
- Top subcategories: football_nfl, newsus, football_ncaa

### Cluster 2

- Users: 7962
- Dominant category: news (0.493 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newsus, newscrime, tv-celebrity

### Cluster 3

- Users: 12210
- Dominant category: lifestyle (0.227 of listed category clicks)
- Top categories: lifestyle, news, tv
- Top subcategories: tv-celebrity, newsus, lifestyleroyals

### Cluster 4

- Users: 9966
- Dominant category: news (0.319 of listed category clicks)
- Top categories: news, finance, lifestyle
- Top subcategories: newsus, newsworld, lifestylebuzz

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.5448
- Max off-diagonal cosine similarity: 0.7536
- Min off-diagonal cosine similarity: 0.4053

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
