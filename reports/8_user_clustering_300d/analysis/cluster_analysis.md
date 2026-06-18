# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 8
- Largest cluster share: 0.149
- Smallest cluster share: 0.095
- Largest/smallest ratio: 1.56

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.0465

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 7307
- Dominant category: news (0.556 of listed category clicks)
- Top categories: news, sports, finance
- Top subcategories: newspolitics, newsus, newsworld

### Cluster 1

- Users: 4670
- Dominant category: news (0.311 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: lifestyleroyals, newsus, tv-celebrity

### Cluster 2

- Users: 5480
- Dominant category: sports (0.318 of listed category clicks)
- Top categories: sports, news, tv
- Top subcategories: football_nfl, newsus, tv-celebrity

### Cluster 3

- Users: 5906
- Dominant category: news (0.202 of listed category clicks)
- Top categories: news, lifestyle, foodanddrink
- Top subcategories: newsus, lifestylebuzz, foodnews

### Cluster 4

- Users: 6999
- Dominant category: news (0.370 of listed category clicks)
- Top categories: news, finance, autos
- Top subcategories: newsus, newsworld, newscrime

### Cluster 5

- Users: 6698
- Dominant category: sports (0.456 of listed category clicks)
- Top categories: sports, news, finance
- Top subcategories: football_nfl, newsus, football_ncaa

### Cluster 6

- Users: 6045
- Dominant category: tv (0.198 of listed category clicks)
- Top categories: tv, news, lifestyle
- Top subcategories: tv-celebrity, newsus, lifestylebuzz

### Cluster 7

- Users: 6003
- Dominant category: news (0.521 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newsus, newscrime, newsworld

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.5443
- Max off-diagonal cosine similarity: 0.8374
- Min off-diagonal cosine similarity: 0.2076

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
