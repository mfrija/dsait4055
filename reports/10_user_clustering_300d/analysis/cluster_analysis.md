# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 10
- Largest cluster share: 0.124
- Smallest cluster share: 0.081
- Largest/smallest ratio: 1.53

## Visual Separation

- Sampled silhouette on saved PCA coordinates: -0.0057

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 4097
- Dominant category: news (0.514 of listed category clicks)
- Top categories: news, finance, sports
- Top subcategories: newsus, newsworld, newscrime

### Cluster 1

- Users: 6069
- Dominant category: tv (0.197 of listed category clicks)
- Top categories: tv, news, lifestyle
- Top subcategories: tv-celebrity, newsus, lifestylebuzz

### Cluster 2

- Users: 5743
- Dominant category: news (0.207 of listed category clicks)
- Top categories: news, lifestyle, foodanddrink
- Top subcategories: newsus, lifestylebuzz, foodnews

### Cluster 3

- Users: 4101
- Dominant category: sports (0.332 of listed category clicks)
- Top categories: sports, news, tv
- Top subcategories: football_nfl, newsus, newscrime

### Cluster 4

- Users: 4970
- Dominant category: news (0.292 of listed category clicks)
- Top categories: news, autos, lifestyle
- Top subcategories: newsus, autosenthusiasts, lifestylebuzz

### Cluster 5

- Users: 4583
- Dominant category: news (0.594 of listed category clicks)
- Top categories: news, finance, sports
- Top subcategories: newspolitics, newsus, newsworld

### Cluster 6

- Users: 5147
- Dominant category: news (0.302 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: lifestyleroyals, newsus, tv-celebrity

### Cluster 7

- Users: 4812
- Dominant category: sports (0.471 of listed category clicks)
- Top categories: sports, news, finance
- Top subcategories: football_nfl, football_ncaa, newsus

### Cluster 8

- Users: 3979
- Dominant category: news (0.370 of listed category clicks)
- Top categories: news, sports, finance
- Top subcategories: football_nfl, newspolitics, newsus

### Cluster 9

- Users: 5607
- Dominant category: news (0.502 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newscrime, newsus, tv-celebrity

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.4256
- Max off-diagonal cosine similarity: 0.7757
- Min off-diagonal cosine similarity: 0.0257

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
