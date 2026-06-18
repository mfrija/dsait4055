# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 8
- Largest cluster share: 0.152
- Smallest cluster share: 0.100
- Largest/smallest ratio: 1.52

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.1189

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 6299
- Dominant category: lifestyle (0.212 of listed category clicks)
- Top categories: lifestyle, news, foodanddrink
- Top subcategories: newsus, lifestylebuzz, tv-celebrity

### Cluster 1

- Users: 5691
- Dominant category: news (0.588 of listed category clicks)
- Top categories: news, sports, finance
- Top subcategories: newspolitics, newsus, newsworld

### Cluster 2

- Users: 6690
- Dominant category: lifestyle (0.243 of listed category clicks)
- Top categories: lifestyle, news, tv
- Top subcategories: tv-celebrity, lifestyleroyals, newsus

### Cluster 3

- Users: 4916
- Dominant category: news (0.451 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newspolitics, newsus, tv-celebrity

### Cluster 4

- Users: 6272
- Dominant category: news (0.486 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newsus, newscrime, tv-celebrity

### Cluster 5

- Users: 5945
- Dominant category: sports (0.500 of listed category clicks)
- Top categories: sports, news, finance
- Top subcategories: football_nfl, football_ncaa, newsus

### Cluster 6

- Users: 5836
- Dominant category: sports (0.318 of listed category clicks)
- Top categories: sports, news, tv
- Top subcategories: football_nfl, newsus, tv-celebrity

### Cluster 7

- Users: 7459
- Dominant category: news (0.343 of listed category clicks)
- Top categories: news, finance, autos
- Top subcategories: newsus, newsworld, newscrime

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.8083
- Max off-diagonal cosine similarity: 0.9268
- Min off-diagonal cosine similarity: 0.5214

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
