# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 10
- Largest cluster share: 0.110
- Smallest cluster share: 0.083
- Largest/smallest ratio: 1.34

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.0928

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 4907
- Dominant category: news (0.517 of listed category clicks)
- Top categories: news, sports, finance
- Top subcategories: newsus, newscrime, newsworld

### Cluster 1

- Users: 4242
- Dominant category: news (0.440 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newspolitics, newsus, tv-celebrity

### Cluster 2

- Users: 5357
- Dominant category: news (0.591 of listed category clicks)
- Top categories: news, sports, finance
- Top subcategories: newspolitics, newsus, newsworld

### Cluster 3

- Users: 4808
- Dominant category: news (0.300 of listed category clicks)
- Top categories: news, finance, sports
- Top subcategories: newsus, football_nfl, newsworld

### Cluster 4

- Users: 5271
- Dominant category: news (0.429 of listed category clicks)
- Top categories: news, lifestyle, tv
- Top subcategories: newscrime, newsus, tv-celebrity

### Cluster 5

- Users: 5271
- Dominant category: news (0.235 of listed category clicks)
- Top categories: news, lifestyle, foodanddrink
- Top subcategories: newsus, lifestylebuzz, foodnews

### Cluster 6

- Users: 5325
- Dominant category: sports (0.331 of listed category clicks)
- Top categories: sports, news, tv
- Top subcategories: football_nfl, newsus, tv-celebrity

### Cluster 7

- Users: 4450
- Dominant category: lifestyle (0.229 of listed category clicks)
- Top categories: lifestyle, news, tv
- Top subcategories: lifestylebuzz, tv-celebrity, newsus

### Cluster 8

- Users: 5424
- Dominant category: sports (0.516 of listed category clicks)
- Top categories: sports, news, finance
- Top subcategories: football_nfl, football_ncaa, newsus

### Cluster 9

- Users: 4053
- Dominant category: lifestyle (0.269 of listed category clicks)
- Top categories: lifestyle, tv, news
- Top subcategories: lifestyleroyals, tv-celebrity, newsus

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.8028
- Max off-diagonal cosine similarity: 0.9276
- Min off-diagonal cosine similarity: 0.4646

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
