# User Clustering Analysis

## Size Balance

- Total users: 49108
- Cluster count: 10
- Largest cluster share: 0.264
- Smallest cluster share: 0.000
- Largest/smallest ratio: 2595.00

## Visual Separation

- Sampled silhouette on saved PCA coordinates: 0.1059

This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.

## Content Summaries

### Cluster 0

- Users: 4697
- Dominant category: news (0.389 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, newspolitics, football_nfl

### Cluster 1

- Users: 3184
- Dominant category: news (0.354 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, newspolitics, football_nfl

### Cluster 2

- Users: 2416
- Dominant category: news (0.399 of listed category clicks)
- Top categories: news, sports, tv
- Top subcategories: newsus, newscrime, tv-celebrity

### Cluster 3

- Users: 11402
- Dominant category: news (0.323 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, football_nfl, newspolitics

### Cluster 4

- Users: 3718
- Dominant category: news (0.281 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, football_nfl, tv-celebrity

### Cluster 5

- Users: 12975
- Dominant category: news (0.391 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, newscrime, newspolitics

### Cluster 6

- Users: 2734
- Dominant category: news (0.327 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsopinion, newsus, football_nfl

### Cluster 7

- Users: 5
- Dominant category: tv (0.800 of listed category clicks)
- Top categories: tv, finance
- Top subcategories: tv-celebrity, finance-companies, tv-gallery

### Cluster 8

- Users: 6145
- Dominant category: news (0.337 of listed category clicks)
- Top categories: news, sports, lifestyle
- Top subcategories: newsus, football_nfl, newspolitics

### Cluster 9

- Users: 1832
- Dominant category: news (0.372 of listed category clicks)
- Top categories: news, tv, lifestyle
- Top subcategories: newsus, tv-celebrity, newscrime

## Centroid Similarity

- Mean off-diagonal cosine similarity: 0.6654
- Max off-diagonal cosine similarity: 0.9989
- Min off-diagonal cosine similarity: -0.6815

High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.
