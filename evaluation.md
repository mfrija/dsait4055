# Evaluation Pipeline

The evaluation code measures how well a recommendation model ranks the clicked news articles inside each MIND impression.

Each impression contains:

- a `user_id`
- a clicked-history list
- a candidate news list
- one binary label per candidate, where `1` means clicked and `0` means not clicked

The model receives the impression and returns one score per candidate. The evaluator sorts candidates by score and computes ranking metrics from the sorted labels.

## Data Flow

The evaluation flow is:

1. Read the evaluation `behaviors.tsv`.
2. Convert each row into a `MindImpression`.
3. Pass each `MindImpression` to a scorer.
4. The scorer returns one score per candidate.
5. Convert the impression into a `ScoredImpression`.
6. Compute ranking metrics per impression.
7. Assign the impression to a user group.
8. Average metrics across impressions in each group.
9. Save the final report and plots.

## MIND User Groups

The grouped evaluation follows the MIND-style split:

- `overall`
  - All evaluated impressions.

- `overlap_users`
  - Impressions from users that also appear in the training behaviors file.

- `unseen_users`
  - Impressions from users that do not appear in the training behaviors file.

The train-user set is built from:

```txt
data/mind-small/MINDsmall_train/MINDsmall_train/behaviors.tsv
```

The default evaluation file is:

```txt
data/mind-small/MINDsmall_dev/MINDsmall_dev/behaviors.tsv
```

## Metrics

All metrics are computed per impression first, then averaged across impressions in each group.

### Precision@k

Precision@k measures how many of the top `k` ranked candidates are clicked items.

```txt
Precision@k = clicked items in top k / k
```

This is useful when the user interface would only show the first `k` recommendations.

### Recall@k

Recall@k measures how many of the clicked candidates were recovered in the top `k`.

```txt
Recall@k = clicked items in top k / total clicked items in the impression
```

This is useful when an impression can contain multiple clicked candidates.

### NDCG@k

NDCG@k measures ranking quality with position discounting. Clicked items near the top contribute more than clicked items lower in the ranking.

The implementation computes DCG from the model ranking and divides it by the ideal DCG for the same labels.

### MRR

MRR, mean reciprocal rank, uses the rank of the first clicked candidate.

```txt
MRR = 1 / rank of first clicked candidate
```

For example, if the first clicked candidate is ranked first, MRR is `1.0`. If it is ranked fourth, MRR is `0.25`.

### AUC

AUC measures whether clicked candidates tend to receive higher scores than non-clicked candidates.

The implementation uses `sklearn.metrics.roc_auc_score`.

If an impression has only one label class, AUC is undefined for that impression and is stored as `nan`. During aggregation, `nan` values are ignored unless all values in a group are `nan`.

## NRMS Evaluation

`NRMSScorer` evaluates the saved NRMS model. It does not train the model.

It performs these steps:

1. Reads train and dev `news.tsv` files.
2. Builds the vocabulary from training news titles, matching `nrms/pipeline.py`.
3. Encodes all known train/dev news titles.
4. Builds the NRMS architecture.
5. Loads the checkpoint state dict from `nrms_simple.pt`.
6. For each impression:
   - pads/truncates history to `HISTORY_SIZE`
   - encodes candidate news IDs
   - runs `model(history, candidates)`
   - returns scores as Python floats

The adapter uses the same constants from `nrms/pipeline.py`, including:

- `TITLE_SIZE`
- `HISTORY_SIZE`
- `PAD_NEWS`
- `PAD_WORD`

This keeps evaluation encoding consistent with the current NRMS implementation.

## Running Evaluation

```python
from evaluation import NRMSScorer, evaluate_mind

scorer = NRMSScorer.from_mind_data_dir(
    data_dir="data/mind-small",
    checkpoint_path="nrms_simple.pt",
)

report = evaluate_mind(
    behaviors_path="data/mind-small/MINDsmall_dev/MINDsmall_dev/behaviors.tsv",
    train_behaviors_path="data/mind-small/MINDsmall_train/MINDsmall_train/behaviors.tsv",
    scorer=scorer,
    k_values=(5, 10),
)

print(report)
```

## Outputs

When `save_outputs=True`, the helper writes three files to the output directory.

### report.json

Structured metric report.

### metrics.png

Bar charts for each metric, comparing:

- `overall`
- `overlap_users`
- `unseen_users`

### counts.png

Bar charts for:

- `impression_count`
- `user_count`
- `empty_history_count`

This helps verify whether the overlap/unseen split is balanced or heavily skewed.



## Important Notes

The current NRMS model is a simplified implementation that uses only news titles. Therefore, the evaluation scores measure this simplified title-only model, not a full MIND benchmark model.

The evaluation is impression-level. It ranks only the candidates provided in each MIND impression. It does not retrieve candidates from the full news corpus.
