import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

try:
    from nrms import data as nrms_data
except ModuleNotFoundError:
    import data as nrms_data


PAD_NEWS = nrms_data.PAD_NEWS
PAD_WORD = nrms_data.PAD_WORD


DATA_DIR = Path("data/mind-small")
ARTICLE_SIZE = 80
HISTORY_SIZE = 50
NEGATIVES_PER_POSITIVE = 4
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 64
EPOCHS = 5
MAX_STEPS = None
EVAL_IMPRESSIONS = 5000
MODEL_PATH = "nrms_simple_glove.pt"
CHECKPOINT_DIR = Path("checkpoints/nrms_glove")
TRAINING_HISTORY_PATH = CHECKPOINT_DIR / "training_history_glove.json"
LOG_INTERVAL_STEPS = 50
CHECKPOINT_INTERVAL_STEPS = None
VALIDATION_K_VALUES = (5, 10)
VALIDATION_NEWS_BATCH_SIZE = 512
BEST_MODEL_METRIC = "auc"

EMBEDDING_DIM = 100
ATTENTION_HEADS = 4
ATTENTION_HIDDEN_DIM = 200
DROPOUT = 0.2
LEARNING_RATE = 0.0001
USE_GLOVE = True
GLOVE_PATH = Path("data/GloVe/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt")
GLOVE_INIT_STD = 0.1

# TO RUN FASTER
ARTICLE_SIZE = 50
HISTORY_SIZE = 30
EMBEDDING_DIM = 100
ATTENTION_HEADS = 4
ATTENTION_HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 3
MAX_STEPS = None
EVAL_IMPRESSIONS = 1000

def keep_eval_items_unbatched(batch):
    return batch[0]


def save_model_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"saved checkpoint to {path}", flush=True)


def write_training_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def ranked_labels(labels, scores):
    ranking = np.argsort(-scores, kind="mergesort")
    return labels[ranking]


def ndcg_at_k(labels, scores, k):
    ranked = ranked_labels(labels, scores)
    cutoff = min(k, len(ranked))
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    dcg = float(np.sum(ranked[:cutoff] * discounts))
    ideal_hits = min(int(np.sum(ranked)), cutoff)
    if ideal_hits == 0:
        return float("nan")
    ideal_dcg = float(np.sum(discounts[:ideal_hits]))
    return dcg / ideal_dcg


def mrr(labels, scores):
    ranked = ranked_labels(labels, scores)
    positive_ranks = np.flatnonzero(ranked > 0)
    if len(positive_ranks) == 0:
        return float("nan")
    return float(1.0 / (positive_ranks[0] + 1))


def aggregate_metric_rows(rows):
    if not rows:
        return {}

    aggregated = {"impression_count": len(rows)}
    for metric_name in rows[0]:
        values = np.asarray([row[metric_name] for row in rows], dtype=np.float64)
        aggregated[metric_name] = (
            float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
        )
    return aggregated


def metric_is_better(current, best):
    if current is None or np.isnan(current):
        return False
    if best is None or np.isnan(best):
        return True
    return current > best


def format_validation_metrics(metrics):
    parts = []
    for metric_name in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        if metric_name in metrics:
            parts.append(f"{metric_name}={metrics[metric_name]:.4f}")
    return ", ".join(parts)


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, ATTENTION_HIDDEN_DIM)
        self.query = nn.Linear(ATTENTION_HIDDEN_DIM, 1, bias=False)

    def forward(self, vectors, mask):
        scores = self.query(torch.tanh(self.projection(vectors))).squeeze(-1)
        valid_rows = mask.any(dim=1)
        scores = scores.masked_fill(~mask, -1e9)
        scores = torch.where(valid_rows.unsqueeze(1), scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), vectors).squeeze(1)
        return pooled.masked_fill(~valid_rows.unsqueeze(1), 0.0)


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            expected_shape = (vocab_size, EMBEDDING_DIM)
            if tuple(pretrained_embeddings.shape) != expected_shape:
                raise ValueError(
                    f"Expected pretrained embeddings with shape {expected_shape}, "
                    f"got {tuple(pretrained_embeddings.shape)}"
                )
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,
                padding_idx=PAD_WORD,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_WORD)

        self.self_attention = nn.MultiheadAttention(
            EMBEDDING_DIM,
            ATTENTION_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.attention_pooling = AdditiveAttention(EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, articles):
        word_mask = articles.ne(PAD_WORD)
        valid_articles = word_mask.any(dim=1)

        word_vectors = self.dropout(self.embedding(articles))
        padding_mask = ~word_mask
        padding_mask = padding_mask.masked_fill(~valid_articles.unsqueeze(1), False)

        contextual_words, _ = self.self_attention(
            word_vectors,
            word_vectors,
            word_vectors,
            key_padding_mask=padding_mask,
        )

        news_vectors = self.attention_pooling(contextual_words, word_mask)
        return news_vectors.masked_fill(~valid_articles.unsqueeze(1), 0.0)


class NRMS(nn.Module):
    def __init__(self, vocab_size, pretrained_embeddings=None):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, pretrained_embeddings=pretrained_embeddings)
        self.user_self_attention = nn.MultiheadAttention(
            EMBEDDING_DIM,
            ATTENTION_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.user_attention_pooling = AdditiveAttention(EMBEDDING_DIM)

    def encode_user(self, history):
        batch_size, history_size, article_size = history.shape

        clicked_news = history.reshape(batch_size * history_size, article_size)
        clicked_vectors = self.news_encoder(clicked_news)
        clicked_vectors = clicked_vectors.reshape(batch_size, history_size, EMBEDDING_DIM)

        history_mask = history.ne(PAD_WORD).any(dim=2)
        return self.encode_user_vectors(clicked_vectors, history_mask)

    def encode_user_vectors(self, clicked_vectors, history_mask):
        valid_users = history_mask.any(dim=1)
        padding_mask = ~history_mask
        padding_mask = padding_mask.masked_fill(~valid_users.unsqueeze(1), False)

        contextual_clicks, _ = self.user_self_attention(
            clicked_vectors,
            clicked_vectors,
            clicked_vectors,
            key_padding_mask=padding_mask,
        )

        return self.user_attention_pooling(contextual_clicks, history_mask)

    def forward(self, history, candidates):
        batch_size, candidate_count, article_size = candidates.shape

        user_vector = self.encode_user(history)
        candidate_articles = candidates.reshape(batch_size * candidate_count, article_size)
        candidate_vectors = self.news_encoder(candidate_articles)
        candidate_vectors = candidate_vectors.reshape(batch_size, candidate_count, EMBEDDING_DIM)

        return torch.bmm(candidate_vectors, user_vector.unsqueeze(2)).squeeze(2)


@torch.no_grad()
def evaluate_validation(model, dataset, device, k_values=VALIDATION_K_VALUES):
    model.eval()
    metric_rows = []
    loader = DataLoader(dataset, batch_size=1, collate_fn=keep_eval_items_unbatched)

    for index, (history, candidates, labels) in enumerate(loader):
        if EVAL_IMPRESSIONS is not None and index == EVAL_IMPRESSIONS:
            break

        scores = model(
            history.unsqueeze(0).to(device),
            candidates.unsqueeze(0).to(device),
        )

        labels_array = labels.numpy()
        scores_array = scores.squeeze(0).cpu().numpy()
        row = {
            "auc": float(roc_auc_score(labels_array, scores_array)),
            "mrr": mrr(labels_array, scores_array),
        }
        for k in k_values:
            row[f"ndcg@{k}"] = ndcg_at_k(labels_array, scores_array, k)
        metric_rows.append(row)

    return aggregate_metric_rows(metric_rows)


def limited_eval_impressions(dataset):
    if EVAL_IMPRESSIONS is None:
        return dataset.impressions
    return dataset.impressions[:EVAL_IMPRESSIONS]


def collect_eval_news_ids(impressions):
    news_ids = {PAD_NEWS}
    for history, candidates, _ in impressions:
        news_ids.update(history[-HISTORY_SIZE:])
        news_ids.update(candidates)
    return news_ids


@torch.no_grad()
def precompute_news_vectors(model, news, news_ids, device, batch_size=VALIDATION_NEWS_BATCH_SIZE):
    model.eval()

    ordered_news_ids = [PAD_NEWS]
    ordered_news_ids.extend(
        news_id
        for news_id in news
        if news_id != PAD_NEWS and news_id in news_ids
    )
    id_to_index = {news_id: index for index, news_id in enumerate(ordered_news_ids)}

    vectors = []
    valid_news = []
    for start in range(0, len(ordered_news_ids), batch_size):
        batch_ids = ordered_news_ids[start : start + batch_size]
        batch_articles = [news.get(news_id, news[PAD_NEWS]) for news_id in batch_ids]
        articles = torch.tensor(batch_articles, dtype=torch.long, device=device)
        vectors.append(model.news_encoder(articles))
        valid_news.extend(any(word_id != PAD_WORD for word_id in article) for article in batch_articles)

    return (
        id_to_index,
        torch.cat(vectors, dim=0),
        torch.tensor(valid_news, dtype=torch.bool, device=device),
    )


def news_indexes(news_ids, id_to_index, pad_index, device):
    return torch.tensor(
        [id_to_index.get(news_id, pad_index) for news_id in news_ids],
        dtype=torch.long,
        device=device,
    )


def padded_history_ids(history):
    history = history[-HISTORY_SIZE:]
    return [PAD_NEWS] * (HISTORY_SIZE - len(history)) + history


@torch.no_grad()
def evaluate_validation_fast(model, dataset, device, k_values=VALIDATION_K_VALUES):
    model.eval()
    impressions = limited_eval_impressions(dataset)
    news_ids = collect_eval_news_ids(impressions)
    id_to_index, news_vectors, valid_news = precompute_news_vectors(
        model=model,
        news=dataset.news,
        news_ids=news_ids,
        device=device,
    )
    pad_index = id_to_index[PAD_NEWS]

    metric_rows = []
    for history, candidates, labels in impressions:
        history_indices = news_indexes(
            padded_history_ids(history),
            id_to_index,
            pad_index,
            device,
        )
        candidate_indices = news_indexes(candidates, id_to_index, pad_index, device)

        clicked_vectors = news_vectors[history_indices].unsqueeze(0)
        history_mask = valid_news[history_indices].unsqueeze(0)
        candidate_vectors = news_vectors[candidate_indices]

        user_vector = model.encode_user_vectors(clicked_vectors, history_mask)
        scores = torch.matmul(candidate_vectors, user_vector.squeeze(0))

        labels_array = np.asarray(labels, dtype=np.int64)
        scores_array = scores.cpu().numpy()
        row = {
            "auc": float(roc_auc_score(labels_array, scores_array)),
            "mrr": mrr(labels_array, scores_array),
        }
        for k in k_values:
            row[f"ndcg@{k}"] = ndcg_at_k(labels_array, scores_array, k)
        metric_rows.append(row)

    return aggregate_metric_rows(metric_rows)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_dir = DATA_DIR / "MINDsmall_train" / "MINDsmall_train"
    train_behaviors_path = train_dir / "behaviors.tsv"

    train_texts = nrms_data.read_news_texts(train_dir / "news.tsv")
    vocab = nrms_data.build_vocab(
        train_texts,
        max_vocab_size=MAX_VOCAB_SIZE,
    )
    news = nrms_data.encode_all_news(
        train_texts,
        vocab,
        article_size=ARTICLE_SIZE,
    )
    pretrained_embeddings = None
    glove_stats = None

    if USE_GLOVE:
        print(f"loading GloVe embeddings from {GLOVE_PATH}...", flush=True)
        pretrained_embeddings, glove_stats = nrms_data.load_glove_embedding_matrix(
            vocab=vocab,
            glove_path=GLOVE_PATH,
            embedding_dim=EMBEDDING_DIM,
            init_std=GLOVE_INIT_STD,
        )
        print(
            f"GloVe coverage: {glove_stats['matched_words']}/{glove_stats['vocab_size']} "
            f"vocab entries ({glove_stats['coverage']:.2%})",
            flush=True,
        )

    train_behavior_lines, validation_behavior_lines = nrms_data.mind_last_day_behavior_split(
        train_behaviors_path
    )
    split_metadata = {
        "strategy": "mind_last_training_day",
        "train_behavior_rows": len(train_behavior_lines),
        "validation_behavior_rows": len(validation_behavior_lines),
        "train_start": nrms_data.behavior_timestamp(train_behavior_lines[0]).isoformat(),
        "train_end": nrms_data.behavior_timestamp(train_behavior_lines[-1]).isoformat(),
        "validation_start": nrms_data.behavior_timestamp(
            validation_behavior_lines[0]
        ).isoformat(),
        "validation_end": nrms_data.behavior_timestamp(
            validation_behavior_lines[-1]
        ).isoformat(),
        "validation_date": nrms_data.behavior_timestamp(
            validation_behavior_lines[0]
        ).date().isoformat(),
        "source": str(train_behaviors_path),
        "final_evaluation_split": "MINDsmall_dev",
    }

    train_data = nrms_data.MindTrainDataset(
        train_behavior_lines,
        news,
        history_size=HISTORY_SIZE,
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
    )
    validation_data = nrms_data.MindEvalDataset(
        validation_behavior_lines,
        news,
        history_size=HISTORY_SIZE,
    )
    del train_behavior_lines, validation_behavior_lines

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NRMS(len(vocab), pretrained_embeddings=pretrained_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    print(f"device: {device}")
    print(f"vocabulary words: {len(vocab)}")
    print(f"training samples: {len(train_data)}")
    print(
        f"internal split: {split_metadata['train_behavior_rows']} train rows, "
        f"{split_metadata['validation_behavior_rows']} validation rows",
    )
    print(
        f"internal validation period: "
        f"{split_metadata['validation_start']} to {split_metadata['validation_end']}",
    )
    print(f"validation impressions: {len(validation_data)}")
    print(f"validation sample cap: {EVAL_IMPRESSIONS if EVAL_IMPRESSIONS is not None else 'all'}")
    print("MINDsmall_dev is reserved for final evaluation")

    step = 0
    total_batches = len(train_loader)
    best_metric_value = None
    training_history = {
        "config": {
            "article_size": ARTICLE_SIZE,
            "history_size": HISTORY_SIZE,
            "negative_per_positive": NEGATIVES_PER_POSITIVE,
            "max_vocab_size": MAX_VOCAB_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_steps": MAX_STEPS,
            "eval_impressions": EVAL_IMPRESSIONS,
            "embedding_dim": EMBEDDING_DIM,
            "attention_heads": ATTENTION_HEADS,
            "attention_hidden_dim": ATTENTION_HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "best_model_metric": BEST_MODEL_METRIC,
            "validation_mode": "fast_precomputed_news",
            "validation_news_batch_size": VALIDATION_NEWS_BATCH_SIZE,
            "use_glove": USE_GLOVE,
            "glove": glove_stats,
        },
        "epochs": [],
        "best_epoch": None,
        "best_metric_value": None,
        "best_model_path": MODEL_PATH,
        "data_split": split_metadata,
    }

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        epoch_started_at = time.time()

        for batch_index, (history, candidates, labels) in enumerate(train_loader, start=1):
            history = history.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)

            scores = model(history, candidates)
            loss = loss_function(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % LOG_INTERVAL_STEPS == 0 or batch_index == total_batches:
                elapsed_minutes = (time.time() - epoch_started_at) / 60
                average_loss = float(np.mean(losses[-LOG_INTERVAL_STEPS:]))
                print(
                    f"epoch {epoch + 1}/{EPOCHS} "
                    f"batch {batch_index}/{total_batches} "
                    f"step {step} "
                    f"recent_loss={average_loss:.4f} "
                    f"elapsed={elapsed_minutes:.1f}m",
                    flush=True,
                )

            if CHECKPOINT_INTERVAL_STEPS and step % CHECKPOINT_INTERVAL_STEPS == 0:
                save_model_checkpoint(
                    model,
                    CHECKPOINT_DIR / f"nrms_step_{step:06d}.pt",
                )

            if MAX_STEPS and step >= MAX_STEPS:
                break

        save_model_checkpoint(
            model,
            CHECKPOINT_DIR / f"nrms_epoch_{epoch + 1:02d}.pt",
        )

        print("evaluating validation metrics...", flush=True)
        validation_metrics = evaluate_validation_fast(model, validation_data, device)
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        metric_value = validation_metrics.get(BEST_MODEL_METRIC)

        epoch_record = {
            "epoch": epoch + 1,
            "step": step,
            "loss": epoch_loss,
            "validation": validation_metrics,
            "elapsed_minutes": (time.time() - epoch_started_at) / 60,
        }
        training_history["epochs"].append(epoch_record)

        print(
            f"epoch {epoch + 1}: "
            f"loss={epoch_loss:.4f}, "
            f"{format_validation_metrics(validation_metrics)}",
            flush=True,
        )

        if metric_is_better(metric_value, best_metric_value):
            best_metric_value = metric_value
            training_history["best_epoch"] = epoch + 1
            training_history["best_metric_value"] = best_metric_value
            if MODEL_PATH:
                save_model_checkpoint(model, Path(MODEL_PATH))
                save_model_checkpoint(model, CHECKPOINT_DIR / "nrms_best.pt")
                print(
                    f"new best model: epoch {epoch + 1}, "
                    f"{BEST_MODEL_METRIC}={best_metric_value:.4f}",
                    flush=True,
                )

        save_model_checkpoint(model, CHECKPOINT_DIR / "nrms_last.pt")
        write_training_history(training_history, TRAINING_HISTORY_PATH)

        if MAX_STEPS and step >= MAX_STEPS:
            break

    write_training_history(training_history, TRAINING_HISTORY_PATH)
    if training_history["best_epoch"] is None:
        print("warning: no best validation checkpoint was selected", flush=True)
    else:
        print(
            f"best epoch: {training_history['best_epoch']} "
            f"{BEST_MODEL_METRIC}={training_history['best_metric_value']:.4f}",
            flush=True,
        )
        print(f"saved training history to {TRAINING_HISTORY_PATH}", flush=True)


if __name__ == "__main__":
    main()
