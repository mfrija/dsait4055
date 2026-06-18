import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from nrms import data as nrms_data
from nrms import pipeline


# Select the trained model and clustering experiment here.
MODEL_NAME = "3_100d"
CLUSTER_COUNT = 3
MODEL_CONFIGS = {
    "3_100d": {
        "checkpoint_path": Path("nrms_100d.pt"),
        "training_history_path": Path("checkpoints/nrms_100d/training_history.json"),
        "output_dir": Path("reports/3_user_clustering_100d"),
    },
    "5_100d": {
        "checkpoint_path": Path("nrms_100d.pt"),
        "training_history_path": Path("checkpoints/nrms_100d/training_history.json"),
        "output_dir": Path("reports/5_user_clustering_100d"),
    },
    "8_100d": {
        "checkpoint_path": Path("nrms_100d.pt"),
        "training_history_path": Path("checkpoints/nrms_100d/training_history.json"),
        "output_dir": Path("reports/8_user_clustering_100d"),
    },
    "10_100d": {
        "checkpoint_path": Path("nrms_100d.pt"),
        "training_history_path": Path("checkpoints/nrms_100d/training_history.json"),
        "output_dir": Path("reports/10_user_clustering_100d"),
    },
    "300d": {
        "checkpoint_path": Path("nrms_300d.pt"),
        "training_history_path": Path("checkpoints/nrms_300d/training_history.json"),
        "output_dir": Path("reports/user_clustering_300d"),
    },
}

DATA_DIR = Path("data/mind-small")
BEHAVIORS_SPLIT = "train"
MAX_USERS = None
USER_VECTOR_BATCH_SIZE = 128
TOP_N_SUMMARY_ITEMS = 8
RANDOM_SEED = 42
KMEANS_N_INIT = 10
DEVICE = torch.device("cpu")

REQUIRED_TRAINING_CONFIG = {
    "article_size",
    "history_size",
    "max_vocab_size",
    "embedding_dim",
    "attention_heads",
    "attention_hidden_dim",
}
CONFIG_TO_PIPELINE_CONSTANT = {
    "article_size": "ARTICLE_SIZE",
    "history_size": "HISTORY_SIZE",
    "max_vocab_size": "MAX_VOCAB_SIZE",
    "embedding_dim": "EMBEDDING_DIM",
    "attention_heads": "ATTENTION_HEADS",
    "attention_hidden_dim": "ATTENTION_HIDDEN_DIM",
    "dropout": "DROPOUT",
}


def current_pipeline_config():
    return {
        "article_size": pipeline.ARTICLE_SIZE,
        "history_size": pipeline.HISTORY_SIZE,
        "max_vocab_size": pipeline.MAX_VOCAB_SIZE,
        "embedding_dim": pipeline.EMBEDDING_DIM,
        "attention_heads": pipeline.ATTENTION_HEADS,
        "attention_hidden_dim": pipeline.ATTENTION_HIDDEN_DIM,
        "dropout": pipeline.DROPOUT,
    }


def read_training_config(path):
    path = Path(path)
    history = json.loads(path.read_text(encoding="utf-8"))
    config = history.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Training history has no config object: {path}")

    missing = sorted(REQUIRED_TRAINING_CONFIG - config.keys())
    if missing:
        raise ValueError(f"Training history {path} is missing config values: {missing}")
    return config


def apply_training_config(config):
    for config_key, constant_name in CONFIG_TO_PIPELINE_CONSTANT.items():
        if config_key in config and config[config_key] is not None:
            setattr(pipeline, constant_name, config[config_key])
    pipeline.validate_training_config()


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_dimensions(state_dict):
    embedding = state_dict["news_encoder.embedding.weight"]
    projection = state_dict["news_encoder.attention_pooling.projection.weight"]
    return {
        "vocab_size": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]),
        "attention_hidden_dim": int(projection.shape[0]),
    }


def read_news_metadata(path):
    metadata = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 5:
                continue
            metadata[columns[0]] = {
                "category": columns[1],
                "subcategory": columns[2],
                "title": columns[3],
            }
    return metadata


def read_latest_user_histories(path, max_users=None):
    histories = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 5:
                continue

            user_id = columns[1]
            history = columns[3].split()
            if history:
                histories[user_id] = history

            if max_users and len(histories) >= max_users:
                break

    return histories


def build_news_lookup(data_dir, config=None):
    config = config or current_pipeline_config()
    train_dir = data_dir / "MINDsmall_train" / "MINDsmall_train"
    dev_dir = data_dir / "MINDsmall_dev" / "MINDsmall_dev"

    train_texts = nrms_data.read_news_texts(train_dir / "news.tsv")
    dev_texts = nrms_data.read_news_texts(dev_dir / "news.tsv")
    vocab = nrms_data.build_vocab(
        train_texts,
        max_vocab_size=int(config["max_vocab_size"]),
    )
    encoded_news = nrms_data.encode_all_news(
        {**train_texts, **dev_texts},
        vocab,
        article_size=int(config["article_size"]),
    )

    metadata = read_news_metadata(train_dir / "news.tsv")
    metadata.update(read_news_metadata(dev_dir / "news.tsv"))

    return vocab, encoded_news, metadata


def encode_history(history, encoded_news, history_size=None):
    history_size = history_size or pipeline.HISTORY_SIZE
    history = history[-history_size:]
    history = [nrms_data.PAD_NEWS] * (history_size - len(history)) + history
    return [
        encoded_news.get(news_id, encoded_news[nrms_data.PAD_NEWS])
        for news_id in history
    ]


def load_model(vocab_size, checkpoint_path, device, config=None):
    config = config or current_pipeline_config()
    state_dict = load_state_dict(checkpoint_path, device)
    dimensions = checkpoint_dimensions(state_dict)
    expected_dimensions = {
        "vocab_size": vocab_size,
        "embedding_dim": int(config["embedding_dim"]),
        "attention_hidden_dim": int(config["attention_hidden_dim"]),
    }
    if dimensions != expected_dimensions:
        raise ValueError(
            "Checkpoint dimensions do not match its training configuration. "
            f"checkpoint={dimensions}, expected={expected_dimensions}"
        )

    model = pipeline.NRMS(vocab_size).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(
            "The checkpoint does not match the current NRMS settings. "
            "Retrain nrms/pipeline.py first, then rerun clustering."
        ) from error
    model.eval()
    return model


@torch.no_grad()
def extract_user_vectors(
    model,
    user_histories,
    encoded_news,
    history_size,
    device,
    batch_size,
):
    user_ids = list(user_histories)
    vectors = []

    for start in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[start : start + batch_size]
        batch_histories = [
            encode_history(user_histories[user_id], encoded_news, history_size)
            for user_id in batch_user_ids
        ]
        history_tensor = torch.tensor(batch_histories, dtype=torch.long, device=device)
        user_vectors = model.encode_user(history_tensor)
        vectors.append(user_vectors.cpu().numpy())

    return user_ids, np.vstack(vectors)


def summarize_clusters(user_ids, labels, user_histories, metadata, top_n):
    cluster_users = defaultdict(list)
    for user_id, label in zip(user_ids, labels):
        cluster_users[int(label)].append(user_id)

    summaries = []
    for cluster_id in sorted(cluster_users):
        users = cluster_users[cluster_id]
        category_counts = Counter()
        subcategory_counts = Counter()
        title_word_counts = Counter()

        for user_id in users:
            for news_id in user_histories[user_id]:
                news = metadata.get(news_id)
                if not news:
                    continue
                category_counts[news["category"]] += 1
                subcategory_counts[news["subcategory"]] += 1
                title_word_counts.update(nrms_data.tokenize(news["title"]))

        summaries.append(
            {
                "cluster": cluster_id,
                "user_count": len(users),
                "top_categories": category_counts.most_common(top_n),
                "top_subcategories": subcategory_counts.most_common(top_n),
                "top_title_words": title_word_counts.most_common(top_n),
            }
        )

    return summaries


def save_user_clusters(path, user_ids, labels, points_2d=None):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["user_id", "cluster"]
        if points_2d is not None:
            header += ["x", "y"]
        writer.writerow(header)

        for index, (user_id, label) in enumerate(zip(user_ids, labels)):
            row = [user_id, int(label)]
            if points_2d is not None:
                row += [float(points_2d[index, 0]), float(points_2d[index, 1])]
            writer.writerow(row)


def save_cluster_summary(path, summaries):
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"clusters": summaries}, file, indent=2)


def save_cluster_centroids(path, centroids):
    np.save(path, centroids)


def raw_centroids_from_labels(vectors, labels, cluster_count):
    centroids = []
    for cluster_id in range(cluster_count):
        cluster_vectors = vectors[labels == cluster_id]
        centroids.append(cluster_vectors.mean(axis=0))
    return np.vstack(centroids)


def run_user_clustering(
    checkpoint_path,
    training_history_path,
    output_dir,
    data_dir=DATA_DIR,
    behaviors_split=BEHAVIORS_SPLIT,
    cluster_count=CLUSTER_COUNT,
    max_users=MAX_USERS,
    batch_size=USER_VECTOR_BATCH_SIZE,
    top_n=TOP_N_SUMMARY_ITEMS,
    seed=RANDOM_SEED,
    kmeans_n_init=KMEANS_N_INIT,
    device=DEVICE,
):
    checkpoint_path = Path(checkpoint_path)
    training_history_path = Path(training_history_path)
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)

    if behaviors_split not in {"train", "dev"}:
        raise ValueError("behaviors_split must be 'train' or 'dev'")
    if cluster_count <= 1:
        raise ValueError("cluster_count must be greater than 1")

    config = read_training_config(training_history_path)
    apply_training_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    behaviors_dir = "MINDsmall_train" if behaviors_split == "train" else "MINDsmall_dev"
    behaviors_path = data_dir / behaviors_dir / behaviors_dir / "behaviors.tsv"

    print(f"checkpoint: {checkpoint_path}")
    print(f"training history: {training_history_path}")
    print(
        "model config: "
        f"embedding_dim={config['embedding_dim']} "
        f"attention_heads={config['attention_heads']} "
        f"article_size={config['article_size']} "
        f"history_size={config['history_size']}"
    )
    print("loading news and vocabulary...")
    vocab, encoded_news, metadata = build_news_lookup(data_dir, config)

    print("loading user histories...")
    user_histories = read_latest_user_histories(behaviors_path, max_users=max_users)
    if len(user_histories) < cluster_count:
        raise ValueError("Need at least as many users as clusters.")

    print(f"device: {device}")
    print(f"users: {len(user_histories)}")
    print(f"clusters: {cluster_count}")

    print("loading NRMS checkpoint...")
    model = load_model(len(vocab), checkpoint_path, device, config=config)

    print("extracting user interest vectors...")
    user_ids, raw_user_vectors = extract_user_vectors(
        model=model,
        user_histories=user_histories,
        encoded_news=encoded_news,
        history_size=int(config["history_size"]),
        device=device,
        batch_size=batch_size,
    )

    normalized_user_vectors = normalize(raw_user_vectors)

    print("clustering users...")
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=seed,
        n_init=kmeans_n_init,
    )
    labels = kmeans.fit_predict(normalized_user_vectors)
    raw_centroids = raw_centroids_from_labels(raw_user_vectors, labels, cluster_count)

    print("summarizing clusters...")
    summaries = summarize_clusters(
        user_ids=user_ids,
        labels=labels,
        user_histories=user_histories,
        metadata=metadata,
        top_n=top_n,
    )

    points_2d = PCA(n_components=2, random_state=seed).fit_transform(normalized_user_vectors)
    save_user_clusters(output_dir / "user_clusters.csv", user_ids, labels, points_2d)
    save_cluster_summary(output_dir / "cluster_summary.json", summaries)
    save_cluster_centroids(output_dir / "cluster_centroids.npy", raw_centroids)

    print(f"saved {output_dir / 'user_clusters.csv'}")
    print(f"saved {output_dir / 'cluster_summary.json'}")
    print(f"saved {output_dir / 'cluster_centroids.npy'}")
    for summary in summaries:
        top_categories = ", ".join(category for category, _ in summary["top_categories"][:3])
        print(f"cluster {summary['cluster']}: users={summary['user_count']} top_categories={top_categories}")

    return {
        "config": config,
        "user_ids": user_ids,
        "labels": labels,
        "raw_user_vectors": raw_user_vectors,
        "centroids": raw_centroids,
        "summaries": summaries,
        "output_dir": output_dir,
    }


def main():
    if MODEL_NAME not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown MODEL_NAME {MODEL_NAME!r}; choose one of {sorted(MODEL_CONFIGS)}"
        )

    model_config = MODEL_CONFIGS[MODEL_NAME]
    run_user_clustering(
        checkpoint_path=model_config["checkpoint_path"],
        training_history_path=model_config["training_history_path"],
        output_dir=model_config["output_dir"],
    )


if __name__ == "__main__":
    main()
