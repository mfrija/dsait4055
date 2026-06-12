import argparse
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

from nrms import pipeline


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


def build_news_lookup(data_dir):
    train_dir = data_dir / "MINDsmall_train" / "MINDsmall_train"
    dev_dir = data_dir / "MINDsmall_dev" / "MINDsmall_dev"

    train_texts = pipeline.read_news_texts(train_dir / "news.tsv")
    dev_texts = pipeline.read_news_texts(dev_dir / "news.tsv")
    vocab = pipeline.build_vocab(train_texts)
    encoded_news = pipeline.encode_all_news({**train_texts, **dev_texts}, vocab)

    metadata = read_news_metadata(train_dir / "news.tsv")
    metadata.update(read_news_metadata(dev_dir / "news.tsv"))

    return vocab, encoded_news, metadata


def encode_history(history, encoded_news):
    history = history[-pipeline.HISTORY_SIZE :]
    history = [pipeline.PAD_NEWS] * (pipeline.HISTORY_SIZE - len(history)) + history
    return [encoded_news.get(news_id, encoded_news[pipeline.PAD_NEWS]) for news_id in history]


def load_model(vocab_size, checkpoint_path, device):
    model = pipeline.NRMS(vocab_size).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
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
def extract_user_vectors(model, user_histories, encoded_news, device, batch_size):
    user_ids = list(user_histories)
    vectors = []

    for start in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[start : start + batch_size]
        batch_histories = [
            encode_history(user_histories[user_id], encoded_news)
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
                title_word_counts.update(pipeline.tokenize(news["title"]))

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


def main():
    parser = argparse.ArgumentParser(description="Cluster users from trained NRMS user interest vectors.")
    parser.add_argument("--data-dir", default="data/mind-small")
    parser.add_argument("--checkpoint-path", default=pipeline.MODEL_PATH)
    parser.add_argument("--behaviors-split", choices=["train", "dev"], default="train")
    parser.add_argument("--output-dir", default="reports/user_clustering")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    behaviors_dir = "MINDsmall_train" if args.behaviors_split == "train" else "MINDsmall_dev"
    behaviors_path = data_dir / behaviors_dir / behaviors_dir / "behaviors.tsv"

    print("loading news and vocabulary...")
    vocab, encoded_news, metadata = build_news_lookup(data_dir)

    print("loading user histories...")
    user_histories = read_latest_user_histories(behaviors_path, max_users=args.max_users)
    if len(user_histories) < args.clusters:
        raise ValueError("Need at least as many users as clusters.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"users: {len(user_histories)}")
    print(f"clusters: {args.clusters}")

    print("loading NRMS checkpoint...")
    try:
        model = load_model(len(vocab), args.checkpoint_path, device)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("extracting user interest vectors...")
    user_ids, raw_user_vectors = extract_user_vectors(
        model=model,
        user_histories=user_histories,
        encoded_news=encoded_news,
        device=device,
        batch_size=args.batch_size,
    )

    normalized_user_vectors = normalize(raw_user_vectors)

    print("clustering users...")
    kmeans = KMeans(n_clusters=args.clusters, random_state=args.seed, n_init=10)
    labels = kmeans.fit_predict(normalized_user_vectors)
    raw_centroids = raw_centroids_from_labels(raw_user_vectors, labels, args.clusters)

    print("summarizing clusters...")
    summaries = summarize_clusters(
        user_ids=user_ids,
        labels=labels,
        user_histories=user_histories,
        metadata=metadata,
        top_n=args.top_n,
    )

    points_2d = PCA(n_components=2, random_state=args.seed).fit_transform(normalized_user_vectors)
    save_user_clusters(output_dir / "user_clusters.csv", user_ids, labels, points_2d)
    save_cluster_summary(output_dir / "cluster_summary.json", summaries)
    save_cluster_centroids(output_dir / "cluster_centroids.npy", raw_centroids)

    print(f"saved {output_dir / 'user_clusters.csv'}")
    print(f"saved {output_dir / 'cluster_summary.json'}")
    print(f"saved {output_dir / 'cluster_centroids.npy'}")
    for summary in summaries:
        top_categories = ", ".join(category for category, _ in summary["top_categories"][:3])
        print(f"cluster {summary['cluster']}: users={summary['user_count']} top_categories={top_categories}")


if __name__ == "__main__":
    main()
