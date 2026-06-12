import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from nrms import pipeline
from nrms.user_clustering import build_news_lookup, encode_history, load_model
from src.evaluation.evaluator import iter_mind_impressions
from src.evaluation.metrics import evaluate_ranking


HISTORY_GROUPS = (
    ("1-3", 1, 3),
    ("4-10", 4, 10),
    ("11-20", 11, 20),
    ("20+", 21, None),
)

DEFAULT_ALPHA_BY_GROUP = {
    "1-3": 0.3,
    "4-10": 0.5,
    "11-20": 0.7,
    "20+": 0.8,
}


def history_group(history_length):
    for name, minimum, maximum in HISTORY_GROUPS:
        if history_length >= minimum and (maximum is None or history_length <= maximum):
            return name
    return None


def load_user_clusters(path):
    clusters = {}
    with open(path, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            clusters[row["user_id"]] = int(row["cluster"])
    return clusters


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def aggregate_metric_rows(rows, k_values):
    metric_names = ["mrr", "auc"]
    for k in k_values:
        metric_names += [f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]

    if not rows:
        return {metric_name: float("nan") for metric_name in metric_names}

    aggregated = {}
    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in rows], dtype=np.float64)
        aggregated[metric_name] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    return aggregated


def encode_candidates(candidates, encoded_news, device):
    encoded = [encoded_news.get(news_id, encoded_news[pipeline.PAD_NEWS]) for news_id in candidates]
    return torch.tensor(encoded, dtype=torch.long, device=device)


@torch.no_grad()
def score_impression(model, impression, encoded_news, centroids, normalized_centroids, user_clusters, device):
    history = torch.tensor(
        encode_history(impression.history, encoded_news),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    candidates = encode_candidates(impression.candidates, encoded_news, device).unsqueeze(0)

    personal_vector = model.encode_user(history)

    candidate_count, article_size = candidates.squeeze(0).shape
    candidate_vectors = model.news_encoder(candidates.reshape(candidate_count, article_size))

    cluster_id = user_clusters.get(impression.user_id)
    if cluster_id is None:
        normalized_personal = normalize(personal_vector.cpu().numpy())
        cluster_id = int(np.argmax(cosine_similarity(normalized_personal, normalized_centroids)[0]))

    cluster_vector = torch.tensor(
        centroids[cluster_id],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    group = history_group(len(impression.history))
    alpha = DEFAULT_ALPHA_BY_GROUP[group]
    hybrid_vector = alpha * personal_vector + (1.0 - alpha) * cluster_vector

    return {
        "individual": torch.matmul(candidate_vectors, personal_vector.squeeze(0)).cpu().tolist(),
        "cluster_only": torch.matmul(candidate_vectors, cluster_vector.squeeze(0)).cpu().tolist(),
        "hybrid": torch.matmul(candidate_vectors, hybrid_vector.squeeze(0)).cpu().tolist(),
    }


def evaluate_hybrid(
    model,
    behaviors_path,
    encoded_news,
    centroids,
    normalized_centroids,
    user_clusters,
    device,
    max_impressions=None,
    k_values=(5, 10),
):
    rows = {
        group_name: {
            "individual": [],
            "cluster_only": [],
            "hybrid": [],
        }
        for group_name, _, _ in HISTORY_GROUPS
    }
    counts = defaultdict(int)

    for index, impression in enumerate(iter_mind_impressions(behaviors_path)):
        if max_impressions is not None and index >= max_impressions:
            break

        group = history_group(len(impression.history))
        if group is None:
            continue

        scores_by_model = score_impression(
            model=model,
            impression=impression,
            encoded_news=encoded_news,
            centroids=centroids,
            normalized_centroids=normalized_centroids,
            user_clusters=user_clusters,
            device=device,
        )

        counts[group] += 1
        for model_name, scores in scores_by_model.items():
            rows[group][model_name].append(evaluate_ranking(impression.labels, scores, k_values=k_values))

    report = {}
    for group_name in rows:
        report[group_name] = {"impression_count": counts[group_name]}
        for model_name, metric_rows in rows[group_name].items():
            report[group_name][model_name] = aggregate_metric_rows(metric_rows, k_values)

    return report


def write_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report_path


def print_report(report):
    for group_name, group_report in report.items():
        print(f"\n{group_name} clicks ({group_report['impression_count']} impressions)")
        for model_name in ("individual", "cluster_only", "hybrid"):
            metrics = group_report[model_name]
            print(
                f"  {model_name}: "
                f"auc={metrics['auc']:.4f} "
                f"mrr={metrics['mrr']:.4f} "
                f"ndcg@5={metrics['ndcg@5']:.4f} "
                f"ndcg@10={metrics['ndcg@10']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare individual NRMS, cluster-only, and hybrid scoring by history length."
    )
    parser.add_argument("--data-dir", default="data/mind-small")
    parser.add_argument("--checkpoint-path", default=pipeline.MODEL_PATH)
    parser.add_argument("--cluster-dir", default="reports/user_clustering")
    parser.add_argument("--output-dir", default="reports/hybrid_evaluation")
    parser.add_argument("--behaviors-path", default="data/mind-small/MINDsmall_dev/MINDsmall_dev/behaviors.tsv")
    parser.add_argument("--max-impressions", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir)

    centroids_path = cluster_dir / "cluster_centroids.npy"
    user_clusters_path = cluster_dir / "user_clusters.csv"

    if not centroids_path.exists() or not user_clusters_path.exists():
        raise SystemExit(
            "Missing cluster files. Run `python nrms/user_clustering.py --clusters 10` first."
        )

    print("loading news and vocabulary...")
    vocab, encoded_news, _ = build_news_lookup(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading NRMS checkpoint...")
    try:
        model = load_model(len(vocab), args.checkpoint_path, device)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("loading clusters...")
    user_clusters = load_user_clusters(user_clusters_path)
    centroids = np.load(centroids_path)
    normalized_centroids = normalize(centroids)

    print("evaluating hybrid models...")
    report = evaluate_hybrid(
        model=model,
        behaviors_path=args.behaviors_path,
        encoded_news=encoded_news,
        centroids=centroids,
        normalized_centroids=normalized_centroids,
        user_clusters=user_clusters,
        device=device,
        max_impressions=args.max_impressions,
    )

    report_path = write_report(report, output_dir)
    print_report(report)
    print(f"\nsaved {report_path}")


if __name__ == "__main__":
    main()
