import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from nrms import data as nrms_data
from nrms.user_clustering import (
    apply_training_config,
    build_news_lookup,
    encode_history,
    load_model,
    read_training_config,
)
from src.evaluation.evaluator import iter_mind_impressions, read_user_ids
from src.evaluation.metrics import evaluate_ranking


MODEL_NAME = "10_100d"
MODEL_CONFIGS = {
    "3_100d": {
        "checkpoint_path": Path("nrms_100d.pt"),
        "training_history_path": Path("checkpoints/nrms_100d/training_history.json"),
        "cluster_dir": Path("reports/3_user_clustering_100d"),
    },
    "5_100d": {
        "checkpoint_path": Path("nrms_100d.pt"),
        "training_history_path": Path("checkpoints/nrms_100d/training_history.json"),
        "cluster_dir": Path("reports/5_user_clustering_100d"),
    },
    "8_100d": {
        "checkpoint_path": Path("nrms_100d.pt"),
        "training_history_path": Path("checkpoints/nrms_100d/training_history.json"),
        "cluster_dir": Path("reports/8_user_clustering_100d"),
    },
    "10_100d": {
        "checkpoint_path": Path("nrms_100d.pt"),
        "training_history_path": Path("checkpoints/nrms_100d/training_history.json"),
        "cluster_dir": Path("reports/10_user_clustering_100d"),
    },
    "300d": {
        "checkpoint_path": Path("nrms_300d.pt"),
        "training_history_path": Path("checkpoints/nrms_300d/training_history.json"),
        "cluster_dir": Path("reports/user_clustering_300d"),
    },
}

DATA_DIR = Path("data/mind-small")
BEHAVIORS_PATH = DATA_DIR / "MINDsmall_dev" / "MINDsmall_dev" / "behaviors.tsv"
TRAIN_BEHAVIORS_PATH = (
    DATA_DIR / "MINDsmall_train" / "MINDsmall_train" / "behaviors.tsv"
)
OUTPUT_DIR = Path("reports") / f"hybrid_evaluation_{MODEL_NAME}"
MAX_IMPRESSIONS = None
K_VALUES = (5, 10)
INCLUDE_OVERALL = True
INCLUDE_USER_SEGMENTS = True
INCLUDE_HISTORY_USER_SEGMENTS = True
DEVICE = torch.device("cpu")

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
    encoded = [
        encoded_news.get(news_id, encoded_news[nrms_data.PAD_NEWS])
        for news_id in candidates
    ]
    return torch.tensor(encoded, dtype=torch.long, device=device)


@torch.no_grad()
def score_impression(
    model,
    impression,
    encoded_news,
    centroids,
    normalized_centroids,
    user_clusters,
    history_size,
    device,
):
    history = torch.tensor(
        encode_history(impression.history, encoded_news, history_size),
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
    history_size,
    device,
    max_impressions=None,
    k_values=(5, 10),
    include_overall=True,
    train_user_ids=None,
    include_history_user_segments=True,
):
    rows = {
        group_name: {
            "individual": [],
            "cluster_only": [],
            "hybrid": [],
        }
        for group_name, _, _ in HISTORY_GROUPS
    }
    if include_overall:
        rows["overall"] = {
            "individual": [],
            "cluster_only": [],
            "hybrid": [],
        }
    if train_user_ids is not None:
        rows["overlap_users"] = {
            "individual": [],
            "cluster_only": [],
            "hybrid": [],
        }
        if include_history_user_segments:
            for group_name, _, _ in HISTORY_GROUPS:
                for user_segment in ("overlap_users", "unseen_users"):
                    rows[f"{group_name}_{user_segment}"] = {
                        "individual": [],
                        "cluster_only": [],
                        "hybrid": [],
                    }
        rows["unseen_users"] = {
            "individual": [],
            "cluster_only": [],
            "hybrid": [],
        }
    counts = defaultdict(int)
    group_users = defaultdict(set)

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
            history_size=history_size,
            device=device,
        )

        target_groups = [group]
        if include_overall:
            target_groups.append("overall")
        if train_user_ids is not None:
            user_segment = (
                "overlap_users"
                if impression.user_id in train_user_ids
                else "unseen_users"
            )
            target_groups.append(user_segment)
            if include_history_user_segments:
                target_groups.append(f"{group}_{user_segment}")

        for target_group in target_groups:
            counts[target_group] += 1
            group_users[target_group].add(impression.user_id)

        for model_name, scores in scores_by_model.items():
            metrics = evaluate_ranking(impression.labels, scores, k_values=k_values)
            for target_group in target_groups:
                rows[target_group][model_name].append(metrics)

    report = {}
    for group_name in rows:
        report[group_name] = {
            "impression_count": counts[group_name],
            "user_count": len(group_users[group_name]),
        }
        for model_name, metric_rows in rows[group_name].items():
            report[group_name][model_name] = aggregate_metric_rows(metric_rows, k_values)

    return report


def write_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report_path


def write_spreadsheet(report, output_dir):
    rows = []
    for group_name, group_report in report.items():
        for model_name in ("individual", "cluster_only", "hybrid"):
            rows.append(
                {
                    "evaluation_group": group_name,
                    "model": model_name,
                    "impression_count": group_report["impression_count"],
                    "user_count": group_report["user_count"],
                    **group_report[model_name],
                }
            )

    spreadsheet_path = output_dir / "hybrid_evaluation.xlsx"
    results = pd.DataFrame(rows)
    with pd.ExcelWriter(spreadsheet_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="results", index=False)
        worksheet = writer.sheets["results"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column in worksheet.columns:
            width = max(len(str(cell.value or "")) for cell in column) + 2
            worksheet.column_dimensions[column[0].column_letter].width = min(width, 22)

    return spreadsheet_path


def print_report(report):
    for group_name, group_report in report.items():
        print(
            f"\n{group_name} "
            f"({group_report['impression_count']} impressions, "
            f"{group_report['user_count']} users)"
        )
        for model_name in ("individual", "cluster_only", "hybrid"):
            metrics = group_report[model_name]
            print(
                f"  {model_name}: "
                f"auc={metrics['auc']:.4f} "
                f"mrr={metrics['mrr']:.4f} "
                f"ndcg@5={metrics['ndcg@5']:.4f} "
                f"ndcg@10={metrics['ndcg@10']:.4f}"
            )


def run_hybrid_evaluation(
    checkpoint_path,
    training_history_path,
    cluster_dir,
    output_dir,
    data_dir=DATA_DIR,
    behaviors_path=BEHAVIORS_PATH,
    train_behaviors_path=TRAIN_BEHAVIORS_PATH,
    max_impressions=MAX_IMPRESSIONS,
    k_values=K_VALUES,
    include_overall=INCLUDE_OVERALL,
    include_user_segments=INCLUDE_USER_SEGMENTS,
    include_history_user_segments=INCLUDE_HISTORY_USER_SEGMENTS,
    device=DEVICE,
):
    checkpoint_path = Path(checkpoint_path)
    training_history_path = Path(training_history_path)
    cluster_dir = Path(cluster_dir)
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    behaviors_path = Path(behaviors_path)
    train_behaviors_path = Path(train_behaviors_path)

    centroids_path = cluster_dir / "cluster_centroids.npy"
    user_clusters_path = cluster_dir / "user_clusters.csv"

    if not centroids_path.exists() or not user_clusters_path.exists():
        raise FileNotFoundError(
            f"Missing cluster files in {cluster_dir}. Run nrms/user_clustering.py first."
        )

    config = read_training_config(training_history_path)
    apply_training_config(config)

    print(f"checkpoint: {checkpoint_path}")
    print(f"training history: {training_history_path}")
    print(f"cluster directory: {cluster_dir}")
    print(
        "model config: "
        f"embedding_dim={config['embedding_dim']} "
        f"attention_heads={config['attention_heads']} "
        f"article_size={config['article_size']} "
        f"history_size={config['history_size']}"
    )
    print("loading news and vocabulary...")
    vocab, encoded_news, _ = build_news_lookup(data_dir, config)

    print(f"device: {device}")

    print("loading NRMS checkpoint...")
    model = load_model(len(vocab), checkpoint_path, device, config=config)

    print("loading clusters...")
    user_clusters = load_user_clusters(user_clusters_path)
    centroids = np.load(centroids_path)
    expected_centroid_dim = int(config["embedding_dim"])
    if centroids.ndim != 2 or centroids.shape[1] != expected_centroid_dim:
        raise ValueError(
            "Cluster centroids do not match the selected NRMS model. "
            f"centroid shape={centroids.shape}, expected second dimension={expected_centroid_dim}"
        )
    if user_clusters and max(user_clusters.values()) >= len(centroids):
        raise ValueError("user_clusters.csv contains a cluster not present in cluster_centroids.npy")

    normalized_centroids = normalize(centroids)
    train_user_ids = (
        read_user_ids(train_behaviors_path) if include_user_segments else None
    )

    print("evaluating hybrid models...")
    report = evaluate_hybrid(
        model=model,
        behaviors_path=behaviors_path,
        encoded_news=encoded_news,
        centroids=centroids,
        normalized_centroids=normalized_centroids,
        user_clusters=user_clusters,
        history_size=int(config["history_size"]),
        device=device,
        max_impressions=max_impressions,
        k_values=k_values,
        include_overall=include_overall,
        train_user_ids=train_user_ids,
        include_history_user_segments=include_history_user_segments,
    )

    report_path = write_report(report, output_dir)
    spreadsheet_path = write_spreadsheet(report, output_dir)
    print_report(report)
    print(f"\nsaved {report_path}")
    print(f"saved {spreadsheet_path}")
    return report


def main():
    if MODEL_NAME not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown MODEL_NAME {MODEL_NAME!r}; choose one of {sorted(MODEL_CONFIGS)}"
        )

    model_config = MODEL_CONFIGS[MODEL_NAME]
    run_hybrid_evaluation(
        checkpoint_path=model_config["checkpoint_path"],
        training_history_path=model_config["training_history_path"],
        cluster_dir=model_config["cluster_dir"],
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
