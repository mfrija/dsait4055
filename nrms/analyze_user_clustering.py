import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CLUSTER_DIR = Path("reports/3_user_clusters_preview")
DEFAULT_OUTPUT_DIR = DEFAULT_CLUSTER_DIR / "analysis"


def load_user_clusters(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "user_id": row["user_id"],
                    "cluster": int(row["cluster"]),
                    "x": float(row["x"]) if row.get("x") else None,
                    "y": float(row["y"]) if row.get("y") else None,
                }
            )
    return rows


def load_cluster_summary(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)["clusters"]


def save_scatter_plot(rows, path):
    points = np.asarray([[row["x"], row["y"]] for row in rows], dtype=np.float64)
    labels = np.asarray([row["cluster"] for row in rows], dtype=np.int64)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="tab20",
        s=8,
        alpha=0.65,
        linewidths=0,
    )
    plt.title("User Clusters Projected with PCA")
    plt.xlabel("PCA x")
    plt.ylabel("PCA y")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_cluster_size_plot(cluster_sizes, path):
    cluster_ids = sorted(cluster_sizes)
    sizes = [cluster_sizes[cluster_id] for cluster_id in cluster_ids]

    plt.figure(figsize=(9, 5))
    plt.bar([str(cluster_id) for cluster_id in cluster_ids], sizes, color="#4C78A8")
    plt.title("Users per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("User count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_centroid_similarity_plot(centroids, path):
    similarities = cosine_similarity(centroids)

    plt.figure(figsize=(7, 6))
    image = plt.imshow(similarities, cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.title("Cluster Centroid Cosine Similarity")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.colorbar(image, label="Cosine similarity")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def top_items(items, limit=3):
    return [name for name, _ in items[:limit]]


def summarize_cluster_content(cluster_summary):
    summaries = []
    for cluster in cluster_summary:
        categories = cluster["top_categories"]
        subcategories = cluster["top_subcategories"]
        listed_category_total = sum(count for _, count in categories)
        dominant_category, dominant_count = categories[0]
        dominant_share = (
            dominant_count / listed_category_total if listed_category_total else 0.0
        )
        summaries.append(
            {
                "cluster": cluster["cluster"],
                "user_count": cluster["user_count"],
                "dominant_category": dominant_category,
                "dominant_category_share_among_listed": dominant_share,
                "top_categories": top_items(categories),
                "top_subcategories": top_items(subcategories),
            }
        )
    return summaries


def sampled_silhouette_2d(rows, sample_size=5000, seed=42):
    points = np.asarray([[row["x"], row["y"]] for row in rows], dtype=np.float64)
    labels = np.asarray([row["cluster"] for row in rows], dtype=np.int64)

    if len(set(labels)) < 2:
        return None

    if len(rows) > sample_size:
        rng = np.random.default_rng(seed)
        indexes = rng.choice(len(rows), size=sample_size, replace=False)
        points = points[indexes]
        labels = labels[indexes]

    return float(silhouette_score(points, labels))


def build_analysis(rows, cluster_summary, centroids=None):
    cluster_sizes = dict(sorted(Counter(row["cluster"] for row in rows).items()))
    total_users = len(rows)
    largest_cluster = max(cluster_sizes.values())
    smallest_cluster = min(cluster_sizes.values())

    analysis = {
        "total_users": total_users,
        "cluster_count": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "largest_cluster_share": largest_cluster / total_users if total_users else 0.0,
        "smallest_cluster_share": smallest_cluster / total_users if total_users else 0.0,
        "largest_to_smallest_ratio": (
            largest_cluster / smallest_cluster if smallest_cluster else None
        ),
        "silhouette_2d_pca_sampled": sampled_silhouette_2d(rows),
        "content_summaries": summarize_cluster_content(cluster_summary),
    }

    if centroids is not None:
        similarity = cosine_similarity(centroids)
        off_diagonal = similarity[~np.eye(similarity.shape[0], dtype=bool)]
        analysis["centroid_cosine_similarity"] = {
            "mean_off_diagonal": float(np.mean(off_diagonal)),
            "max_off_diagonal": float(np.max(off_diagonal)),
            "min_off_diagonal": float(np.min(off_diagonal)),
        }

    return analysis


def write_markdown_report(analysis, path):
    silhouette = analysis["silhouette_2d_pca_sampled"]
    silhouette_text = f"{silhouette:.4f}" if silhouette is not None else "not available"
    ratio = analysis["largest_to_smallest_ratio"]
    ratio_text = f"{ratio:.2f}" if ratio is not None else "not available"

    lines = [
        "# User Clustering Analysis",
        "",
        "## Size Balance",
        "",
        f"- Total users: {analysis['total_users']}",
        f"- Cluster count: {analysis['cluster_count']}",
        f"- Largest cluster share: {analysis['largest_cluster_share']:.3f}",
        f"- Smallest cluster share: {analysis['smallest_cluster_share']:.3f}",
        f"- Largest/smallest ratio: {ratio_text}",
        "",
        "## Visual Separation",
        "",
        f"- Sampled silhouette on saved PCA coordinates: {silhouette_text}",
        "",
        "This silhouette score is computed only on the saved 2D PCA projection. It is useful for checking visual separation, but final cluster selection should use the original user-vector space and downstream recommendation metrics.",
        "",
        "## Content Summaries",
        "",
    ]

    for summary in analysis["content_summaries"]:
        lines.extend(
            [
                f"### Cluster {summary['cluster']}",
                "",
                f"- Users: {summary['user_count']}",
                f"- Dominant category: {summary['dominant_category']} ({summary['dominant_category_share_among_listed']:.3f} of listed category clicks)",
                f"- Top categories: {', '.join(summary['top_categories'])}",
                f"- Top subcategories: {', '.join(summary['top_subcategories'])}",
                "",
            ]
        )

    if "centroid_cosine_similarity" in analysis:
        similarity = analysis["centroid_cosine_similarity"]
        lines.extend(
            [
                "## Centroid Similarity",
                "",
                f"- Mean off-diagonal cosine similarity: {similarity['mean_off_diagonal']:.4f}",
                f"- Max off-diagonal cosine similarity: {similarity['max_off_diagonal']:.4f}",
                f"- Min off-diagonal cosine similarity: {similarity['min_off_diagonal']:.4f}",
                "",
                "High centroid similarity means the stereotype vectors are close together, which weakens the interpretation that clusters represent distinct user stereotypes.",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_clustering(cluster_dir=DEFAULT_CLUSTER_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    cluster_dir = Path(cluster_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_clusters_path = cluster_dir / "user_clusters.csv"
    cluster_summary_path = cluster_dir / "cluster_summary.json"
    centroids_path = cluster_dir / "cluster_centroids.npy"

    rows = load_user_clusters(user_clusters_path)
    cluster_summary = load_cluster_summary(cluster_summary_path)
    centroids = np.load(centroids_path) if centroids_path.exists() else None

    save_scatter_plot(rows, output_dir / "cluster_scatter_pca.png")
    cluster_sizes = Counter(row["cluster"] for row in rows)
    save_cluster_size_plot(cluster_sizes, output_dir / "cluster_sizes.png")
    if centroids is not None:
        save_centroid_similarity_plot(centroids, output_dir / "centroid_similarity.png")

    analysis = build_analysis(rows, cluster_summary, centroids)
    (output_dir / "cluster_analysis.json").write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(analysis, output_dir / "cluster_analysis.md")
    return analysis


if __name__ == "__main__":
    analyze_clustering()
