"""Programmatic NRMS evaluation with JSON and plot outputs."""

from __future__ import annotations

import json
import math
from itertools import islice
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .evaluator import (
    ScoredImpression,
    evaluate_scored_impressions,
    iter_mind_impressions,
    read_user_ids,
)
from .nrms_adapter import NRMSScorer


# Programmatic defaults. Edit these values here or assign new values after
# importing this module. Explicit run_nrms_evaluation arguments take precedence.
DEFAULT_CHECKPOINT = Path("nrms_100d.pt")
DEFAULT_TRAINING_HISTORY: Path | None = None  # None discovers it automatically.
DEFAULT_DATA_DIR = Path("data/mind-small")
DEFAULT_DEV_BEHAVIORS = (
    DEFAULT_DATA_DIR / "MINDsmall_dev" / "MINDsmall_dev" / "behaviors.tsv"
)
DEFAULT_TRAIN_BEHAVIORS = (
    DEFAULT_DATA_DIR / "MINDsmall_train" / "MINDsmall_train" / "behaviors.tsv"
)
DEFAULT_OUTPUT_DIR = Path("reports/evaluation-nrms-100d")
DEFAULT_K_VALUES = (5, 10)
DEFAULT_MAX_IMPRESSIONS: int | None = None
DEFAULT_EMPTY_HISTORY_STRATEGY = "model"
DEFAULT_PRECOMPUTE_NEWS_VECTORS = True
DEFAULT_NEWS_VECTOR_BATCH_SIZE = 512
DEFAULT_SAVE_OUTPUTS = True
DEFAULT_PRINT_RESULTS = True


EvaluationReport = dict[str, dict[str, float | int]]
_USE_DEFAULT = object()


def scored_impressions(
    behaviors_path: str | Path,
    scorer: NRMSScorer,
    max_impressions: int | None,
):
    impressions = iter_mind_impressions(behaviors_path)
    if max_impressions is not None:
        impressions = islice(impressions, max_impressions)

    for impression in impressions:
        yield ScoredImpression(
            impression_id=impression.impression_id,
            user_id=impression.user_id,
            labels=impression.labels,
            scores=scorer(impression),
            has_history=bool(impression.history),
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_report(report: EvaluationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )


def write_run_config(config: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "run_config.json").write_text(
        json.dumps(json_safe(config), indent=2),
        encoding="utf-8",
    )


def plot_metrics(report: EvaluationReport, output_dir: Path) -> None:
    groups = list(report)
    metric_names = [
        name
        for name in next(iter(report.values()))
        if not name.endswith("_count")
    ]
    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(9, max(3, len(metric_names) * 2.1)),
    )
    if len(metric_names) == 1:
        axes = [axes]

    for axis, metric_name in zip(axes, metric_names):
        values = [report[group].get(metric_name, float("nan")) for group in groups]
        axis.bar(
            groups,
            values,
            color=["#4c78a8", "#f58518", "#54a24b"][: len(groups)],
        )
        axis.set_title(metric_name)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", dpi=160)
    plt.close(fig)


def plot_counts(report: EvaluationReport, output_dir: Path) -> None:
    groups = list(report)
    count_names = [
        name for name in next(iter(report.values())) if name.endswith("_count")
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    fig, axes = plt.subplots(1, len(count_names), figsize=(5 * len(count_names), 4))
    if len(count_names) == 1:
        axes = [axes]

    for axis, count_name, color in zip(axes, count_names, colors):
        axis.bar(groups, [report[group][count_name] for group in groups], color=color)
        axis.set_title(count_name)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()
    fig.savefig(output_dir / "counts.png", dpi=160)
    plt.close(fig)


def print_report(report: EvaluationReport) -> None:
    groups = list(report)
    metric_names = list(next(iter(report.values())))
    width = 18
    print("".join(["metric".ljust(width), *[group.rjust(width) for group in groups]]))
    print("-" * (width * (len(groups) + 1)))

    for metric_name in metric_names:
        cells = [metric_name.ljust(width)]
        for group in groups:
            value = report[group].get(metric_name)
            if isinstance(value, float):
                cells.append(
                    ("nan" if math.isnan(value) else f"{value:.6f}").rjust(width)
                )
            else:
                cells.append(str(value).rjust(width))
        print("".join(cells))


def run_nrms_evaluation(
    checkpoint: str | Path | None = None,
    training_history: str | Path | None = None,
    data_dir: str | Path | None = None,
    behaviors: str | Path | None = None,
    train_behaviors: str | Path | None = None,
    output_dir: str | Path | None = None,
    k_values: tuple[int, ...] | None = None,
    max_impressions: int | None | object = _USE_DEFAULT,
    empty_history_strategy: str | None = None,
    precompute_news_vectors: bool | None = None,
    news_vector_batch_size: int | None = None,
    save_outputs: bool | None = None,
    print_results: bool | None = None,
) -> EvaluationReport:
    checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT)
    training_history = (
        Path(training_history)
        if training_history is not None
        else DEFAULT_TRAINING_HISTORY
    )
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    behaviors = Path(behaviors or DEFAULT_DEV_BEHAVIORS)
    train_behaviors = Path(train_behaviors or DEFAULT_TRAIN_BEHAVIORS)
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    k_values = k_values or DEFAULT_K_VALUES
    if max_impressions is _USE_DEFAULT:
        max_impressions = DEFAULT_MAX_IMPRESSIONS
    empty_history_strategy = (
        empty_history_strategy or DEFAULT_EMPTY_HISTORY_STRATEGY
    )
    precompute_news_vectors = (
        DEFAULT_PRECOMPUTE_NEWS_VECTORS
        if precompute_news_vectors is None
        else precompute_news_vectors
    )
    news_vector_batch_size = (
        news_vector_batch_size or DEFAULT_NEWS_VECTOR_BATCH_SIZE
    )
    save_outputs = DEFAULT_SAVE_OUTPUTS if save_outputs is None else save_outputs
    print_results = (
        DEFAULT_PRINT_RESULTS if print_results is None else print_results
    )

    scorer = NRMSScorer.from_mind_data_dir(
        data_dir=data_dir,
        checkpoint_path=checkpoint,
        training_history_path=training_history,
        empty_history_strategy=empty_history_strategy,
        precompute_news_vectors=precompute_news_vectors,
        news_vector_batch_size=news_vector_batch_size,
    )
    report = evaluate_scored_impressions(
        scored_impressions(behaviors, scorer, max_impressions),
        train_user_ids=read_user_ids(train_behaviors),
        k_values=k_values,
    )

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_report(report, output_dir)
        write_run_config(
            {
                "checkpoint": checkpoint,
                "training_history": scorer.training_history_path,
                "data_dir": data_dir,
                "behaviors": behaviors,
                "train_behaviors": train_behaviors,
                "output_dir": output_dir,
                "k_values": k_values,
                "max_impressions": max_impressions,
                "empty_history_strategy": empty_history_strategy,
                "precompute_news_vectors": precompute_news_vectors,
                "news_vector_batch_size": news_vector_batch_size,
                "model_config": scorer.config,
            },
            output_dir,
        )
        plot_metrics(report, output_dir)
        plot_counts(report, output_dir)

    if print_results:
        print(
            f"checkpoint: {checkpoint}\n"
            f"training history: {scorer.training_history_path}"
        )
        print_report(report)
        if save_outputs:
            print(f"\nSaved report, config, and plots to {output_dir}")
    return report


if __name__ == "__main__":
    run_nrms_evaluation()
