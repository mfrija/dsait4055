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


DEFAULT_DEV_BEHAVIORS = "data/mind-small/MINDsmall_dev/MINDsmall_dev/behaviors.tsv"
DEFAULT_TRAIN_BEHAVIORS = "data/mind-small/MINDsmall_train/MINDsmall_train/behaviors.tsv"
EvaluationReport = dict[str, dict[str, float | int]]


def scored_impressions(behaviors_path: str, scorer: NRMSScorer, max_impressions: int | None):
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
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_report(report: EvaluationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")


def plot_metrics(report: EvaluationReport, output_dir: Path) -> None:
    groups = list(report.keys())
    metric_names = [
        metric_name
        for metric_name in next(iter(report.values())).keys()
        if not metric_name.endswith("_count")
    ]

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(9, max(3, len(metric_names) * 2.1)))
    if len(metric_names) == 1:
        axes = [axes]

    for axis, metric_name in zip(axes, metric_names):
        values = [report[group].get(metric_name, float("nan")) for group in groups]
        axis.bar(groups, values, color=["#4c78a8", "#f58518", "#54a24b"][: len(groups)])
        axis.set_title(metric_name)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", dpi=160)
    plt.close(fig)


def plot_counts(report: EvaluationReport, output_dir: Path) -> None:
    groups = list(report.keys())
    count_names = [name for name in next(iter(report.values())).keys() if name.endswith("_count")]
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, axes = plt.subplots(1, len(count_names), figsize=(5 * len(count_names), 4))
    if len(count_names) == 1:
        axes = [axes]

    for axis, count_name, color in zip(axes, count_names, colors):
        counts = [report[group][count_name] for group in groups]
        axis.bar(groups, counts, color=color)
        axis.set_title(count_name)
        axis.grid(axis="y", alpha=0.3)

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()
    fig.savefig(output_dir / "counts.png", dpi=160)
    plt.close(fig)


def print_report(report: EvaluationReport) -> None:
    groups = list(report.keys())
    metric_names = list(next(iter(report.values())).keys())
    width = 18

    print("".join(["metric".ljust(width), *[group.rjust(width) for group in groups]]))
    print("-" * (width * (len(groups) + 1)))

    for metric_name in metric_names:
        cells = [metric_name.ljust(width)]
        for group in groups:
            value = report[group].get(metric_name)
            if isinstance(value, float):
                cells.append(("nan" if math.isnan(value) else f"{value:.6f}").rjust(width))
            else:
                cells.append(str(value).rjust(width))
        print("".join(cells))


def run_nrms_evaluation(
    checkpoint: str | Path = "nrms_simple.pt",
    data_dir: str | Path = "data/mind-small",
    behaviors: str | Path = DEFAULT_DEV_BEHAVIORS,
    train_behaviors: str | Path = DEFAULT_TRAIN_BEHAVIORS,
    output_dir: str | Path | None = "reports/evaluation",
    k_values: tuple[int, ...] = (5, 10),
    max_impressions: int | None = None,
    empty_history_strategy: str = "model",
    save_outputs: bool = True,
    print_results: bool = True,
) -> EvaluationReport:
    scorer = NRMSScorer.from_mind_data_dir(
        data_dir=data_dir,
        checkpoint_path=checkpoint,
        empty_history_strategy=empty_history_strategy,
    )

    report = evaluate_scored_impressions(
        scored_impressions(str(behaviors), scorer, max_impressions),
        train_user_ids=read_user_ids(train_behaviors),
        k_values=k_values,
    )

    if save_outputs:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save_outputs=True")
        output_path = Path(output_dir)
        write_report(report, output_path)
        plot_metrics(report, output_path)
        plot_counts(report, output_path)

    if print_results:
        print_report(report)
        if save_outputs:
            print(f"\nSaved report and plots to {output_dir}")

    return report


if __name__ == "__main__":
    run_nrms_evaluation(
        checkpoint="nrms_simple.pt",
        output_dir="reports/evaluation",
        # max_impressions=200,
    )
