"""Model-agnostic evaluation helpers for MIND-style recommendation data."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import evaluate_ranking


@dataclass(frozen=True)
class MindImpression:
    impression_id: str
    user_id: str
    time: str
    history: list[str]
    candidates: list[str]
    labels: list[int]


@dataclass(frozen=True)
class ScoredImpression:
    impression_id: str
    user_id: str
    labels: Sequence[int]
    scores: Sequence[float]
    has_history: bool | None = None


Scorer = Callable[[MindImpression], Sequence[float]]


def parse_impressions(text: str) -> tuple[list[str], list[int]]:
    candidates = []
    labels = []

    for item in text.split():
        news_id, label = item.rsplit("-", 1)
        candidates.append(news_id)
        labels.append(int(label))

    return candidates, labels


def iter_mind_impressions(behaviors_path: str | Path) -> Iterable[MindImpression]:
    with open(behaviors_path, encoding="utf-8") as file:
        for line in file:
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 5:
                raise ValueError(f"Expected 5 columns in behaviors file, got {len(columns)}")

            impression_id, user_id, time, history_text, impressions_text = columns
            candidates, labels = parse_impressions(impressions_text)

            yield MindImpression(
                impression_id=impression_id,
                user_id=user_id,
                time=time,
                history=history_text.split() if history_text else [],
                candidates=candidates,
                labels=labels,
            )


def read_user_ids(behaviors_path: str | Path) -> set[str]:
    return {impression.user_id for impression in iter_mind_impressions(behaviors_path)}


def empty_metric_row(k_values: Sequence[int] = (5, 10)) -> dict[str, float]:
    metrics = {
        "mrr": float("nan"),
        "auc": float("nan"),
    }

    for k in k_values:
        metrics[f"precision@{k}"] = float("nan")
        metrics[f"recall@{k}"] = float("nan")
        metrics[f"ndcg@{k}"] = float("nan")

    return metrics


def aggregate_metric_rows(
    rows: Sequence[dict[str, float]],
    k_values: Sequence[int] = (5, 10),
) -> dict[str, float]:
    if not rows:
        return empty_metric_row(k_values)

    metric_names = rows[0].keys()
    aggregated = {}

    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in rows], dtype=np.float64)
        aggregated[metric_name] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")

    return aggregated


def evaluate_scored_impressions(
    impressions: Iterable[ScoredImpression],
    train_user_ids: set[str] | None = None,
    k_values: Sequence[int] = (5, 10),
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, float]]] = {"overall": []}
    group_users: dict[str, set[str]] = {"overall": set()}
    empty_history_counts = {"overall": 0}

    if train_user_ids is not None:
        groups["overlap_users"] = []
        groups["unseen_users"] = []
        group_users["overlap_users"] = set()
        group_users["unseen_users"] = set()
        empty_history_counts["overlap_users"] = 0
        empty_history_counts["unseen_users"] = 0

    for impression in impressions:
        metrics = evaluate_ranking(impression.labels, impression.scores, k_values=k_values)
        groups["overall"].append(metrics)
        group_users["overall"].add(impression.user_id)
        if impression.has_history is False:
            empty_history_counts["overall"] += 1

        if train_user_ids is not None:
            group_name = "overlap_users" if impression.user_id in train_user_ids else "unseen_users"
            groups[group_name].append(metrics)
            group_users[group_name].add(impression.user_id)
            if impression.has_history is False:
                empty_history_counts[group_name] += 1

    report = {}
    for group_name, rows in groups.items():
        report[group_name] = {
            "impression_count": len(rows),
            "user_count": len(group_users[group_name]),
            "empty_history_count": empty_history_counts[group_name],
            **aggregate_metric_rows(rows, k_values=k_values),
        }

    return report


def evaluate_mind(
    behaviors_path: str | Path,
    scorer: Scorer,
    train_behaviors_path: str | Path | None = None,
    k_values: Sequence[int] = (5, 10),
) -> dict[str, dict[str, float | int]]:
    train_user_ids = read_user_ids(train_behaviors_path) if train_behaviors_path is not None else None

    return evaluate_scored_impressions(
        _score_mind_impressions(iter_mind_impressions(behaviors_path), scorer),
        train_user_ids=train_user_ids,
        k_values=k_values,
    )


def _score_mind_impressions(
    impressions: Iterable[MindImpression],
    scorer: Scorer,
) -> Iterable[ScoredImpression]:
    for impression in impressions:
        yield ScoredImpression(
            impression_id=impression.impression_id,
            user_id=impression.user_id,
            labels=impression.labels,
            scores=scorer(impression),
            has_history=bool(impression.history),
        )
