"""Adapter that exposes a trained NRMS checkpoint as an evaluation scorer."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from nrms import data as nrms_data
from nrms import pipeline as nrms_pipeline

from .evaluator import MindImpression


CONFIG_TO_PIPELINE_CONSTANT = {
    "article_size": "ARTICLE_SIZE",
    "history_size": "HISTORY_SIZE",
    "max_vocab_size": "MAX_VOCAB_SIZE",
    "embedding_dim": "EMBEDDING_DIM",
    "attention_heads": "ATTENTION_HEADS",
    "attention_hidden_dim": "ATTENTION_HIDDEN_DIM",
    "dropout": "DROPOUT",
}
REQUIRED_CONFIG_KEYS = {
    "article_size",
    "history_size",
    "max_vocab_size",
    "embedding_dim",
    "attention_heads",
    "attention_hidden_dim",
}


def read_training_history(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    history = json.loads(path.read_text(encoding="utf-8"))
    config = history.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Training history has no config object: {path}")

    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"Training history {path} is missing config values: {missing}")
    return history


def find_training_history(
    checkpoint_path: str | Path,
    training_history_path: str | Path | None = None,
    search_root: str | Path = "checkpoints",
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if training_history_path is not None:
        path = Path(training_history_path)
        if not path.exists():
            raise FileNotFoundError(f"Training history not found: {path}")
        return path

    candidates = []
    if checkpoint_path.parent != Path("."):
        candidates.extend(sorted(checkpoint_path.parent.glob("training_history*.json")))

    search_root = Path(search_root)
    named_directory = search_root / checkpoint_path.stem
    candidates.extend(sorted(named_directory.glob("training_history*.json")))

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.exists():
            return candidate
        seen.add(resolved)

    if search_root.exists():
        for candidate in sorted(search_root.rglob("training_history*.json")):
            try:
                history = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            best_model_path = history.get("best_model_path")
            if best_model_path and Path(best_model_path).name == checkpoint_path.name:
                return candidate

    raise FileNotFoundError(
        f"Could not automatically find training history for {checkpoint_path}. "
        "Pass training_history_path explicitly."
    )


def apply_training_config(config: dict[str, Any]) -> None:
    for config_key, constant_name in CONFIG_TO_PIPELINE_CONSTANT.items():
        if config_key in config and config[config_key] is not None:
            setattr(nrms_pipeline, constant_name, config[config_key])
    nrms_pipeline.validate_training_config()


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_dimensions(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    embedding = state_dict["news_encoder.embedding.weight"]
    projection = state_dict["news_encoder.attention_pooling.projection.weight"]
    return {
        "vocab_size": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]),
        "attention_hidden_dim": int(projection.shape[0]),
    }


class NRMSScorer:
    """CPU scorer for MIND impressions using one trained NRMS checkpoint."""

    def __init__(
        self,
        model: nrms_pipeline.NRMS,
        news: dict[str, list[int]],
        config: dict[str, Any],
        checkpoint_path: str | Path,
        training_history_path: str | Path,
        empty_history_strategy: str = "model",
        precompute_news_vectors: bool = True,
        news_vector_batch_size: int = 512,
    ):
        if empty_history_strategy not in {"model", "raise"}:
            raise ValueError("empty_history_strategy must be one of: model, raise")

        self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.news = news
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.training_history_path = Path(training_history_path)
        self.article_size = int(config["article_size"])
        self.history_size = int(config["history_size"])
        self.empty_history_strategy = empty_history_strategy
        self.id_to_index: dict[str, int] | None = None
        self.news_vectors: torch.Tensor | None = None
        self.valid_news: torch.Tensor | None = None
        self.pad_index: int | None = None

        if precompute_news_vectors:
            self.precompute_news_vectors(news_vector_batch_size)

    @classmethod
    def from_news_files(
        cls,
        train_news_path: str | Path,
        eval_news_paths: Iterable[str | Path],
        checkpoint_path: str | Path,
        training_history_path: str | Path | None = None,
        empty_history_strategy: str = "model",
        precompute_news_vectors: bool = True,
        news_vector_batch_size: int = 512,
        strict: bool = True,
    ) -> "NRMSScorer":
        checkpoint_path = Path(checkpoint_path)
        resolved_history_path = find_training_history(
            checkpoint_path,
            training_history_path=training_history_path,
        )
        history = read_training_history(resolved_history_path)
        config = history["config"]
        apply_training_config(config)

        train_texts = nrms_data.read_news_texts(train_news_path)
        all_texts = dict(train_texts)
        for news_path in eval_news_paths:
            all_texts.update(nrms_data.read_news_texts(news_path))

        vocab = nrms_data.build_vocab(
            train_texts,
            max_vocab_size=int(config["max_vocab_size"]),
        )
        news = nrms_data.encode_all_news(
            all_texts,
            vocab,
            article_size=int(config["article_size"]),
        )

        state_dict = load_state_dict(checkpoint_path)
        dimensions = checkpoint_dimensions(state_dict)
        expected = {
            "vocab_size": len(vocab),
            "embedding_dim": int(config["embedding_dim"]),
            "attention_hidden_dim": int(config["attention_hidden_dim"]),
        }
        if dimensions != expected:
            raise ValueError(
                "Checkpoint dimensions do not match its training history. "
                f"checkpoint={dimensions}, history/vocab={expected}"
            )

        model = nrms_pipeline.NRMS(len(vocab))
        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as error:
            raise RuntimeError(
                f"Failed to load {checkpoint_path} using config from "
                f"{resolved_history_path}: {error}"
            ) from error

        return cls(
            model=model,
            news=news,
            config=config,
            checkpoint_path=checkpoint_path,
            training_history_path=resolved_history_path,
            empty_history_strategy=empty_history_strategy,
            precompute_news_vectors=precompute_news_vectors,
            news_vector_batch_size=news_vector_batch_size,
        )

    @classmethod
    def from_mind_data_dir(
        cls,
        data_dir: str | Path,
        checkpoint_path: str | Path,
        training_history_path: str | Path | None = None,
        empty_history_strategy: str = "model",
        precompute_news_vectors: bool = True,
        news_vector_batch_size: int = 512,
        strict: bool = True,
    ) -> "NRMSScorer":
        data_dir = Path(data_dir)
        train_news_path = data_dir / "MINDsmall_train" / "MINDsmall_train" / "news.tsv"
        dev_news_path = data_dir / "MINDsmall_dev" / "MINDsmall_dev" / "news.tsv"
        return cls.from_news_files(
            train_news_path=train_news_path,
            eval_news_paths=(dev_news_path,),
            checkpoint_path=checkpoint_path,
            training_history_path=training_history_path,
            empty_history_strategy=empty_history_strategy,
            precompute_news_vectors=precompute_news_vectors,
            news_vector_batch_size=news_vector_batch_size,
            strict=strict,
        )

    def __call__(self, impression: MindImpression) -> list[float]:
        if not impression.candidates:
            return []
        if not impression.history and self.empty_history_strategy == "raise":
            raise ValueError(f"Impression {impression.impression_id} has no user history")
        if self.news_vectors is not None:
            return self.score_with_precomputed_news(impression)

        history = torch.as_tensor(
            self.encode_history(impression.history),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        candidates = torch.as_tensor(
            self.encode_candidates(impression.candidates),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(history, candidates).squeeze(0)
        return scores.detach().cpu().tolist()

    @torch.no_grad()
    def precompute_news_vectors(self, batch_size: int) -> None:
        ordered_news_ids = [nrms_data.PAD_NEWS]
        ordered_news_ids.extend(
            news_id for news_id in self.news if news_id != nrms_data.PAD_NEWS
        )
        self.id_to_index = {
            news_id: index for index, news_id in enumerate(ordered_news_ids)
        }
        self.pad_index = self.id_to_index[nrms_data.PAD_NEWS]

        vectors = []
        valid_news = []
        for start in range(0, len(ordered_news_ids), batch_size):
            batch_ids = ordered_news_ids[start : start + batch_size]
            batch_articles = [
                self.news.get(news_id, self.news[nrms_data.PAD_NEWS])
                for news_id in batch_ids
            ]
            articles = torch.as_tensor(
                batch_articles,
                dtype=torch.long,
                device=self.device,
            )
            vectors.append(self.model.news_encoder(articles))
            valid_news.extend(
                any(word_id != nrms_data.PAD_WORD for word_id in article)
                for article in batch_articles
            )

        self.news_vectors = torch.cat(vectors, dim=0)
        self.valid_news = torch.as_tensor(
            valid_news,
            dtype=torch.bool,
            device=self.device,
        )

    @torch.no_grad()
    def score_with_precomputed_news(self, impression: MindImpression) -> list[float]:
        if (
            self.id_to_index is None
            or self.news_vectors is None
            or self.valid_news is None
            or self.pad_index is None
        ):
            raise RuntimeError("News vectors were not precomputed")

        history_ids = list(impression.history[-self.history_size :])
        history_ids = [nrms_data.PAD_NEWS] * (
            self.history_size - len(history_ids)
        ) + history_ids
        history_indices = torch.as_tensor(
            [
                self.id_to_index.get(news_id, self.pad_index)
                for news_id in history_ids
            ],
            dtype=torch.long,
            device=self.device,
        )
        candidate_indices = torch.as_tensor(
            [
                self.id_to_index.get(news_id, self.pad_index)
                for news_id in impression.candidates
            ],
            dtype=torch.long,
            device=self.device,
        )

        clicked_vectors = self.news_vectors[history_indices].unsqueeze(0)
        history_mask = self.valid_news[history_indices].unsqueeze(0)
        candidate_vectors = self.news_vectors[candidate_indices]
        user_vector = self.model.encode_user_vectors(clicked_vectors, history_mask)
        scores = torch.matmul(candidate_vectors, user_vector.squeeze(0))
        return scores.detach().cpu().tolist()

    def encode_history(self, history: Sequence[str]) -> list[list[int]]:
        history = list(history[-self.history_size :])
        history = [nrms_data.PAD_NEWS] * (
            self.history_size - len(history)
        ) + history
        return [self.encode_news(news_id) for news_id in history]

    def encode_candidates(self, candidates: Sequence[str]) -> list[list[int]]:
        return [self.encode_news(news_id) for news_id in candidates]

    def encode_news(self, news_id: str) -> list[int]:
        return self.news.get(news_id, self.news[nrms_data.PAD_NEWS])
