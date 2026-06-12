"""Adapter that exposes the NRMS model as an evaluation scorer."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from nrms import pipeline as nrms_pipeline

from .evaluator import MindImpression


class NRMSScorer:
    """Callable scorer for MIND impressions using the NRMS implementation."""

    def __init__(
        self,
        model: nrms_pipeline.NRMS,
        news: dict[str, list[int]],
        empty_history_strategy: str = "model",
    ):
        if empty_history_strategy not in {"model", "raise"}:
            raise ValueError("empty_history_strategy must be one of: model, raise")

        self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.news = news
        self.empty_history_strategy = empty_history_strategy

    @classmethod
    def from_news_files(
        cls,
        train_news_path: str | Path,
        eval_news_paths: Iterable[str | Path] = (),
        checkpoint_path: str | Path | None = None,
        empty_history_strategy: str = "model",
        strict: bool = True,
    ) -> "NRMSScorer":
        train_texts = nrms_pipeline.read_news_texts(train_news_path)
        all_texts = dict(train_texts)

        for news_path in eval_news_paths:
            all_texts.update(nrms_pipeline.read_news_texts(news_path))

        vocab = nrms_pipeline.build_vocab(train_texts)
        news = nrms_pipeline.encode_all_news(all_texts, vocab)
        model = nrms_pipeline.NRMS(len(vocab))

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=strict)

        return cls(
            model=model,
            news=news,
            empty_history_strategy=empty_history_strategy,
        )

    @classmethod
    def from_mind_data_dir(
        cls,
        data_dir: str | Path = "data/mind-small",
        checkpoint_path: str | Path | None = None,
        empty_history_strategy: str = "model",
        strict: bool = True,
    ) -> "NRMSScorer":
        data_dir = Path(data_dir)
        train_news_path = data_dir / "MINDsmall_train" / "MINDsmall_train" / "news.tsv"
        dev_news_path = data_dir / "MINDsmall_dev" / "MINDsmall_dev" / "news.tsv"

        return cls.from_news_files(
            train_news_path=train_news_path,
            eval_news_paths=(dev_news_path,),
            checkpoint_path=checkpoint_path,
            empty_history_strategy=empty_history_strategy,
            strict=strict,
        )

    def __call__(self, impression: MindImpression) -> list[float]:
        if not impression.candidates:
            return []

        if not impression.history:
            if self.empty_history_strategy == "raise":
                raise ValueError(f"Impression {impression.impression_id} has no user history")

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

        return [float(score) for score in scores.detach().cpu().tolist()]

    def encode_history(self, history: Sequence[str]) -> list[list[int]]:
        history = list(history[-nrms_pipeline.HISTORY_SIZE :])
        history = [nrms_pipeline.PAD_NEWS] * (nrms_pipeline.HISTORY_SIZE - len(history)) + history
        return [self.encode_news(news_id) for news_id in history]

    def encode_candidates(self, candidates: Sequence[str]) -> list[list[int]]:
        return [self.encode_news(news_id) for news_id in candidates]

    def encode_news(self, news_id: str) -> list[int]:
        return self.news.get(news_id, self.news[nrms_pipeline.PAD_NEWS])
