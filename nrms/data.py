import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_WORD = 0
UNK_WORD = 1
PAD_NEWS = "<PAD_NEWS>"
MIND_TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def read_news_texts(path):
    news = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            columns = line.rstrip("\n").split("\t")
            news_id = columns[0]
            title = columns[3]
            abstract = columns[4] if len(columns) > 4 else ""
            news[news_id] = tokenize(f"{title} {abstract}")
    return news


def build_vocab(news_texts, max_vocab_size):
    word_counts = Counter(word for article in news_texts.values() for word in article)
    vocab = {"<PAD>": PAD_WORD, "<UNK>": UNK_WORD}

    for word, _ in word_counts.most_common(max_vocab_size - len(vocab)):
        vocab[word] = len(vocab)

    return vocab


def load_glove_embedding_matrix(
    vocab,
    glove_path,
    embedding_dim,
    init_std=0.1,
    seed=42,
):
    glove_path = Path(glove_path)
    if not glove_path.exists():
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    rng = np.random.default_rng(seed)
    embedding_matrix = rng.normal(
        loc=0.0,
        scale=init_std,
        size=(len(vocab), embedding_dim),
    ).astype(np.float32)
    embedding_matrix[PAD_WORD] = 0.0

    matched_words = 0
    wrong_dimension_matches = 0

    with glove_path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip().split()
            if not parts:
                continue

            word = parts[0]
            vocab_index = vocab.get(word)
            if vocab_index is None:
                continue

            vector_values = parts[1:]
            if len(vector_values) != embedding_dim:
                wrong_dimension_matches += 1
                continue

            embedding_matrix[vocab_index] = np.asarray(vector_values, dtype=np.float32)
            matched_words += 1

    if matched_words == 0:
        raise ValueError(
            f"No vocabulary words from this run matched {glove_path}. "
            "Check tokenizer/vocabulary compatibility and embedding dimension."
        )

    coverage_denominator = max(len(vocab) - 2, 1)
    stats = {
        "path": str(glove_path),
        "embedding_dim": embedding_dim,
        "matched_words": matched_words,
        "vocab_size": len(vocab),
        "coverage": matched_words / coverage_denominator,
        "wrong_dimension_matches": wrong_dimension_matches,
    }
    return torch.tensor(embedding_matrix, dtype=torch.float32), stats


def encode_article(tokens, vocab, article_size):
    ids = [vocab.get(word, UNK_WORD) for word in tokens[:article_size]]
    return ids + [PAD_WORD] * (article_size - len(ids))


def encode_all_news(news_texts, vocab, article_size):
    encoded = {
        news_id: encode_article(tokens, vocab, article_size)
        for news_id, tokens in news_texts.items()
    }
    encoded[PAD_NEWS] = [PAD_WORD] * article_size
    return encoded


def parse_impression_list(text):
    impressions = []
    for item in text.split():
        news_id, label = item.rsplit("-", 1)
        impressions.append((news_id, int(label)))
    return impressions


def iter_behavior_lines(source):
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as file:
            yield from file
        return

    yield from source


def behavior_timestamp(line):
    columns = line.rstrip("\n").split("\t")
    if len(columns) != 5:
        raise ValueError(f"Expected 5 columns in behaviors row, got {len(columns)}")
    return datetime.strptime(columns[2], MIND_TIMESTAMP_FORMAT)


def mind_last_day_behavior_split(path):
    with open(path, encoding="utf-8") as file:
        lines = list(file)

    if len(lines) < 2:
        raise ValueError("Need at least two behavior rows to create train and validation splits")

    lines.sort(key=behavior_timestamp)
    validation_date = behavior_timestamp(lines[-1]).date()
    split_index = len(lines) - 1
    while split_index > 0 and behavior_timestamp(lines[split_index - 1]).date() == validation_date:
        split_index -= 1

    if split_index == 0:
        raise ValueError("Last-day split left no behavior rows for training")

    return lines[:split_index], lines[split_index:]


class MindTrainDataset(Dataset):
    def __init__(
        self,
        behaviors_source,
        news,
        history_size,
        negatives_per_positive,
    ):
        self.news = news
        self.history_size = history_size
        self.negatives_per_positive = negatives_per_positive
        self.samples = []

        for line in iter_behavior_lines(behaviors_source):
            columns = line.rstrip("\n").split("\t")
            history = columns[3].split()
            impressions = parse_impression_list(columns[4])

            clicked = [news_id for news_id, label in impressions if label == 1]
            ignored = [news_id for news_id, label in impressions if label == 0]

            if history and clicked and ignored:
                for clicked_news in clicked:
                    self.samples.append((history, clicked_news, ignored))

    def __len__(self):
        return len(self.samples)

    def news_tensor(self, news_id):
        return self.news.get(news_id, self.news[PAD_NEWS])

    def history_tensor(self, history):
        history = history[-self.history_size :]
        history = [PAD_NEWS] * (self.history_size - len(history)) + history
        return [self.news_tensor(news_id) for news_id in history]

    def __getitem__(self, index):
        history, clicked_news, ignored_news = self.samples[index]
        negatives = random.choices(
            ignored_news,
            k=self.negatives_per_positive,
        )
        candidates = [clicked_news] + negatives

        return (
            torch.tensor(self.history_tensor(history)),
            torch.tensor([self.news_tensor(news_id) for news_id in candidates]),
            torch.tensor(0),
        )


class MindEvalDataset(Dataset):
    def __init__(self, behaviors_source, news, history_size):
        self.news = news
        self.history_size = history_size
        self.impressions = []

        for line in iter_behavior_lines(behaviors_source):
            columns = line.rstrip("\n").split("\t")
            history = columns[3].split()
            impression = parse_impression_list(columns[4])
            labels = [label for _, label in impression]

            if history and len(set(labels)) == 2:
                candidates = [news_id for news_id, _ in impression]
                self.impressions.append((history, candidates, labels))

    def __len__(self):
        return len(self.impressions)

    def news_tensor(self, news_id):
        return self.news.get(news_id, self.news[PAD_NEWS])

    def history_tensor(self, history):
        history = history[-self.history_size :]
        history = [PAD_NEWS] * (self.history_size - len(history)) + history
        return [self.news_tensor(news_id) for news_id in history]

    def __getitem__(self, index):
        history, candidates, labels = self.impressions[index]
        return (
            torch.tensor(self.history_tensor(history)),
            torch.tensor([self.news_tensor(news_id) for news_id in candidates]),
            torch.tensor(labels),
        )
