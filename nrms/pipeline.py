import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path("data/mind-small")
TITLE_SIZE = 20
HISTORY_SIZE = 20
NEGATIVES_PER_POSITIVE = 4
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 64
EPOCHS = 1
MAX_STEPS = 500
EVAL_IMPRESSIONS = 1000

EMBEDDING_DIM = 64
ATTENTION_HEADS = 4
ATTENTION_HIDDEN_DIM = 100
DROPOUT = 0.2
LEARNING_RATE = 0.0001

PAD_WORD = 0
UNK_WORD = 1
PAD_NEWS = "<PAD_NEWS>"


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def read_news_titles(path):
    news = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            columns = line.rstrip("\n").split("\t")
            news_id, title = columns[0], columns[3]
            news[news_id] = tokenize(title)
    return news


def build_vocab(news_titles):
    word_counts = Counter(word for title in news_titles.values() for word in title)
    vocab = {"<PAD>": PAD_WORD, "<UNK>": UNK_WORD}

    for word, _ in word_counts.most_common(MAX_VOCAB_SIZE - len(vocab)):
        vocab[word] = len(vocab)

    return vocab


def encode_title(tokens, vocab):
    ids = [vocab.get(word, UNK_WORD) for word in tokens[:TITLE_SIZE]]
    return ids + [PAD_WORD] * (TITLE_SIZE - len(ids))


def encode_all_news(news_titles, vocab):
    encoded = {news_id: encode_title(tokens, vocab) for news_id, tokens in news_titles.items()}
    encoded[PAD_NEWS] = [PAD_WORD] * TITLE_SIZE
    return encoded


def parse_impression_list(text):
    impressions = []
    for item in text.split():
        news_id, label = item.rsplit("-", 1)
        impressions.append((news_id, int(label)))
    return impressions


class MindTrainDataset(Dataset):
    def __init__(self, behaviors_path, news):
        self.news = news
        self.samples = []

        with open(behaviors_path, encoding="utf-8") as file:
            for line in file:
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
        history = history[-HISTORY_SIZE:]
        history = [PAD_NEWS] * (HISTORY_SIZE - len(history)) + history
        return [self.news_tensor(news_id) for news_id in history]

    def __getitem__(self, index):
        history, clicked_news, ignored_news = self.samples[index]
        negatives = random.choices(ignored_news, k=NEGATIVES_PER_POSITIVE)
        candidates = [clicked_news] + negatives

        return (
            torch.tensor(self.history_tensor(history)),
            torch.tensor([self.news_tensor(news_id) for news_id in candidates]),
            torch.tensor(0),  # candidate 0 is always the clicked one
        )


class MindEvalDataset(Dataset):
    def __init__(self, behaviors_path, news):
        self.news = news
        self.impressions = []

        with open(behaviors_path, encoding="utf-8") as file:
            for line in file:
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
        history = history[-HISTORY_SIZE:]
        history = [PAD_NEWS] * (HISTORY_SIZE - len(history)) + history
        return [self.news_tensor(news_id) for news_id in history]

    def __getitem__(self, index):
        history, candidates, labels = self.impressions[index]
        return (
            torch.tensor(self.history_tensor(history)),
            torch.tensor([self.news_tensor(news_id) for news_id in candidates]),
            torch.tensor(labels),
        )


def keep_eval_items_unbatched(batch):
    return batch[0]


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, ATTENTION_HIDDEN_DIM)
        self.query = nn.Linear(ATTENTION_HIDDEN_DIM, 1, bias=False)

    def forward(self, vectors, mask):
        scores = self.query(torch.tanh(self.projection(vectors))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), vectors).squeeze(1)


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_WORD)
        self.self_attention = nn.MultiheadAttention(
            EMBEDDING_DIM,
            ATTENTION_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.attention_pooling = AdditiveAttention(EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, titles):
        word_mask = titles.ne(PAD_WORD)
        valid_titles = word_mask.any(dim=1)

        word_vectors = self.dropout(self.embedding(titles))
        padding_mask = ~word_mask
        padding_mask = padding_mask.masked_fill(~valid_titles.unsqueeze(1), False)

        contextual_words, _ = self.self_attention(
            word_vectors,
            word_vectors,
            word_vectors,
            key_padding_mask=padding_mask,
        )

        news_vectors = self.attention_pooling(contextual_words, word_mask)
        return news_vectors.masked_fill(~valid_titles.unsqueeze(1), 0.0)


class NRMS(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size)
        self.user_self_attention = nn.MultiheadAttention(
            EMBEDDING_DIM,
            ATTENTION_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.user_attention_pooling = AdditiveAttention(EMBEDDING_DIM)

    def encode_user(self, history):
        batch_size, history_size, title_size = history.shape

        clicked_news = history.reshape(batch_size * history_size, title_size)
        clicked_vectors = self.news_encoder(clicked_news)
        clicked_vectors = clicked_vectors.reshape(batch_size, history_size, EMBEDDING_DIM)

        history_mask = history.ne(PAD_WORD).any(dim=2)
        contextual_clicks, _ = self.user_self_attention(
            clicked_vectors,
            clicked_vectors,
            clicked_vectors,
            key_padding_mask=~history_mask,
        )

        return self.user_attention_pooling(contextual_clicks, history_mask)

    def forward(self, history, candidates):
        batch_size, candidate_count, title_size = candidates.shape

        user_vector = self.encode_user(history)
        candidate_titles = candidates.reshape(batch_size * candidate_count, title_size)
        candidate_vectors = self.news_encoder(candidate_titles)
        candidate_vectors = candidate_vectors.reshape(batch_size, candidate_count, EMBEDDING_DIM)

        return torch.bmm(candidate_vectors, user_vector.unsqueeze(2)).squeeze(2)


@torch.no_grad()
def evaluate_auc(model, dataset, device):
    model.eval()
    auc_scores = []
    loader = DataLoader(dataset, batch_size=1, collate_fn=keep_eval_items_unbatched)

    for index, (history, candidates, labels) in enumerate(loader):
        if index == EVAL_IMPRESSIONS:
            break

        scores = model(
            history.unsqueeze(0).to(device),
            candidates.unsqueeze(0).to(device),
        )

        auc_scores.append(roc_auc_score(labels.numpy(), scores.squeeze(0).cpu().numpy()))

    return float(np.mean(auc_scores))


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_dir = DATA_DIR / "MINDsmall_train" / "MINDsmall_train"
    dev_dir = DATA_DIR / "MINDsmall_dev" / "MINDsmall_dev"

    train_titles = read_news_titles(train_dir / "news.tsv")
    dev_titles = read_news_titles(dev_dir / "news.tsv")
    vocab = build_vocab(train_titles)
    news = encode_all_news({**train_titles, **dev_titles}, vocab)

    train_data = MindTrainDataset(train_dir / "behaviors.tsv", news)
    dev_data = MindEvalDataset(dev_dir / "behaviors.tsv", news)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NRMS(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    print(f"device: {device}")
    print(f"vocabulary words: {len(vocab)}")
    print(f"training samples: {len(train_data)}")
    print(f"validation impressions: {len(dev_data)}")

    step = 0
    for epoch in range(EPOCHS):
        model.train()
        losses = []

        for history, candidates, labels in train_loader:
            history = history.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)

            scores = model(history, candidates)
            loss = loss_function(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if MAX_STEPS and step >= MAX_STEPS:
                break

        auc = evaluate_auc(model, dev_data, device)
        print(f"epoch {epoch + 1}: loss={np.mean(losses):.4f}, validation_auc={auc:.4f}")

        if MAX_STEPS and step >= MAX_STEPS:
            break

    torch.save(model.state_dict(), "nrms_simple.pt")
    print("saved model to nrms_simple.pt")


if __name__ == "__main__":
    main()
