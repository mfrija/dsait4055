from torch.utils.data import Dataset

class MINDTrainDataset(Dataset):
    def __init__(self, train_samples_df, news_store, max_history_len=50):
        self.samples = train_samples_df
        self.news_store = news_store
        self.max_history_len = max_history_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        user_id = row["user_id"]

        # look up token IDs for history, candidate and negatives
        history = [self.news_store[nid] for nid in row['history'][-self.max_history_len:]]
        candidate = self.news_store[row['candidate']]
        negatives = [self.news_store[nid] for nid in row['neg_samples']]

        return {
            'user_id': user_id, # user id
            'history': history,       # list of representations of the articles previously clicked
            'candidate': candidate,   # representation of candidate article
            'negatives': negatives    # list of representations of the negative samples
        }

class MINDTestDataset(Dataset):
    def __init__(self, behaviors_df, news_store, max_history_len=50):
        self.impressions = behaviors_df  # one row = one full impression
        self.news_store = news_store
        self.max_history_len = max_history_len

    def __len__(self):
        return len(self.impressions)

    def __getitem__(self, idx):
        row = self.impressions.iloc[idx]

        user_id = row["user_id"]

        history = [self.news_store[nid] for nid in row['history'][-self.max_history_len:]]
        
        # unpack list of (nid, label) tuples
        candidates = [self.news_store[nid] for nid, _ in row['impressions']]
        labels     = [label for _, label in row['impressions']]

        return {
            'user_id': user_id, # user id
            'history':    history, # list of representations of the articles previously clicked
            'candidates': candidates, # list of representations of the candidate articles
            'labels':     labels # labels of the candidate articles
        }
