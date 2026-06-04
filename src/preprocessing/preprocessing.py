import pandas as pd
import random 


def load_tsv_files(dataset_folder_path: str):
    """
    Load TSV files from the MIND dataset and print statistics.
    
    Args:
        dataset_folder_path: Path to the dataset folder (e.g., 'data/mind-small/MINDsmall_train')
    
    Returns:
        Tuple of (behaviors_df, news_df)
    """
    # Extract the last folder name from the path
    behaviors_path = f"{dataset_folder_path}/behaviors.tsv"
    news_path = f"{dataset_folder_path}/news.tsv"
    
    # Define column names based on dataset.md specifications
    behaviors_columns = ["impression_id", "user_id", "time", "history", "impressions"]
    news_columns = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    
    # Load TSV files into dataframes
    print(f"Loading data from {dataset_folder_path}...")
    behaviors_df = pd.read_csv(behaviors_path, sep='\t', names=behaviors_columns)
    news_df = pd.read_csv(news_path, sep='\t', names=news_columns)

    def parse_impression(x):
        """Parse impressions string into list of (news_id, click) tuples."""
        kv_pairs = [tuple(pair.split("-")) for pair in x.split(" ")]
        return kv_pairs

    # Parse the history and impressions fields of the original file
    behaviors_df["history"] = behaviors_df["history"].apply(lambda x: str(x).split(" "))
    behaviors_df["impressions"] = behaviors_df["impressions"].apply(lambda x: parse_impression(str(x)))

    # Calculate and print statistics from behaviors data
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    total_rows = len(behaviors_df)
    unique_users = behaviors_df["user_id"].nunique()
    
    # Calculate rows per user
    rows_per_user = behaviors_df.groupby("user_id").size()
    min_rows_per_user = rows_per_user.min()
    max_rows_per_user = rows_per_user.max()
    avg_rows_per_user = rows_per_user.mean()
    
    print(f"Total number of rows (impressions):     {total_rows:,}")
    print(f"Unique number of user IDs:             {unique_users:,}")
    print(f"Min impression rows per user:          {min_rows_per_user}")
    print(f"Max impression rows per user:          {max_rows_per_user}")
    print(f"Average impression rows per user:      {avg_rows_per_user:.2f}")

    print(behaviors_df.head(10))
    print("="*70 + "\n")

    # extract_urls(news_df=news_df, dataset_folder_path=dataset_folder_path)
    
    return behaviors_df, news_df


def build_training_samples(behaviors_df, neg_ratio=4):
    """
    Constructs a training dataset from MIND behaviors dataframe with negative sampling.
    
    This method creates training samples by:
    - Using each positive impression as an anchor
    - Randomly sampling negative impressions from the same impression (with replacement)
    - Each row contains a user, their history, a candidate news article, and negative samples
    
    Args:
        behaviors_df (pd.DataFrame): Behaviors dataframe with columns:
            - impression_id, user_id, time, history, impressions
            - history: list of news IDs clicked by the user
            - impressions: list of (news_id, label) tuples
        neg_ratio (int): Number of negative samples to draw per positive sample (default: 4)
    
    Returns:
        pd.DataFrame: Training samples with columns:
            - user_id: user identifier
            - history: list of clicked news IDs
            - candidate: positive news ID
            - neg_samples: list of negative news IDs (sampled with replacement)
            - label: always 1 (positive sample)
    """
    random.seed(42)
    
    training_samples = []
    
    for _, row in behaviors_df.iterrows():
        user_id = row['user_id']
        history = row['history']
        impressions = row['impressions']
        
        # Handle NaN or empty history
        if not isinstance(history, list) or history == ['nan']:
            history = []
        
        # Separate positives and negatives
        positives = [news_id for news_id, label in impressions if label == '1']
        negatives = [news_id for news_id, label in impressions if label == '0']
        
        # Skip if no positives or no negatives
        if not positives or not negatives:
            continue
        
        # For each positive, create a training sample
        for positive_news_id in positives:
            # Sample negatives with replacement
            neg_samples = random.choices(negatives, k=neg_ratio)
            
            training_samples.append({
                'user_id': user_id,
                'history': history,
                'candidate': positive_news_id,
                'neg_samples': neg_samples,
                'label': 1
            })
    
    return pd.DataFrame(training_samples)

def inspect_user_behavior(behaviors_df: pd.DataFrame):
    def check_time_consistency(g: pd.Series):
        lengths = g.map(len).tolist()
        return all(a <= b for a, b in zip(lengths[:-1], lengths[1:]))

    return ~behaviors_df.sort_values(by="time").groupby("user_id")["history"].agg(check_time_consistency).any()


def extract_urls(news_df: pd.DataFrame, dataset_folder_path: str):
    news_id_url_df = news_df[["news_id", "url"]]
    news_id_url_df.to_csv(f"{dataset_folder_path}/news_urls.csv")

def run():
    dataset_folder_path = "../../data/mind-small/MINDsmall_{split}/MINDsmall_{split}"

    train_behaviors_df, train_news_df = load_tsv_files(dataset_folder_path=dataset_folder_path.format(split="train"))
    val_behaviors_df, val_news_df = load_tsv_files(dataset_folder_path=dataset_folder_path.format(split="dev"))    
    
    print(inspect_user_behavior(train_behaviors_df))
    print(inspect_user_behavior(val_behaviors_df))


if __name__=="__main__":
    run()



