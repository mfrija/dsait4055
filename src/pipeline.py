import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from preprocessing.preprocessing import load_tsv_files, build_training_samples
from preprocessing.utils import MINDTrainDataset, MINDTestDataset
from loss import Loss
from loss import create_loss_fn
import numpy as np
from sklearn.metrics import roc_auc_score


def data_loading():
    """Main pipeline to load and process MIND datasets."""
    
    # Define dataset paths
    train_dataset_path = "../data/mind-small/MINDsmall_train/MINDsmall_train"
    dev_dataset_path = "../data/mind-small/MINDsmall_dev/MINDsmall_dev"
    train_split_path = "../data/mind-small/MINDsmall_train/training_samples.csv"
    
    print("\n" + "#"*70)
    print("# LOADING MIND-SMALL TRAINING DATASET")
    print("#"*70)
    train_behaviors_df, train_news_df = load_tsv_files(train_dataset_path)
    
    print("\n" + "#"*70)
    print("# LOADING MIND-SMALL DEV DATASET")
    print("#"*70)
    dev_behaviors_df, dev_news_df = load_tsv_files(dev_dataset_path)
    
    # Split dev set 50-50 into validation and test sets
    print("\n" + "#"*70)
    print("# SPLITTING DEV SET INTO VALIDATION (50%) AND TEST (50%)")
    print("#"*70)
    split_idx = len(dev_behaviors_df) // 2
    
    val_behaviors_df = dev_behaviors_df.iloc[:split_idx].reset_index(drop=True)
    test_behaviors_df = dev_behaviors_df.iloc[split_idx:].reset_index(drop=True)
    
    val_news_df = dev_news_df
    test_news_df = dev_news_df
    
    print(f"Validation behaviors shape: {val_behaviors_df.shape}")
    print(f"Test behaviors shape: {test_behaviors_df.shape}")
    
    # Check if training samples file already exists
    print("\n" + "#"*70)
    print("# BUILDING TRAINING SAMPLES")
    print("#"*70)
    
    if os.path.exists(train_split_path):
        print(f"Loading existing training samples from {train_split_path}...")
        train_samples_df = pd.read_csv(train_split_path)
        print(f"Loaded training samples shape: {train_samples_df.shape}")
    else:
        print(f"Creating training samples (this may take a while)...")
        train_samples_df = build_training_samples(train_behaviors_df, neg_ratio=4)
        print(f"Created training samples shape: {train_samples_df.shape}")
        
        # Save training samples to CSV
        print(f"Saving training samples to {train_split_path}...")
        train_samples_df.to_csv(train_split_path, index=False)
        print("Training samples saved successfully!")
    
    print("\n" + "#"*70)
    print("# SUMMARY")
    print("#"*70)
    print(f"Train behaviors shape: {train_behaviors_df.shape}")
    print(f"Train news shape: {train_news_df.shape}")
    print(f"Train samples shape: {train_samples_df.shape}")
    print(f"Validation behaviors shape: {val_behaviors_df.shape}")
    print(f"Validation news shape: {val_news_df.shape}")
    print(f"Test behaviors shape: {test_behaviors_df.shape}")
    print(f"Test news shape: {test_news_df.shape}")
    print("#"*70 + "\n")
    
    return (train_behaviors_df, train_news_df, train_samples_df, 
            val_behaviors_df, val_news_df, 
            test_behaviors_df, test_news_df)



def train_epoch(model, train_dataloader, optimizer, loss_fn, device):
    """
    Execute one training epoch.
    
    Args:
        model: Neural network model
        train_dataloader: DataLoader for training samples
        optimizer: Optimizer (e.g., Adam)
        loss_fn: Loss function (KPlusOneClassificationLoss)
        device: Device to run on (cpu or cuda)
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        # Move batch to device
        # batch contains: user_id, history, candidate, neg_samples
        
        # TODO: Prepare model input from batch
        # logits = model(batch)
        
        # TODO: Compute loss
        # loss = loss_fn(logits)
        
        # Placeholder for demonstration
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx + 1}/{len(train_dataloader)} - Avg Loss: {avg_loss:.4f}")
    
    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_epoch_loss

def mrr_score(labels, scores):
    # sort by score descending, find rank of first relevant item
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for rank, idx in enumerate(sorted_indices, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0

def ndcg_score(labels, scores, k=10):
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    dcg  = sum(labels[idx] / np.log2(rank + 2) for rank, idx in enumerate(sorted_indices))
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(sum(labels), k)))
    return dcg / idcg if idcg > 0 else 0.0

def validate(model, val_dataloader, loss_fn: Loss, device):
    model.eval()
    
    all_aucs  = []
    all_mrrs  = []
    all_ndcgs = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for impression in val_dataloader:
            # each impression is one full set of candidates with labels
            history    = impression['history'].to(device)
            candidates = impression['candidates'].to(device)
            labels     = impression['labels']     # keep on cpu for sklearn
            num_batches += 1

            # score each candidate independently
            # shape: (num_candidates,)
            scores = model(history, candidates)
            scores = scores.cpu().numpy()
            labels = labels.numpy()

            # skip impressions with no positive labels
            if labels.sum() == 0:
                continue
            
            total_loss += loss_fn(scores, labels)
            all_aucs.append(roc_auc_score(labels, scores))
            all_mrrs.append(mrr_score(labels, scores))
            all_ndcgs.append(ndcg_score(labels, scores, k=5))

    metrics = {
        'loss': total_loss / num_batches,
        'auc':  np.mean(all_aucs),
        'mrr':  np.mean(all_mrrs),
        'ndcg': np.mean(all_ndcgs)
    }

    return metrics


def main():
    """Main pipeline execution."""
    # Load data
    train_behaviors_df, train_news_df, train_samples_df, \
    val_behaviors_df, val_news_df, \
    test_behaviors_df, test_news_df = data_loading()
    
    print("\n" + "#"*70)
    print("# TRAINING SETUP")
    print("#"*70)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize loss function
    loss_fn = create_loss_fn()
    print(f"Loss function: (k+1) Classification Cross Entropy Loss")
    
    # TODO: Initialize model
    # model = NewsRecommendationModel(...)
    # model = model.to(device)
    
    # TODO: Initialize optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoaders
    train_dataset = MINDTrainDataset(train_samples_df=train_samples_df, news_store=train_news_df)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = MINDTestDataset(behaviors_df=val_behaviors_df, news_store=val_news_df)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("\n" + "#"*70)
    print("# TRAINING LOOP STRUCTURE")
    print("#"*70)
    print("Training and validation loops are prepared.")
    print("Model implementation is required to proceed with actual training.")
    print("Expected loop structure:")
    print("  1. train_epoch() - Executes one training epoch")
    print("  2. validate() - Evaluates on validation set")
    print("  3. Early stopping / checkpoint logic (to be implemented)")
    print("#"*70 + "\n")
    
    # Example training loop structure (commented out)
    """
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation
        val_metrics = validate(model, val_dataloader, loss_fn, device)
        val_loss = val_metrics['loss']
        print(f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Model checkpoint saved!")
    """


if __name__ == "__main__":
    main()