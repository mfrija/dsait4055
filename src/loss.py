import torch
import torch.nn as nn


class Loss(nn.Module):
    """
    Combined loss for (k+1) classification task in news recommendation.
    
    Training mode (labels=None):
        - Cross entropy over (k+1) candidates
        - Positive sample is always at position 0
        - Targets are always 0 (index of positive)
    
    Validation mode (labels provided):
        - Binary cross entropy over all candidates independently
        - Labels are 0/1 for each candidate
    """
    
    def __init__(self):
        super(Loss, self).__init__()
        self.train_criterion = nn.CrossEntropyLoss()
        self.val_criterion   = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, labels=None):
        """
        Args:
            logits (torch.Tensor): 
                Training:   shape [batch_size, k+1]
                Validation: shape [num_candidates]
            labels (torch.Tensor, optional):
                None for training.
                Binary tensor of shape [num_candidates] for validation.
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        if labels is None:
            # training — target is always index 0
            batch_size = logits.size(0)
            targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
            return self.train_criterion(logits, targets)
        else:
            # validation — binary CE over all candidates independently
            labels = labels.float().to(logits.device)
            return self.val_criterion(logits, labels)


def create_loss_fn():
    """Factory function to create the (k+1) classification loss."""
    return Loss()