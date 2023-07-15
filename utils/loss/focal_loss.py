import torch
import torch.nn as nn


def focal_loss(logits, targets, alpha=0.25, gamma=2):
    """Compute the focal loss for binary classification.

    Args:
        logits (torch.Tensor): Predicted logits from the model.
        targets (torch.Tensor): True labels.
        alpha (float, optional): Weighting factor for the positive class.
        Defaults to 0.25.
        gamma (float, optional): Focusing parameter. Defaults to 2.

    Returns:
        torch.Tensor: Computed focal loss.
    """
    cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")
    ce_loss = cross_entropy_loss(logits, targets)
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss

    return loss
