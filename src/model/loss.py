import torch
import torch.nn as nn
import torch.nn.functional as F


# Compute a smooth loss that combines L1 and L2 behavior
class HuberLoss(nn.Module):
    # Intiailizer
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, preds, targets):
        assert preds.shape == targets.shape

        # Convert delta to a tensor
        delta = torch.tensor(self.delta, dtype=preds.dtype, device=preds.device)

        # Compute the absolute error
        abs_error = (preds - targets).abs()

        # Check if the absolute error is within the delta threshold
        within_delta = abs_error <= delta

        # Compute loss in the quadratic region (L2)
        quadratic_loss = 0.5 * (abs_error**2)

        # Compute loss in the linear region (L1)
        linear_loss = delta * (abs_error - 0.5 * delta)

        # Select loss based on whether the error is within delta
        loss = torch.where(within_delta, quadratic_loss, linear_loss)

        return loss.mean()


# Compute a modulated cross-entropy loss that focuses on hard examples
class FocalLoss(nn.Module):
    # Initializer
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, preds, targets):
        assert preds.dim() == 2 and targets.dim() == 1
        assert preds.size(0) == targets.size(0)

        # Compute softmax probabilities
        log_probs = F.log_softmax(preds, dim=1)

        # Gather the probabilities of the true class
        probs = torch.exp(log_probs)
        true_probs = probs.gather(dim=1, index=targets.unsqueeze(-1)).squeeze(1)
        true_probs = true_probs.clamp(min=1e-8, max=1.0)

        # Compute the focal weight
        focal_weight = (1.0 - true_probs) ** self.gamma

        # Compute the focal loss
        true_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(-1)).squeeze(1)
        loss = (-1.0) * focal_weight * true_log_probs

        return loss.mean()
