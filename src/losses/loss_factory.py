import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(loss_cfg, device):
    name = loss_cfg["name"]

    class_weights = loss_cfg.get("class_weights", None)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if name == "ce":
        return nn.CrossEntropyLoss(weight=weight_tensor)

    if name == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        return FocalLoss(alpha=weight_tensor, gamma=gamma)

    raise ValueError(f"Unsupported loss: {name}")