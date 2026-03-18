import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_logits(model_output):
    if isinstance(model_output, dict):
        if "logits" not in model_output:
            raise ValueError("Model output dict must contain 'logits'")
        return model_output["logits"]
    return model_output


def extract_embeddings(model_output):
    if isinstance(model_output, dict):
        return model_output.get("embedding")
    return None


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
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassificationLossWrapper(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, model_output, targets):
        logits = extract_logits(model_output)
        return self.base_loss(logits, targets)


class BalancedSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, labels):
        if embeddings is None:
            raise ValueError("BalancedSupConLoss requires embeddings from the model output")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings of shape [B, D], got {tuple(embeddings.shape)}")

        labels = labels.view(-1)
        if embeddings.size(0) != labels.size(0):
            raise ValueError("Batch size mismatch between embeddings and labels")

        if embeddings.size(0) < 2:
            return embeddings.new_tensor(0.0)

        features = F.normalize(embeddings, dim=1)
        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        device = embeddings.device
        batch_size = embeddings.size(0)
        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask

        valid_anchor_mask = positive_mask.any(dim=1)
        if not valid_anchor_mask.any():
            return embeddings.new_tensor(0.0)

        logits_mask = (~self_mask).float()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        positive_mask_float = positive_mask.float()
        mean_log_prob_pos = (positive_mask_float * log_prob).sum(dim=1) / (
            positive_mask_float.sum(dim=1) + self.eps
        )

        class_counts = torch.bincount(labels, minlength=int(labels.max().item()) + 1).float().to(device)
        anchor_weights = 1.0 / class_counts[labels].clamp_min(1.0)
        anchor_weights = anchor_weights * valid_anchor_mask.float()
        anchor_weights = anchor_weights / anchor_weights.sum().clamp_min(self.eps)

        loss = -(anchor_weights * mean_log_prob_pos).sum()
        return loss


class ClassificationContrastiveLoss(nn.Module):
    def __init__(self, cls_loss, contrastive_loss, cls_weight=1.0, contrastive_weight=0.1):
        super().__init__()
        self.cls_loss = cls_loss
        self.contrastive_loss = contrastive_loss
        self.cls_weight = cls_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, model_output, targets):
        logits = extract_logits(model_output)
        embeddings = extract_embeddings(model_output)

        cls_term = self.cls_loss(logits, targets)
        contrastive_term = self.contrastive_loss(embeddings, targets)
        return self.cls_weight * cls_term + self.contrastive_weight * contrastive_term


def build_base_classification_loss(loss_cfg, device):
    name = loss_cfg["name"]

    class_weights = loss_cfg.get("class_weights")
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if name == "ce":
        return nn.CrossEntropyLoss(weight=weight_tensor)

    if name == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        return FocalLoss(alpha=weight_tensor, gamma=gamma)

    raise ValueError(f"Unsupported loss: {name}")


def build_loss(loss_cfg, device):
    cls_loss = build_base_classification_loss(loss_cfg, device)

    aux_cfg = loss_cfg.get("auxiliary")
    if not aux_cfg or not aux_cfg.get("enabled", False):
        return ClassificationLossWrapper(cls_loss)

    aux_name = aux_cfg.get("name", "balanced_supcon")
    if aux_name != "balanced_supcon":
        raise ValueError(f"Unsupported auxiliary loss: {aux_name}")

    contrastive_loss = BalancedSupConLoss(
        temperature=aux_cfg.get("temperature", 0.07),
    )

    return ClassificationContrastiveLoss(
        cls_loss=cls_loss,
        contrastive_loss=contrastive_loss,
        cls_weight=aux_cfg.get("cls_weight", 1.0),
        contrastive_weight=aux_cfg.get("contrastive_weight", 0.1),
    )
