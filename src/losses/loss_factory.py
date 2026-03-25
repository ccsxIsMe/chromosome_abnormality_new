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


def extract_anomaly_scores(model_output):
    if isinstance(model_output, dict):
        return model_output.get("anomaly_score")
    return None


def extract_prototype_distances(model_output):
    if isinstance(model_output, dict):
        return model_output.get("prototype_distances")
    return None


def extract_all_prototypes(model_output):
    if isinstance(model_output, dict):
        return model_output.get("all_prototypes")
    return None


def extract_pair_distance(model_output):
    if isinstance(model_output, dict):
        return model_output.get("pair_distance")
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


class PairContrastiveLoss(nn.Module):
    """
    Classic pairwise contrastive objective for homologous pair learning.

    Label semantics in this project:
    - 0: normal pair, should be close
    - 1: abnormal pair, should be farther than a margin
    """

    def __init__(
        self,
        cls_loss,
        cls_weight=1.0,
        pair_weight=0.1,
        margin=0.5,
        normal_weight=1.0,
        abnormal_weight=1.0,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.cls_weight = cls_weight
        self.pair_weight = pair_weight
        self.margin = margin
        self.normal_weight = normal_weight
        self.abnormal_weight = abnormal_weight

    def forward(self, model_output, targets):
        logits = extract_logits(model_output)
        pair_distance = extract_pair_distance(model_output)

        if pair_distance is None:
            raise ValueError("PairContrastiveLoss requires 'pair_distance' in model output")

        if pair_distance.ndim != 1:
            pair_distance = pair_distance.view(-1)

        targets = targets.view(-1)
        if pair_distance.size(0) != targets.size(0):
            raise ValueError("Batch size mismatch between pair_distance and targets")

        cls_term = self.cls_loss(logits, targets)
        total = self.cls_weight * cls_term

        normal_mask = targets == 0
        abnormal_mask = targets == 1

        pair_term = pair_distance.new_tensor(0.0)
        if normal_mask.any():
            pair_term = pair_term + self.normal_weight * pair_distance[normal_mask].pow(2).mean()
        if abnormal_mask.any():
            pair_term = pair_term + self.abnormal_weight * F.relu(self.margin - pair_distance[abnormal_mask]).pow(2).mean()

        total = total + self.pair_weight * pair_term
        return total


class MultiPrototypeMetricLoss(nn.Module):
    """
    Loss for multi-prototype metric learning.

    Main idea:
    - normal samples should be close to at least one prototype of their chromosome type
    - abnormal samples should be farther than a margin from all type-specific normal prototypes
    - optional CE auxiliary branch
    - optional prototype diversity regularization
    """

    def __init__(
        self,
        cls_loss=None,
        cls_weight=0.5,
        normal_weight=1.0,
        abnormal_weight=1.0,
        abnormal_margin=0.35,
        diversity_weight=0.05,
        diversity_margin=0.2,
        eps=1e-8,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.cls_weight = cls_weight
        self.normal_weight = normal_weight
        self.abnormal_weight = abnormal_weight
        self.abnormal_margin = abnormal_margin
        self.diversity_weight = diversity_weight
        self.diversity_margin = diversity_margin
        self.eps = eps

    def _prototype_diversity_penalty(self, all_prototypes):
        """
        all_prototypes: [T, K, D]
        Encourage different prototypes of one chromosome type to not collapse.
        """
        if all_prototypes is None:
            return None

        protos = F.normalize(all_prototypes, dim=-1)
        sim = torch.matmul(protos, protos.transpose(-1, -2))  # [T, K, K]

        k = sim.size(-1)
        eye = torch.eye(k, device=sim.device, dtype=torch.bool).unsqueeze(0)
        off_diag = sim.masked_select(~eye)

        if off_diag.numel() == 0:
            return sim.new_tensor(0.0)

        penalty = F.relu(off_diag - self.diversity_margin).mean()
        return penalty

    def forward(self, model_output, targets):
        prototype_dists = extract_prototype_distances(model_output)
        if prototype_dists is None:
            raise ValueError("MultiPrototypeMetricLoss requires 'prototype_distances' in model output")

        if prototype_dists.ndim != 2:
            raise ValueError(
                f"Expected prototype_distances with shape [B, K], got {tuple(prototype_dists.shape)}"
            )

        min_dist = prototype_dists.min(dim=1).values
        normal_mask = targets == 0
        abnormal_mask = targets == 1

        total_loss = min_dist.new_tensor(0.0)

        if self.cls_loss is not None and self.cls_weight > 0:
            logits = extract_logits(model_output)
            cls_term = self.cls_loss(logits, targets)
            total_loss = total_loss + self.cls_weight * cls_term

        if normal_mask.any():
            normal_term = min_dist[normal_mask].mean()
            total_loss = total_loss + self.normal_weight * normal_term

        if abnormal_mask.any():
            abnormal_term = F.relu(self.abnormal_margin - min_dist[abnormal_mask]).mean()
            total_loss = total_loss + self.abnormal_weight * abnormal_term

        all_prototypes = extract_all_prototypes(model_output)
        if self.diversity_weight > 0:
            diversity_term = self._prototype_diversity_penalty(all_prototypes)
            if diversity_term is not None:
                total_loss = total_loss + self.diversity_weight * diversity_term

        return total_loss


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


def build_loss(loss_cfg, device, experiment_mode="classifier"):
    cls_loss = build_base_classification_loss(loss_cfg, device)

    if experiment_mode == "multi_prototype_metric":
        metric_cfg = loss_cfg.get("metric", {})
        return MultiPrototypeMetricLoss(
            cls_loss=cls_loss,
            cls_weight=metric_cfg.get("cls_weight", 0.5),
            normal_weight=metric_cfg.get("normal_weight", 1.0),
            abnormal_weight=metric_cfg.get("abnormal_weight", 1.0),
            abnormal_margin=metric_cfg.get("abnormal_margin", 0.35),
            diversity_weight=metric_cfg.get("diversity_weight", 0.05),
            diversity_margin=metric_cfg.get("diversity_margin", 0.2),
        )

    aux_cfg = loss_cfg.get("auxiliary")
    if not aux_cfg or not aux_cfg.get("enabled", False):
        return ClassificationLossWrapper(cls_loss)

    aux_name = aux_cfg.get("name", "balanced_supcon")
    if aux_name == "balanced_supcon":
        contrastive_loss = BalancedSupConLoss(
            temperature=aux_cfg.get("temperature", 0.07),
        )

        return ClassificationContrastiveLoss(
            cls_loss=cls_loss,
            contrastive_loss=contrastive_loss,
            cls_weight=aux_cfg.get("cls_weight", 1.0),
            contrastive_weight=aux_cfg.get("contrastive_weight", 0.1),
        )

    if aux_name == "pair_contrastive":
        return PairContrastiveLoss(
            cls_loss=cls_loss,
            cls_weight=aux_cfg.get("cls_weight", 1.0),
            pair_weight=aux_cfg.get("pair_weight", 0.1),
            margin=aux_cfg.get("margin", 0.5),
            normal_weight=aux_cfg.get("normal_weight", 1.0),
            abnormal_weight=aux_cfg.get("abnormal_weight", 1.0),
        )

    if aux_name not in {"balanced_supcon", "pair_contrastive"}:
        raise ValueError(f"Unsupported auxiliary loss: {aux_name}")
