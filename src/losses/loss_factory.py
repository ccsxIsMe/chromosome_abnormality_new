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


def extract_side_logits(model_output):
    if isinstance(model_output, dict):
        return model_output.get("side_logits")
    return None


def extract_structure_logits(model_output):
    if isinstance(model_output, dict):
        return model_output.get("structure_logits")
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

    def forward(self, model_output, targets, batch=None):
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

    def forward(self, model_output, targets, batch=None):
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

    def forward(self, model_output, targets, batch=None):
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


class PairDetectionSideLoss(nn.Module):
    def __init__(self, cls_loss, side_weight=0.5, side_class_weights=None):
        super().__init__()
        self.cls_loss = cls_loss
        self.side_weight = side_weight
        self.side_class_weights = side_class_weights

    def forward(self, model_output, targets, batch=None):
        logits = extract_logits(model_output)
        total = self.cls_loss(logits, targets)

        if self.side_weight <= 0 or batch is None or "side_label" not in batch:
            return total

        side_logits = extract_side_logits(model_output)
        if side_logits is None:
            return total

        side_labels = batch["side_label"].to(logits.device)
        valid_mask = side_labels >= 0
        if not valid_mask.any():
            return total

        if self.side_class_weights is not None:
            weight = torch.tensor(
                self.side_class_weights,
                dtype=torch.float32,
                device=logits.device,
            )
            side_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            side_loss = nn.CrossEntropyLoss()

        side_term = side_loss(side_logits[valid_mask], side_labels[valid_mask])
        return total + self.side_weight * side_term


class PairStructuredAttributeLoss(nn.Module):
    def __init__(
        self,
        cls_loss,
        cls_weight=1.0,
        pair_weight=0.2,
        margin=0.5,
        normal_weight=1.0,
        abnormal_weight=1.0,
        structure_weight=0.5,
        pericentric_weight=1.0,
        arm_weight=1.0,
        major_weight=1.0,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.cls_weight = cls_weight
        self.pair_weight = pair_weight
        self.margin = margin
        self.normal_weight = normal_weight
        self.abnormal_weight = abnormal_weight
        self.structure_weight = structure_weight
        self.pericentric_weight = pericentric_weight
        self.arm_weight = arm_weight
        self.major_weight = major_weight

    def _masked_ce(self, logits, labels):
        valid_mask = labels >= 0
        if logits is None or not valid_mask.any():
            return None
        return F.cross_entropy(logits[valid_mask], labels[valid_mask])

    def forward(self, model_output, targets, batch=None):
        logits = extract_logits(model_output)
        pair_distance = extract_pair_distance(model_output)
        structure_logits = extract_structure_logits(model_output)

        if pair_distance is None:
            raise ValueError("PairStructuredAttributeLoss requires 'pair_distance' in model output")

        targets = targets.view(-1)
        total = self.cls_weight * self.cls_loss(logits, targets)

        normal_mask = targets == 0
        abnormal_mask = targets == 1

        pair_term = pair_distance.new_tensor(0.0)
        if normal_mask.any():
            pair_term = pair_term + self.normal_weight * pair_distance[normal_mask].pow(2).mean()
        if abnormal_mask.any():
            pair_term = pair_term + self.abnormal_weight * F.relu(self.margin - pair_distance[abnormal_mask]).pow(2).mean()
        total = total + self.pair_weight * pair_term

        if self.structure_weight <= 0 or structure_logits is None or batch is None:
            return total

        structure_total = pair_distance.new_tensor(0.0)
        structure_terms = 0

        pericentric_labels = batch.get("pericentric_label")
        if pericentric_labels is not None:
            loss = self._masked_ce(
                structure_logits.get("pericentric_logits"),
                pericentric_labels.to(logits.device),
            )
            if loss is not None:
                structure_total = structure_total + self.pericentric_weight * loss
                structure_terms += self.pericentric_weight

        bp1_arm_labels = batch.get("bp1_arm_label")
        if bp1_arm_labels is not None:
            loss = self._masked_ce(
                structure_logits.get("bp1_arm_logits"),
                bp1_arm_labels.to(logits.device),
            )
            if loss is not None:
                structure_total = structure_total + self.arm_weight * loss
                structure_terms += self.arm_weight

        bp2_arm_labels = batch.get("bp2_arm_label")
        if bp2_arm_labels is not None:
            loss = self._masked_ce(
                structure_logits.get("bp2_arm_logits"),
                bp2_arm_labels.to(logits.device),
            )
            if loss is not None:
                structure_total = structure_total + self.arm_weight * loss
                structure_terms += self.arm_weight

        bp1_major_labels = batch.get("bp1_major_label")
        if bp1_major_labels is not None:
            loss = self._masked_ce(
                structure_logits.get("bp1_major_logits"),
                bp1_major_labels.to(logits.device),
            )
            if loss is not None:
                structure_total = structure_total + self.major_weight * loss
                structure_terms += self.major_weight

        bp2_major_labels = batch.get("bp2_major_label")
        if bp2_major_labels is not None:
            loss = self._masked_ce(
                structure_logits.get("bp2_major_logits"),
                bp2_major_labels.to(logits.device),
            )
            if loss is not None:
                structure_total = structure_total + self.major_weight * loss
                structure_terms += self.major_weight

        if structure_terms > 0:
            total = total + self.structure_weight * (structure_total / structure_terms)

        return total


class BPaCoLoss(nn.Module):
    """
    Practical BPaCo-style adaptation for long-tailed subtype classification.

    Design choices for this codebase:
    - keep the standard CE head as the main supervised objective
    - add logit compensation based on class prior
    - maintain class centers plus a memory queue of embeddings
    - average queued negatives by class to avoid head-class domination

    This is an adaptation to the current single-forward training framework,
    not an exact reproduction of the original repository training script.
    """

    def __init__(
        self,
        cls_loss,
        num_classes,
        embedding_dim,
        class_counts=None,
        cls_weight=1.0,
        contrastive_weight=0.1,
        temperature=0.07,
        queue_size=1024,
        center_momentum=0.9,
        logit_tau=1.0,
        eps=1e-8,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.cls_weight = cls_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.queue_size = int(queue_size)
        self.center_momentum = center_momentum
        self.logit_tau = logit_tau
        self.eps = eps

        self.register_buffer("class_centers", torch.zeros(self.num_classes, self.embedding_dim))
        self.register_buffer("center_initialized", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("queue", torch.zeros(self.queue_size, self.embedding_dim))
        self.register_buffer("queue_labels", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if class_counts is None:
            class_counts = [1.0] * self.num_classes
        if len(class_counts) != self.num_classes:
            raise ValueError("class_counts length must match num_classes")

        class_count_tensor = torch.tensor(class_counts, dtype=torch.float32)
        class_prior = class_count_tensor / class_count_tensor.sum().clamp_min(1.0)
        self.register_buffer("class_counts", class_count_tensor)
        self.register_buffer("class_prior", class_prior)

    @torch.no_grad()
    def _update_centers(self, features, targets):
        for cls in targets.unique(sorted=True).tolist():
            cls = int(cls)
            cls_mask = targets == cls
            cls_feat = features[cls_mask].mean(dim=0)
            if not self.center_initialized[cls]:
                updated = cls_feat
                self.center_initialized[cls] = True
            else:
                updated = self.center_momentum * self.class_centers[cls] + (1.0 - self.center_momentum) * cls_feat
            self.class_centers[cls] = F.normalize(updated, dim=0)

    @torch.no_grad()
    def _enqueue(self, features, targets):
        batch_size = features.size(0)
        if batch_size == 0:
            return

        ptr = int(self.queue_ptr.item())
        for idx in range(batch_size):
            self.queue[ptr] = features[idx]
            self.queue_labels[ptr] = targets[idx]
            ptr = (ptr + 1) % self.queue_size
        self.queue_ptr[0] = ptr

    def _class_averaged_queue_logits(self, features, queue_features, queue_labels):
        valid_mask = queue_labels >= 0
        if not valid_mask.any():
            return features.new_zeros(features.size(0), self.num_classes)

        queue_feat = queue_features[valid_mask]
        queue_labels = queue_labels[valid_mask]
        sim = torch.matmul(features, queue_feat.t()) / self.temperature

        aggregated = features.new_zeros(features.size(0), self.num_classes)
        for cls in range(self.num_classes):
            cls_mask = queue_labels == cls
            if cls_mask.any():
                cls_logits = sim[:, cls_mask]
                aggregated[:, cls] = torch.logsumexp(cls_logits, dim=1) - torch.log(
                    cls_logits.new_tensor(float(cls_mask.sum().item()))
                )
        return aggregated

    def forward(self, model_output, targets, batch=None):
        logits = extract_logits(model_output)
        embeddings = extract_embeddings(model_output)
        if embeddings is None:
            raise ValueError("BPaCoLoss requires model output to contain 'embedding'")

        features = F.normalize(embeddings, dim=1)
        targets = targets.view(-1)

        # Use buffer snapshots for the current forward pass, then update the
        # real buffers after loss computation. This avoids in-place version
        # conflicts during backward.
        center_initialized = self.center_initialized.detach().clone()
        class_centers = self.class_centers.detach().clone()
        queue_features = self.queue.detach().clone()
        queue_labels = self.queue_labels.detach().clone()
        class_prior = self.class_prior.detach().clone()

        center_logits = features.new_zeros(features.size(0), self.num_classes)
        valid_center_mask = center_initialized
        if valid_center_mask.any():
            valid_centers = F.normalize(class_centers[valid_center_mask], dim=1)
            center_logits[:, valid_center_mask] = torch.matmul(features, valid_centers.t()) / self.temperature

        queue_logits = self._class_averaged_queue_logits(features, queue_features, queue_labels)
        complement_logits = torch.logsumexp(
            torch.stack([center_logits, queue_logits], dim=0),
            dim=0,
        )

        prior_adjust = self.logit_tau * torch.log(class_prior.clamp_min(self.eps))
        cls_term = self.cls_loss(logits - prior_adjust.unsqueeze(0), targets)
        contrastive_term = F.cross_entropy(complement_logits - prior_adjust.unsqueeze(0), targets)

        with torch.no_grad():
            self._update_centers(features.detach(), targets.detach())
            self._enqueue(features.detach(), targets.detach())

        return self.cls_weight * cls_term + self.contrastive_weight * contrastive_term


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
        structure_weight=0.0,
        pericentric_weight=1.0,
        arm_weight=1.0,
        major_weight=1.0,
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
        self.structure_weight = structure_weight
        self.pericentric_weight = pericentric_weight
        self.arm_weight = arm_weight
        self.major_weight = major_weight
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

    def _masked_ce(self, logits, labels):
        valid_mask = labels >= 0
        if logits is None or not valid_mask.any():
            return None
        return F.cross_entropy(logits[valid_mask], labels[valid_mask])

    def forward(self, model_output, targets, batch=None):
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

        if self.structure_weight > 0 and batch is not None:
            structure_logits = extract_structure_logits(model_output)
            if structure_logits is not None:
                structure_total = min_dist.new_tensor(0.0)
                structure_terms = 0.0

                pericentric_labels = batch.get("pericentric_label")
                if pericentric_labels is not None:
                    loss = self._masked_ce(
                        structure_logits.get("pericentric_logits"),
                        pericentric_labels.to(min_dist.device),
                    )
                    if loss is not None:
                        structure_total = structure_total + self.pericentric_weight * loss
                        structure_terms += self.pericentric_weight

                bp1_arm_labels = batch.get("bp1_arm_label")
                if bp1_arm_labels is not None:
                    loss = self._masked_ce(
                        structure_logits.get("bp1_arm_logits"),
                        bp1_arm_labels.to(min_dist.device),
                    )
                    if loss is not None:
                        structure_total = structure_total + self.arm_weight * loss
                        structure_terms += self.arm_weight

                bp2_arm_labels = batch.get("bp2_arm_label")
                if bp2_arm_labels is not None:
                    loss = self._masked_ce(
                        structure_logits.get("bp2_arm_logits"),
                        bp2_arm_labels.to(min_dist.device),
                    )
                    if loss is not None:
                        structure_total = structure_total + self.arm_weight * loss
                        structure_terms += self.arm_weight

                bp1_major_labels = batch.get("bp1_major_label")
                if bp1_major_labels is not None:
                    loss = self._masked_ce(
                        structure_logits.get("bp1_major_logits"),
                        bp1_major_labels.to(min_dist.device),
                    )
                    if loss is not None:
                        structure_total = structure_total + self.major_weight * loss
                        structure_terms += self.major_weight

                bp2_major_labels = batch.get("bp2_major_label")
                if bp2_major_labels is not None:
                    loss = self._masked_ce(
                        structure_logits.get("bp2_major_logits"),
                        bp2_major_labels.to(min_dist.device),
                    )
                    if loss is not None:
                        structure_total = structure_total + self.major_weight * loss
                        structure_terms += self.major_weight

                if structure_terms > 0:
                    total_loss = total_loss + self.structure_weight * (structure_total / structure_terms)

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


def build_loss(loss_cfg, device, experiment_mode="classifier", model=None):
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
            structure_weight=metric_cfg.get("structure_weight", 0.0),
            pericentric_weight=metric_cfg.get("pericentric_weight", 1.0),
            arm_weight=metric_cfg.get("arm_weight", 1.0),
            major_weight=metric_cfg.get("major_weight", 1.0),
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

    if aux_name == "pair_side":
        return PairDetectionSideLoss(
            cls_loss=cls_loss,
            side_weight=aux_cfg.get("side_weight", 0.5),
            side_class_weights=aux_cfg.get("side_class_weights"),
        )

    if aux_name == "bpaco":
        if model is None:
            raise ValueError("BPaCo loss requires the instantiated model")
        if not hasattr(model, "embedding_dim"):
            raise ValueError("BPaCo loss requires model.embedding_dim")

        return BPaCoLoss(
            cls_loss=cls_loss,
            num_classes=aux_cfg["num_classes"],
            embedding_dim=model.embedding_dim,
            class_counts=aux_cfg.get("class_counts"),
            cls_weight=aux_cfg.get("cls_weight", 1.0),
            contrastive_weight=aux_cfg.get("contrastive_weight", 0.1),
            temperature=aux_cfg.get("temperature", 0.07),
            queue_size=aux_cfg.get("queue_size", 1024),
            center_momentum=aux_cfg.get("center_momentum", 0.9),
            logit_tau=aux_cfg.get("logit_tau", 1.0),
        )

    if aux_name == "pair_structured":
        return PairStructuredAttributeLoss(
            cls_loss=cls_loss,
            cls_weight=aux_cfg.get("cls_weight", 1.0),
            pair_weight=aux_cfg.get("pair_weight", 0.2),
            margin=aux_cfg.get("margin", 0.5),
            normal_weight=aux_cfg.get("normal_weight", 1.0),
            abnormal_weight=aux_cfg.get("abnormal_weight", 1.0),
            structure_weight=aux_cfg.get("structure_weight", 0.5),
            pericentric_weight=aux_cfg.get("pericentric_weight", 1.0),
            arm_weight=aux_cfg.get("arm_weight", 1.0),
            major_weight=aux_cfg.get("major_weight", 1.0),
        )

    if aux_name not in {"balanced_supcon", "pair_contrastive", "pair_side", "bpaco", "pair_structured"}:
        raise ValueError(f"Unsupported auxiliary loss: {aux_name}")
