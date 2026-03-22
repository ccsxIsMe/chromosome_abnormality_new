import torch
import torch.nn.functional as F


class PerTypePrototypeBank:
    """
    Per-chromosome-type prototype bank.

    Current version:
    - fit normal-only prototypes per chromosome type
    - optional global normal fallback
    - anomaly score = distance to corresponding normal prototype
      (higher score => more abnormal)
    """

    def __init__(
        self,
        distance="cosine",
        normalize_embedding=True,
        fallback="global_normal",
        eps=1e-8,
    ):
        if distance not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported distance: {distance}")

        if fallback not in {"global_normal", "none"}:
            raise ValueError(f"Unsupported fallback: {fallback}")

        self.distance = distance
        self.normalize_embedding = normalize_embedding
        self.fallback = fallback
        self.eps = eps

        self.type_prototypes = {}
        self.global_normal_prototype = None
        self.fitted = False

    def _maybe_normalize(self, x):
        if self.normalize_embedding:
            return F.normalize(x, dim=-1)
        return x

    def fit(self, embeddings, chr_indices, labels, fit_on="normal_only"):
        """
        Args:
            embeddings: Tensor [N, D]
            chr_indices: Tensor [N]
            labels: Tensor [N]
            fit_on:
                - "normal_only": use only label == 0
                - "all": use all samples (not recommended for current task)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [N, D], got {tuple(embeddings.shape)}")

        if chr_indices.ndim != 1 or labels.ndim != 1:
            raise ValueError("chr_indices and labels must be 1D tensors")

        if not (embeddings.size(0) == chr_indices.size(0) == labels.size(0)):
            raise ValueError("Batch size mismatch among embeddings / chr_indices / labels")

        embeddings = self._maybe_normalize(embeddings)

        if fit_on == "normal_only":
            mask = labels == 0
        elif fit_on == "all":
            mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported fit_on: {fit_on}")

        if not mask.any():
            raise ValueError("No samples available for prototype fitting")

        fit_embeddings = embeddings[mask]
        fit_chr_indices = chr_indices[mask]

        self.type_prototypes = {}
        unique_types = fit_chr_indices.unique().tolist()

        for t in unique_types:
            type_mask = fit_chr_indices == t
            proto = fit_embeddings[type_mask].mean(dim=0)
            proto = self._maybe_normalize(proto.unsqueeze(0)).squeeze(0)
            self.type_prototypes[int(t)] = proto.detach().cpu()

        if self.fallback == "global_normal":
            global_proto = fit_embeddings.mean(dim=0)
            global_proto = self._maybe_normalize(global_proto.unsqueeze(0)).squeeze(0)
            self.global_normal_prototype = global_proto.detach().cpu()
        else:
            self.global_normal_prototype = None

        self.fitted = True

    def _get_type_prototype(self, chr_idx):
        chr_idx = int(chr_idx)
        if chr_idx in self.type_prototypes:
            return self.type_prototypes[chr_idx]

        if self.fallback == "global_normal" and self.global_normal_prototype is not None:
            return self.global_normal_prototype

        raise KeyError(f"No prototype found for chr_idx={chr_idx}")

    def score(self, embeddings, chr_indices):
        """
        Args:
            embeddings: Tensor [B, D]
            chr_indices: Tensor [B]

        Returns:
            scores: Tensor [B], higher => more abnormal
        """
        if not self.fitted:
            raise RuntimeError("Prototype bank has not been fitted yet")

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [B, D], got {tuple(embeddings.shape)}")

        if chr_indices.ndim != 1:
            raise ValueError("chr_indices must be 1D")

        if embeddings.size(0) != chr_indices.size(0):
            raise ValueError("Batch size mismatch between embeddings and chr_indices")

        device = embeddings.device
        embeddings = self._maybe_normalize(embeddings)

        proto_list = []
        for idx in chr_indices.detach().cpu().tolist():
            proto_list.append(self._get_type_prototype(idx))

        prototypes = torch.stack(proto_list, dim=0).to(device)
        prototypes = self._maybe_normalize(prototypes)

        if self.distance == "cosine":
            sim = F.cosine_similarity(embeddings, prototypes, dim=1, eps=self.eps)
            scores = 1.0 - sim
        else:  # euclidean
            scores = torch.norm(embeddings - prototypes, p=2, dim=1)

        return scores