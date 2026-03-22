import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPrototypeMetricModel(nn.Module):
    """
    Wrap a base model that outputs:
        {
            "logits": ...,
            "embedding": ...
        }

    and add per-chromosome-type learnable normal prototypes.

    For each sample:
    - select prototypes of its chromosome type
    - compute distances to K prototypes
    - anomaly_score = min distance to type-specific prototypes
    """

    def __init__(
        self,
        base_model,
        num_chromosome_types,
        embedding_dim,
        num_prototypes=4,
        distance="cosine",
        normalize_embedding=True,
    ):
        super().__init__()

        if num_chromosome_types is None:
            raise ValueError("num_chromosome_types must be provided for MultiPrototypeMetricModel")

        if embedding_dim is None:
            raise ValueError("embedding_dim must be provided for MultiPrototypeMetricModel")

        if distance not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported distance: {distance}")

        self.base_model = base_model
        self.num_chromosome_types = num_chromosome_types
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.distance = distance
        self.normalize_embedding = normalize_embedding

        self.prototypes = nn.Parameter(
            torch.randn(num_chromosome_types, num_prototypes, embedding_dim) * 0.02
        )

    def _maybe_normalize(self, x):
        if self.normalize_embedding:
            return F.normalize(x, dim=-1)
        return x

    def _compute_distances(self, embedding, chr_idx):
        """
        embedding: [B, D]
        chr_idx: [B]
        returns:
            dists: [B, K]
            min_dist: [B]
            min_idx: [B]
        """
        embedding = self._maybe_normalize(embedding)
        protos = self.prototypes[chr_idx]  # [B, K, D]
        protos = self._maybe_normalize(protos)

        if self.distance == "cosine":
            emb = embedding.unsqueeze(1)  # [B, 1, D]
            sim = F.cosine_similarity(emb, protos, dim=-1)  # [B, K]
            dists = 1.0 - sim
        else:
            emb = embedding.unsqueeze(1)  # [B, 1, D]
            dists = torch.norm(emb - protos, p=2, dim=-1)  # [B, K]

        min_dist, min_idx = dists.min(dim=1)
        return dists, min_dist, min_idx

    def forward(self, left_image, right_image, chr_idx=None):
        if chr_idx is None:
            raise ValueError("chr_idx is required for MultiPrototypeMetricModel")

        base_output = self.base_model(left_image, right_image, chr_idx)
        if not isinstance(base_output, dict):
            raise ValueError("base_model must return a dict containing at least 'embedding'")

        if "embedding" not in base_output:
            raise ValueError("base_model output must contain 'embedding'")

        embedding = base_output["embedding"]
        dists, min_dist, min_idx = self._compute_distances(embedding, chr_idx)

        output = dict(base_output)
        output["prototype_distances"] = dists
        output["anomaly_score"] = min_dist
        output["nearest_prototype_idx"] = min_idx
        output["all_prototypes"] = self.prototypes
        return output