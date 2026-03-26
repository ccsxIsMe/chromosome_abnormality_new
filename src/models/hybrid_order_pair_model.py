import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.local_pair_comparator import ResNetFeatureExtractor


class HybridOrderAwarePairClassifier(nn.Module):
    """
    Strong pair baseline + order-sensitive branch.

    Rationale:
    - keep the proven global siamese comparison path
    - add an explicit sequence-order branch to capture inversion-like evidence
    - do not force the order branch to solve the whole task alone
    """

    def __init__(
        self,
        backbone_name="resnet18",
        num_classes=2,
        pretrained=False,
        proj_dim=256,
        hidden_dim=256,
        seq_layers=2,
        num_heads=4,
        dropout=0.3,
        use_chromosome_id=False,
        num_chromosome_types=None,
        chr_embed_dim=16,
    ):
        super().__init__()

        self.use_chromosome_id = use_chromosome_id
        self.encoder = ResNetFeatureExtractor(backbone_name=backbone_name, pretrained=pretrained)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=seq_layers)

        if self.use_chromosome_id:
            if num_chromosome_types is None:
                raise ValueError("num_chromosome_types must be provided when use_chromosome_id=True")
            self.chr_embedding = nn.Embedding(num_chromosome_types, chr_embed_dim)
        else:
            chr_embed_dim = 0

        pair_feat_dim = proj_dim * 6 + 3 + chr_embed_dim
        self.embedding_head = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.embedding_dim = hidden_dim

    def _to_sequence(self, feat_map):
        seq = feat_map.mean(dim=3).transpose(1, 2)
        return self.sequence_encoder(seq)

    def forward(self, left_image, right_image, chr_idx=None):
        left_feat = self.feature_proj(self.encoder(left_image))
        right_feat = self.feature_proj(self.encoder(right_image))

        left_global = F.adaptive_avg_pool2d(left_feat, output_size=1).flatten(1)
        right_global = F.adaptive_avg_pool2d(right_feat, output_size=1).flatten(1)
        global_diff = torch.abs(left_global - right_global)
        global_mul = left_global * right_global

        left_seq = self._to_sequence(left_feat)
        right_seq = self._to_sequence(right_feat)
        right_seq_rev = torch.flip(right_seq, dims=[1])

        direct_diff_map = torch.abs(left_seq - right_seq)
        reverse_diff_map = torch.abs(left_seq - right_seq_rev)

        direct_vec = direct_diff_map.mean(dim=1)
        reverse_vec = reverse_diff_map.mean(dim=1)

        direct_dist = direct_diff_map.mean(dim=(1, 2))
        reverse_dist = reverse_diff_map.mean(dim=(1, 2))
        reverse_gain = F.relu(direct_dist - reverse_dist)

        fused = [
            left_global,
            right_global,
            global_diff,
            global_mul,
            direct_vec,
            reverse_vec,
            torch.stack([direct_dist, reverse_dist, reverse_gain], dim=1),
        ]

        if self.use_chromosome_id:
            if chr_idx is None:
                raise ValueError("chr_idx is required when use_chromosome_id=True")
            fused.append(self.chr_embedding(chr_idx))

        embedding = self.embedding_head(torch.cat(fused, dim=1))
        logits = self.classifier(embedding)

        pair_distance = 1.0 - F.cosine_similarity(
            F.normalize(left_global, dim=1),
            F.normalize(right_global, dim=1),
            dim=1,
        )

        return {
            "logits": logits,
            "embedding": embedding,
            "pair_distance": pair_distance,
            "direct_distance": direct_dist,
            "reverse_distance": reverse_dist,
            "reverse_gain": reverse_gain,
        }
