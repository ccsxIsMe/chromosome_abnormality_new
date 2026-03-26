import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.local_pair_comparator import ResNetFeatureExtractor


class OrderAwarePairComparator(nn.Module):
    """
    Pair model with explicit order-sensitive sequence comparison.

    The last CNN feature map is converted into a 1D sequence along the
    chromosome long axis. We compare:
    - direct alignment: left vs right
    - flipped alignment: left vs reversed right

    Inversion-like evidence is captured by how much better the flipped
    alignment matches than the direct alignment.
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
        dropout=0.2,
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

        summary_dim = proj_dim * 4 + 3 + chr_embed_dim
        self.embedding_head = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.embedding_dim = hidden_dim

    def _feature_map_to_sequence(self, feat_map):
        seq = feat_map.mean(dim=3).transpose(1, 2)
        return self.sequence_encoder(seq)

    def _sequence_summary(self, seq_left, seq_right):
        seq_right_rev = torch.flip(seq_right, dims=[1])

        direct_diff = torch.abs(seq_left - seq_right)
        reverse_diff = torch.abs(seq_left - seq_right_rev)

        direct_dist = direct_diff.mean(dim=(1, 2))
        reverse_dist = reverse_diff.mean(dim=(1, 2))
        reverse_gain = F.relu(direct_dist - reverse_dist)

        left_vec = seq_left.mean(dim=1)
        right_vec = seq_right.mean(dim=1)
        direct_vec = direct_diff.mean(dim=1)
        reverse_vec = reverse_diff.mean(dim=1)

        return {
            "left_vec": left_vec,
            "right_vec": right_vec,
            "direct_vec": direct_vec,
            "reverse_vec": reverse_vec,
            "direct_dist": direct_dist,
            "reverse_dist": reverse_dist,
            "reverse_gain": reverse_gain,
        }

    def forward(self, left_image, right_image, chr_idx=None):
        left_feat = self.feature_proj(self.encoder(left_image))
        right_feat = self.feature_proj(self.encoder(right_image))

        seq_left = self._feature_map_to_sequence(left_feat)
        seq_right = self._feature_map_to_sequence(right_feat)
        summary = self._sequence_summary(seq_left, seq_right)

        fused = [
            summary["direct_vec"],
            summary["reverse_vec"],
            torch.abs(summary["left_vec"] - summary["right_vec"]),
            summary["left_vec"] * summary["right_vec"],
            torch.stack(
                [
                    summary["direct_dist"],
                    summary["reverse_dist"],
                    summary["reverse_gain"],
                ],
                dim=1,
            ),
        ]

        if self.use_chromosome_id:
            if chr_idx is None:
                raise ValueError("chr_idx is required when use_chromosome_id=True")
            fused.append(self.chr_embedding(chr_idx))

        embedding = self.embedding_head(torch.cat(fused, dim=1))
        logits = self.classifier(embedding)

        return {
            "logits": logits,
            "embedding": embedding,
            "pair_distance": summary["direct_dist"],
            "direct_distance": summary["direct_dist"],
            "reverse_distance": summary["reverse_dist"],
            "reverse_gain": summary["reverse_gain"],
        }
