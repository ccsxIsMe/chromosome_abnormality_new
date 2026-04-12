import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.local_pair_comparator import ResNetFeatureExtractor
from src.utils.inversion_attributes import get_structure_label_dims


class CorrespondenceIntervalProfilePairClassifier(nn.Module):
    """
    P12-style pair model with an explicit 1D band-profile branch.

    The extra branch keeps the current correspondence-interval backbone but adds
    a more task-aligned inductive bias:
    - collapse width to an ordered long-axis profile
    - compare direct vs reversed homolog profiles
    - aggregate multi-scale 1D evidence before the final embedding head
    """

    def __init__(
        self,
        backbone_name="resnet18",
        num_classes=2,
        pretrained=False,
        proj_dim=192,
        hidden_dim=256,
        seq_layers=2,
        num_heads=4,
        dropout=0.3,
        use_chromosome_id=False,
        num_chromosome_types=None,
        chr_embed_dim=16,
        profile_hidden_dim=48,
        profile_kernel_sizes=(3, 5, 9),
    ):
        super().__init__()

        self.use_chromosome_id = use_chromosome_id
        self.encoder = ResNetFeatureExtractor(backbone_name=backbone_name, pretrained=pretrained)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )

        seq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(seq_encoder_layer, num_layers=seq_layers)

        self.token_fusion = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, proj_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )
        self.interval_attention = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim // 2),
            nn.GELU(),
            nn.Conv2d(proj_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.corr_pool = nn.AdaptiveAvgPool2d((1, 1))

        profile_in_channels = proj_dim * 6
        self.profile_reduce = nn.Sequential(
            nn.Conv1d(profile_in_channels, profile_hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(profile_hidden_dim),
            nn.GELU(),
        )
        self.profile_multiscale = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        profile_hidden_dim,
                        profile_hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(profile_hidden_dim),
                    nn.GELU(),
                )
                for kernel_size in profile_kernel_sizes
            ]
        )
        profile_out_channels = profile_hidden_dim * len(profile_kernel_sizes)
        self.profile_attention = nn.Sequential(
            nn.Conv1d(profile_out_channels, profile_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(profile_hidden_dim),
            nn.GELU(),
            nn.Conv1d(profile_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        if self.use_chromosome_id:
            if num_chromosome_types is None:
                raise ValueError("num_chromosome_types must be provided when use_chromosome_id=True")
            self.chr_embedding = nn.Embedding(num_chromosome_types, chr_embed_dim)
        else:
            chr_embed_dim = 0

        profile_stat_dim = 5
        fusion_dim = proj_dim * 4 + 7 + (profile_out_channels * 2) + profile_stat_dim + chr_embed_dim
        self.embedding_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

        structure_dims = get_structure_label_dims()
        self.structure_heads = nn.ModuleDict(
            {
                "pericentric_logits": nn.Linear(hidden_dim, structure_dims["pericentric"]),
                "bp1_arm_logits": nn.Linear(hidden_dim, structure_dims["bp_arm"]),
                "bp2_arm_logits": nn.Linear(hidden_dim, structure_dims["bp_arm"]),
                "bp1_major_logits": nn.Linear(hidden_dim, structure_dims["bp_major"]),
                "bp2_major_logits": nn.Linear(hidden_dim, structure_dims["bp_major"]),
            }
        )
        self.embedding_dim = hidden_dim

    def _to_sequence(self, feat_map):
        seq = feat_map.mean(dim=3).transpose(1, 2)
        return self.sequence_encoder(seq)

    def _build_profile_features(self, left_feat, right_feat):
        left_profile = left_feat.mean(dim=3)
        right_profile = right_feat.mean(dim=3)
        right_profile_rev = torch.flip(right_profile, dims=[2])

        direct_diff = torch.abs(left_profile - right_profile)
        reverse_diff = torch.abs(left_profile - right_profile_rev)
        reverse_advantage = F.relu(direct_diff - reverse_diff)

        profile_stack = torch.cat(
            [
                left_profile,
                right_profile,
                right_profile_rev,
                direct_diff,
                reverse_diff,
                reverse_advantage,
            ],
            dim=1,
        )
        profile_base = self.profile_reduce(profile_stack)
        profile_multiscale_feat = torch.cat([branch(profile_base) for branch in self.profile_multiscale], dim=1)
        profile_attn = self.profile_attention(profile_multiscale_feat)
        profile_weighted = profile_multiscale_feat * profile_attn
        profile_avg = profile_weighted.mean(dim=2)
        profile_max = profile_weighted.amax(dim=2)
        profile_vec = torch.cat([profile_avg, profile_max], dim=1)

        direct_l1 = direct_diff.mean(dim=(1, 2))
        reverse_l1 = reverse_diff.mean(dim=(1, 2))
        direct_profile_similarity = F.cosine_similarity(left_profile.flatten(1), right_profile.flatten(1), dim=1)
        reverse_profile_similarity = F.cosine_similarity(left_profile.flatten(1), right_profile_rev.flatten(1), dim=1)
        reverse_profile_gain = F.relu(reverse_profile_similarity - direct_profile_similarity)

        profile_stats = torch.stack(
            [
                direct_l1,
                reverse_l1,
                F.relu(direct_l1 - reverse_l1),
                direct_profile_similarity,
                reverse_profile_similarity,
            ],
            dim=1,
        )

        return {
            "profile_vec": profile_vec,
            "profile_stats": profile_stats,
            "profile_attention": profile_attn,
            "direct_profile_similarity": direct_profile_similarity,
            "reverse_profile_similarity": reverse_profile_similarity,
            "reverse_profile_gain": reverse_profile_gain,
        }

    def forward(self, left_image, right_image, chr_idx=None):
        left_feat = self.feature_proj(self.encoder(left_image))
        right_feat = self.feature_proj(self.encoder(right_image))

        left_global = F.adaptive_avg_pool2d(left_feat, output_size=1).flatten(1)
        right_global = F.adaptive_avg_pool2d(right_feat, output_size=1).flatten(1)
        global_diff = torch.abs(left_global - right_global)

        left_seq = self._to_sequence(left_feat)
        right_seq = self._to_sequence(right_feat)
        right_seq_rev = torch.flip(right_seq, dims=[1])

        left_seq_norm = F.normalize(left_seq, dim=-1)
        right_seq_norm = F.normalize(right_seq, dim=-1)
        right_seq_rev_norm = F.normalize(right_seq_rev, dim=-1)

        direct_corr = torch.matmul(left_seq_norm, right_seq_norm.transpose(1, 2))
        reverse_corr = torch.matmul(left_seq_norm, right_seq_rev_norm.transpose(1, 2))
        corr_delta = reverse_corr - direct_corr

        corr_feat = torch.stack([direct_corr, reverse_corr, corr_delta], dim=1)
        corr_feat = self.corr_encoder(corr_feat)
        interval_attn = self.interval_attention(corr_feat)
        corr_vec = self.corr_pool(corr_feat * interval_attn).flatten(1)

        direct_diag = torch.diagonal(direct_corr, dim1=1, dim2=2)
        reverse_diag = torch.diagonal(reverse_corr, dim1=1, dim2=2)
        direct_diag_mean = direct_diag.mean(dim=1)
        reverse_diag_mean = reverse_diag.mean(dim=1)
        reverse_gain = F.relu(reverse_diag_mean - direct_diag_mean)

        direct_token_diff = torch.abs(left_seq - right_seq)
        reverse_token_diff = torch.abs(left_seq - right_seq_rev)
        token_feat = torch.cat([left_seq, right_seq, direct_token_diff, reverse_token_diff], dim=-1)
        token_feat = self.token_fusion(token_feat)
        token_vec = token_feat.mean(dim=1)

        profile_output = self._build_profile_features(left_feat, right_feat)

        fused = [
            global_diff,
            corr_vec,
            token_vec,
            torch.abs(direct_token_diff.mean(dim=1) - reverse_token_diff.mean(dim=1)),
            torch.stack(
                [
                    1.0 - direct_diag_mean,
                    1.0 - reverse_diag_mean,
                    reverse_gain,
                    direct_corr.mean(dim=(1, 2)),
                    reverse_corr.mean(dim=(1, 2)),
                    direct_corr.amax(dim=(1, 2)),
                    reverse_corr.amax(dim=(1, 2)),
                ],
                dim=1,
            ),
            profile_output["profile_vec"],
            profile_output["profile_stats"],
        ]

        if self.use_chromosome_id:
            if chr_idx is None:
                raise ValueError("chr_idx is required when use_chromosome_id=True")
            fused.append(self.chr_embedding(chr_idx))

        embedding = self.embedding_head(torch.cat(fused, dim=1))
        logits = self.classifier(embedding)

        pair_distance = 0.5 * (
            (1.0 - F.cosine_similarity(F.normalize(left_global, dim=1), F.normalize(right_global, dim=1), dim=1))
            + (1.0 - direct_diag_mean)
        )

        structure_logits = {name: head(embedding) for name, head in self.structure_heads.items()}

        return {
            "logits": logits,
            "embedding": embedding,
            "pair_distance": pair_distance,
            "structure_logits": structure_logits,
            "direct_diag_similarity": direct_diag_mean,
            "reverse_diag_similarity": reverse_diag_mean,
            "pair_consistency_direct": direct_diag_mean,
            "pair_consistency_reverse": reverse_diag_mean,
            "reverse_gain": reverse_gain,
            "interval_attention": interval_attn,
            "profile_attention": profile_output["profile_attention"],
            "direct_profile_similarity": profile_output["direct_profile_similarity"],
            "reverse_profile_similarity": profile_output["reverse_profile_similarity"],
            "profile_consistency_direct": profile_output["direct_profile_similarity"],
            "profile_consistency_reverse": profile_output["reverse_profile_similarity"],
            "reverse_profile_gain": profile_output["reverse_profile_gain"],
        }
