import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.local_pair_comparator import ResNetFeatureExtractor
from src.models.pair_mixstyle import PairMixStyle


class LocalGlobalPairComparator(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        num_classes=2,
        pretrained=False,
        proj_dim=256,
        global_heads=4,
        global_dropout=0.1,
        hidden_dim=256,
        dropout=0.3,
        use_chromosome_id=False,
        num_chromosome_types=None,
        chr_embed_dim=16,
        use_pair_mixstyle=False,
        mixstyle_p=0.5,
        mixstyle_alpha=0.1,
    ):
        super().__init__()

        self.use_chromosome_id = use_chromosome_id
        self.use_pair_mixstyle = use_pair_mixstyle
        self.embedding_dim = hidden_dim

        self.encoder = ResNetFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        encoder_channels = self.encoder.out_channels
        self.pair_mixstyle = PairMixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_pair_mixstyle else None

        self.feature_proj = nn.Sequential(
            nn.Conv2d(encoder_channels, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        local_in_channels = proj_dim * 4
        self.local_fusion = nn.Sequential(
            nn.Conv2d(local_in_channels, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.local_attention = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.local_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.left_to_right_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=global_heads,
            dropout=global_dropout,
            batch_first=True,
        )
        self.right_to_left_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=global_heads,
            dropout=global_dropout,
            batch_first=True,
        )
        self.global_norm = nn.LayerNorm(proj_dim)
        self.global_fusion = nn.Sequential(
            nn.Linear(proj_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_chromosome_id:
            if num_chromosome_types is None:
                raise ValueError("num_chromosome_types must be provided when use_chromosome_id=True")
            self.chr_embedding = nn.Embedding(num_chromosome_types, chr_embed_dim)
        else:
            chr_embed_dim = 0

        pair_feat_dim = proj_dim + hidden_dim + chr_embed_dim
        self.embedding_head = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _pool_global_tokens(self, tokens_left, tokens_right):
        attended_lr, _ = self.left_to_right_attn(tokens_left, tokens_right, tokens_right)
        attended_rl, _ = self.right_to_left_attn(tokens_right, tokens_left, tokens_left)

        fused_left = self.global_norm(tokens_left + attended_lr).mean(dim=1)
        fused_right = self.global_norm(tokens_right + attended_rl).mean(dim=1)
        fused_diff = torch.abs(fused_left - fused_right)
        fused_mul = fused_left * fused_right
        fused_global = self.global_fusion(torch.cat([fused_left, fused_right, fused_diff, fused_mul], dim=1))
        return fused_left, fused_right, fused_global

    def forward(self, left_image, right_image, chr_idx=None, return_attention=False):
        left_feat = self.feature_proj(self.encoder(left_image))
        right_feat = self.feature_proj(self.encoder(right_image))

        if self.pair_mixstyle is not None:
            left_feat, right_feat = self.pair_mixstyle(left_feat, right_feat)

        local_diff = torch.abs(left_feat - right_feat)
        local_mul = left_feat * right_feat
        local_feat = torch.cat([left_feat, right_feat, local_diff, local_mul], dim=1)
        local_feat = self.local_fusion(local_feat)

        attention = self.local_attention(local_feat)
        local_vec = self.local_pool(local_feat * attention).flatten(1)
        left_local_vec = self.local_pool(left_feat).flatten(1)
        right_local_vec = self.local_pool(right_feat).flatten(1)

        left_tokens = left_feat.flatten(2).transpose(1, 2)
        right_tokens = right_feat.flatten(2).transpose(1, 2)
        left_global_vec, right_global_vec, global_vec = self._pool_global_tokens(left_tokens, right_tokens)

        local_distance = 1.0 - F.cosine_similarity(
            F.normalize(left_local_vec, dim=1),
            F.normalize(right_local_vec, dim=1),
            dim=1,
        )
        global_distance = 1.0 - F.cosine_similarity(
            F.normalize(left_global_vec, dim=1),
            F.normalize(right_global_vec, dim=1),
            dim=1,
        )
        pair_distance = 0.5 * (local_distance + global_distance)

        fused = [local_vec, global_vec]
        if self.use_chromosome_id:
            if chr_idx is None:
                raise ValueError("chr_idx is required when use_chromosome_id=True")
            fused.append(self.chr_embedding(chr_idx))

        embedding = self.embedding_head(torch.cat(fused, dim=1))
        logits = self.classifier(embedding)

        output = {
            "logits": logits,
            "embedding": embedding,
            "pair_distance": pair_distance,
            "local_distance": local_distance,
            "global_distance": global_distance,
        }
        if return_attention:
            output["attention"] = attention
        return output
