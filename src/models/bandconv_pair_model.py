import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 3), stride=(1, 1), dropout=0.0):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class BandConvPairClassifier(nn.Module):
    """
    Lightweight pair classifier on top of band representation v2.

    Inputs
    - left_band: [B, 1, L, W]
    - right_band: [B, 1, L, W]
    - left_profile: [B, 1, L]
    - right_profile: [B, 1, L]
    - left_width: [B, 1, L]
    - right_width: [B, 1, L]
    """

    def __init__(self, hidden_dim=256, dropout=0.2, num_classes=2):
        super().__init__()

        self.band_encoder = nn.Sequential(
            ConvBlock2D(1, 16, kernel_size=(7, 3), stride=(1, 1), dropout=dropout),
            ConvBlock2D(16, 32, kernel_size=(5, 3), stride=(2, 1), dropout=dropout),
            ConvBlock2D(32, 64, kernel_size=(5, 3), stride=(2, 2), dropout=dropout),
            ConvBlock2D(64, 96, kernel_size=(3, 3), stride=(2, 2), dropout=dropout),
        )
        self.band_pool = nn.AdaptiveAvgPool2d((4, 2))
        self.band_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.seq_encoder = nn.Sequential(
            ConvBlock1D(2, 32, kernel_size=7, stride=1, dropout=dropout),
            ConvBlock1D(32, 64, kernel_size=5, stride=2, dropout=dropout),
            ConvBlock1D(64, 96, kernel_size=5, stride=2, dropout=dropout),
        )
        self.seq_pool = nn.AdaptiveAvgPool1d(8)
        self.seq_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 8, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # direct / reverse / delta are all concatenated explicitly, so the final
        # fusion width is much larger than a single pairwise block.
        fusion_dim = (128 * 9) + (96 * 9) + 7
        self.embedding_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.embedding_dim = hidden_dim

    def _encode_single_band(self, band):
        feat = self.band_encoder(band)
        feat = self.band_pool(feat)
        return self.band_proj(feat)

    def _encode_single_seq(self, profile, width):
        seq = torch.cat([profile, width], dim=1)
        feat = self.seq_encoder(seq)
        feat = self.seq_pool(feat)
        return self.seq_proj(feat)

    @staticmethod
    def _pair_stats(left_profile, right_profile, left_width, right_width):
        left_profile_z = F.normalize(left_profile.flatten(1), dim=1)
        right_profile_z = F.normalize(right_profile.flatten(1), dim=1)
        reverse_profile_z = F.normalize(torch.flip(right_profile, dims=[2]).flatten(1), dim=1)

        direct_corr = (left_profile_z * right_profile_z).sum(dim=1)
        reverse_corr = (left_profile_z * reverse_profile_z).sum(dim=1)
        width_direct_l1 = torch.mean(torch.abs(left_width - right_width), dim=(1, 2))
        width_reverse_l1 = torch.mean(torch.abs(left_width - torch.flip(right_width, dims=[2])), dim=(1, 2))
        profile_direct_l1 = torch.mean(torch.abs(left_profile - right_profile), dim=(1, 2))
        profile_reverse_l1 = torch.mean(torch.abs(left_profile - torch.flip(right_profile, dims=[2])), dim=(1, 2))
        reverse_gain = reverse_corr - direct_corr
        return torch.stack(
            [
                direct_corr,
                reverse_corr,
                reverse_gain,
                profile_direct_l1,
                profile_reverse_l1,
                width_direct_l1,
                width_reverse_l1,
            ],
            dim=1,
        )

    def forward(self, left_band, right_band, left_profile, right_profile, left_width, right_width):
        right_band_rev = torch.flip(right_band, dims=[2])
        right_profile_rev = torch.flip(right_profile, dims=[2])
        right_width_rev = torch.flip(right_width, dims=[2])

        left_band_vec = self._encode_single_band(left_band)
        right_band_vec = self._encode_single_band(right_band)
        right_band_rev_vec = self._encode_single_band(right_band_rev)

        left_seq_vec = self._encode_single_seq(left_profile, left_width)
        right_seq_vec = self._encode_single_seq(right_profile, right_width)
        right_seq_rev_vec = self._encode_single_seq(right_profile_rev, right_width_rev)

        direct_band = torch.cat(
            [
                left_band_vec,
                right_band_vec,
                torch.abs(left_band_vec - right_band_vec),
            ],
            dim=1,
        )
        reverse_band = torch.cat(
            [
                left_band_vec,
                right_band_rev_vec,
                torch.abs(left_band_vec - right_band_rev_vec),
            ],
            dim=1,
        )
        band_pair_vec = torch.cat(
            [
                direct_band,
                reverse_band,
                torch.abs(direct_band - reverse_band),
            ],
            dim=1,
        )

        direct_seq = torch.cat(
            [
                left_seq_vec,
                right_seq_vec,
                torch.abs(left_seq_vec - right_seq_vec),
            ],
            dim=1,
        )
        reverse_seq = torch.cat(
            [
                left_seq_vec,
                right_seq_rev_vec,
                torch.abs(left_seq_vec - right_seq_rev_vec),
            ],
            dim=1,
        )
        seq_pair_vec = torch.cat(
            [
                direct_seq,
                reverse_seq,
                torch.abs(direct_seq - reverse_seq),
            ],
            dim=1,
        )

        stats = self._pair_stats(left_profile, right_profile, left_width, right_width)
        fused = torch.cat([band_pair_vec, seq_pair_vec, stats], dim=1)
        embedding = self.embedding_head(fused)
        logits = self.classifier(embedding)

        return {
            "logits": logits,
            "embedding": embedding,
            "pair_stats": stats,
        }
