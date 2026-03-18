import torch
import torch.nn as nn


class PairMixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, left_feat, right_feat):
        if (not self.training) or self.p <= 0.0:
            return left_feat, right_feat

        if torch.rand(1, device=left_feat.device).item() > self.p:
            return left_feat, right_feat

        if left_feat.size(0) < 2:
            return left_feat, right_feat

        left_mu = left_feat.mean(dim=(2, 3), keepdim=True)
        left_std = left_feat.var(dim=(2, 3), keepdim=True, unbiased=False).add(self.eps).sqrt()
        right_mu = right_feat.mean(dim=(2, 3), keepdim=True)
        right_std = right_feat.var(dim=(2, 3), keepdim=True, unbiased=False).add(self.eps).sqrt()

        # Use pair-level statistics so both sides of one chromosome pair share the same mixed style.
        pair_mu = 0.5 * (left_mu + right_mu)
        pair_std = 0.5 * (left_std + right_std)

        perm = torch.randperm(left_feat.size(0), device=left_feat.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((left_feat.size(0), 1, 1, 1)).to(left_feat.device)

        mixed_pair_mu = lam * pair_mu + (1.0 - lam) * pair_mu[perm]
        mixed_pair_std = lam * pair_std + (1.0 - lam) * pair_std[perm]

        left_norm = (left_feat - left_mu) / left_std
        right_norm = (right_feat - right_mu) / right_std

        mixed_left = left_norm * mixed_pair_std + mixed_pair_mu
        mixed_right = right_norm * mixed_pair_std + mixed_pair_mu
        return mixed_left, mixed_right
