import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiamesePairClassifier(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        num_classes=2,
        pretrained=False,
        hidden_dim=256,
        dropout=0.3,
        use_chromosome_id=False,
        num_chromosome_types=None,
        chr_embed_dim=16,
        use_side_head=False,
        num_side_classes=2,
    ):
        super().__init__()

        self.use_chromosome_id = use_chromosome_id
        self.use_side_head = use_side_head

        if backbone_name == "resnet18":
            if pretrained:
                try:
                    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                except Exception:
                    print("[Warning] Failed to load pretrained ResNet18 weights, using random init.")
                    backbone = models.resnet18(weights=None)
            else:
                backbone = models.resnet18(weights=None)

            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.encoder = backbone

        elif backbone_name == "resnet50":
            if pretrained:
                try:
                    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                except Exception:
                    print("[Warning] Failed to load pretrained ResNet50 weights, using random init.")
                    backbone = models.resnet50(weights=None)
            else:
                backbone = models.resnet50(weights=None)

            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.encoder = backbone

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if self.use_chromosome_id:
            if num_chromosome_types is None:
                raise ValueError("num_chromosome_types must be provided when use_chromosome_id=True")
            self.chr_embedding = nn.Embedding(num_chromosome_types, chr_embed_dim)
        else:
            chr_embed_dim = 0

        pair_feat_dim = feat_dim * 4 + chr_embed_dim

        self.embedding_head = nn.Sequential(
            nn.Linear(pair_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.side_classifier = nn.Linear(hidden_dim, num_side_classes) if use_side_head else None
        self.embedding_dim = hidden_dim

    def forward(self, left_image, right_image, chr_idx=None):
        f_left = self.encoder(left_image)
        f_right = self.encoder(right_image)
        f_diff = torch.abs(f_left - f_right)
        f_mul = f_left * f_right

        feats = [f_left, f_right, f_diff, f_mul]

        if self.use_chromosome_id:
            if chr_idx is None:
                raise ValueError("chr_idx is required when use_chromosome_id=True")
            chr_feat = self.chr_embedding(chr_idx)
            feats.append(chr_feat)

        feat = torch.cat(feats, dim=1)
        embedding = self.embedding_head(feat)
        logits = self.classifier(embedding)
        pair_distance = 1.0 - F.cosine_similarity(
            F.normalize(f_left, dim=1),
            F.normalize(f_right, dim=1),
            dim=1,
        )
        output = {
            "logits": logits,
            "embedding": embedding,
            "pair_distance": pair_distance,
            "left_feature": f_left,
            "right_feature": f_right,
        }
        if self.side_classifier is not None:
            output["side_logits"] = self.side_classifier(embedding)
        return output
