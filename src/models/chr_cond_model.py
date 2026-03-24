import torch
import torch.nn as nn
from torchvision import models


class ChromosomeConditionalClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = False,
        num_chromosome_types: int = 25,
        chr_embed_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone_name = backbone_name

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
            self.backbone = backbone

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
            self.backbone = backbone

        elif backbone_name == "densenet121":
            if pretrained:
                try:
                    backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                except Exception:
                    print("[Warning] Failed to load pretrained DenseNet121 weights, using random init.")
                    backbone = models.densenet121(weights=None)
            else:
                backbone = models.densenet121(weights=None)

            feat_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        else:
            raise ValueError(
                f"Unsupported backbone for offline conditional model: {backbone_name}. "
                f"Currently supported: resnet18, resnet50, densenet121"
            )

        self.chr_embedding = nn.Embedding(
            num_embeddings=num_chromosome_types,
            embedding_dim=chr_embed_dim
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim + chr_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.embedding_dim = 256

    def forward(self, images, chr_idx):
        img_feat = self.backbone(images)
        chr_feat = self.chr_embedding(chr_idx)
        feat = torch.cat([img_feat, chr_feat], dim=1)
        embedding = self.embedding_head(feat)
        logits = self.classifier(embedding)
        return {
            "logits": logits,
            "embedding": embedding,
        }
