import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """
    输出最后一层卷积特征图，而不是 global pooled feature
    ResNet18 输入 300x300 时，大致输出 [B, 512, 10, 10]
    """
    def __init__(self, backbone_name="resnet18", pretrained=False):
        super().__init__()

        if backbone_name == "resnet18":
            if pretrained:
                try:
                    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                except Exception:
                    print("[Warning] Failed to load pretrained ResNet18 weights, using random init.")
                    backbone = models.resnet18(weights=None)
            else:
                backbone = models.resnet18(weights=None)

            self.out_channels = 512

        elif backbone_name == "resnet50":
            if pretrained:
                try:
                    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                except Exception:
                    print("[Warning] Failed to load pretrained ResNet50 weights, using random init.")
                    backbone = models.resnet50(weights=None)
            else:
                backbone = models.resnet50(weights=None)

            self.out_channels = 2048

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 去掉 avgpool 和 fc，只保留卷积主干
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, C, H, W]


class LocalPairComparator(nn.Module):
    """
    Local Pair Comparator v1

    输入:
        left_image:  [B, 3, H, W]
        right_image: [B, 3, H, W]

    输出:
        logits: [B, num_classes]

    核心:
        F_L, F_R -> local interactions -> conv aggregation -> GAP -> classifier
    """
    def __init__(
        self,
        backbone_name="resnet18",
        num_classes=2,
        pretrained=False,
        hidden_channels=256,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = ResNetFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        c = self.encoder.out_channels

        # 局部交互:
        # [F_L, F_R, |F_L-F_R|, F_L*F_R] => 4C channels
        in_channels = c * 4

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # 可选的局部异常注意图，帮助模型学会“哪里不同”
        self.attention_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, left_image, right_image, return_attention=False):
        f_left = self.encoder(left_image)    # [B, C, H, W]
        f_right = self.encoder(right_image)  # [B, C, H, W]

        f_diff = torch.abs(f_left - f_right)
        f_mul = f_left * f_right

        feat = torch.cat([f_left, f_right, f_diff, f_mul], dim=1)
        feat = self.fusion(feat)

        attention = self.attention_head(feat)   # [B, 1, H, W]
        feat = feat * attention

        pooled = self.pool(feat).flatten(1)     # [B, hidden_channels]
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attention
        return logits