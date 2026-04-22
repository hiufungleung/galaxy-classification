# src/models/image.py

import torch.nn as nn
from torchvision import models


NUM_CLASSES = 3
ENCODER_DIM = 512  # ResNet-18 penultimate layer output size


def build_image_encoder(pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 with the final FC layer replaced by an identity.
    Returns 512-d feature vector per image.
    Used standalone in image baseline and as encoder branch in fusion.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    encoder = models.resnet18(weights=weights)
    encoder.fc = nn.Identity()
    return encoder


class ImageClassifier(nn.Module):
    """Full image classifier: ResNet-18 encoder + classification head."""

    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = build_image_encoder(pretrained)
        self.head    = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ENCODER_DIM, num_classes),
        ) if dropout > 0 else nn.Linear(ENCODER_DIM, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)
