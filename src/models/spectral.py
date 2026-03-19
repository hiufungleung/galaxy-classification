# src/models/spectral.py
#
# 1D-CNN for SDSS spectra.
# Input: (batch, 1, 2700) — single-channel 1D signal
#
# Architecture:
#   Conv1d(1,  32, k=15) -> BN -> ReLU -> MaxPool1d(4)  # 2700 -> 675
#   Conv1d(32, 64, k=9)  -> BN -> ReLU -> MaxPool1d(4)  # 675  -> 168
#   Conv1d(64,128, k=5)  -> BN -> ReLU -> MaxPool1d(4)  # 168  -> 42
#   GlobalAvgPool1d -> Linear(128, 3)
#
# Kernel sizes decrease with depth (coarse -> fine features).
# GlobalAvgPool instead of Flatten makes the encoder input-length agnostic.

import torch
import torch.nn as nn


NUM_CLASSES  = 3
ENCODER_DIM  = 128


class CNN1DEncoder(nn.Module):
    """
    1D-CNN encoder. Returns 128-d feature vector per spectrum.
    Used standalone in spectral baseline and as encoder branch in fusion.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,   32, kernel_size=15, padding=7), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32,  64, kernel_size=9,  padding=4), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5,  padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool -> (batch, 128, 1)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return x.squeeze(-1)  # (batch, 128)


class SpectralClassifier(nn.Module):
    """Full spectral classifier: CNN1D encoder + classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = CNN1DEncoder()
        self.head    = nn.Linear(ENCODER_DIM, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))
