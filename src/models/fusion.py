# src/models/fusion.py
#
# Late fusion model: concatenates encoded vectors from all three modalities
# and passes them through an MLP classification head.
#
# Fusion vector:
#   ResNet-18 image encoder  -> 512-d
#   CNN1D spectral encoder   -> 128-d
#   Tabular MLP encoder      ->  64-d
#   has_spectrum flag        ->   1-d
#                               ------
#                               705-d  -> Linear(705, 256) -> ReLU -> Dropout -> Linear(256, 3)
#
# Training procedure (see src/train/fusion.py):
#   Phase 1: freeze image + spectral encoders, train tabular encoder + head only (10 epochs)
#   Phase 2: unfreeze all, joint training at LR=1e-5

import torch
import torch.nn as nn

from src.models.image import ENCODER_DIM as IMG_DIM
from src.models.image import build_image_encoder
from src.models.spectral import CNN1DEncoder
from src.models.spectral import ENCODER_DIM as SPEC_DIM


TAB_HIDDEN  = 64
FUSION_DIM  = IMG_DIM + SPEC_DIM + TAB_HIDDEN + 1  # 705
NUM_CLASSES = 3


class TabularEncoder(nn.Module):
    """Small MLP to project tabular features into a fixed-size embedding."""

    def __init__(self, n_features: int, hidden: int = TAB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, hidden),     nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class LateFusionModel(nn.Module):
    """
    Late fusion classifier.

    Parameters
    ----------
    n_tab_features : number of tabular input features (from clean.get_feature_cols)
    num_classes    : 3 (elliptical, spiral, merger)
    """

    def __init__(self, n_tab_features: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.image_encoder   = build_image_encoder(pretrained=True)
        self.spectral_encoder = CNN1DEncoder()
        self.tabular_encoder  = TabularEncoder(n_tab_features)

        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, spec, tab, has_spec):
        f_img  = self.image_encoder(img)                      # (B, 512)
        f_spec = self.spectral_encoder(spec)                  # (B, 128)
        f_tab  = self.tabular_encoder(tab)                    # (B, 64)
        hs     = has_spec.unsqueeze(1)                        # (B, 1)

        fused = torch.cat([f_img, f_spec, f_tab, hs], dim=1)  # (B, 705)
        return self.head(fused)

    def forward_from_feats(self, img_feat, spec_feat, tab, has_spec):
        """Phase 1 fast path: use pre-extracted encoder features, skip image/spectral encoders."""
        f_tab = self.tabular_encoder(tab)                     # (B, 64)
        hs    = has_spec.unsqueeze(1)                         # (B, 1)
        fused = torch.cat([img_feat, spec_feat, f_tab, hs], dim=1)  # (B, 705)
        return self.head(fused)

    def freeze_encoders(self):
        """Phase 1: freeze pretrained image and spectral encoders."""
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.spectral_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        """Phase 2: unfreeze all parameters for joint fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True
