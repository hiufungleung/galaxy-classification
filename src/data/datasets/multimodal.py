# src/data/datasets/multimodal.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.datasets.image import get_train_transform, get_val_transform
from src.data.datasets.spectral import N_BINS, load_spectrum
from src.data.labels import LABEL_MAP


class MultiModalDataset(Dataset):
    """
    PyTorch Dataset returning all three modalities for a given galaxy.

    Returns
    -------
    img_tensor  : (3, 224, 224) float32
    spec_tensor : (1, 2700)     float32  — zeros if spectrum missing
    tab_tensor  : (n_features,) float32
    has_spec    : float scalar  — 1.0 if spectrum present, 0.0 otherwise
    label       : int

    Parameters
    ----------
    df          : DataFrame with objid, label, spec_path, has_spectrum, and all
                  tabular feature columns (output of clean.get_feature_cols)
    img_dir     : directory containing {objid}.jpeg files
    feature_cols: ordered list of tabular feature column names
    train       : if True, use augmentation transforms for images
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str | Path,
        feature_cols: list[str],
        train: bool = False,
    ):
        self.df           = df.reset_index(drop=True)
        self.img_dir      = Path(img_dir)
        self.feature_cols = feature_cols
        self.transform    = get_train_transform() if train else get_val_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = LABEL_MAP[row["label"]]

        # ── Image ──────────────────────────────────────────────────────────
        img_path = self.img_dir / f"{row['objid']}.jpeg"
        img      = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # ── Spectrum ───────────────────────────────────────────────────────
        has_spec = float(row.get("has_spectrum", False))
        if has_spec and Path(row["spec_path"]).exists():
            flux = load_spectrum(row["spec_path"])
        else:
            flux     = np.zeros(N_BINS, dtype=np.float32)
            has_spec = 0.0
        spec_tensor = torch.from_numpy(flux).unsqueeze(0)

        # ── Tabular ────────────────────────────────────────────────────────
        tab = row[self.feature_cols].values.astype(np.float32)
        tab_tensor = torch.from_numpy(tab)

        has_spec_tensor = torch.tensor(has_spec, dtype=torch.float32)

        return img_tensor, spec_tensor, tab_tensor, has_spec_tensor, label
