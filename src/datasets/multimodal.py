# src/datasets/multimodal.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.image import get_train_transform, get_val_transform
from src.datasets.spectral import N_BINS, PRELOAD_WORKERS, load_spectrum, preload_spectra
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
                  tabular feature columns (output of features.get_feature_cols)
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

        # Pre-load all spectra into shared memory so DataLoader workers can
        # access without pickling the full array (required on Windows spawn).
        print(f"  Pre-loading {len(self.df):,} spectra into shared memory ({PRELOAD_WORKERS} threads)...", flush=True)
        paths    = self.df["spec_path"].tolist()
        has_list = self.df["has_spectrum"].tolist()
        spectra_np = preload_spectra(paths, has_list)

        # Convert to torch tensors in shared memory
        self.spectra  = torch.from_numpy(spectra_np).share_memory_()
        self.has_spec = torch.tensor(
            self.df["has_spectrum"].astype(float).values, dtype=torch.float32
        ).share_memory_()

        # Tabular features and labels as shared tensors too
        self.tab_data = torch.tensor(
            self.df[feature_cols].values.astype(np.float32)
        ).share_memory_()
        self.labels = torch.tensor(
            [LABEL_MAP[l] for l in self.df["label"]], dtype=torch.long
        ).share_memory_()

        print(f"  Done. {int(self.has_spec.sum()):,} spectra in shared memory.", flush=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # ── Image (loaded from disk by worker) ─────────────────────────────
        objid      = self.df.iloc[idx]["objid"]
        img_path   = self.img_dir / f"{objid}.jpeg"
        img        = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # ── Spectrum / tabular / label from shared memory (no pickling) ────
        spec_tensor     = self.spectra[idx].unsqueeze(0)
        tab_tensor      = self.tab_data[idx]
        has_spec_tensor = self.has_spec[idx]
        label           = int(self.labels[idx])

        return img_tensor, spec_tensor, tab_tensor, has_spec_tensor, label
