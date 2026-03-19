# src/data/datasets/spectral.py

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.labels import LABEL_MAP


# Resampling grid: 3800–9200 Å at 2 Å/pixel = 2700 bins
WAVE_MIN  = 3800.0
WAVE_MAX  = 9200.0
WAVE_STEP = 2.0
N_BINS    = int((WAVE_MAX - WAVE_MIN) / WAVE_STEP)  # 2700


def load_spectrum(fits_path: str | Path) -> np.ndarray:
    """
    Load and normalise an SDSS FITS spectrum to a fixed 2700-bin array.

    Tries HDU 1 BinTable (standard SDSS format) first, falls back to HDU 0
    image with COEFF0/COEFF1 header keywords.

    Returns
    -------
    flux : np.ndarray of shape (2700,), dtype float32
        Resampled, clipped, and median-normalised flux.
        Returns zeros if the file cannot be read.
    """
    from astropy.io import fits

    target_wave = np.arange(WAVE_MIN, WAVE_MAX, WAVE_STEP)

    try:
        with fits.open(fits_path) as hdul:
            # Standard SDSS lite format: HDU 1 is a BinTable with loglam + flux
            if len(hdul) > 1 and hdul[1].data is not None:
                data    = hdul[1].data
                loglam  = data["loglam"].astype(np.float64)
                flux    = data["flux"].astype(np.float64)
                wave    = 10.0 ** loglam
            else:
                # Fallback: HDU 0 image with COEFF0/COEFF1 log-wavelength keywords
                flux   = hdul[0].data.astype(np.float64).flatten()
                coeff0 = hdul[0].header["COEFF0"]
                coeff1 = hdul[0].header["COEFF1"]
                loglam = coeff0 + coeff1 * np.arange(len(flux))
                wave   = 10.0 ** loglam

        # Resample onto uniform grid via linear interpolation
        resampled = np.interp(target_wave, wave, flux, left=0.0, right=0.0)

        # Clip outliers at ±5σ
        median = np.median(resampled)
        std    = np.std(resampled)
        if std > 0:
            resampled = np.clip(resampled, median - 5 * std, median + 5 * std)

        # Normalise by median absolute flux (avoid division by zero)
        mad = np.median(np.abs(resampled))
        if mad > 0:
            resampled = resampled / mad

        return resampled.astype(np.float32)

    except Exception:
        return np.zeros(N_BINS, dtype=np.float32)


class SpectralDataset(Dataset):
    """
    PyTorch Dataset for SDSS 1D spectra.

    Parameters
    ----------
    df       : DataFrame with columns 'objid', 'label', 'spec_path', 'has_spectrum'
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch

        row   = self.df.iloc[idx]
        label = LABEL_MAP[row["label"]]

        if row.get("has_spectrum", False) and Path(row["spec_path"]).exists():
            flux = load_spectrum(row["spec_path"])
        else:
            flux = np.zeros(N_BINS, dtype=np.float32)

        # Shape: (1, N_BINS) — channel dimension for Conv1d
        flux_tensor = torch.from_numpy(flux).unsqueeze(0)
        return flux_tensor, label
