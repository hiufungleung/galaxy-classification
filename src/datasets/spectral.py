# src/datasets/spectral.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.labels import LABEL_MAP

PRELOAD_WORKERS = 32  # parallel FITS readers; matches logical processor count


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


def preload_spectra(paths: list, has_spectrum: list) -> np.ndarray:
    """
    Load all spectra in parallel using a thread pool.
    Returns float32 array of shape (N, N_BINS).
    Threads work well here: FITS parsing releases the GIL during file I/O.
    """
    n = len(paths)
    out = np.zeros((n, N_BINS), dtype=np.float32)

    def _load(i):
        if has_spectrum[i]:
            return i, load_spectrum(paths[i])
        return i, None

    with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as ex:
        futures = {ex.submit(_load, i): i for i in range(n)}
        done = 0
        for f in as_completed(futures):
            i, flux = f.result()
            if flux is not None:
                out[i] = flux
            done += 1
            if done % 10000 == 0:
                print(f"    {done:,} / {n:,}", flush=True)

    return out


class SpectralDataset(Dataset):
    """
    PyTorch Dataset for SDSS 1D spectra.

    Pre-loads all spectra into a numpy array at init so __getitem__ is a
    pure array lookup — eliminates per-sample FITS I/O that starves the GPU.

    Parameters
    ----------
    df : DataFrame with columns 'label', 'spec_path', 'has_spectrum'
    """

    def __init__(self, df: pd.DataFrame):
        self.df     = df.reset_index(drop=True)
        self.labels = np.array([LABEL_MAP[l] for l in self.df["label"]], dtype=np.int64)

        print(f"  Pre-loading {len(self.df):,} spectra into RAM ({PRELOAD_WORKERS} threads)...", flush=True)
        paths        = self.df["spec_path"].tolist()
        has_spectrum = self.df["has_spectrum"].tolist()
        self.spectra = preload_spectra(paths, has_spectrum)
        print(f"  Done. Array size: {self.spectra.nbytes / 1e6:.0f} MB", flush=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch
        flux_tensor = torch.from_numpy(self.spectra[idx]).unsqueeze(0)
        return flux_tensor, int(self.labels[idx])
