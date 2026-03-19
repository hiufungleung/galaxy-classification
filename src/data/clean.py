# src/data/clean.py
#
# Feature engineering and missing value handling for the tabular pipeline.
#
# IMPORTANT — columns that must NEVER be used as ML features:
#   - p_el, p_cw, p_acw, p_edge, p_mg, p_cs, p_dk  (Galaxy Zoo labels / label source)
#   - nvote_tot, nvote_*                             (vote metadata)
#   - class, sdssPrimary, clean, zWarning            (filter columns)
#   - objid, ra, dec, specObjID, plate, spec_mjd, fiberid  (identifiers)

import numpy as np
import pandas as pd


# ── Columns to drop after loading ────────────────────────────────────────────

ID_COLS = [
    "objid", "ra", "dec", "specObjID", "plate", "spec_mjd", "fiberid",
]

FILTER_COLS = [
    "class", "sdssPrimary", "clean", "nvote_tot", "p_dk", "zWarning",
]

LABEL_COLS = [
    "p_el", "p_cw", "p_acw", "p_edge", "p_mg", "p_cs",
]

# Duplicate/renamed collision columns (all dropped)
DUP_COLS = [
    "spec_objid_dup", "spec_ra_dup", "spec_dec_dup",
    "spec_cx_dup", "spec_cy_dup", "spec_cz_dup",
    "zoo_objid_dup", "zoo_ra_dup", "zoo_dec_dup",
    "zoo_col12_dup", "zoo_col13_dup", "zoo_col14_dup",
]


# ── Raw ML feature columns (before derived features are added) ────────────────

TABULAR_FEATURES = [
    # Extinction-corrected model magnitudes (dered = modelMag - Schlegel dust map)
    "dered_u", "dered_g", "dered_r", "dered_i", "dered_z",

    # Petrosian magnitudes — robust total flux for extended sources
    "petroMag_u", "petroMag_g", "petroMag_r", "petroMag_i", "petroMag_z",
    "petroMagErr_u", "petroMagErr_g", "petroMagErr_r", "petroMagErr_i", "petroMagErr_z",

    # Petrosian radii — r-band only (reference band, highest S/N)
    "petroR50_r",    # half-light radius (arcsec)
    "petroR50Err_r",
    "petroR90_r",    # 90%-light radius (arcsec) — used to derive concentration
    "petroR90Err_r",

    # Profile weight — strongest single photometric feature
    # 1.0 = pure de Vaucouleurs (elliptical), 0.0 = pure exponential disc (spiral)
    "fracDeV_r", "fracDeV_g",

    # Profile fit log-likelihoods — r and g bands
    "lnLDeV_r", "lnLExp_r",
    "lnLDeV_g", "lnLExp_g",

    # Profile shape — r-band axis ratios and effective radii
    "deVAB_r",   # de Vaucouleurs axis ratio b/a
    "expAB_r",   # exponential disc axis ratio b/a
    "deVRad_r",  # de Vaucouleurs effective radius (arcsec)
    "expRad_r",  # exponential disc effective radius (arcsec)

    # Adaptive moment ellipticity — r-band
    "mE1_r", "mE2_r",  # shape asymmetry; merger indicator

    # Spectroscopic features
    "spec_z",       # spectroscopic redshift (renamed from Column8)
    "velDisp",      # stellar velocity dispersion (km/s)
    "velDispErr",   # uncertainty on velDisp
    "velDispChi2",  # chi2 of velDisp fit — flag unreliable if > 10

    # Eigenspectrum PCA coefficients (theta_0 tracks early/late-type)
    "theta_0", "theta_1", "theta_2", "theta_3", "theta_4",

    # Spectral quality
    "snMedian_r",  # r-band S/N per pixel
    "snMedian",    # overall S/N per pixel
    "wCoverage",   # spectral wavelength coverage completeness

    # Spectroscopic subclass (categorical — one-hot encoded in engineer_features)
    "subClass",
]

# Derived features added by engineer_features()
DERIVED_FEATURES = [
    "color_g_r", "color_r_i", "color_u_g",
    "concentration",
    "deV_exp_ratio",
    "velDisp_measured",
]

# One-hot columns produced from subClass
SUBCLASS_DUMMIES = ["subClass_STARFORMING", "subClass_AGN", "subClass_BROADLINE"]


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features. Must be called after handle_missing().

    Colour indices: red g-r (~0.7) -> ellipticals | blue g-r (~0.4) -> spirals
    Concentration:  high (~3.5-5) -> ellipticals  | low (~2-3.5)    -> spirals
    deV_exp_ratio:  > 0 -> bulge-dominated (elliptical)
    """
    df = df.copy()

    df["color_g_r"] = df["dered_g"] - df["dered_r"]
    df["color_r_i"] = df["dered_r"] - df["dered_i"]
    df["color_u_g"] = df["dered_u"] - df["dered_g"]

    df["concentration"] = df["petroR90_r"] / df["petroR50_r"].clip(lower=0.01)

    df["deV_exp_ratio"] = df["lnLDeV_r"] - df["lnLExp_r"]

    # One-hot encode subClass; fill unknown/null with all zeros
    if "subClass" in df.columns:
        dummies = pd.get_dummies(
            df["subClass"].fillna("UNKNOWN").str.strip().str.upper(),
            prefix="subClass",
        )
        for col in SUBCLASS_DUMMIES:
            df[col] = dummies[col].astype(float) if col in dummies.columns else 0.0
        df = df.drop(columns=["subClass"])

    return df


# ── Missing value handling ────────────────────────────────────────────────────

SENTINEL = -9999.0
SENTINEL_THRESHOLD = -100.0  # any value below this is treated as missing


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace SDSS sentinel values (-9999) with NaN, then impute.

    SDSS stores -9999.0 for failed measurements. Values < -100 are sentinels.
    This is common for u-band (faint, high noise) and blended objects.

    Strategy:
      - Drop rows missing r-band backbone (petroMag_r, dered_r, fracDeV_r, concentration)
      - Impute remaining NaNs with column median (fitted on the full labeled set here;
        in the train/val/test pipeline, fit on train split only — see src/train/tabular.py)
      - velDisp = 0 means not measured; zero-fill and add binary flag
    """
    df = df.copy()

    numeric_features = [c for c in TABULAR_FEATURES + DERIVED_FEATURES
                        if c in df.columns and df[c].dtype != object]

    # Replace sentinels with NaN
    for col in numeric_features:
        df[col] = df[col].where(df[col] > SENTINEL_THRESHOLD, other=np.nan)

    # Drop rows missing the r-band backbone
    backbone = ["petroMag_r", "dered_r", "fracDeV_r"]
    backbone_present = [c for c in backbone if c in df.columns]
    if "concentration" in df.columns:
        backbone_present.append("concentration")
    df = df.dropna(subset=backbone_present)

    # Impute remaining NaNs with column median
    for col in numeric_features:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # velDisp: 0 or sentinel means not measured
    if "velDisp" in df.columns:
        df["velDisp_measured"] = (
            (df["velDisp"] > 0) & (df["velDispErr"] > 0)
        ).astype(float)
        df["velDisp"]    = df["velDisp"].clip(lower=0)
        df["velDispErr"] = df["velDispErr"].clip(lower=0)

    return df


# ── Final feature list (after engineering) ───────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature columns present in df, ready for model input."""
    candidates = (
        [c for c in TABULAR_FEATURES if c != "subClass"] +
        DERIVED_FEATURES +
        SUBCLASS_DUMMIES
    )
    return [c for c in candidates if c in df.columns]
