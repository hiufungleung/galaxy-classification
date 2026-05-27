# src/data/prepare.py
#
# Stage 0: Generate labeled_index.csv, working_set.csv, and the initial
# train/val/test split from the raw merged CSV.
#
# This must run before split.py (which only expands train/val to full data).
#
# Usage:
#   uv run python -m src.data.prepare

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.rename import apply_renames
from src.data.labels import make_labels

TABLES  = Path("input/tables")
RAW_CSV = TABLES / "dr19.csv"
IMG_DIR = Path("input/images")
SPEC_DIR = Path("input/spectra")

LABELED_INDEX = TABLES / "labeled_index.csv"
WORKING_SET   = TABLES / "working_set.csv"
SPLIT_TRAIN   = TABLES / "split_train.csv"
SPLIT_VAL     = TABLES / "split_val.csv"
SPLIT_TEST    = TABLES / "split_test.csv"


def main(seed: int = 42) -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Follow the Data Acquisition steps in README.md "
            "to download the CSV from SDSS CasJobs."
        )

    # ── Load and rename ──────────────────────────────────────────────────────
    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, on_bad_lines="skip")
    df = apply_renames(df)
    print(f"  Raw rows: {len(df):,}")

    # ── Assign labels ────────────────────────────────────────────────────────
    labeled = make_labels(df)

    # ── Check image/spectrum availability ────────────────────────────────────
    labeled["has_image"] = labeled["objid"].apply(
        lambda x: os.path.exists(IMG_DIR / f"{x}.jpeg")
    )
    labeled["has_spectrum"] = labeled.apply(
        lambda r: os.path.exists(
            SPEC_DIR / f"spec-{int(r['plate']):04d}-{int(r['spec_mjd'])}-{int(r['fiberid']):04d}.fits"
        ),
        axis=1,
    )
    labeled["has_both"] = labeled["has_image"] & labeled["has_spectrum"]

    n_img  = labeled["has_image"].sum()
    n_spec = labeled["has_spectrum"].sum()
    n_both = labeled["has_both"].sum()
    print(f"\nData availability:")
    print(f"  has_image:    {n_img:,} / {len(labeled):,}  ({100*n_img/len(labeled):.1f}%)")
    print(f"  has_spectrum: {n_spec:,} / {len(labeled):,}  ({100*n_spec/len(labeled):.1f}%)")
    print(f"  has_both:     {n_both:,} / {len(labeled):,}  ({100*n_both/len(labeled):.1f}%)")

    # ── Save labeled_index.csv ───────────────────────────────────────────────
    TABLES.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(LABELED_INDEX, index=False)
    print(f"\nSaved {LABELED_INDEX}  ({len(labeled):,} rows)")

    # ── Save working_set.csv (rows with images) ─────────────────────────────
    working = labeled[labeled["has_image"]].copy()
    working.to_csv(WORKING_SET, index=False)
    print(f"Saved {WORKING_SET}  ({len(working):,} rows)")

    # ── Initial stratified 70/15/15 split ────────────────────────────────────
    if SPLIT_TEST.exists():
        print(f"\n{SPLIT_TEST} already exists — skipping initial split (use split.py for full-data expansion)")
        return

    train_val, test = train_test_split(
        working, test_size=0.15, stratify=working["label"], random_state=seed,
    )
    train, val = train_test_split(
        train_val, test_size=0.15/0.85, stratify=train_val["label"], random_state=seed,
    )

    train.to_csv(SPLIT_TRAIN, index=False)
    val.to_csv(SPLIT_VAL, index=False)
    test.to_csv(SPLIT_TEST, index=False)

    print(f"\nInitial split (70/15/15):")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        counts = part["label"].value_counts().to_dict()
        print(f"  {name:5s}: {len(part):,}  {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: prepare labeled_index, working_set, and initial split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
