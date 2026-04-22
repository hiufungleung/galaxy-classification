# src/data/split.py
#
# Expands train/val to all 214k labeled rows while keeping the existing test set fixed.
# Run once; all models reuse split_test.csv so test comparisons are valid.
#
# Usage:
#   uv run python -m src.data.split

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TABLES = Path("input/tables")

LABELED_INDEX    = TABLES / "labeled_index.csv"
SPLIT_TEST       = TABLES / "split_test.csv"
SPLIT_TRAIN_FULL = TABLES / "split_train_full.csv"
SPLIT_VAL_FULL   = TABLES / "split_val_full.csv"


def make_full_split(seed: int = 42) -> None:
    """
    Expand train/val to all 214k labeled rows while keeping the existing test set fixed.

    Strategy:
      1. Load labeled_index.csv (all labeled galaxies, including those without spectra)
      2. Remove rows whose objid appears in the existing split_test.csv
      3. Stratified 85/15 split of the remainder → train_full / val_full
      4. Save split_train_full.csv and split_val_full.csv
         (split_test.csv is unchanged — same test set for all models)
    """
    labeled = pd.read_csv(LABELED_INDEX)
    test_df  = pd.read_csv(SPLIT_TEST)
    print(f"Labeled index : {len(labeled):,} rows")
    print(f"Existing test : {len(test_df):,} rows (kept fixed)")

    # Add has_image / has_spectrum flags from working_set (rows that have both).
    # Rows in labeled_index but NOT in working_set lack a spectrum; images are assumed
    # present (94% download rate for all 507k rows).
    working = pd.read_csv(TABLES / "working_set.csv", usecols=["objid", "has_image", "has_spectrum"])
    labeled = labeled.merge(working[["objid", "has_image", "has_spectrum"]], on="objid", how="left")
    labeled["has_image"]    = labeled["has_image"].fillna(True).astype(bool)
    labeled["has_spectrum"] = labeled["has_spectrum"].fillna(False).astype(bool)
    labeled["has_both"]     = labeled["has_image"] & labeled["has_spectrum"]

    test_ids = set(test_df["objid"])
    remainder = labeled[~labeled["objid"].isin(test_ids)].reset_index(drop=True)
    print(f"Remainder for train+val: {len(remainder):,} rows")

    train_full, val_full = train_test_split(
        remainder,
        test_size=0.15,
        stratify=remainder["label"],
        random_state=seed,
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    train_full.to_csv(SPLIT_TRAIN_FULL, index=False)
    val_full.to_csv(SPLIT_VAL_FULL,     index=False)

    print(f"\nFull-data split saved:")
    print(f"  train_full : {len(train_full):,}  -> {SPLIT_TRAIN_FULL}")
    print(f"  val_full   : {len(val_full):,}  -> {SPLIT_VAL_FULL}")
    print(f"  test       : {len(test_df):,}  -> {SPLIT_TEST}  (unchanged)")
    for name, part in [("train_full", train_full), ("val_full", val_full)]:
        counts = part["label"].value_counts().to_dict()
        print(f"  {name} class counts: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Create full-data train/val split (test set fixed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_full_split(seed=args.seed)


if __name__ == "__main__":
    main()
