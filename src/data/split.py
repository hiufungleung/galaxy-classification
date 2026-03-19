# src/data/split.py
#
# Creates a fixed stratified 70/15/15 train/val/test split from working_set.csv.
# Run once; all four models reuse the same split so test comparisons are valid.
#
# Usage:
#   uv run python -m src.data.split

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TABLES = Path("input/tables")

WORKING_SET   = TABLES / "working_set.csv"
SPLIT_TRAIN   = TABLES / "split_train.csv"
SPLIT_VAL     = TABLES / "split_val.csv"
SPLIT_TEST    = TABLES / "split_test.csv"


def make_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split on 'label' column.

    Returns (train_df, val_df, test_df).
    Split is stratified so class proportions are preserved across all three sets.
    random_state=42 is fixed — do not change after the split is created.
    """
    # First split: train vs (val + test)
    val_test_size = val_size + test_size
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # Second split: val vs test (relative size within val+test)
    relative_test_size = test_size / val_test_size
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test_size,
        stratify=val_test_df["label"],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Create stratified train/val/test split")
    parser.add_argument("--input", default=str(WORKING_SET))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    train_df, val_df, test_df = make_split(
        df, args.val_size, args.test_size, args.seed
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLIT_TRAIN, index=False)
    val_df.to_csv(SPLIT_VAL,   index=False)
    test_df.to_csv(SPLIT_TEST,  index=False)

    print(f"Split saved:")
    print(f"  train : {len(train_df):,}  -> {SPLIT_TRAIN}")
    print(f"  val   : {len(val_df):,}  -> {SPLIT_VAL}")
    print(f"  test  : {len(test_df):,}  -> {SPLIT_TEST}")

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = part["label"].value_counts()
        print(f"  {name} class counts: {counts.to_dict()}")


if __name__ == "__main__":
    main()
