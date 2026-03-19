# src/train/tabular.py
#
# XGBoost tabular baseline — runs on CPU in minutes.
#
# Usage:
#   uv run python -m src.train.tabular

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.data.clean import engineer_features, get_feature_cols, handle_missing
from src.data.labels import make_labels
from src.data.rename import apply_renames
from src.models.tabular import TabularPipeline
from src.utils.metrics import evaluate, print_results


TABLES      = Path("input/tables")
CHECKPOINTS = Path("checkpoints/tabular")


def load_split(split_csv: Path, feature_df: pd.DataFrame) -> tuple:
    """Join split index with feature dataframe on objid."""
    split = pd.read_csv(split_csv)[["objid", "label", "label_id"]]
    merged = split.merge(feature_df, on="objid", how="inner")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost tabular baseline")
    parser.add_argument("--data",  default=str(TABLES / "DATA7901_DR19_merged.csv"))
    parser.add_argument("--train", default=str(TABLES / "split_train.csv"))
    parser.add_argument("--val",   default=str(TABLES / "split_val.csv"))
    parser.add_argument("--test",  default=str(TABLES / "split_test.csv"))
    args = parser.parse_args()

    # ── Load and preprocess ──────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False)
    df = apply_renames(df)

    print("Labelling...")
    df = make_labels(df)

    print("Cleaning + engineering features...")
    df = handle_missing(df)
    df = engineer_features(df)

    feature_cols = get_feature_cols(df)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]} ...")

    # ── Load splits ──────────────────────────────────────────────────────
    train_df = load_split(Path(args.train), df)
    val_df   = load_split(Path(args.val),   df)
    test_df  = load_split(Path(args.test),  df)

    print(f"Split sizes — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    X_train, y_train = train_df[feature_cols].values, train_df["label_id"].values
    X_val,   y_val   = val_df[feature_cols].values,   val_df["label_id"].values
    X_test,  y_test  = test_df[feature_cols].values,  test_df["label_id"].values

    # ── Train ────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    pipeline = TabularPipeline()
    pipeline.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ─────────────────────────────────────────────────────────
    val_results  = evaluate(y_val,  pipeline.predict(X_val))
    test_results = evaluate(y_test, pipeline.predict(X_test))

    print_results(val_results,  "Validation")
    print_results(test_results, "Test")

    # ── Save ─────────────────────────────────────────────────────────────
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINTS / "xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "feature_cols": feature_cols}, f)
    print(f"\nModel saved: {model_path}")

    results_path = CHECKPOINTS / "test_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(test_results, f)


if __name__ == "__main__":
    main()
