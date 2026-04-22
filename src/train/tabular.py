# src/train/tabular.py
#
# XGBoost tabular baseline — runs on CPU in minutes.
#
# Usage:
#   uv run python -m src.train.tabular                         # default (working set split)
#   uv run python -m src.train.tabular --full-data             # train on all 214k labeled rows
#   uv run python -m src.train.tabular --grid-search           # grid search then retrain best
#   uv run python -m src.train.tabular --full-data --grid-search  # both

import argparse
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.features import engineer_features, get_feature_cols, handle_missing
from src.data.labels import make_labels
from src.data.rename import apply_renames
from src.models.tabular import TabularPipeline, compute_class_weights
from src.utils.metrics import evaluate, print_results


TABLES      = Path("input/tables")
CHECKPOINTS = Path("checkpoints/tabular")

GRID = {
    "n_estimators":  [100, 300, 500],
    "max_depth":     [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
}


def load_split(split_csv: Path, feature_df: pd.DataFrame) -> tuple:
    """Join split index with feature dataframe on objid."""
    split = pd.read_csv(split_csv)[["objid", "label", "label_id"]]
    # Drop label columns from feature_df to avoid _x/_y suffix conflicts on merge
    feat = feature_df.drop(columns=["label", "label_id"], errors="ignore")
    merged = split.merge(feat, on="objid", how="inner")
    return merged


def run_grid_search(X_train, y_train, grid: dict, n_splits: int = 5) -> dict:
    """5-fold stratified CV grid search. Returns best params dict."""
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(product(*values))
    print(f"\nGrid search: {len(combos)} combinations × {n_splits}-fold CV")

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train)
    sw     = np.array([compute_class_weights(y_train)[yi] for yi in y_train])
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score, best_params = -1.0, {}
    results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        fold_f1s = []

        clf = XGBClassifier(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

        for train_idx, val_idx in skf.split(X_s, y_train):
            clf.fit(X_s[train_idx], y_train[train_idx],
                    sample_weight=sw[train_idx], verbose=False)
            res = evaluate(y_train[val_idx], clf.predict(X_s[val_idx]))
            fold_f1s.append(res["macro_f1"])

        mean_f1 = float(np.mean(fold_f1s))
        results.append({**params, "cv_macro_f1": round(mean_f1, 4)})
        print(f"  [{i+1:2d}/{len(combos)}] {params}  cv_f1={mean_f1:.4f}")

        if mean_f1 > best_score:
            best_score  = mean_f1
            best_params = params

    # Save grid search results
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values("cv_macro_f1", ascending=False).to_csv(
        CHECKPOINTS / "grid_search_results.csv", index=False
    )
    print(f"\nBest params (cv_f1={best_score:.4f}): {best_params}")
    print(f"Grid results saved: {CHECKPOINTS / 'grid_search_results.csv'}")
    return best_params


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost tabular baseline")
    parser.add_argument("--data",        default=str(TABLES / "DATA7901_DR19_merged.csv"))
    parser.add_argument("--train",       default=str(TABLES / "split_train_full.csv"))
    parser.add_argument("--val",         default=str(TABLES / "split_val_full.csv"))
    parser.add_argument("--test",        default=str(TABLES / "split_test.csv"))
    parser.add_argument("--grid-search", action="store_true",
                        help="Run 5-fold CV grid search before final training")
    args = parser.parse_args()

    # ── Load and preprocess ──────────────────────────────────────────────
    print("Loading data...", flush=True)
    df = pd.read_csv(args.data, low_memory=False, on_bad_lines="skip")
    df = apply_renames(df)

    print("Labelling...", flush=True)
    df = make_labels(df)

    print("Cleaning + engineering features...", flush=True)
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

    # ── Grid search (optional) ───────────────────────────────────────────
    best_params = {}
    if args.grid_search:
        best_params = run_grid_search(X_train, y_train, GRID)

    # ── Train ────────────────────────────────────────────────────────────
    print("\nTraining XGBoost (best params)...", flush=True)
    pipeline = TabularPipeline(
        n_estimators  = best_params.get("n_estimators",  500),
        max_depth     = best_params.get("max_depth",     6),
        learning_rate = best_params.get("learning_rate", 0.05),
    )
    pipeline.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ─────────────────────────────────────────────────────────
    val_results  = evaluate(y_val,  pipeline.predict(X_val))
    test_results = evaluate(y_test, pipeline.predict(X_test))

    print_results(val_results,  "Validation")
    print_results(test_results, "Test")

    # ── Save ─────────────────────────────────────────────────────────────
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    suffix = "_full" if args.full_data else ""
    model_path = CHECKPOINTS / f"xgb_model{suffix}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "feature_cols": feature_cols,
                     "best_params": best_params}, f)
    print(f"\nModel saved: {model_path}")

    results_path = CHECKPOINTS / f"test_results{suffix}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(test_results, f)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
