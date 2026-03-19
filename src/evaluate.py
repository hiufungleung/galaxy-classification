# src/evaluate.py
#
# Loads all four saved model checkpoints, runs on the identical test split,
# and produces a summary comparison table.
#
# Usage:
#   uv run python -m src.evaluate

import pickle
from pathlib import Path

import pandas as pd

from src.utils.metrics import CLASSES, print_results


CHECKPOINTS = Path("checkpoints")

MODEL_RESULTS = {
    "tabular":  CHECKPOINTS / "tabular/test_results.pkl",
    "image":    CHECKPOINTS / "image/test_results.pkl",
    "spectral": CHECKPOINTS / "spectral/test_results.pkl",
    "fusion":   CHECKPOINTS / "fusion/test_results.pkl",
}


def load_results(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_summary(all_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, results in all_results.items():
        if results is None:
            continue
        row = {
            "model":       model_name,
            "macro_f1":    results["macro_f1"],
            "weighted_f1": results["weighted_f1"],
            "accuracy":    results["accuracy"],
        }
        for cls in CLASSES:
            if cls in results["per_class"]:
                row[f"{cls}_f1"] = round(results["per_class"][cls]["f1-score"], 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def main():
    print("Loading test results...\n")
    all_results = {name: load_results(path) for name, path in MODEL_RESULTS.items()}

    for name, results in all_results.items():
        if results is not None:
            print_results(results, name)

    summary = build_summary(all_results)

    out = CHECKPOINTS / "results_summary.csv"
    summary.to_csv(out)

    print("\n── Summary ─────────────────────────────────────────────")
    print(summary.to_string())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
