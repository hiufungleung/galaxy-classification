# src/utils/metrics.py

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASSES = ["elliptical", "spiral", "merger"]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Returns dict with:
      macro_f1, weighted_f1, accuracy
      per_class: {class_name: {precision, recall, f1, support}}
      confusion_matrix: 2D list
    """
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy    = np.mean(y_true == y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy":    round(accuracy, 4),
        "per_class":   {cls: report[cls] for cls in CLASSES if cls in report},
        "confusion_matrix": cm,
    }


def print_results(results: dict, model_name: str = "") -> None:
    header = f"── {model_name} " if model_name else "── "
    print(f"\n{header}{'─' * (50 - len(header))}")
    print(f"  Macro F1    : {results['macro_f1']:.4f}")
    print(f"  Weighted F1 : {results['weighted_f1']:.4f}")
    print(f"  Accuracy    : {results['accuracy']:.4f}")
    print()
    for cls, m in results["per_class"].items():
        print(f"  {cls:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}  n={int(m['support'])}")
    print()
    cm = np.array(results["confusion_matrix"])
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':12s}  " + "  ".join(f"{c[:4]:>6}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i]:12s}  " + "  ".join(f"{v:6d}" for v in row))
