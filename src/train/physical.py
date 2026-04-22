# src/train/physical.py
#
# Physical classification stretch task: star-forming vs passive galaxies.
#
# Labels from colour: g-r < 0.65 → star-forming (blue cloud)
#                     g-r >= 0.65 → passive (red sequence)
# This is a proxy for sSFR (specific star formation rate), not morphology.
#
# Hypothesis: spectral model should outperform tabular here, since spectra
# encode star-formation indicators (Hα emission, Dn4000 break) directly.
#
# Usage:
#   uv run python -m src.train.physical

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.features import engineer_features, get_feature_cols, handle_missing
from src.data.labels import make_labels
from src.data.rename import apply_renames
from src.datasets.spectral import preload_spectra, PRELOAD_WORKERS
from src.models.spectral import SpectralClassifier
from torch.utils.data import Dataset
import torch


class PhysicalSpectralDataset(Dataset):
    """SpectralDataset variant that uses phys_label_id (0/1) instead of LABEL_MAP."""

    def __init__(self, df):
        self.df     = df.reset_index(drop=True)
        self.labels = self.df["phys_label_id"].values.astype("int64")
        print(f"  Pre-loading {len(self.df):,} spectra into RAM ({PRELOAD_WORKERS} threads)...", flush=True)
        self.spectra = preload_spectra(
            self.df["spec_path"].tolist(),
            self.df["has_spectrum"].tolist(),
        )
        print(f"  Done. Array size: {self.spectra.nbytes / 1e6:.0f} MB", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        flux = torch.from_numpy(self.spectra[idx]).unsqueeze(0)
        return flux, int(self.labels[idx])
from src.utils.io import get_device, load_checkpoint, save_checkpoint

# Local 2-class evaluation (overrides the 3-class shared evaluate)
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PHYS_CLASSES = ["star_forming", "passive"]

def evaluate(y_true, y_pred):
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy    = float(np.mean(y_true == y_pred))
    report = classification_report(y_true, y_pred, target_names=PHYS_CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy":    round(accuracy, 4),
        "per_class":   {cls: report[cls] for cls in PHYS_CLASSES if cls in report},
        "confusion_matrix": cm,
    }

def print_results(results, model_name=""):
    header = f"── {model_name} " if model_name else "── "
    print(f"\n{header}{'─' * (50 - len(header))}")
    print(f"  Macro F1    : {results['macro_f1']:.4f}")
    print(f"  Weighted F1 : {results['weighted_f1']:.4f}")
    print(f"  Accuracy    : {results['accuracy']:.4f}\n")
    for cls, m in results["per_class"].items():
        print(f"  {cls:13s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}  n={int(m['support'])}")


TABLES      = Path("input/tables")
CHECKPOINTS = Path("checkpoints/physical")
SPEC_DIR    = Path("input/spectra")

COLOUR_THRESHOLD = 0.65   # g-r cut: blue cloud vs red sequence
BATCH_SIZE  = 1024
LR          = 1e-4
MAX_EPOCHS  = 30
PATIENCE    = 5
NUM_WORKERS = 0


def make_physical_labels(df: pd.DataFrame, threshold: float = COLOUR_THRESHOLD) -> pd.DataFrame:
    """
    Assign star-forming / passive label from g-r colour.
    Only rows that already passed morphological label filters are used.
    Returns df with 'phys_label' (str) and 'phys_label_id' (0/1).
    """
    df = df.copy()
    df["color_g_r"] = df["dered_g"] - df["dered_r"]
    df = df.dropna(subset=["color_g_r"])
    # Exclude ambiguous mid-green valley (optional — keep all for now)
    df["phys_label"]    = np.where(df["color_g_r"] < threshold, "star_forming", "passive")
    df["phys_label_id"] = np.where(df["color_g_r"] < threshold, 0, 1)
    return df


def class_weights_tensor(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for specs, labels in tqdm(loader, leave=False, desc="train" if train else "eval"):
            specs, labels = specs.to(device), labels.to(device)
            logits = model(specs)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    results  = evaluate(np.array(all_labels), np.array(all_preds))
    return avg_loss, results


def main():
    parser = argparse.ArgumentParser(description="Physical classification: star-forming vs passive")
    parser.add_argument("--data", default=str(TABLES / "DATA7901_DR19_merged.csv"))
    args = parser.parse_args()

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # ── Load and label ────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False, on_bad_lines="skip")
    df = apply_renames(df)
    df = make_labels(df)          # apply redshift/vote filters (same sample as morphology task)
    df = handle_missing(df)
    df = engineer_features(df)
    df = make_physical_labels(df)

    counts = df["phys_label"].value_counts()
    print(f"Physical label counts:\n{counts}")
    print(f"  star_forming fraction: {counts.get('star_forming', 0) / len(df):.1%}")

    feature_cols = get_feature_cols(df)

    # Fixed split matching morphology task test set
    test_df  = pd.read_csv(TABLES / "split_test.csv")
    test_ids = set(test_df["objid"])

    test_phys  = df[df["objid"].isin(test_ids)]
    remainder  = df[~df["objid"].isin(test_ids)]
    train_phys, val_phys = train_test_split(
        remainder, test_size=0.15, stratify=remainder["phys_label"], random_state=42
    )

    print(f"Split — train: {len(train_phys):,}  val: {len(val_phys):,}  test: {len(test_phys):,}")

    # ── Tabular baseline (XGBoost) ────────────────────────────────────────
    print("\n── Tabular (XGBoost) ──")
    X_tr = train_phys[feature_cols].values;  y_tr = train_phys["phys_label_id"].values
    X_va = val_phys[feature_cols].values;    y_va = val_phys["phys_label_id"].values
    X_te = test_phys[feature_cols].values;   y_te = test_phys["phys_label_id"].values

    # Remove color_g_r from features to avoid trivial classification
    # (it's derived from dered_g - dered_r which IS in features, but the label IS color_g_r)
    # Actually dered_g and dered_r are separate features — the model will trivially learn
    # g-r. Document this: tabular model has direct access to the label-defining feature.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.2,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    counts_arr = np.bincount(y_tr)
    sw = np.where(y_tr == 0, counts_arr[1] / counts_arr[0], 1.0)
    clf.fit(X_tr_s, y_tr, sample_weight=sw,
            eval_set=[(X_va_s, y_va)], verbose=False)

    tab_results = evaluate(y_te, clf.predict(X_te_s))
    print_results(tab_results, "Tabular (XGBoost) — Physical")
    with open(CHECKPOINTS / "tabular_results.pkl", "wb") as f:
        pickle.dump(tab_results, f)

    # ── Spectral baseline (1D-CNN) ────────────────────────────────────────
    print("\n── Spectral (1D-CNN, 2-class) ──")
    # Build spectral DataFrames with spec_path
    def prep_spectral(part_df):
        part = part_df[part_df["objid"].isin(
            pd.read_csv(TABLES / "working_set.csv")["objid"]
        )].copy()
        part["label_id"] = part["phys_label_id"]
        part["spec_path"] = part.apply(
            lambda r: str(SPEC_DIR / f"spec-{int(r.plate)}-{int(r.spec_mjd)}-{int(r.fiberid):04d}.fits"),
            axis=1,
        )
        part = part[part["spec_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        part["has_spectrum"] = True
        return part

    tr_spec = prep_spectral(train_phys)
    va_spec  = prep_spectral(val_phys)
    te_spec  = prep_spectral(test_phys)
    print(f"Spectral rows — train: {len(tr_spec):,}  val: {len(va_spec):,}  test: {len(te_spec):,}")

    if len(tr_spec) > 0:
        tr_ds = PhysicalSpectralDataset(tr_spec)
        va_ds = PhysicalSpectralDataset(va_spec)
        te_ds = PhysicalSpectralDataset(te_spec)

        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model     = SpectralClassifier(num_classes=2).to(device)
        weights   = class_weights_tensor(tr_spec["phys_label_id"].values).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = AdamW(model.parameters(), lr=LR)
        scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

        ckpt_path = CHECKPOINTS / "cnn1d_physical_best.pth"
        best_f1, patience_counter = 0.0, 0

        for epoch in range(1, MAX_EPOCHS + 1):
            tr_loss, tr_res = run_epoch(model, tr_ld, criterion, optimizer, device, train=True)
            va_loss, va_res = run_epoch(model, va_ld, criterion, optimizer, device, train=False)
            scheduler.step()
            val_f1 = va_res["macro_f1"]
            print(f"  Epoch {epoch:3d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_macro_f1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1; patience_counter = 0
                save_checkpoint(model, ckpt_path, epoch=epoch, val_macro_f1=best_f1)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        load_checkpoint(model, ckpt_path, device)
        _, spec_results = run_epoch(model, te_ld, criterion, optimizer, device, train=False)
        print_results(spec_results, "Spectral (1D-CNN) — Physical")
        with open(CHECKPOINTS / "spectral_results.pkl", "wb") as f:
            pickle.dump(spec_results, f)
    else:
        print("  No spectral data available — skipping spectral baseline.")
        spec_results = None

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Physical Classification Summary ──")
    print(f"  Tabular macro F1 : {tab_results['macro_f1']:.4f}")
    if spec_results:
        print(f"  Spectral macro F1: {spec_results['macro_f1']:.4f}")
        print(f"  Spectral advantage: {spec_results['macro_f1'] - tab_results['macro_f1']:+.4f}")
    print(f"\nNote: tabular model has access to dered_g and dered_r (raw inputs for the g-r colour")
    print(f"label). The label is therefore partially derivable from raw features. Document this in report.")


if __name__ == "__main__":
    main()
