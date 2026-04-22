# src/train/spectral.py
#
# 1D-CNN spectral baseline — GPU helpful (~30 min on T4).
#
# Usage:
#   uv run python -m src.train.spectral

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.spectral import SpectralDataset
from src.models.spectral import SpectralClassifier
from src.utils.io import get_device, load_checkpoint, save_checkpoint
from src.utils.metrics import evaluate, print_results


TABLES      = Path("input/tables")
CHECKPOINTS = Path("checkpoints/spectral")

BATCH_SIZE  = 1024
LR          = 1e-4
MAX_EPOCHS  = 50
PATIENCE    = 5
NUM_WORKERS = 0
SPEC_DIR    = Path("input/spectra")

CONFIGS = {
    "A": dict(lr=1e-4, base_filters=32, dropout=0.0),   # baseline
    "B": dict(lr=1e-3, base_filters=32, dropout=0.0),   # higher LR
    "C": dict(lr=1e-4, base_filters=64, dropout=0.0),   # wider filters (64/128/256)
    "D": dict(lr=1e-4, base_filters=32, dropout=0.3),   # add dropout
}


def class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    results  = evaluate(np.array(all_labels), np.array(all_preds))
    return avg_loss, results


def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN spectral baseline")
    parser.add_argument("--train",       default=str(TABLES / "split_train.csv"))
    parser.add_argument("--val",         default=str(TABLES / "split_val.csv"))
    parser.add_argument("--test",        default=str(TABLES / "split_test.csv"))
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",      type=int,   default=MAX_EPOCHS)
    parser.add_argument("--patience",    type=int,   default=PATIENCE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--base-filters",type=int,   default=32)
    parser.add_argument("--dropout",     type=float, default=0.0)
    parser.add_argument("--config",      default=None, choices=list(CONFIGS.keys()),
                        help="Named hyperparam config (A/B/C/D)")
    args = parser.parse_args()

    if args.config:
        cfg = CONFIGS[args.config]
        args.lr           = cfg["lr"]
        args.base_filters = cfg["base_filters"]
        args.dropout      = cfg["dropout"]
        print(f"Config {args.config}: {cfg}")

    device = get_device()
    print(f"Device: {device}")

    train_df = pd.read_csv(args.train)
    val_df   = pd.read_csv(args.val)
    test_df  = pd.read_csv(args.test)

    def prepare(df):
        df = df[df["has_spectrum"]].reset_index(drop=True)
        df["spec_path"] = df.apply(
            lambda r: str(SPEC_DIR / f"spec-{int(r.plate)}-{int(r.spec_mjd)}-{int(r.fiberid):04d}.fits"),
            axis=1,
        )
        return df

    train_df = prepare(train_df)
    val_df   = prepare(val_df)
    test_df  = prepare(test_df)
    print(f"Spectral rows — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    train_ds = SpectralDataset(train_df)
    val_ds   = SpectralDataset(val_df)
    test_ds  = SpectralDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = SpectralClassifier(base_filters=args.base_filters, dropout=args.dropout).to(device)
    print(f"Model: base_filters={args.base_filters}  lr={args.lr}  dropout={args.dropout}")

    weights   = class_weights(train_df["label_id"].values).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    suffix       = f"_config{args.config}" if args.config else ""
    ckpt_path    = CHECKPOINTS / f"cnn1d{suffix}_best.pth"
    results_path = CHECKPOINTS / f"test_results{suffix}.pkl"
    best_f1, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_res = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_res   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        val_f1 = val_res["macro_f1"]
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, ckpt_path, epoch=epoch, val_macro_f1=best_f1)
            print(f"Checkpoint saved: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    load_checkpoint(model, ckpt_path, device)
    _, test_results = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print_results(test_results, f"Test — config {args.config or 'default'}")

    with open(results_path, "wb") as f:
        pickle.dump(test_results, f)


if __name__ == "__main__":
    main()
