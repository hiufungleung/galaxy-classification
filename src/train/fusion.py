# src/train/fusion.py
#
# Late fusion training — requires GPU (~1 hr on T4).
# Loads pretrained image and spectral encoder weights before training.
#
# Usage:
#   uv run python -m src.train.fusion

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

from src.data.clean import engineer_features, get_feature_cols, handle_missing
from src.data.datasets.multimodal import MultiModalDataset
from src.data.labels import make_labels
from src.data.rename import apply_renames
from src.models.fusion import LateFusionModel
from src.utils.io import get_device, load_checkpoint, save_checkpoint
from src.utils.metrics import evaluate, print_results


TABLES      = Path("input/tables")
IMG_DIR     = Path("input/images")
CHECKPOINTS = Path("checkpoints/fusion")

IMG_CKPT  = Path("checkpoints/image/resnet18_best.pth")
SPEC_CKPT = Path("checkpoints/spectral/cnn1d_best.pth")

BATCH_SIZE   = 32
LR_PHASE1    = 1e-4   # head only
LR_PHASE2    = 1e-5   # all params, 10× lower
PHASE1_EPOCHS = 10
MAX_EPOCHS    = 50
PATIENCE      = 5
NUM_WORKERS   = 4


def class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, spec, tab, has_spec, labels in tqdm(loader, leave=False):
            img, spec, tab = img.to(device), spec.to(device), tab.to(device)
            has_spec, labels = has_spec.to(device), labels.to(device)

            logits = model(img, spec, tab, has_spec)
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
    parser = argparse.ArgumentParser(description="Train late fusion model")
    parser.add_argument("--train",      default=str(TABLES / "split_train.csv"))
    parser.add_argument("--val",        default=str(TABLES / "split_val.csv"))
    parser.add_argument("--test",       default=str(TABLES / "split_test.csv"))
    parser.add_argument("--data",       default=str(TABLES / "DATA7901_DR19_merged.csv"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Build feature table ───────────────────────────────────────────────
    print("Loading tabular features...")
    df = pd.read_csv(args.data, low_memory=False)
    df = apply_renames(df)
    df = make_labels(df)
    df = handle_missing(df)
    df = engineer_features(df)
    feature_cols = get_feature_cols(df)

    def load_split(path):
        split = pd.read_csv(path)
        return split.merge(df[["objid"] + feature_cols], on="objid", how="inner")

    train_df = load_split(args.train)
    val_df   = load_split(args.val)
    test_df  = load_split(args.test)

    # Filter to rows that have images
    def has_img(d): return d[d["objid"].apply(lambda x: (IMG_DIR / f"{x}.jpeg").exists())]
    train_df, val_df, test_df = has_img(train_df), has_img(val_df), has_img(test_df)

    train_ds = MultiModalDataset(train_df, IMG_DIR, feature_cols, train=True)
    val_ds   = MultiModalDataset(val_df,   IMG_DIR, feature_cols, train=False)
    test_ds  = MultiModalDataset(test_df,  IMG_DIR, feature_cols, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = LateFusionModel(n_tab_features=len(feature_cols)).to(device)

    # Load pretrained encoder weights
    load_checkpoint(model.image_encoder,    IMG_CKPT,  device)
    load_checkpoint(model.spectral_encoder, SPEC_CKPT, device)
    print("Loaded pretrained encoder weights.")

    weights   = class_weights(train_df["label_id"].values).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    best_f1, patience_counter = 0.0, 0

    # ── Phase 1: train head only ──────────────────────────────────────────
    print("\n── Phase 1: frozen encoders, training head only ──")
    model.freeze_encoders()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_res = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"  Phase1 epoch {epoch:2d}  train_loss={train_loss:.4f}  val_f1={val_res['macro_f1']:.4f}")

    # ── Phase 2: unfreeze all, joint fine-tuning ──────────────────────────
    print("\n── Phase 2: all params unfrozen, joint fine-tuning ──")
    model.unfreeze_encoders()
    optimizer = AdamW(model.parameters(), lr=LR_PHASE2)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_res = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        val_f1 = val_res["macro_f1"]
        print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, CHECKPOINTS / "late_fusion_best.pth", epoch=epoch, val_macro_f1=best_f1)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Test ──────────────────────────────────────────────────────────────
    load_checkpoint(model, CHECKPOINTS / "late_fusion_best.pth", device)
    _, test_results = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print_results(test_results, "Test (best checkpoint)")

    with open(CHECKPOINTS / "test_results.pkl", "wb") as f:
        pickle.dump(test_results, f)


if __name__ == "__main__":
    main()
