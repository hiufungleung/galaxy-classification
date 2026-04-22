# src/train/fusion.py
#
# Late fusion training — requires GPU (~1 hr on T4).
# Loads pretrained image and spectral encoder weights before training.
#
# Optimal Phase 1: pre-extract frozen encoder features once → cache in RAM
# → train head + tabular encoder on cached vectors (no image I/O bottleneck).
# Phase 2: unfreeze all, joint fine-tuning with full image loading.
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.features import engineer_features, get_feature_cols, handle_missing
from src.datasets.multimodal import MultiModalDataset
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

BATCH_SIZE    = 256   # RTX 5070 Ti 16GB
BATCH_SIZE_P1 = 2048  # Phase 1: all in-memory, no I/O — use large batch
LR_PHASE1     = 1e-4  # head + tabular encoder
LR_PHASE2     = 1e-5  # all params, 10× lower
PHASE1_EPOCHS = 10
MAX_EPOCHS    = 50
PATIENCE      = 5
NUM_WORKERS   = 14    # image JPEG decode + transforms (Phase 2 only)
SPEC_DIR      = Path("input/spectra")


def class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, ablate: str = ""):
    """Standard epoch: full forward pass through all encoders (Phase 2 / test).
    ablate: one of '' | 'image' | 'spectral' | 'tabular' — zeros that modality's input.
    """
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, spec, tab, has_spec, labels in tqdm(loader, leave=False):
            img, spec, tab = img.to(device), spec.to(device), tab.to(device)
            has_spec, labels = has_spec.to(device), labels.to(device)

            if ablate == "image":
                img = torch.zeros_like(img)
            elif ablate == "spectral":
                spec     = torch.zeros_like(spec)
                has_spec = torch.zeros_like(has_spec)
            elif ablate == "tabular":
                tab = torch.zeros_like(tab)

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


def run_epoch_cached(model, loader, criterion, optimizer, device, train: bool):
    """Phase 1 epoch: uses pre-extracted img/spec features — no image I/O."""
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img_feat, spec_feat, tab, has_spec, labels in tqdm(loader, leave=False):
            img_feat  = img_feat.to(device)
            spec_feat = spec_feat.to(device)
            tab       = tab.to(device)
            has_spec  = has_spec.to(device)
            labels    = labels.to(device)

            logits = model.forward_from_feats(img_feat, spec_feat, tab, has_spec)
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


def extract_encoder_features(model, loader, device):
    """
    One-time pass through frozen image + spectral encoders to cache embeddings.
    Returns img_feats (N, 512), spec_feats (N, 128), tab (N, F), has_spec (N,), labels (N,).
    """
    model.eval()
    img_list, spec_list, tab_list, hs_list, lbl_list = [], [], [], [], []

    with torch.no_grad():
        for img, spec, tab, has_spec, labels in tqdm(loader, desc="  Extracting embeddings", leave=False):
            img_feat  = model.image_encoder(img.to(device)).cpu()   # (B, 512)
            spec_feat = model.spectral_encoder(spec.to(device)).cpu()  # (B, 128)
            img_list.append(img_feat)
            spec_list.append(spec_feat)
            tab_list.append(tab)
            hs_list.append(has_spec)
            lbl_list.append(labels)

    return (
        torch.cat(img_list),    # (N, 512)
        torch.cat(spec_list),   # (N, 128)
        torch.cat(tab_list),    # (N, F)
        torch.cat(hs_list),     # (N,)
        torch.cat(lbl_list),    # (N,)
    )


def make_cached_loader(img_feats, spec_feats, tabs, has_specs, labels, batch_size, shuffle):
    """Build an in-memory DataLoader from cached encoder features."""
    ds = TensorDataset(img_feats, spec_feats, tabs, has_specs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main():
    parser = argparse.ArgumentParser(description="Train late fusion model")
    parser.add_argument("--train",      default=str(TABLES / "split_train.csv"))
    parser.add_argument("--val",        default=str(TABLES / "split_val.csv"))
    parser.add_argument("--test",       default=str(TABLES / "split_test.csv"))
    parser.add_argument("--data",       default=str(TABLES / "DATA7901_DR19_merged.csv"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-ckpt",   default=str(IMG_CKPT),
                        help="Path to pretrained image encoder checkpoint")
    parser.add_argument("--spec-ckpt",  default=str(SPEC_CKPT),
                        help="Path to pretrained spectral encoder checkpoint")
    parser.add_argument("--suffix",     default="",
                        help="Suffix appended to output checkpoint/results filenames")
    parser.add_argument("--ablate",     default="", choices=["", "image", "spectral", "tabular"],
                        help="Zero out this modality during training and test (ablation study)")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training; load saved checkpoint and run test evaluation only")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Build feature table ───────────────────────────────────────────────
    print("Loading tabular features...", flush=True)
    print("  Reading CSV...", flush=True)
    df = pd.read_csv(args.data, low_memory=False, on_bad_lines="skip")
    print(f"  CSV loaded: {len(df):,} rows", flush=True)
    df = apply_renames(df)
    print("  Making labels...", flush=True)
    df = make_labels(df)
    print(f"  After labels: {len(df):,} rows", flush=True)
    df = handle_missing(df)
    df = engineer_features(df)
    feature_cols = get_feature_cols(df)
    print(f"  Features ready: {len(feature_cols)} columns", flush=True)

    def load_split(path):
        split = pd.read_csv(path)
        feat  = df[["objid"] + feature_cols].copy()
        merged = split.merge(feat, on="objid", how="inner")
        merged = merged[merged["has_image"]].reset_index(drop=True)
        merged["spec_path"] = merged.apply(
            lambda r: str(SPEC_DIR / f"spec-{int(r.plate)}-{int(r.spec_mjd)}-{int(r.fiberid):04d}.fits"),
            axis=1,
        )
        return merged

    train_df = load_split(args.train)
    val_df   = load_split(args.val)
    test_df  = load_split(args.test)
    print(f"Fusion rows — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    # Full multimodal loaders (used for embedding extraction + Phase 2 + test)
    train_ds = MultiModalDataset(train_df, IMG_DIR, feature_cols, train=True)
    val_ds   = MultiModalDataset(val_df,   IMG_DIR, feature_cols, train=False)
    test_ds  = MultiModalDataset(test_df,  IMG_DIR, feature_cols, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True, prefetch_factor=4)
    # num_workers=0 for test: avoids Windows pagefile exhaustion when spawning workers after Phase 2
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = LateFusionModel(n_tab_features=len(feature_cols)).to(device)

    if args.eval_only:
        print("Eval-only mode: loading checkpoint and running test...", flush=True)
        weights   = class_weights(train_df["label_id"].values).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = AdamW(model.parameters(), lr=LR_PHASE2)
        ckpt_name    = f"late_fusion{args.suffix}_best.pth"
        results_name = f"test_results{args.suffix}.pkl"
        load_checkpoint(model, CHECKPOINTS / ckpt_name, device)
        print(f"  Running test on {len(test_df):,} rows (num_workers=0)...", flush=True)
        _, test_results = run_epoch(model, test_loader, criterion, optimizer, device, train=False, ablate=args.ablate)
        print_results(test_results, f"Test (best checkpoint{args.suffix})")
        with open(CHECKPOINTS / results_name, "wb") as f:
            pickle.dump(test_results, f)
        print(f"Saved: checkpoints/fusion/{results_name}", flush=True)
        return

    def load_encoder(encoder, ckpt_path, prefix):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
        encoder.load_state_dict(sd)
        print(f"  Loaded {ckpt_path} (stripped prefix '{prefix}')")

    load_encoder(model.image_encoder,    Path(args.img_ckpt),  "encoder.")
    load_encoder(model.spectral_encoder, Path(args.spec_ckpt), "encoder.")
    print("Loaded pretrained encoder weights.")

    weights   = class_weights(train_df["label_id"].values).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    best_f1, patience_counter = 0.0, 0

    # ── Phase 1: pre-extract embeddings, train head + tabular encoder ──────
    print("\n── Phase 1: extracting encoder embeddings (one-time pass) ──")
    model.freeze_encoders()

    # Extract from a non-shuffled version of train to keep index alignment
    extract_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=False,
                                persistent_workers=True, prefetch_factor=4)
    print("  Train set:")
    tr_img, tr_spec, tr_tab, tr_hs, tr_lbl = extract_encoder_features(model, extract_loader, device)
    print("  Val set:")
    va_img, va_spec, va_tab, va_hs, va_lbl = extract_encoder_features(model, val_loader, device)

    # Move to shared memory for DataLoader (num_workers=0 so not strictly needed, but consistent)
    for t in [tr_img, tr_spec, tr_tab, tr_hs, tr_lbl, va_img, va_spec, va_tab, va_hs, va_lbl]:
        t.share_memory_()

    cached_train_loader = make_cached_loader(tr_img, tr_spec, tr_tab, tr_hs, tr_lbl, BATCH_SIZE_P1, shuffle=True)
    cached_val_loader   = make_cached_loader(va_img, va_spec, va_tab, va_hs, va_lbl, BATCH_SIZE_P1, shuffle=False)

    print(f"\n── Phase 1: training head + tabular encoder ({PHASE1_EPOCHS} epochs, batch={BATCH_SIZE_P1}) ──")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, _ = run_epoch_cached(model, cached_train_loader, criterion, optimizer, device, train=True)
        val_loss, val_res = run_epoch_cached(model, cached_val_loader, criterion, optimizer, device, train=False)
        print(f"  Phase1 epoch {epoch:2d}  train_loss={train_loss:.4f}  val_f1={val_res['macro_f1']:.4f}")

    # Free cached tensors to recover RAM before Phase 2
    del tr_img, tr_spec, tr_tab, tr_hs, tr_lbl
    del va_img, va_spec, va_tab, va_hs, va_lbl
    del cached_train_loader, cached_val_loader, extract_loader

    # ── Phase 2: unfreeze all, joint fine-tuning ──────────────────────────
    print("\n── Phase 2: all params unfrozen, joint fine-tuning ──")
    model.unfreeze_encoders()
    optimizer = AdamW(model.parameters(), lr=LR_PHASE2)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True,  ablate=args.ablate)
        val_loss, val_res = run_epoch(model, val_loader, criterion, optimizer, device, train=False, ablate=args.ablate)
        scheduler.step()

        val_f1 = val_res["macro_f1"]
        print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            ckpt_name = f"late_fusion{args.suffix}_best.pth"
            save_checkpoint(model, CHECKPOINTS / ckpt_name, epoch=epoch, val_macro_f1=best_f1)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Test ──────────────────────────────────────────────────────────────
    ckpt_name    = f"late_fusion{args.suffix}_best.pth"
    results_name = f"test_results{args.suffix}.pkl"
    load_checkpoint(model, CHECKPOINTS / ckpt_name, device)
    _, test_results = run_epoch(model, test_loader, criterion, optimizer, device, train=False, ablate=args.ablate)
    print_results(test_results, f"Test (best checkpoint{args.suffix})")

    with open(CHECKPOINTS / results_name, "wb") as f:
        pickle.dump(test_results, f)
    print(f"Saved: checkpoints/fusion/{results_name}")


if __name__ == "__main__":
    main()
