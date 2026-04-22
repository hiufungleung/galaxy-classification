#!/usr/bin/env python
# run_pipeline.py
#
# Runs the full galaxy classification pipeline in order.
# Each stage is skipped automatically if its output files already exist.
#
# Usage:
#   uv run python run_pipeline.py          # skip stages whose output exists
#   uv run python run_pipeline.py --force  # re-run everything from scratch

import argparse
import subprocess
import sys
import time
from pathlib import Path

TABLES   = Path("input/tables")
CKPT_TAB = Path("checkpoints/tabular")
CKPT_IMG = Path("checkpoints/image")
CKPT_SPEC = Path("checkpoints/spectral")
CKPT_FUS = Path("checkpoints/fusion")
CKPT_PHYS = Path("checkpoints/physical")

FORCE = False  # set by --force; used in done()


def uv(*args):
    """Run `uv run python -m <args>` and stream output."""
    cmd = ["uv", "run", "python", "-m", *args]
    print(f"\n{'='*60}\n  {' '.join(cmd)}\n{'='*60}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] exit {result.returncode} after {elapsed:.0f}s", flush=True)
        sys.exit(result.returncode)
    print(f"\n[OK] {elapsed:.0f}s", flush=True)


def done(label: str, *sentinel_paths: Path) -> bool:
    """Return True (skip stage) if all sentinel files exist and --force is not set."""
    if not FORCE and all(p.exists() for p in sentinel_paths):
        print(f"\n[SKIP] {label} — output already exists")
        return True
    return False


# ── Stage 1 — full-data split ────────────────────────────────────────────────
if not done("split-full",
            TABLES / "split_train_full.csv",
            TABLES / "split_val_full.csv"):
    uv("src.data.split")

# ── Stage 2 — tabular baseline (XGBoost + grid search) ──────────────────────
if not done("tabular",
            CKPT_TAB / "xgb_model_full.pkl",
            CKPT_TAB / "test_results_full.pkl"):
    uv("src.train.tabular", "--grid-search")

# ── Stage 3 — image baseline (ResNet-18, configs A–D) ───────────────────────
if not done("image", CKPT_IMG / "test_results_configC.pkl"):
    for extra, sentinel in [
        ((),                                              CKPT_IMG / "resnet18_best.pth"),
        (("--lr","5e-5","--suffix","configB"),            CKPT_IMG / "test_results_configB.pkl"),
        (("--dropout","0.3","--suffix","configC"),        CKPT_IMG / "resnet18_configC_best.pth"),
        (("--no-pretrain","--dropout","0.3","--suffix","configD"), CKPT_IMG / "test_results_configD.pkl"),
    ]:
        if not done(f"  image {extra}", sentinel):
            uv("src.train.image", *extra)

# ── Stage 4 — spectral baseline (1D-CNN, configs A–D) ───────────────────────
if not done("spectral", CKPT_SPEC / "test_results_configC.pkl"):
    for extra, sentinel in [
        ((),                                          CKPT_SPEC / "cnn1d_best.pth"),
        (("--lr","5e-5","--suffix","configB"),        CKPT_SPEC / "test_results_configB.pkl"),
        (("--base-filters","64","--suffix","configC"),CKPT_SPEC / "test_results_configC.pkl"),
        (("--dropout","0.3","--suffix","configD"),    CKPT_SPEC / "test_results_configD.pkl"),
    ]:
        if not done(f"  spectral {extra}", sentinel):
            uv("src.train.spectral", *extra)

# ── Stage 5 — late fusion (best encoders) ────────────────────────────────────
if not done("fusion", CKPT_FUS / "test_results_bestenc.pkl"):
    uv("src.train.fusion",
       "--img-ckpt",  "checkpoints/image/resnet18_configC_best.pth",
       "--spec-ckpt", "checkpoints/spectral/cnn1d_best.pth",
       "--suffix",    "_bestenc")

# ── Stage 6 — ablation study (3 variants) ────────────────────────────────────
for modality, suffix in [
    ("spectral", "_ablate_nospec"),
    ("image",    "_ablate_noimg"),
    ("tabular",  "_ablate_notab"),
]:
    if not done(f"ablation-{modality}", CKPT_FUS / f"test_results{suffix}.pkl"):
        uv("src.train.fusion",
           "--img-ckpt",  "checkpoints/image/resnet18_configC_best.pth",
           "--spec-ckpt", "checkpoints/spectral/cnn1d_best.pth",
           "--ablate",    modality,
           "--suffix",    suffix)

# ── Stage 7 — interpretability ───────────────────────────────────────────────
if not done("interpret",
            CKPT_TAB / "feature_importance.csv",
            CKPT_IMG / "gradcam_samples.png"):
    uv("src.interpret")

# ── Stage 8 — physical classification ───────────────────────────────────────
if not done("physical",
            CKPT_PHYS / "tabular_results.pkl",
            CKPT_PHYS / "spectral_results.pkl"):
    uv("src.train.physical")

# ── Stage 9 — final evaluation summary ──────────────────────────────────────
if not done("evaluate", Path("checkpoints/results_summary.csv")):
    uv("src.evaluate")

print(f"\n{'='*60}\n  Pipeline complete.\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full galaxy classification pipeline.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages even if output already exists")
    args = parser.parse_args()
    FORCE = args.force
