# src/interpret.py
#
# Interpretability analysis:
#   1. XGBoost feature importance (gain) — saved to checkpoints/tabular/feature_importance.csv
#   2. GradCAM activation maps on ResNet-18 — saved to checkpoints/image/gradcam_samples.png
#
# Usage:
#   uv run python -m src.interpret

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.datasets.image import get_val_transform
from src.models.image import ImageClassifier
from src.utils.io import get_device, load_checkpoint


TABLES      = Path("input/tables")
IMG_DIR     = Path("input/images")
TAB_CKPT    = Path("checkpoints/tabular/xgb_model_full.pkl")
IMG_CKPT    = Path("checkpoints/image/resnet18_configC_best.pth")
OUT_TAB     = Path("checkpoints/tabular/feature_importance.csv")
OUT_IMG     = Path("checkpoints/image/gradcam_samples.png")

LABEL_NAMES = ["elliptical", "spiral", "merger"]
N_SAMPLES   = 6   # GradCAM: 2 per class


# ── 1. XGBoost feature importance ─────────────────────────────────────────────

def xgb_feature_importance():
    print("Loading XGBoost model...")
    with open(TAB_CKPT, "rb") as f:
        pkg = pickle.load(f)
    pipeline     = pkg["pipeline"]
    feature_cols = pkg["feature_cols"]

    # XGBoost gain importance (average gain per split using each feature)
    scores = pipeline.model.get_booster().get_score(importance_type="gain")
    # Map from f0/f1/... keys to column names
    fi = pd.DataFrame([
        {"feature": feature_cols[int(k[1:])], "importance": v}
        for k, v in scores.items()
    ]).sort_values("importance", ascending=False).reset_index(drop=True)

    fi.to_csv(OUT_TAB, index=False)
    print(f"Feature importance saved: {OUT_TAB}  ({len(fi)} features)")
    print(fi.head(10).to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    top = fi.head(15)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost — Top-15 Feature Importances (Gain)")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    out_bar = Path("checkpoints/tabular/feature_importance.png")
    fig.savefig(out_bar, dpi=150)
    plt.close(fig)
    print(f"Feature importance chart saved: {out_bar}")


# ── 2. GradCAM on ResNet-18 ───────────────────────────────────────────────────

class GradCAM:
    """
    Registers hooks on the last conv layer of ResNet-18 (layer4[-1])
    and computes class activation maps via gradient weighting.
    """
    def __init__(self, model: ImageClassifier):
        self.model   = model
        self.grads   = None
        self.acts    = None
        target_layer = model.encoder.layer4[-1]
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.acts = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns CAM (H, W) normalised to [0, 1]."""
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor.requires_grad_(True)

        logits = self.model(img_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.acts).sum(dim=1).squeeze()   # (H, W)
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def gradcam_grid():
    print("\nBuilding GradCAM grid...")
    device = get_device()

    model = ImageClassifier(pretrained=False, dropout=0.3).to(device)
    load_checkpoint(model, IMG_CKPT, device)
    model.eval()
    cam_fn = GradCAM(model)

    test_df = pd.read_csv(TABLES / "split_test.csv")
    test_df = test_df[test_df["has_image"]].reset_index(drop=True)
    transform = get_val_transform()

    # Collect 2 correct predictions per class
    samples = {0: [], 1: [], 2: []}   # class_idx -> list of (img_path, objid)
    NEED = 2

    label_map = {"elliptical": 0, "spiral": 1, "merger": 2}
    for _, row in test_df.iterrows():
        c = label_map.get(row["label"], -1)
        if c < 0 or len(samples[c]) >= NEED:
            continue
        path = IMG_DIR / f"{int(row['objid'])}.jpeg"
        if path.exists():
            samples[c].append((path, int(row["objid"])))
        if all(len(v) >= NEED for v in samples.values()):
            break

    nrows = sum(len(v) for v in samples.values())
    fig, axes = plt.subplots(nrows, 3, figsize=(9, nrows * 3))
    row_idx = 0

    for class_idx, paths in samples.items():
        for img_path, objid in paths:
            pil = Image.open(img_path).convert("RGB")
            tensor = transform(pil).to(device)

            with torch.no_grad():
                pred = model(tensor.unsqueeze(0)).argmax(1).item()

            cam = cam_fn(tensor, class_idx)

            # Overlay CAM on image
            img_np = np.array(pil.resize((224, 224))).astype(float) / 255.0
            import cv2 as _cv2  # optional dependency
            cam_up = _cv2.resize(cam, (224, 224))
            heatmap = plt.cm.jet(cam_up)[..., :3]
            overlay = 0.5 * img_np + 0.5 * heatmap

            axes[row_idx, 0].imshow(img_np)
            axes[row_idx, 0].set_title(f"objid={objid}\ntrue={LABEL_NAMES[class_idx]}", fontsize=8)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(cam_up, cmap="jet")
            axes[row_idx, 1].set_title("GradCAM", fontsize=8)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(np.clip(overlay, 0, 1))
            axes[row_idx, 2].set_title(f"pred={LABEL_NAMES[pred]}", fontsize=8)
            axes[row_idx, 2].axis("off")

            row_idx += 1

    fig.suptitle("GradCAM — ResNet-18 Galaxy Classifier", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_IMG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"GradCAM grid saved: {OUT_IMG}")


def main():
    xgb_feature_importance()
    try:
        gradcam_grid()
    except ImportError:
        print("opencv-python not installed — skipping GradCAM overlay. "
              "Install with: uv add opencv-python")
        # Fallback: save CAMs without overlay
        _gradcam_no_cv2()


def _gradcam_no_cv2():
    """GradCAM without cv2 — uses PIL resize instead."""
    print("Running GradCAM (no-cv2 fallback)...")
    device = get_device()

    model = ImageClassifier(pretrained=False, dropout=0.3).to(device)
    load_checkpoint(model, IMG_CKPT, device)
    model.eval()
    cam_fn = GradCAM(model)

    test_df = pd.read_csv(TABLES / "split_test.csv")
    test_df = test_df[test_df["has_image"]].reset_index(drop=True)
    transform = get_val_transform()

    label_map = {"elliptical": 0, "spiral": 1, "merger": 2}
    samples = {0: [], 1: [], 2: []}
    NEED = 2

    for _, row in test_df.iterrows():
        c = label_map.get(row["label"], -1)
        if c < 0 or len(samples[c]) >= NEED:
            continue
        path = IMG_DIR / f"{int(row['objid'])}.jpeg"
        if path.exists():
            samples[c].append((path, int(row["objid"])))
        if all(len(v) >= NEED for v in samples.values()):
            break

    nrows = sum(len(v) for v in samples.values())
    fig, axes = plt.subplots(nrows, 2, figsize=(6, nrows * 3))
    row_idx = 0

    for class_idx, paths in samples.items():
        for img_path, objid in paths:
            pil = Image.open(img_path).convert("RGB")
            tensor = transform(pil).to(device)

            with torch.no_grad():
                pred = model(tensor.unsqueeze(0)).argmax(1).item()

            cam = cam_fn(tensor, class_idx)

            # Resize CAM with PIL
            cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            cam_np  = np.array(cam_img) / 255.0

            axes[row_idx, 0].imshow(np.array(pil.resize((224, 224))))
            axes[row_idx, 0].set_title(f"objid={objid}\ntrue={LABEL_NAMES[class_idx]}  pred={LABEL_NAMES[pred]}", fontsize=8)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(cam_np, cmap="jet")
            axes[row_idx, 1].set_title("GradCAM", fontsize=8)
            axes[row_idx, 1].axis("off")

            row_idx += 1

    fig.suptitle("GradCAM — ResNet-18 Galaxy Classifier", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_IMG, dpi=150)
    plt.close(fig)
    print(f"GradCAM saved: {OUT_IMG}")


if __name__ == "__main__":
    main()
