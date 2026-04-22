# src/utils/io.py

from pathlib import Path

import torch


def save_checkpoint(model, path: str | Path, **meta) -> None:
    """Save model state dict and optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, path: str | Path, device: str | torch.device = "cpu") -> dict:
    """Load state dict into model. Returns metadata dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return {k: v for k, v in ckpt.items() if k != "state_dict"}


def get_device() -> torch.device:
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
