# src/datasets/image.py

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.labels import LABEL_MAP


# ImageNet normalisation statistics (used because ResNet-18 is pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),   # galaxies have no preferred orientation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class GalaxyImageDataset(Dataset):
    """
    PyTorch Dataset for galaxy JPEG cutouts.

    Parameters
    ----------
    df       : DataFrame with columns 'objid' and 'label' (string)
    img_dir  : directory containing {objid}.jpeg files
    transform: torchvision transform; use get_train_transform() or get_val_transform()
    """

    def __init__(self, df: pd.DataFrame, img_dir: str | Path, transform=None):
        self.df      = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform or get_val_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        objid = row["objid"]
        label = LABEL_MAP[row["label"]]

        path = self.img_dir / f"{objid}.jpeg"
        img  = Image.open(path).convert("RGB")
        img  = self.transform(img)

        return img, label
