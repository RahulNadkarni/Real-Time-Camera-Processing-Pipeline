"""
Trains a lightweight U-Net for saliency map prediction with MPS acceleration.
Main training loop with wandb logging; uses SALICON-style data.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import wandb

from config import Config
from utils import create_dataloaders
from utils.model_utils import save_checkpoint


class SaliencyDataset(Dataset):
    """Dataset that yields (image, saliency_map) from a directory. Expects data_path/images and data_path/maps."""

    def __init__(self, data_path: str, split: str, transform=None, target_size=(224, 224)):
        self.data_path = Path(data_path)
        self.split = split  # "train" or "val"
        self.transform = transform
        self.target_size = target_size
        self.image_dir = self.data_path / "images" / split
        self.map_dir = self.data_path / "maps" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Saliency data not found at {self.image_dir}. "
                "Expect data_path/images/train, data_path/images/val, data_path/maps/train, data_path/maps/val."
            )
        self.samples = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
        if not self.samples:
            self.samples = sorted(self.image_dir.glob("*"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        map_path = self.map_dir / (img_path.stem + img_path.suffix)
        if not map_path.exists():
            map_path = self.map_dir / (img_path.stem + ".png")
        image = Image.open(img_path).convert("RGB")
        saliency = Image.open(map_path).convert("L") if map_path.exists() else Image.new("L", image.size, 0)
        if self.transform:
            image = self.transform(image)
        saliency = TF.resize(TF.to_tensor(saliency), list(self.target_size))
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return image, saliency


def load_salicon_data(data_path: str, batch_size: int, num_workers: int):
    """
    Load SALICON-style saliency dataset (images + saliency/fixation maps).

    Parameters
    ----------
    data_path : str
        Root path (expects images/train, images/val, maps/train, maps/val).
    batch_size : int
        Batch size for loaders.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    Tuple of (train_loader, val_loader).
    """
    size = Config.saliency_input_size
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = SaliencyDataset(data_path, "train", transform=transform, target_size=size)
    val_dataset = SaliencyDataset(data_path, "val", transform=transform, target_size=size)
    train_loader = create_dataloaders(train_dataset, batch_size, num_workers, train=True)
    val_loader = create_dataloaders(val_dataset, batch_size, num_workers, train=False)
    return train_loader, val_loader


def build_unet(in_channels: int = 3, out_channels: int = 1):
    """
    Build a lightweight U-Net for saliency (heatmap) prediction.

    Parameters
    ----------
    in_channels : int
        Input channels (3 for RGB).
    out_channels : int
        Output channels (1 for single saliency map).

    Returns
    -------
    torch.nn.Module
        U-Net model (minimal encoder + 1-channel output; extend with decoder for full U-Net).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(64, out_channels, kernel_size=1),
        nn.Sigmoid(),
    )



def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    """
    Run one training epoch for saliency model.

    Returns
    -------
    Tuple[float, float]
        (average loss, average NSS).
    """
    model.train()
    total_loss = 0.0
    total_nss = 0.0
    for images, saliency_maps in loader:
        images = images.to(device)
        saliency_maps = saliency_maps.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, saliency_maps)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_nss += compute_nss(outputs.detach(), saliency_maps)
    n_batches = max(len(loader), 1)
    return total_loss / n_batches, total_nss / n_batches


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    """
    Evaluate saliency model (loss and NSS).

    Parameters
    ----------
    model : torch.nn.Module
        Saliency model.
    loader : torch.utils.data.DataLoader
        Validation loader.
    criterion : torch.nn.Module
        Loss (e.g. BCELoss).
    device : torch.device
        Device.

    Returns
    -------
    Tuple[float, float]
        (average_loss, average_NSS).
    """
    model.eval()
    total_loss = 0.0
    total_nss = 0.0
    with torch.no_grad():
        for images, saliency_maps in loader:
            images = images.to(device)
            saliency_maps = saliency_maps.to(device)
            outputs = model(images)
            loss = criterion(outputs, saliency_maps)
            total_loss += loss.item()
            total_nss += compute_nss(outputs, saliency_maps)
    n_batches = max(len(loader), 1)
    return total_loss / n_batches, total_nss / n_batches



def compute_nss(
    predicted_map: Union[torch.Tensor, np.ndarray],
    fixation_map: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute Normalized Scanpath Saliency (NSS): mean of (pred - mu) / sigma over the map.
    For strict NSS, one would sample at fixation locations only; this version uses the full map.

    Parameters
    ----------
    predicted_map : torch.Tensor or np.ndarray
        Predicted saliency (B, 1, H, W) or (B, H, W).
    fixation_map : torch.Tensor or np.ndarray
        Ground truth (same shape).

    Returns
    -------
    float
        NSS value (higher is better).
    """
    if isinstance(predicted_map, torch.Tensor):
        predicted_map = predicted_map.detach().cpu().numpy()
    if isinstance(fixation_map, torch.Tensor):
        fixation_map = fixation_map.cpu().numpy()
    pred = np.squeeze(predicted_map).astype(np.float64)
    fix = np.squeeze(fixation_map).astype(np.float64)
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
        fix = fix.reshape(fix.shape[0], -1)
        mu = pred.mean(axis=1, keepdims=True)
        sigma = pred.std(axis=1, keepdims=True) + 1e-8
        nss_per_sample = ((pred - mu) / sigma * fix).sum(axis=1) / (fix.sum(axis=1) + 1e-8)
        return float(nss_per_sample.mean())
    mu, sigma = pred.mean(), pred.std() + 1e-8
    return float(((pred - mu) / sigma * fix).sum() / (fix.sum() + 1e-8))


def main():
    """Main training loop: load SALICON-style data, build U-Net, train with wandb logging."""
    cfg = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader = load_salicon_data(
        cfg.salicon_data_path, cfg.batch_size, cfg.num_workers
    )
    model = build_unet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(cfg.epochs):
        train_loss, train_nss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_nss = evaluate(model, val_loader, criterion, device)
        wandb.log({
            "train_loss": train_loss,
            "train_nss": train_nss,
            "val_loss": val_loss,
            "val_nss": val_nss,
        })
        save_checkpoint(
            model,
            optimizer,
            epoch,
            {"train_loss": train_loss, "train_nss": train_nss, "val_loss": val_loss, "val_nss": val_nss},
            cfg.saliency_checkpoint_path,
        )
    return model


if __name__ == "__main__":
    main()
