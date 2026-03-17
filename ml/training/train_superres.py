"""
Trains SRCNN on DIV2K dataset with MPS acceleration.
Optional alternative: ESRGAN-tiny. Main loop logs PSNR and SSIM per epoch via wandb.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import wandb

from config import Config
from utils.model_utils import save_checkpoint
SUPERRES_HR_SIZE = (256, 256)


class SuperResDataset(Dataset):
    """Dataset of (LR, HR) pairs: load image as HR, downsample to LR, return tensors."""

    def __init__(self, data_path: str, scale_factor: int, split: str, hr_size: Tuple[int, int] = SUPERRES_HR_SIZE):
        self.data_path = Path(data_path)
        self.scale_factor = scale_factor
        self.hr_size = hr_size
        self.lr_size = (hr_size[0] // scale_factor, hr_size[1] // scale_factor)
        dir_train = self.data_path / "DIV2K_train_HR" if (self.data_path / "DIV2K_train_HR").exists() else self.data_path / "train"
        dir_val = self.data_path / "DIV2K_valid_HR" if (self.data_path / "DIV2K_valid_HR").exists() else self.data_path / "val"
        img_dir = dir_train if split == "train" else dir_val
        if not img_dir.exists():
            raise FileNotFoundError(f"DIV2K data not found at {img_dir}. Expect {data_path}/DIV2K_train_HR and DIV2K_valid_HR (or train/ and val/).")
        self.paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = TF.to_tensor(img)
        img = TF.resize(img, self.hr_size, interpolation=T.InterpolationMode.BICUBIC)
        hr = img
        lr = TF.resize(img, self.lr_size, interpolation=T.InterpolationMode.BICUBIC)
        lr_up = TF.resize(lr, self.hr_size, interpolation=T.InterpolationMode.BICUBIC)
        return lr_up, hr


def load_div2k_data(data_path: str, scale_factor: int, batch_size: int, num_workers: int):
    """
    Load DIV2K dataset for super-resolution (LR/HR pairs).

    Returns
    -------
    Tuple of (train_loader, val_loader).
    """
    train_dataset = SuperResDataset(data_path, scale_factor, split="train")
    val_dataset = SuperResDataset(data_path, scale_factor, split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_srcnn():
    """
    Build SRCNN model (input: bicubic-upsampled LR, output: refined HR; same spatial size).

    Returns
    -------
    torch.nn.Module
        SRCNN model. Output in [0, 1] via sigmoid (or unbounded; train with MSE on [0,1] targets).
    """
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=9, padding=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 32, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 3, kernel_size=5, padding=2),
    )



def build_esrgan_tiny():
    """
    Build a small ESRGAN-style generator (alternative to SRCNN).

    Returns
    -------
    torch.nn.Module
        Lightweight ESRGAN-like model for real-time use.
    """
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, padding=1),
    )



def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float, float]:
    """
    Run one training epoch for SR model. Returns (average loss, average PSNR, average SSIM).
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        output = model(lr)
        loss = criterion(output, hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            total_psnr += compute_psnr(output, hr)
            total_ssim += compute_ssim(output, hr)
    n = max(len(loader), 1)
    return total_loss / n, total_psnr / n, total_ssim / n


def evaluate(model, loader, criterion, device) -> Tuple[float, float, float]:
    """
    Evaluate SR model: loss, PSNR, SSIM.

    Returns
    -------
    Tuple[float, float, float]
        (average_loss, average_PSNR_dB, average_SSIM).
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            output = model(lr)
            loss = criterion(output, hr)
            total_loss += loss.item()
            total_psnr += compute_psnr(output, hr)
            total_ssim += compute_ssim(output, hr)
    n = max(len(loader), 1)
    return total_loss / n, total_psnr / n, total_ssim / n


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR in dB. Assumes pred and target in same range (e.g. [0, 1]).

    Returns
    -------
    float
        PSNR in dB (batch-averaged).
    """
    mse = F.mse_loss(pred, target, reduction="none").view(pred.size(0), -1).mean(dim=1)
    mse = mse.clamp(min=1e-8)
    psnr = 10.0 * torch.log10(max_val ** 2 / mse)
    return psnr.mean().item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute SSIM (structural similarity) between pred and target; batch-averaged.

    Returns
    -------
    float
        SSIM in [0, 1] (simplified; for full SSIM use skimage or piq).
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        return 0.0
    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()
    vals = []
    for i in range(pred_np.shape[0]):
        v = ssim_fn(pred_np[i], target_np[i], channel_axis=2, data_range=max_val)
        vals.append(v)
    return float(np.mean(vals))  


def main():
    """Main training loop: load DIV2K, build SRCNN (or ESRGAN-tiny), train with wandb (PSNR, SSIM)."""
    wandb.init(project="camera-pipeline", config={})  # no-op when WANDB_MODE=disabled
    cfg = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader = load_div2k_data(
        cfg.div2k_data_path, cfg.superres_scale_factor, cfg.batch_size, cfg.num_workers
    )
    model = build_srcnn()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(cfg.epochs):
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, criterion, device)
        wandb.log({
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "train_ssim": train_ssim,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
        })
        save_checkpoint(
            model,
            optimizer,
            epoch,
            {
                "train_loss": train_loss,
                "train_psnr": train_psnr,
                "train_ssim": train_ssim,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            },
            cfg.superres_checkpoint_path,
        )
    return model


if __name__ == "__main__":
    main()
