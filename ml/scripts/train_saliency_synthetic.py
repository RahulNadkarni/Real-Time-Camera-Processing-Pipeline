"""
Train the saliency U-Net on synthetic in-memory data (no disk I/O).
Use when SALICON data is on a slow/network volume and times out.
Produces a model with real variation so the pipeline overlay is non-uniform.
"""

import sys
from pathlib import Path

# run from ml/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import Config
from training.train_saliency import build_unet


class SyntheticSaliencyDataset(Dataset):
    """In-memory (image, saliency) pairs: random RGB + center-biased Gaussian heatmap."""

    def __init__(self, num_samples: int, size=(224, 224)):
        self.num_samples = num_samples
        self.size = size
        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        # Random RGB image in [0,1]
        image = torch.rand(3, self.size[0], self.size[1])
        # Center-biased saliency (Gaussian blob) so the model learns some structure
        h, w = self.size[0], self.size[1]
        y = torch.linspace(-1, 1, h)
        x = torch.linspace(-1, 1, w)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        # Random offset and scale per sample
        cx = (idx % 7) / 7.0 * 0.8 - 0.4
        cy = ((idx * 3) % 5) / 5.0 * 0.8 - 0.4
        sigma = 0.3 + (idx % 10) / 50.0
        saliency = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency = saliency.unsqueeze(0)  # (1, H, W)
        if self.transform is not None:
            image = self.transform(image)
        return image, saliency


def main():
    cfg = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    size = cfg.saliency_input_size

    train_ds = SyntheticSaliencyDataset(400, size=size)
    val_ds = SyntheticSaliencyDataset(100, size=size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for images, maps in train_loader:
            images, maps = images.to(device), maps.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, maps)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, maps in val_loader:
                images, maps = images.to(device), maps.to(device)
                out = model(images)
                val_loss += criterion(out, maps).item()
        n_train = len(train_loader)
        n_val = len(val_loader)
        print(f"Epoch {epoch + 1}: train_loss={total_loss / max(n_train, 1):.4f} val_loss={val_loss / max(n_val, 1):.4f}")

    Path(cfg.saliency_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, cfg.saliency_checkpoint_path)
    print(f"Saved {cfg.saliency_checkpoint_path}")


if __name__ == "__main__":
    main()
