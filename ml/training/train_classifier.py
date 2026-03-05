"""
Trains MobileNetV3-Small fine-tuned on CIFAR-10 with MPS acceleration.
Main training loop with wandb logging of loss and accuracy per epoch.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.datasets as datasets
import wandb

from config import Config
from utils import create_dataloaders
from utils.model_utils import save_checkpoint, count_parameters


def load_data(data_path: str, batch_size: int, num_workers: int):
    """
    Load CIFAR-10 train/val datasets and return train and val DataLoaders.

    Parameters
    ----------
    data_path : str
        Root path where CIFAR-10 is or will be downloaded.
    batch_size : int
        Batch size for both loaders.
    num_workers : int
        Number of dataloader workers.

    Returns
    -------
    Tuple of (train_loader, val_loader).

    Side effects
    ------------
    May download CIFAR-10 to data_path if not present.
    """
    # ImageNet normalization for pretrained MobileNet; resize to 224x224
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize(Config.classifier_input_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    val_transform = T.Compose([
        T.Resize(Config.classifier_input_size),
        T.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(root=data_path, train=False, transform=val_transform, download=True)
    train_loader = create_dataloaders(train_dataset, batch_size, num_workers, train=True)
    val_loader = create_dataloaders(val_dataset, batch_size, num_workers, train=False)
    return train_loader, val_loader


def build_model(num_classes: int = 10, pretrained: bool = True):
    """
    Build MobileNetV3-Small for classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (10 for CIFAR-10).
    pretrained : bool
        Whether to load ImageNet pretrained weights.

    Returns
    -------
    torch.nn.Module
        The model (on CPU; caller moves to device).
    """
    
    model = models.mobilenet_v3_small(pretrained=pretrained)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    """
    Run one training epoch; update model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Classification model.
    loader : torch.utils.data.DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    criterion : torch.nn.Module
        Loss (e.g. CrossEntropyLoss).
    device : torch.device
        Device (e.g. mps).

    Returns
    -------
    Tuple[float, float]
        (average loss over the epoch, training accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    num_batches = max(len(loader), 1)
    return total_loss / num_batches, (correct / total) if total else 0.0


def evaluate(
    model, loader, criterion, device
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset (no grad).

    Parameters
    ----------
    model : torch.nn.Module
        Classification model.
    loader : torch.utils.data.DataLoader
        Validation/test data loader.
    criterion : torch.nn.Module
        Loss (e.g. CrossEntropyLoss).
    device : torch.device
        Device.

    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    num_batches = max(len(loader), 1)
    return total_loss / num_batches, (correct / total) if total else 0.0

def _save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Save model state dict and optionally optimizer, epoch, metrics. Writes to disk."""
    state = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)

def main():
    """Main training loop: load data, build model, train for N epochs with wandb logging."""
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = len(config.classifier_class_names)

    train_loader, val_loader = load_data(
        config.cifar10_path, config.batch_size, config.num_workers
    )
    model = build_model(num_classes=num_classes, pretrained=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        _save_checkpoint(
            model,
            config.classifier_checkpoint_path,
            epoch=epoch,
            metrics={
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            optimizer=optimizer,
        )
    return model

if __name__ == "__main__":
    main()
