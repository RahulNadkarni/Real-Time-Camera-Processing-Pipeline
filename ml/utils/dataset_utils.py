"""
Shared dataset utilities: download CIFAR-10 and DIV2K, transforms, and dataloader creation.
"""

from pathlib import Path
from typing import Tuple, Optional

import torchvision 
import torch 
import PIL 

def download_cifar10(save_path: str) -> Path:
    """
    Download CIFAR-10 to save_path if not already present.

    Parameters
    ----------
    save_path : str
        Directory to save CIFAR-10 (e.g. data/cifar10).

    Returns
    -------
    Path
        Path to the root of the CIFAR-10 data.

    Side effects
    ------------
    Downloads and extracts CIFAR-10 to save_path.
    """
    return torchvision.datasets.CIFAR10(root=save_path, download_cifar10 = True) 


def download_div2k(save_path: str) -> Path:
    """
    Download DIV2K dataset to save_path if not already present.

    Parameters
    ----------
    save_path : str
        Directory to save DIV2K.

    Returns
    -------
    Path
        Path to DIV2K root.

    Side effects
    ------------
    May download from DIV2K URL and extract (or provide instructions if manual download).
    """
    return torchvision.datasets.DIV2K(root=save_path, download_div2k = True) 


def get_transforms(
    mode: str, input_size: Tuple[int, int]
) -> Tuple[Optional["torchvision.transforms.Compose"], Optional["torchvision.transforms.Compose"]]:
    """
    Return train and validation transform pipelines for images.

    Parameters
    ----------
    mode : str
        One of "classifier", "saliency", "superres" (different normalizations/crops).
    input_size : Tuple[int, int]
        (height, width) for resize.

    Returns
    -------
    Tuple of (train_transform, val_transform).
    train_transform can include augmentation (random crop, flip); val_transform is deterministic.
    """
    return (torchvision.transforms.Resize(input_size), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))


def create_dataloaders(
    dataset: "torch.utils.data.Dataset",
    batch_size: int,
    num_workers: int = 0,
    train: bool = True,
) -> "torch.utils.data.DataLoader":
    """
    Create a DataLoader for the given dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset (e.g. CIFAR-10, or Subset).
    batch_size : int
        Batch size.
    num_workers : int
        Number of worker processes.
    train : bool
        If True, shuffle and drop_last for training.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader instance.
    """
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = train, num_workers = num_workers, drop_last = train)