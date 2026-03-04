"""
Trains MobileNetV3-Small fine-tuned on CIFAR-10 with MPS acceleration.
Main training loop with wandb logging of loss and accuracy per epoch.
"""

from typing import Tuple, Optional

# TODO: add imports (torch, torchvision, wandb, config, dataset_utils, model_utils)


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
    # TODO: implement — use dataset_utils.download_cifar10, get_transforms, create_dataloaders
    pass


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
    # TODO: implement — torchvision.models.mobilenet_v3_small, replace classifier head
    pass


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
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
    float
        Average loss over the epoch.
    """
    # TODO: implement — set model.train(), iterate loader, backward, step optimizer
    pass


def evaluate(model, loader, device) -> Tuple[float, float]:
    """
    Evaluate model on a dataset (no grad).

    Parameters
    ----------
    model : torch.nn.Module
        Classification model.
    loader : torch.utils.data.DataLoader
        Validation/test data loader.
    device : torch.device
        Device.

    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy).
    """
    # TODO: implement — model.eval(), torch.no_grad(), compute loss and correct count
    pass


def save_checkpoint(model, path: str, epoch: Optional[int] = None, metrics: Optional[dict] = None):
    """
    Save model state dict (and optionally epoch/metrics) to path.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    path : str
        Output file path (.pt).
    epoch : int, optional
        Current epoch to store.
    metrics : dict, optional
        Metrics to store (e.g. val_accuracy).

    Side effects
    ------------
    Writes file to disk.
    """
    # TODO: implement — torch.save(state_dict or full checkpoint dict)
    pass


def main():
    """Main training loop: load data, build model, train for N epochs with wandb logging."""
    # TODO: implement — config, device MPS, load_data, build_model, optimizer, criterion;
    #       for each epoch: train_one_epoch, evaluate, wandb.log(loss, accuracy), save_checkpoint
    pass


if __name__ == "__main__":
    main()
