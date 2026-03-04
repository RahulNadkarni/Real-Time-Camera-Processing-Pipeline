"""
Trains a lightweight U-Net for saliency map prediction with MPS acceleration.
Main training loop with wandb logging; uses SALICON-style data.
"""

from typing import Tuple

# TODO: add imports (torch, wandb, config, dataset_utils, model_utils)


def load_salicon_data(data_path: str):
    """
    Load SALICON (or similar) saliency dataset.

    Parameters
    ----------
    data_path : str
        Root path to dataset (images and fixation/saliency maps).

    Returns
    -------
    Tuple of (train_loader, val_loader) or (train_dataset, val_dataset).
    """
    # TODO: implement — build Dataset that returns (image, saliency_map or fixation_map)
    pass


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
        U-Net model.
    """
    # TODO: implement — encoder-decoder with skip connections; keep small for real-time
    pass


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """
    Run one training epoch for saliency model.

    Parameters
    ----------
    model : torch.nn.Module
        U-Net saliency model.
    loader : torch.utils.data.DataLoader
        Training loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    criterion : torch.nn.Module
        Loss (e.g. BCE, or CC loss for saliency).
    device : torch.device
        Device (mps).

    Returns
    -------
    float
        Average loss over the epoch.
    """
    # TODO: implement — train loop
    pass


def evaluate(model, loader, device) -> Tuple[float, float]:
    """
    Evaluate saliency model (loss and optionally NSS).

    Parameters
    ----------
    model : torch.nn.Module
        Saliency model.
    loader : torch.utils.data.DataLoader
        Validation loader.
    device : torch.device
        Device.

    Returns
    -------
    Tuple[float, float]
        (average_loss, average_NSS or 0 if not computed).
    """
    # TODO: implement — eval loop, optionally compute NSS per batch
    pass


def compute_nss(predicted_map, fixation_map) -> float:
    """
    Compute Normalized Scanpath Saliency (NSS) metric.

    Parameters
    ----------
    predicted_map : torch.Tensor or np.ndarray
        Predicted saliency map (2D or 3D with batch).
    fixation_map : torch.Tensor or np.ndarray
        Ground truth fixation map (binary or density).

    Returns
    -------
    float
        NSS value (higher is better).
    """
    # TODO: implement — NSS formula: mean of (pred - mu) / sigma at fixation locations
    pass


def main():
    """Main training loop: load SALICON data, build U-Net, train with wandb logging."""
    # TODO: implement — config, device MPS, load_salicon_data, build_unet,
    #       train_one_epoch, evaluate, wandb.log(loss, nss), save_checkpoint
    pass


if __name__ == "__main__":
    main()
