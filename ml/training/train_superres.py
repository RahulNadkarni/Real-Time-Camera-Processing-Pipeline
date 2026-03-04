"""
Trains SRCNN on DIV2K dataset with MPS acceleration.
Optional alternative: ESRGAN-tiny. Main loop logs PSNR and SSIM per epoch via wandb.
"""

from typing import Tuple

# TODO: add imports (torch, wandb, config, dataset_utils, model_utils)


def load_div2k_data(data_path: str, scale_factor: int):
    """
    Load DIV2K dataset for super-resolution (LR/HR pairs).

    Parameters
    ----------
    data_path : str
        Root path to DIV2K (train/val images).
    scale_factor : int
        Downsampling scale (e.g. 2 for 2x SR).

    Returns
    -------
    Tuple of (train_loader, val_loader).
    """
    # TODO: implement — Dataset that returns (lr_patch, hr_patch); optional augmentation
    pass


def build_srcnn():
    """
    Build SRCNN model (input: LR image, output: HR image same spatial size as LR upscaled).

    Returns
    -------
    torch.nn.Module
        SRCNN model.
    """
    # TODO: implement — classic SRCNN: conv layers for patch-to-patch mapping; input can be bicubic-upsampled LR
    pass


def build_esrgan_tiny():
    """
    Build a small ESRGAN-style generator (alternative to SRCNN).

    Returns
    -------
    torch.nn.Module
        Lightweight ESRGAN-like model for real-time use.
    """
    # TODO: implement — reduced residual blocks, optional upsampling subpixel; keep tiny for latency
    pass


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """
    Run one training epoch for SR model.

    Parameters
    ----------
    model : torch.nn.Module
        SRCNN or ESRGAN-tiny.
    loader : torch.utils.data.DataLoader
        Training loader (lr, hr).
    optimizer : torch.optim.Optimizer
        Optimizer.
    criterion : torch.nn.Module
        Loss (e.g. L1 or L2).
    device : torch.device
        Device (mps).

    Returns
    -------
    float
        Average loss.
    """
    # TODO: implement — train loop
    pass


def evaluate(model, loader, device) -> Tuple[float, float, float]:
    """
    Evaluate SR model: loss, PSNR, SSIM.

    Parameters
    ----------
    model : torch.nn.Module
        SR model.
    loader : torch.utils.data.DataLoader
        Validation loader.
    device : torch.device
        Device.

    Returns
    -------
    Tuple[float, float, float]
        (average_loss, average_PSNR_dB, average_SSIM).
    """
    # TODO: implement — eval loop, compute_psnr and compute_ssim per batch or per image
    pass


def compute_psnr(pred: "torch.Tensor", target: "torch.Tensor") -> float:
    """
    Compute PSNR in dB (assuming pixel range 0–1 or 0–255; use same for both).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted HR image.
    target : torch.Tensor
        Ground truth HR image.

    Returns
    -------
    float
        PSNR in dB.
    """
    # TODO: implement — MSE then 10 * log10(max^2 / MSE)
    pass


def compute_ssim(pred: "torch.Tensor", target: "torch.Tensor") -> float:
    """
    Compute SSIM (structural similarity) between pred and target.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted HR image.
    target : torch.Tensor
        Ground truth HR image.

    Returns
    -------
    float
        SSIM in [0, 1].
    """
    # TODO: implement — use torch or skimage SSIM; handle batch
    pass


def main():
    """Main training loop: load DIV2K, build SRCNN (or ESRGAN-tiny), train with wandb (PSNR, SSIM)."""
    # TODO: implement — config, device MPS, load_div2k_data, build_srcnn (or build_esrgan_tiny),
    #       train_one_epoch, evaluate, wandb.log(loss, psnr, ssim), save_checkpoint
    pass


if __name__ == "__main__":
    main()
