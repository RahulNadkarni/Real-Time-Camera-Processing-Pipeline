"""
Evaluation suite for the super-resolution model.
Computes PSNR, SSIM, LPIPS; visualizes LR vs SR vs HR comparison.
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from export.export_superres import load_model
from training.train_superres import load_div2k_data


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Compute PSNR in dB. Assumes pred and target in same range (e.g. [0, 1]).

    Parameters
    ----------
    pred : np.ndarray
        Predicted HR image.
    target : np.ndarray
        Ground truth HR image.

    Returns
    -------
    float
        PSNR in dB.
    """
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2) + 1e-10
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Compute SSIM between pred and target (numpy, HWC or CHW).

    Parameters
    ----------
    pred : np.ndarray
        Predicted HR image.
    target : np.ndarray
        Ground truth HR image.

    Returns
    -------
    float
        SSIM in [0, 1].
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        return 0.0
    if pred.ndim == 4:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]
    if pred.shape[0] == 3:
        pred = np.transpose(pred, (1, 2, 0))
    if target.shape[0] == 3:
        target = np.transpose(target, (1, 2, 0))
    return float(ssim_fn(pred, target, channel_axis=2, data_range=max_val))


def compute_lpips(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute LPIPS (perceptual similarity) if lpips package available.
    Lower is better. Returns -1.0 if lpips not available or on error.

    Parameters
    ----------
    pred : np.ndarray
        Predicted HR image (HWC or CHW, 0–1).
    target : np.ndarray
        Ground truth HR image.

    Returns
    -------
    float
        LPIPS value, or -1.0 if unavailable.
    """
    try:
        import lpips
    except ImportError:
        return -1.0
    try:
        if pred.ndim == 3 and pred.shape[-1] == 3:
            pred = np.transpose(pred, (2, 0, 1))
        if target.ndim == 3 and target.shape[-1] == 3:
            target = np.transpose(target, (2, 0, 1))
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]
        if pred.ndim == 3:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pred_t = torch.from_numpy(pred).float().to(dev)
        target_t = torch.from_numpy(target).float().to(dev)
        loss_fn = lpips.LPIPS(net="alex").to(dev)
        with torch.no_grad():
            d = loss_fn(pred_t, target_t)
        return float(d.mean().item())
    except Exception:
        return -1.0


def visualize_comparison(
    lr_image: np.ndarray,
    sr_image: np.ndarray,
    hr_image: np.ndarray,
    save_path: str,
) -> None:
    """
    Create side-by-side figure: low-res (upscaled), super-res, ground truth HR; save to file.

    Parameters
    ----------
    lr_image : np.ndarray
        Low-res input, upscaled to HR size (C, H, W) or (H, W, C).
    sr_image : np.ndarray
        Model super-resolved output.
    hr_image : np.ndarray
        Ground truth high-res.
    save_path : str
        Path to save the figure (.png).
    """
    def to_hwc(x: np.ndarray) -> np.ndarray:
        x = np.squeeze(x)
        if x.ndim == 3 and x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))
        return np.clip(x, 0, 1)

    lr_img = to_hwc(lr_image)
    sr_img = to_hwc(sr_image)
    hr_img = to_hwc(hr_image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(lr_img)
    axes[0].set_title("LR (bicubic up)")
    axes[0].axis("off")
    axes[1].imshow(sr_img)
    axes[1].set_title("SR (model)")
    axes[1].axis("off")
    axes[2].imshow(hr_img)
    axes[2].set_title("HR (ground truth)")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_evaluation(
    model_path: str,
    data_path: str,
    output_dir: Optional[str] = None,
    scale_factor: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    num_visualize: int = 5,
) -> dict:
    """
    Run full SR evaluation: load model, run on val set, compute PSNR/SSIM/LPIPS, save comparison images.

    Parameters
    ----------
    model_path : str
        Path to checkpoint (.pt).
    data_path : str
        Path to DIV2K (or similar) dataset.
    output_dir : str, optional
        Directory for output; defaults to eval_output/eval_superres.
    scale_factor : int, optional
        SR scale factor; defaults to Config.superres_scale_factor.
    batch_size : int
        Batch size for dataloader.
    num_workers : int
        DataLoader workers.
    num_visualize : int
        Number of sample comparison images to save.

    Returns
    -------
    dict
        Metrics (psnr, ssim, lpips) and paths to saved visualizations.
    """
    if output_dir is None:
        output_dir = "eval_output/eval_superres"
    if scale_factor is None:
        scale_factor = Config.superres_scale_factor

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(model_path)
    model.to(device)
    model.eval()

    _, val_loader = load_div2k_data(data_path, scale_factor, batch_size, num_workers)

    psnr_list: List[float] = []
    ssim_list: List[float] = []
    lpips_list: List[float] = []
    saved_paths: List[str] = []
    viz_count = 0

    with torch.no_grad():
        for lr_up, hr in val_loader:
            lr_up = lr_up.to(device)
            hr = hr.to(device)
            sr = model(lr_up)
            pred_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()
            lr_np = lr_up.cpu().numpy()
            for i in range(pred_np.shape[0]):
                p = pred_np[i]
                t = hr_np[i]
                psnr_list.append(compute_psnr(p, t))
                ssim_list.append(compute_ssim(p, t))
                lpips_val = compute_lpips(p, t)
                if lpips_val >= 0:
                    lpips_list.append(lpips_val)
                if viz_count < num_visualize:
                    path = os.path.join(output_dir, f"comparison_{viz_count}.png")
                    visualize_comparison(lr_np[i], p, t, path)
                    saved_paths.append(path)
                    viz_count += 1

    psnr_mean = float(np.mean(psnr_list)) if psnr_list else 0.0
    ssim_mean = float(np.mean(ssim_list)) if ssim_list else 0.0
    lpips_mean = float(np.mean(lpips_list)) if lpips_list else -1.0

    metrics = {
        "psnr_db": psnr_mean,
        "ssim": ssim_mean,
        "lpips": lpips_mean,
        "comparison_paths": saved_paths,
    }
    return metrics


if __name__ == "__main__":
    config = Config()
    run_evaluation(
        config.superres_checkpoint_path,
        config.div2k_data_path,
        output_dir="eval_output/eval_superres",
        scale_factor=config.superres_scale_factor,
    )
