"""
Evaluation suite for the saliency model.
Computes NSS, AUC, KL divergence; saves metric plots.
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from export.export_saliency import load_model
from training.train_saliency import load_salicon_data


def compute_nss(pred: np.ndarray, fixations: np.ndarray) -> float:
    """
    Compute Normalized Scanpath Saliency: (pred - mu) / sigma at fixation locations, averaged.

    Parameters
    ----------
    pred : np.ndarray
        Predicted saliency map (2D or batched).
    fixations : np.ndarray
        Ground truth fixation map (binary or density), same shape.

    Returns
    -------
    float
        NSS value (higher is better).
    """
    pred = np.squeeze(pred).astype(np.float64)
    fix = np.squeeze(fixations).astype(np.float64)
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
        fix = fix.reshape(fix.shape[0], -1)
        mu = pred.mean(axis=1, keepdims=True)
        sigma = pred.std(axis=1, keepdims=True) + 1e-8
        nss_per = ((pred - mu) / sigma * fix).sum(axis=1) / (fix.sum(axis=1) + 1e-8)
        return float(nss_per.mean())
    mu, sigma = pred.mean(), pred.std() + 1e-8
    return float(((pred - mu) / sigma * fix).sum() / (fix.sum() + 1e-8))


def compute_auc(pred: np.ndarray, fixations: np.ndarray) -> float:
    """
    Compute AUC (AUC-Judd style): threshold pred at many levels, ROC curve, area under curve.

    Parameters
    ----------
    pred : np.ndarray
        Predicted saliency map (2D).
    fixations : np.ndarray
        Binary fixation map (2D).

    Returns
    -------
    float
        AUC in [0, 1] (higher is better).
    """
    pred = np.squeeze(pred).flatten().astype(np.float64)
    fix = np.squeeze(fixations).flatten().astype(np.float64)
    fix_bin = (fix > 0.5).astype(np.float64)
    n_pos = fix_bin.sum()
    n_neg = len(fix_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    thresholds = np.unique(pred)[::-1]
    if len(thresholds) <= 1:
        return 0.5
    tpr, fpr = [], []
    for t in thresholds:
        pred_bin = (pred >= t).astype(np.float64)
        tp = (pred_bin * fix_bin).sum()
        fp = (pred_bin * (1 - fix_bin)).sum()
        tpr.append(tp / (n_pos + 1e-8))
        fpr.append(fp / (n_neg + 1e-8))
    tpr, fpr = np.array(tpr), np.array(fpr)
    return float(np.trapz(tpr, fpr))


def compute_kl_divergence(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL(gt || pred): sum(gt * (log(gt) - log(pred))). Both normalized to sum to 1.

    Parameters
    ----------
    pred : np.ndarray
        Predicted saliency (flattened, will be normalized).
    gt : np.ndarray
        Ground truth saliency (flattened, will be normalized).

    Returns
    -------
    float
        KL divergence (lower is better).
    """
    pred = np.squeeze(pred).flatten().astype(np.float64) + eps
    gt = np.squeeze(gt).flatten().astype(np.float64) + eps
    pred = pred / pred.sum()
    gt = gt / gt.sum()
    return float((gt * (np.log(gt) - np.log(pred))).sum())


def run_evaluation(
    model_path: str,
    data_path: str,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """
    Run full saliency evaluation: load model, run on val set, compute NSS/AUC/KL, save plots.

    Parameters
    ----------
    model_path : str
        Path to checkpoint (.pt).
    data_path : str
        Path to saliency dataset (images/val, maps/val).
    output_dir : str, optional
        Directory for output plots; defaults to eval_output/eval_saliency.
    batch_size : int
        Batch size for dataloader.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    dict
        Metrics (nss, auc, kl, loss).
    """
    if output_dir is None:
        output_dir = "eval_output/eval_saliency"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(model_path)
    model.to(device)
    model.eval()

    _, val_loader = load_salicon_data(data_path, batch_size, num_workers)
    criterion = torch.nn.BCELoss()

    nss_list: List[float] = []
    auc_list: List[float] = []
    kl_list: List[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for images, saliency_maps in val_loader:
            images = images.to(device)
            saliency_maps = saliency_maps.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, saliency_maps).item()
            pred_np = outputs.cpu().numpy()
            gt_np = saliency_maps.cpu().numpy()
            for i in range(pred_np.shape[0]):
                nss_list.append(compute_nss(pred_np[i], gt_np[i]))
                auc_list.append(compute_auc(pred_np[i], gt_np[i]))
                kl_list.append(compute_kl_divergence(pred_np[i], gt_np[i]))

    n_batches = max(len(val_loader), 1)
    nss_mean = float(np.mean(nss_list)) if nss_list else 0.0
    auc_mean = float(np.mean(auc_list)) if auc_list else 0.5
    kl_mean = float(np.mean(kl_list)) if kl_list else 0.0
    loss_mean = total_loss / n_batches

    metrics = {"nss": nss_mean, "auc": auc_mean, "kl": kl_mean, "loss": loss_mean}

    fig, ax = plt.subplots()
    ax.bar(["NSS", "AUC", "KL"], [nss_mean, auc_mean, kl_mean], color=["#2ecc71", "#3498db", "#e74c3c"])
    ax.set_ylabel("Score")
    ax.set_title("Saliency evaluation metrics")
    fig.savefig(os.path.join(output_dir, "metrics.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    return metrics


if __name__ == "__main__":
    config = Config()
    run_evaluation(
        config.saliency_checkpoint_path,
        config.salicon_data_path,
        output_dir="eval_output/eval_saliency",
    )
