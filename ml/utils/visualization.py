"""
Shared visualization: training curves and overlays on frames (for C++ reference or Python preview).
"""

from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = "Training curves",
) -> None:
    """
    Plot train and validation loss curves and save to file.

    Parameters
    ----------
    train_losses : List[float]
        Per-epoch training loss.
    val_losses : List[float]
        Per-epoch validation loss.
    save_path : str
        Path to save figure (.png).
    title : str
        Plot title.

    Side effects
    ------------
    Saves figure to disk. Creates and closes a new figure (does not affect global pyplot state).
    """
    fig, ax = plt.subplots()
    epochs = range(len(train_losses))
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def overlay_label_on_frame(
    frame: "np.ndarray",
    label: str,
    confidence: float,
    position: Tuple[int, int] = (10, 30),
) -> "np.ndarray":
    """
    Draw a text label and confidence on a frame (BGR image). Used for scene classifier overlay.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3); can be modified in place or a copy returned.
    label : str
        Class label string.
    confidence : float
        Confidence in [0, 1].
    position : Tuple[int, int]
        (x, y) for text origin.

    Returns
    -------
    np.ndarray
        Frame with overlay (may be same as input if in-place).
    """
    cv2.putText(frame, f"{label}: {confidence:.2f}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def overlay_saliency_on_frame(
    frame: "np.ndarray",
    saliency_map: "np.ndarray",
    alpha: float = 0.5,
) -> "np.ndarray":
    """
    Overlay saliency heatmap on frame (e.g. colormap applied to saliency, then blended).

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    saliency_map : np.ndarray
        2D saliency (H, W), values in [0, 1].
    alpha : float
        Blend factor for saliency overlay (0–1).

    Returns
    -------
    np.ndarray
        Frame with saliency overlay (modified in place).
    """
    h, w = frame.shape[:2]
    sm = np.squeeze(saliency_map)
    if sm.ndim != 2:
        raise ValueError("saliency_map must be 2D (H, W).")
    sm = cv2.resize(sm, (w, h))
    
    if np.issubdtype(sm.dtype, np.floating):
        sm = (np.clip(sm, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
    cv2.addWeighted(frame, 1 - alpha, heatmap_bgr, alpha, 0, frame)
    return frame


def overlay_metrics_on_frame(
    frame: "np.ndarray",
    psnr: float,
    ssim: float,
    position: Tuple[int, int] = (10, 60),
) -> "np.ndarray":
    """
    Draw PSNR and SSIM text on frame (for super-resolution overlay).

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    psnr : float
        PSNR in dB.
    ssim : float
        SSIM in [0, 1].
    position : Tuple[int, int]
        (x, y) for text.

    Returns
    -------
    np.ndarray
        Frame with metrics overlay.
    """
    cv2.putText(frame, f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.2f}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame
