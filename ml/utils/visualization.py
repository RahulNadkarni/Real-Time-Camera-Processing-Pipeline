"""
Shared visualization: training curves and overlays on frames (for C++ reference or Python preview).
"""

from typing import List, Optional, Tuple

# TODO: add imports (matplotlib, numpy, cv2 if needed)


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
    Saves figure to disk.
    """
    # TODO: implement — matplotlib plot, x=epoch, legend, savefig
    pass


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
    # TODO: implement — cv2.putText or PIL; optional: draw rectangle background for readability
    pass


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
        Frame with saliency overlay.
    """
    # TODO: implement — resize saliency to frame size if needed, apply colormap, cv2.addWeighted
    pass


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
    # TODO: implement — cv2.putText for "PSNR: X dB, SSIM: Y"
    pass
