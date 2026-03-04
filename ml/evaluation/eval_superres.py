"""
Evaluation suite for the super-resolution model.
Computes PSNR, SSIM, LPIPS; visualizes LR vs SR vs HR comparison.
"""

from typing import Optional

# TODO: add imports (torch, numpy, matplotlib, PIL, load_model, config; optional lpips)


def compute_psnr(pred: "np.ndarray", target: "np.ndarray") -> float:
    """
    Compute PSNR in dB between pred and target (numpy, any range; use same for both).

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
    # TODO: implement — MSE, then 10 * log10(max^2 / MSE)
    pass


def compute_ssim(pred: "np.ndarray", target: "np.ndarray") -> float:
    """
    Compute SSIM between pred and target (numpy).

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
    # TODO: implement — e.g. skimage.metrics.structural_similarity
    pass


def compute_lpips(pred: "np.ndarray", target: "np.ndarray") -> float:
    """
    Compute LPIPS (perceptual similarity) if lpips package available.

    Parameters
    ----------
    pred : np.ndarray
        Predicted HR image (e.g. HWC, 0–1 or 0–255).
    target : np.ndarray
        Ground truth HR image.

    Returns
    -------
    float
        LPIPS value (lower is better); return -1.0 if lpips not available.
    """
    # TODO: implement — convert to tensor, normalize if needed, run LPIPS model
    pass


def visualize_comparison(
    lr_image: "np.ndarray",
    sr_image: "np.ndarray",
    hr_image: "np.ndarray",
    save_path: str,
) -> None:
    """
    Create side-by-side figure: low-res, super-res, and ground truth HR; save to file.

    Parameters
    ----------
    lr_image : np.ndarray
        Low-resolution input (HWC).
    sr_image : np.ndarray
        Model super-resolved output.
    hr_image : np.ndarray
        Ground truth high-res.

    save_path : str
        Path to save the figure (.png).

    Side effects
    ------------
    Saves figure to disk.
    """
    # TODO: implement — matplotlib or PIL; subplot or concat, titles, savefig
    pass


def run_evaluation(
    model_path: str, data_path: str, output_dir: Optional[str] = None
) -> dict:
    """
    Run full SR evaluation: load model, run on dataset, compute PSNR/SSIM/LPIPS, save plots and comparison images.

    Parameters
    ----------
    model_path : str
        Path to checkpoint or ONNX.
    data_path : str
        Path to DIV2K or test set.
    output_dir : str, optional
        Directory for output figures.

    Returns
    -------
    dict
        Metrics (psnr, ssim, lpips) and paths to saved visualizations.
    """
    # TODO: implement — load model, iterate dataset, compute metrics, visualize_comparison for samples,
    #       save plots, return metrics dict
    pass


if __name__ == "__main__":
    # TODO: parse args or config, call run_evaluation
    pass
