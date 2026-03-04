"""
Evaluation suite for the saliency model.
Computes NSS, AUC, KL divergence; saves metric plots.
"""

from typing import Optional

# TODO: add imports (torch, numpy, matplotlib, load_model, config)


def compute_nss(pred: "np.ndarray", fixations: "np.ndarray") -> float:
    """
    Compute Normalized Scanpath Saliency.

    Parameters
    ----------
    pred : np.ndarray
        Predicted saliency map (2D).
    fixations : np.ndarray
        Ground truth fixation map (binary or density).

    Returns
    -------
    float
        NSS value.
    """
    # TODO: implement — same as training module or centralized in utils
    pass


def compute_auc(pred: "np.ndarray", fixations: "np.ndarray") -> float:
    """
    Compute AUC (e.g. AUC-Judd or AUC-Borji) for saliency.

    Parameters
    ----------
    pred : np.ndarray
        Predicted saliency map.
    fixations : np.ndarray
        Ground truth fixations.

    Returns
    -------
    float
        AUC score in [0, 1].
    """
    # TODO: implement — threshold pred at many levels, compute ROC, area under curve
    pass


def compute_kl_divergence(pred: "np.ndarray", gt: "np.ndarray") -> float:
    """
    Compute KL divergence between predicted and ground truth saliency distributions.

    Parameters
    ----------
    pred : np.ndarray
        Predicted saliency (normalized to sum to 1).
    gt : np.ndarray
        Ground truth saliency (normalized).

    Returns
    -------
    float
        KL(gt || pred) or symmetric KL; avoid log(0) with epsilon.
    """
    # TODO: implement — flatten, normalize, kl = sum(gt * (log(gt) - log(pred)))
    pass


def run_evaluation(
    model_path: str, data_path: str, output_dir: Optional[str] = None
) -> dict:
    """
    Run full saliency evaluation: load model, run on dataset, compute NSS/AUC/KL, save plots.

    Parameters
    ----------
    model_path : str
        Path to checkpoint or ONNX.
    data_path : str
        Path to saliency dataset.
    output_dir : str, optional
        Directory for output plots.

    Returns
    -------
    dict
        Metrics (nss, auc, kl, etc.).
    """
    # TODO: implement — load model, iterate dataset, compute metrics, aggregate,
    #       plot (e.g. bar chart of metrics), save, return dict
    pass


if __name__ == "__main__":
    # TODO: parse args or config, call run_evaluation
    pass
