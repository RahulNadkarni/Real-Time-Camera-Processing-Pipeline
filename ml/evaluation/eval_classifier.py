"""
Full evaluation suite for the scene classifier.
Computes accuracy, confusion matrix, per-class metrics, and saves plots.
"""

from typing import List, Optional, Tuple

import numpy as np

# TODO: add imports (torch, matplotlib, config, load_model from training or export)


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute overall accuracy.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class indices (shape (N,)).
    labels : np.ndarray
        Ground truth class indices (shape (N,)).

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    # TODO: implement — (predictions == labels).mean()
    pass


def compute_confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, class_names: List[str]
) -> np.ndarray:
    """
    Compute confusion matrix (rows = true, cols = pred).

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class indices.
    labels : np.ndarray
        Ground truth class indices.
    class_names : List[str]
        Ordered list of class names (index matches class id).

    Returns
    -------
    np.ndarray
        Square confusion matrix of shape (num_classes, num_classes).
    """
    # TODO: implement — np.zeros, iterate and fill, or use sklearn.metrics.confusion_matrix
    pass


def compute_per_class_metrics(confusion_matrix: np.ndarray) -> dict:
    """
    Compute per-class precision, recall, F1 from confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Square confusion matrix.

    Returns
    -------
    dict
        e.g. {"precision": [...], "recall": [...], "f1": [...]} per class.
    """
    # TODO: implement — TP, FP, FN from matrix; precision = TP/(TP+FP), recall = TP/(TP+FN)
    pass


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: List[str], save_path: str
) -> None:
    """
    Plot confusion matrix as heatmap and save to file.

    Parameters
    ----------
    matrix : np.ndarray
        Confusion matrix.
    class_names : List[str]
        Class names for axis labels.
    save_path : str
        Path to save the figure (e.g. .png).

    Side effects
    ------------
    Saves figure to disk.
    """
    # TODO: implement — matplotlib imshow/heatmap, axis labels, colorbar, savefig
    pass


def run_evaluation(model_path: str, data_path: str, output_dir: Optional[str] = None) -> dict:
    """
    Run full evaluation: load model, run on val/test set, compute metrics, save plots.

    Parameters
    ----------
    model_path : str
        Path to checkpoint or ONNX model.
    data_path : str
        Path to dataset (e.g. CIFAR-10 root).
    output_dir : str, optional
        Directory to save plots; defaults to current dir or eval_output/.

    Returns
    -------
    dict
        Metrics (accuracy, per_class_metrics, etc.).
    """
    # TODO: implement — load model, load data, get predictions and labels,
    #       compute_accuracy, compute_confusion_matrix, compute_per_class_metrics,
    #       plot_confusion_matrix, save all plots, return metrics dict
    pass


if __name__ == "__main__":
    # TODO: parse args or use config for model_path, data_path; call run_evaluation
    pass
