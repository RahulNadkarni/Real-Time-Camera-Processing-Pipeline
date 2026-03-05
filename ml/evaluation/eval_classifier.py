"""
Full evaluation suite for the scene classifier.
Computes accuracy, confusion matrix, per-class metrics, and saves plots.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from export.export_classifier import load_model
from training.train_classifier import load_data as load_classifier_data


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
    return (predictions == labels).mean()


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
    confusion_matrix = np.zeros((len(class_names), len(class_names)))
    for i in range(len(predictions)):
        confusion_matrix[labels[i], predictions[i]] += 1
    return confusion_matrix



def compute_per_class_metrics(
    confusion_matrix: np.ndarray, class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1 from confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Square confusion matrix (rows=true, cols=pred).
    class_names : List[str]
        Ordered class names (index = class id).

    Returns
    -------
    dict
        Per-class dict: class_name -> {"precision", "recall", "f1"}.
    """
    n = confusion_matrix.shape[0]
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp  
        fn = confusion_matrix[i, :].sum() - tp 
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        per_class_metrics[class_names[i]] = {"precision": precision, "recall": recall, "f1": f1}
    return per_class_metrics


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: List[str], save_path: str
) -> None:
    """
    Plot confusion matrix as heatmap and save to file.

    Parameters
    ----------
    matrix : np.ndarray
        Confusion matrix (rows=true, cols=pred).
    class_names : List[str]
        Class names for axis labels.
    save_path : str
        Path to save the figure (e.g. .png).

    Side effects
    ------------
    Saves figure to disk.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_evaluation(
    model_path: str,
    data_path: str,
    output_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Run full evaluation: load model, run on val set, compute metrics, save plots.

    Parameters
    ----------
    model_path : str
        Path to checkpoint (.pt).
    data_path : str
        Path to dataset (e.g. CIFAR-10 root).
    output_dir : str, optional
        Directory to save plots; defaults to eval_output/eval_classifier.
    class_names : list, optional
        Class names for labels; defaults to Config.classifier_class_names.

    Returns
    -------
    dict
        Metrics (accuracy, confusion_matrix, per_class_metrics).
    """
    if output_dir is None:
        output_dir = "eval_output/eval_classifier"
    if class_names is None:
        class_names = Config.classifier_class_names

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(model_path)
    model.to(device)
    model.eval()

    _, val_loader = load_classifier_data(data_path, batch_size=128, num_workers=4)
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for images, batch_labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())

    predictions = np.array(all_preds)
    labels_arr = np.array(all_labels)
    accuracy = compute_accuracy(predictions, labels_arr)
    confusion_matrix = compute_confusion_matrix(predictions, labels_arr, class_names)
    per_class_metrics = compute_per_class_metrics(confusion_matrix, class_names)
    plot_confusion_matrix(
        confusion_matrix, class_names, os.path.join(output_dir, "confusion_matrix.png")
    )
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "per_class_metrics": per_class_metrics,
    }


if __name__ == "__main__":
    config = Config()
    run_evaluation(
        config.classifier_checkpoint_path,
        config.cifar10_path,
        output_dir="eval_output/eval_classifier",
    )