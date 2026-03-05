"""
Exports the trained scene classifier to ONNX.
Includes validate_onnx to verify ONNX output matches PyTorch on a sample input.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.onnx
import onnxruntime
import numpy as np
from config import Config
from utils import load_checkpoint


def load_model(checkpoint_path: str):
    """
    Load the trained classifier from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint (state_dict or full checkpoint).

    Returns
    -------
    torch.nn.Module
        Model in eval mode, with weights loaded.
    """
    # TODO: implement — build same architecture as training, load state_dict
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def export_to_onnx(model, output_path: str, input_size: Tuple[int, int], opset_version: int = 14):
    """
    Export model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode.
    output_path : str
        Path for the output .onnx file.
    input_size : Tuple[int, int]
        (height, width) of model input (e.g. (224, 224)).
    opset_version : int
        ONNX opset version.

    Side effects
    ------------
    Writes .onnx file to output_path.
    """
    # TODO: implement — dummy input, torch.onnx.export with input_names, output_names
    pass


def validate_onnx(onnx_path: str, sample_input) -> bool:
    """
    Run a forward pass through onnxruntime and compare output to PyTorch output.

    Parameters
    ----------
    onnx_path : str
        Path to the exported .onnx file.
    sample_input : torch.Tensor or np.ndarray
        Sample input tensor (e.g. 1xCxHxW).

    Returns
    -------
    bool
        True if outputs match within tolerance (e.g. atol=1e-5).
    """
    # TODO: implement — run PyTorch forward, run ONNX Runtime InferenceSession,
    #       compare outputs (logits or softmax) with np.allclose
    pass


def main():
    """Load model from config, export to ONNX, then validate."""
    # TODO: implement — config, load_model, export_to_onnx, validate_onnx, print result
    pass


if __name__ == "__main__":
    main()
