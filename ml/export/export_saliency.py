"""
Exports the trained saliency (U-Net) model to ONNX.
validate_onnx compares heatmap outputs between PyTorch and ONNX Runtime.
"""

from typing import Tuple, Union

import numpy as np
import onnxruntime
import torch
import torch.onnx

from config import Config
from training.train_saliency import build_unet


def load_model(checkpoint_path: str):
    """
    Load trained saliency U-Net from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint (full checkpoint with "model" key or raw state_dict).

    Returns
    -------
    torch.nn.Module
        Model in eval mode.
    """
    model = build_unet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def export_to_onnx(model, output_path: str, input_size: Tuple[int, int], opset_version: int = 14):
    """
    Export saliency model to ONNX.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net.
    output_path : str
        Output .onnx path.
    input_size : Tuple[int, int]
        (H, W) input size.
    opset_version : int
        ONNX opset version.

    Side effects
    ------------
    Writes .onnx file.
    """
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    torch.onnx.export(model, dummy_input, output_path, input_names=['input'], output_names=['output'], opset_version=opset_version)


def validate_onnx(
    onnx_path: str,
    model: torch.nn.Module,
    sample_input: Union[torch.Tensor, np.ndarray],
    atol: float = 1e-5,
) -> bool:
    """
    Compare heatmap output of ONNX vs PyTorch on sample input.

    Parameters
    ----------
    onnx_path : str
        Path to .onnx model.
    model : torch.nn.Module
        PyTorch model (same architecture as exported) in eval mode.
    sample_input : torch.Tensor or np.ndarray
        Sample image tensor (1xCxHxW).
    atol : float
        Absolute tolerance for np.allclose.

    Returns
    -------
    bool
        True if outputs match within tolerance.
    """
    if isinstance(sample_input, np.ndarray):
        sample_tensor = torch.from_numpy(sample_input).float()
        sample_np = sample_input.astype(np.float32)
    else:
        sample_tensor = sample_input.float()
        sample_np = sample_input.detach().numpy()

    with torch.no_grad():
        py_heatmap = model(sample_tensor).numpy()

    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_heatmap = ort_session.run(None, {input_name: sample_np})[0]
    return bool(np.allclose(py_heatmap, ort_heatmap, atol=atol))


def main():
    """Load model, export to ONNX, validate."""
    config = Config()
    model = load_model(config.saliency_checkpoint_path)
    export_to_onnx(
        model,
        config.saliency_onnx_path,
        config.saliency_input_size,
        opset_version=config.onnx_opset_version,
    )
    sample = torch.randn(1, 3, config.saliency_input_size[0], config.saliency_input_size[1])
    if not validate_onnx(config.saliency_onnx_path, model, sample):
        raise RuntimeError("ONNX validation failed: PyTorch and ONNX outputs do not match.")
    print("ONNX export and validation successful.")


if __name__ == "__main__":
    main()
