"""
Exports the trained super-resolution model to ONNX.
validate_onnx compares PyTorch vs ONNX output (allclose). Model input = bicubic-upsampled LR (same size as HR).
"""

from typing import Tuple, Union

import numpy as np
import onnxruntime
import torch
import torch.onnx

from config import Config
from training.train_superres import build_srcnn


def load_model(checkpoint_path: str):
    """
    Load trained SR model (SRCNN) from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint (full checkpoint with "model" key or raw state_dict).

    Returns
    -------
    torch.nn.Module
        Model in eval mode.
    """
    model = build_srcnn()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


# Model input = bicubic-upsampled LR, same spatial size as HR (e.g. 256x256 in training).
SUPERRES_INPUT_SIZE = (256, 256)


def export_to_onnx(
    model, output_path: str, input_size: Tuple[int, int], opset_version: int = 14
):
    """
    Export SR model to ONNX. Input is upscaled-LR image (H, W) = input_size.

    Parameters
    ----------
    model : torch.nn.Module
        Trained SR model.
    output_path : str
        Output .onnx path.
    input_size : Tuple[int, int]
        (H, W) of model input (same as HR size; bicubic-upsampled LR).
    opset_version : int
        ONNX opset version.

    Side effects
    ------------
    Writes .onnx file.
    """
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["input"], output_names=["output"],
        opset_version=opset_version,
    )


def validate_onnx(
    onnx_path: str,
    model: torch.nn.Module,
    sample_input: Union[torch.Tensor, np.ndarray],
    atol: float = 1e-5,
) -> bool:
    """
    Compare PyTorch vs ONNX output on a sample input (upscaled LR); true if allclose.

    Parameters
    ----------
    onnx_path : str
        Path to .onnx model.
    model : torch.nn.Module
        PyTorch model in eval mode.
    sample_input : torch.Tensor or np.ndarray
        Sample input (1xCxHxW).
    atol : float
        Absolute tolerance for np.allclose.

    Returns
    -------
    bool
        True if PyTorch and ONNX outputs match within tolerance.
    """
    if isinstance(sample_input, np.ndarray):
        sample_tensor = torch.from_numpy(sample_input).float()
        sample_np = sample_input.astype(np.float32)
    else:
        sample_tensor = sample_input.float()
        sample_np = sample_input.detach().numpy()
    with torch.no_grad():
        py_output = model(sample_tensor).numpy()
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_output = ort_session.run(None, {input_name: sample_np})[0]
    return bool(np.allclose(py_output, ort_output, atol=atol))


def main():
    """Load model, export to ONNX, validate."""
    config = Config()
    model = load_model(config.superres_checkpoint_path)
    export_to_onnx(
        model,
        config.superres_onnx_path,
        SUPERRES_INPUT_SIZE,
        opset_version=config.onnx_opset_version,
    )
    sample = torch.randn(1, 3, SUPERRES_INPUT_SIZE[0], SUPERRES_INPUT_SIZE[1])
    if not validate_onnx(config.superres_onnx_path, model, sample):
        raise RuntimeError("ONNX validation failed: PyTorch and ONNX outputs do not match.")
    print("ONNX export and validation successful.")


if __name__ == "__main__":
    main()
