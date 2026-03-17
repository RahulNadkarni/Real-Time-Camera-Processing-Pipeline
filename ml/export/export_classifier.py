"""
Exports the trained scene classifier to ONNX.
Includes validate_onnx to verify ONNX output matches PyTorch on a sample input.
"""

from typing import Tuple, Union

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torch.onnx
import torchvision.models as models

from config import Config


def build_model(num_classes: int = 10, pretrained: bool = False):
    """Build MobileNetV3-Small with same architecture as training (no pretrained for export)."""
    model = models.mobilenet_v3_small(pretrained=pretrained)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def load_model(checkpoint_path: str):
    """
    Load the trained classifier from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint (full checkpoint with "model" key or raw state_dict).

    Returns
    -------
    torch.nn.Module
        Model in eval mode, with weights loaded.
    """
    num_classes = len(Config().classifier_class_names)
    model = build_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
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
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    torch.onnx.export(model, dummy_input, output_path, input_names=['input'], output_names=['output'], opset_version=opset_version)


def validate_onnx(
    onnx_path: str,
    model: nn.Module,
    sample_input: Union[torch.Tensor, np.ndarray],
    atol: float = 1e-5,
) -> bool:
    """
    Run a forward pass through PyTorch and ONNX Runtime; compare logits.

    Parameters
    ----------
    onnx_path : str
        Path to the exported .onnx file.
    model : torch.nn.Module
        PyTorch model (same architecture as exported) in eval mode.
    sample_input : torch.Tensor or np.ndarray
        Sample input (e.g. 1xCxHxW). Converted to tensor for PyTorch, numpy for ORT.
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
        py_logits = model(sample_tensor).numpy()

    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_logits = ort_session.run(None, {input_name: sample_np})[0]

    return bool(np.allclose(py_logits, ort_logits, atol=atol))



def main():
    """Load model from config, export to ONNX, then validate."""
    config = Config()
    model = load_model(config.classifier_checkpoint_path)

    export_to_onnx(
        model,
        config.classifier_onnx_path,
        config.classifier_input_size,
        opset_version=config.onnx_opset_version,
    )

    sample = torch.randn(1, 3, config.classifier_input_size[0], config.classifier_input_size[1])
    if not validate_onnx(config.classifier_onnx_path, model, sample):
        raise RuntimeError("ONNX validation failed: PyTorch and ONNX outputs do not match.")
    print("ONNX export and validation successful.")


if __name__ == "__main__":
    main()
