"""
Exports the trained saliency (U-Net) model to ONNX.
validate_onnx compares heatmap outputs between PyTorch and ONNX Runtime.
"""

from typing import Tuple

# TODO: add imports (torch, onnx, onnxruntime, config, numpy)


def load_model(checkpoint_path: str):
    """
    Load trained saliency U-Net from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint.

    Returns
    -------
    torch.nn.Module
        Model in eval mode.
    """
    # TODO: implement — build_unet, load state_dict
    pass


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
    # TODO: implement — torch.onnx.export
    pass


def validate_onnx(onnx_path: str, sample_input) -> bool:
    """
    Compare heatmap output of ONNX vs PyTorch on sample input.

    Parameters
    ----------
    onnx_path : str
        Path to .onnx model.
    sample_input : torch.Tensor or np.ndarray
        Sample image tensor (1xCxHxW).

    Returns
    -------
    bool
        True if outputs match within tolerance.
    """
    # TODO: implement — run PyTorch forward, run ONNX Runtime, compare heatmaps (e.g. L2 or max diff)
    pass


def main():
    """Load model, export to ONNX, validate."""
    # TODO: implement — config, load_model, export_to_onnx, validate_onnx
    pass


if __name__ == "__main__":
    main()
