"""
Exports the trained super-resolution model to ONNX.
validate_onnx compares PSNR of PyTorch vs ONNX output on a sample image.
"""

from typing import Tuple

# TODO: add imports (torch, onnx, onnxruntime, config, numpy)


def load_model(checkpoint_path: str):
    """
    Load trained SR model (SRCNN or ESRGAN-tiny) from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint.

    Returns
    -------
    torch.nn.Module
        Model in eval mode.
    """
    # TODO: implement — build_srcnn or build_esrgan_tiny, load state_dict
    pass


def export_to_onnx(
    model, output_path: str, input_size: Tuple[int, int], scale_factor: int, opset_version: int = 14
):
    """
    Export SR model to ONNX.

    Parameters
    ----------
    model : torch.nn.Module
        Trained SR model.
    output_path : str
        Output .onnx path.
    input_size : Tuple[int, int]
        (H, W) of LR input (e.g. after downscale).
    scale_factor : int
        Scale factor (for documentation or dynamic shape).
    opset_version : int
        ONNX opset version.

    Side effects
    ------------
    Writes .onnx file.
    """
    # TODO: implement — torch.onnx.export; output shape may be scale_factor * H, scale_factor * W
    pass


def validate_onnx(onnx_path: str, sample_input) -> bool:
    """
    Compare PyTorch vs ONNX output on a sample LR image; check PSNR or max diff.

    Parameters
    ----------
    onnx_path : str
        Path to .onnx model.
    sample_input : torch.Tensor or np.ndarray
        Sample LR image (1xCxHxW).

    Returns
    -------
    bool
        True if PSNR between PyTorch and ONNX outputs is above threshold (e.g. > 40 dB) or diff small.
    """
    # TODO: implement — run PyTorch forward, run ONNX Runtime, compute PSNR between two outputs
    pass


def main():
    """Load model, export to ONNX, validate."""
    # TODO: implement — config, load_model, export_to_onnx, validate_onnx
    pass


if __name__ == "__main__":
    main()
