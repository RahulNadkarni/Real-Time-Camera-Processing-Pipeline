"""
Shared model utilities: load/save checkpoints, parameter count, inference benchmark.
"""

from typing import Any, Dict, Optional

import torch
import time


def load_checkpoint(
    path: str,
    model: "torch.nn.Module",
    optimizer: Optional["torch.optim.Optimizer"] = None,
    device: Optional["torch.device"] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint from path into model (and optionally optimizer).

    Parameters
    ----------
    path : str
        Path to .pt file (state_dict or full checkpoint dict).
    model : torch.nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer, optional
        If provided and checkpoint contains optimizer state, load it.
    device : torch.device, optional
        Device to load tensors onto (e.g. when loading GPU checkpoint on CPU/MPS). Uses CPU if None.

    Returns
    -------
    dict
        Checkpoint dict (epoch, metrics, etc.) or empty dict if only state_dict saved.

    Side effects
    ------------
    Modifies model (and optionally optimizer) in place.
    """
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint
    model.load_state_dict(checkpoint, strict=True)
    return {}



def save_checkpoint(
    model: "torch.nn.Module",
    optimizer: Optional["torch.optim.Optimizer"],
    epoch: int,
    metrics: Optional[Dict[str, float]],
    path: str,
) -> None:
    """
    Save model state_dict and optionally optimizer, epoch, metrics to path.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer, optional
        Optimizer state to save.
    epoch : int
        Current epoch.
    metrics : dict, optional
        Metrics to store (e.g. val_loss, val_accuracy).
    path : str
        Output .pt path.

    Side effects
    ------------
    Writes file to disk.
    """
    if not path:
        raise ValueError("save_checkpoint requires a non-empty path.")
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def count_parameters(model: "torch.nn.Module") -> int:
    """
    Return total number of trainable parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model.

    Returns
    -------
    int
        Sum of parameter counts.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


def benchmark_inference_speed(
    model: "torch.nn.Module",
    input_size: tuple,
    device: "torch.device",
    n_runs: int = 100,
    warmup: int = 10,
) -> float:
    """
    Measure average inference time in milliseconds (forward pass only).

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval mode.
    input_size : tuple
        (C, H, W) or (N, C, H, W) for dummy input.
    device : torch.device
        Device (e.g. mps, cuda, cpu).
    n_runs : int
        Number of forward passes to average.
    warmup : int
        Number of warmup runs before timing.

    Returns
    -------
    float
        Average inference time in ms.
    """
    model.eval()
    model.to(device)
    size = input_size if len(input_size) == 4 else (1,) + tuple(input_size)
    dummy = torch.randn(size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()

    return (end - start) / n_runs * 1000.0