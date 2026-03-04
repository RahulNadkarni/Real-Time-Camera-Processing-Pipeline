"""
Shared model utilities: load/save checkpoints, parameter count, inference benchmark.
"""

from pathlib import Path
from typing import Optional, Dict, Any

# TODO: add imports (torch)


def load_checkpoint(
    path: str,
    model: "torch.nn.Module",
    optimizer: Optional["torch.optim.Optimizer"] = None,
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

    Returns
    -------
    dict
        Checkpoint dict (epoch, metrics, etc.) or empty dict if only state_dict saved.

    Side effects
    ------------
    Modifies model (and optionally optimizer) in place.
    """
    # TODO: implement — torch.load, model.load_state_dict; if full checkpoint, load optimizer and return
    pass


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
    # TODO: implement — torch.save({"model": model.state_dict(), "optimizer": ..., "epoch": ..., "metrics": ...})
    pass


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
    # TODO: implement — sum(p.numel() for p in model.parameters() if p.requires_grad)
    pass


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
    # TODO: implement — create dummy tensor, model.to(device), warmup then torch.cuda.synchronize/mps sync,
    #       time n_runs forwards, return mean in ms
    pass
