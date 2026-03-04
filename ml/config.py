"""
Central configuration for the ML layer.
Dataclass covering device (MPS), model paths, training hyperparameters,
dataset paths, ONNX export settings, input sizes, and class names for all three models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    """Central config for training, export, and evaluation."""

    # Device: MPS (Metal Performance Shaders) for Apple Silicon
    device: str = "mps"

    # Model paths (checkpoints and ONNX outputs)
    classifier_checkpoint_path: str = "models/classifier.pt"
    classifier_onnx_path: str = "models/scene_classifier.onnx"
    saliency_checkpoint_path: str = "models/saliency.pt"
    saliency_onnx_path: str = "models/saliency.onnx"
    superres_checkpoint_path: str = "models/superres.pt"
    superres_onnx_path: str = "models/superres.onnx"

    # Training hyperparameters (shared defaults; override per script)
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    num_workers: int = 4

    # Dataset paths
    cifar10_path: str = "data/cifar10"
    salicon_data_path: str = "data/salicon"
    div2k_data_path: str = "data/div2k"

    # ONNX export settings
    onnx_opset_version: int = 14
    onnx_dynamic_axes: bool = False  # Set True for variable input size

    # Input image sizes per model
    classifier_input_size: Tuple[int, int] = (224, 224)
    saliency_input_size: Tuple[int, int] = (224, 224)
    superres_scale_factor: int = 2

    # Class names for scene classifier (CIFAR-10)
    classifier_class_names: List[str] = field(
        default_factory=lambda: [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    )
