# Training scripts for classifier, saliency, and super-resolution models.
# Run from ml/: python training/train_classifier.py (or -m ml.training.train_classifier with PYTHONPATH=.)

from .train_classifier import main as train_classifier
from .train_saliency import main as train_saliency
from .train_superres import main as train_superres

__all__ = ["train_classifier", "train_saliency", "train_superres"]