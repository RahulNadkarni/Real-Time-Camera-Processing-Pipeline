# Evaluation suites for classifier, saliency, and super-resolution.

from .eval_classifier import run_evaluation as run_classifier_evaluation
from .eval_saliency import run_evaluation as run_saliency_evaluation
from .eval_superres import run_evaluation as run_superres_evaluation

__all__ = ["run_classifier_evaluation", "run_saliency_evaluation", "run_superres_evaluation"]   