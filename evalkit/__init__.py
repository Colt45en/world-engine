"""Evaluation kit utilities for classification experiments."""

from .metrics import MetricReport, compute_classification_metrics
from .ab_compare import compare_models, bootstrap_ci, mcnemar_on_accuracy, CI, McNemarResult
from .robustness import evaluate_with_stressors, add_missing, add_outliers, jitter_numeric
from .learning_curve import learning_curve
from .scalability import benchmark_infer

__all__ = [
    "MetricReport",
    "compute_classification_metrics",
    "compare_models",
    "bootstrap_ci",
    "mcnemar_on_accuracy",
    "CI",
    "McNemarResult",
    "evaluate_with_stressors",
    "add_missing",
    "add_outliers",
    "jitter_numeric",
    "learning_curve",
    "benchmark_infer",
]
