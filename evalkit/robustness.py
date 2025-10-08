"""Robustness stressors for evaluating model stability."""
from __future__ import annotations

from typing import Dict, Any, Callable

import numpy as np

from .metrics import compute_classification_metrics


ArrayLike = np.ndarray


def add_missing(X: ArrayLike, rate: float, seed: int = 0) -> ArrayLike:
    rng = np.random.default_rng(seed)
    Xc = np.array(X, copy=True, dtype=float)
    mask = rng.random(Xc.shape) < rate
    Xc[mask] = np.nan
    return Xc


def add_outliers(X: ArrayLike, frac: float, scale: float = 10.0, seed: int = 0) -> ArrayLike:
    rng = np.random.default_rng(seed)
    Xc = np.array(X, copy=True, dtype=float)
    n = Xc.shape[0]
    k = max(1, int(frac * n)) if frac > 0 else 0
    if k == 0:
        return Xc
    idx = rng.choice(n, k, replace=False)
    noise = rng.normal(0, scale, size=Xc[idx].shape)
    Xc[idx] = Xc[idx] + noise
    return Xc


def jitter_numeric(X: ArrayLike, sigma: float = 0.01, seed: int = 0) -> ArrayLike:
    rng = np.random.default_rng(seed)
    return np.array(X, copy=True, dtype=float) + rng.normal(0, sigma, size=np.shape(X))


def evaluate_with_stressors(
    fit_predict: Callable[[ArrayLike, np.ndarray, ArrayLike], np.ndarray],
    X_train: ArrayLike,
    y_train: np.ndarray,
    X_test: ArrayLike,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Run a sequence of data perturbations and collect metric reports."""

    def run(Xtr: ArrayLike, Xte: ArrayLike, name: str):
        y_pred = fit_predict(Xtr, y_train, Xte)
        report = compute_classification_metrics(y_test, y_pred)
        return name, report

    reports: Dict[str, Any] = {}
    key, rep = run(X_train, X_test, "clean")
    reports[key] = rep

    key, rep = run(add_missing(X_train, 0.05), add_missing(X_test, 0.05), "missing_5%")
    reports[key] = rep

    key, rep = run(add_outliers(X_train, 0.05, 8.0), X_test, "outliers_train_5%")
    reports[key] = rep

    key, rep = run(jitter_numeric(X_train, 0.02), jitter_numeric(X_test, 0.02), "jitter_2%")
    reports[key] = rep

    return reports
