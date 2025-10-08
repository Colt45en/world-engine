"""A/B comparison utilities for paired model evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Iterable, Optional, NamedTuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .metrics import compute_classification_metrics


@dataclass
class CI:
    mean: float
    lo: float
    hi: float


class McNemarResult(NamedTuple):
    n00: int
    n01: int
    n10: int
    n11: int
    p_value: Optional[float]


def bootstrap_ci(
    vec: Iterable[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 13,
) -> CI:
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(vec), dtype=float)
    if arr.size == 0:
        raise ValueError("vector must contain at least one value")

    boots = []
    n = arr.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(arr[idx].mean()))
    boots.sort()

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return CI(mean=float(arr.mean()), lo=lo, hi=hi)


def mcnemar_on_accuracy(
    y_true: Iterable[Any],
    y_pred_A: Iterable[Any],
    y_pred_B: Iterable[Any],
) -> McNemarResult:
    arr_true = np.asarray(list(y_true))
    arr_A = np.asarray(list(y_pred_A))
    arr_B = np.asarray(list(y_pred_B))

    if not (arr_true.size == arr_A.size == arr_B.size):
        raise ValueError("All inputs must have the same number of samples")

    correct_A = arr_A == arr_true
    correct_B = arr_B == arr_true

    n11 = int(np.sum(correct_A & correct_B))
    n00 = int(np.sum(~correct_A & ~correct_B))
    n10 = int(np.sum(correct_A & ~correct_B))
    n01 = int(np.sum(~correct_A & correct_B))

    p_val: Optional[float] = None
    try:
        from statsmodels.stats.contingency_tables import mcnemar

        table = [[n11, n10], [n01, n00]]
        exact = (n10 + n01) <= 25
        result = mcnemar(table, exact=exact, correction=True)
        p_val = float(result.pvalue)
    except Exception:  # pragma: no cover - optional dependency
        p_val = None

    return McNemarResult(n00=n00, n01=n01, n10=n10, n11=n11, p_value=p_val)


def compare_models(
    y_true: Iterable[Any],
    y_pred_A: Iterable[Any],
    y_pred_B: Iterable[Any],
    metric_fn: Optional[Callable[[Iterable[Any], Iterable[Any]], Any]] = None,
) -> Dict[str, Any]:
    """Paired comparison on the same dataset.

    Returns deltas and bootstrap confidence intervals for accuracy and macro F1.
    Optionally, `metric_fn` can provide additional per-model metrics that will be
    evaluated on both predictions.
    """
    arr_true = np.asarray(list(y_true))
    arr_A = np.asarray(list(y_pred_A))
    arr_B = np.asarray(list(y_pred_B))

    if not (arr_true.size == arr_A.size == arr_B.size):
        raise ValueError("All inputs must have the same number of samples")

    acc_A = float(accuracy_score(arr_true, arr_A))
    acc_B = float(accuracy_score(arr_true, arr_B))

    f1_A = float(f1_score(arr_true, arr_A, average="macro"))
    f1_B = float(f1_score(arr_true, arr_B, average="macro"))

    correct_A = (arr_A == arr_true).astype(float)
    correct_B = (arr_B == arr_true).astype(float)
    acc_delta_vec = correct_B - correct_A
    acc_ci = bootstrap_ci(acc_delta_vec)

    rng = np.random.default_rng(7)
    n_boot = 2000
    n = arr_true.size
    f1_deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_deltas.append(
            float(f1_score(arr_true[idx], arr_B[idx], average="macro")
                  - f1_score(arr_true[idx], arr_A[idx], average="macro"))
        )
    f1_deltas.sort()
    f1_ci = CI(
        mean=float(f1_B - f1_A),
        lo=float(np.quantile(f1_deltas, 0.025)),
        hi=float(np.quantile(f1_deltas, 0.975)),
    )

    results: Dict[str, Any] = {
        "acc_A": acc_A,
        "acc_B": acc_B,
        "acc_delta_CI": acc_ci,
        "f1_A": f1_A,
        "f1_B": f1_B,
        "f1_delta_CI": f1_ci,
        "mcnemar": mcnemar_on_accuracy(arr_true, arr_A, arr_B)._asdict(),
    }

    if metric_fn is not None:
        results["metric_A"] = metric_fn(arr_true, arr_A)
        results["metric_B"] = metric_fn(arr_true, arr_B)

    return results
