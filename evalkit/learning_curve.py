"""Learning curve utilities for bias/variance analysis."""
from __future__ import annotations

from typing import Callable, Dict, Any, List

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def learning_curve(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    sizes: List[int],
    repeats: int = 5,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Return accuracy mean ± std for each train size on a fixed test split."""
    rng = np.random.default_rng(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    output: List[Dict[str, Any]] = []
    for size in sizes:
        accs = []
        m = min(size, len(X_train))
        for _ in range(repeats):
            idx = rng.choice(len(X_train), size=m, replace=False)
            y_pred = fit_predict(X_train[idx], y_train[idx], X_test)
            accs.append(float(accuracy_score(y_test, y_pred)))
        arr = np.asarray(accs, dtype=float)
        output.append(
            {
                "train_size": int(size),
                "acc_mean": float(arr.mean()),
                "acc_std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            }
        )
    return output
