"""Order-preserving calibration utilities."""

from __future__ import annotations

from typing import Dict, List
import numpy as np

try:  # pragma: no cover - optional dependency
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

__all__ = ["IsotonicCalibrator"]


class IsotonicCalibrator:
    """Order-preserving calibration along a given order of items."""

    def __init__(self, y_min: float = -1.0, y_max: float = 1.0) -> None:
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def fit_transform(self, order: List[str], scores: Dict[str, float]) -> Dict[str, float]:
        if len(order) <= 1:
            return {k: float(np.clip(v, self.y_min, self.y_max)) for k, v in scores.items()}

        x_axis = np.arange(len(order), dtype=np.float32)
        y = np.array([scores.get(w, 0.0) for w in order], dtype=np.float32)

        if _HAS_SK:
            ir = IsotonicRegression(
                y_min=self.y_min,
                y_max=self.y_max,
                increasing=True,
                out_of_bounds="clip",
            )
            y_fit = ir.fit_transform(x_axis, y)
        else:
            y_fit = self._pav(y)
            y_fit = np.clip(y_fit, self.y_min, self.y_max)

        out = dict(scores)
        for i, word in enumerate(order):
            out[word] = float(y_fit[i])
        return out

    @staticmethod
    def _pav(y: np.ndarray) -> np.ndarray:
        y = y.copy()
        n = len(y)
        w = np.ones(n, dtype=np.float64)
        i = 0
        while i < n - 1:
            if y[i] <= y[i + 1] + 1e-12:
                i += 1
            else:
                new_y = (w[i] * y[i] + w[i + 1] * y[i + 1]) / (w[i] + w[i + 1])
                y[i] = y[i + 1] = new_y
                w[i] += w[i + 1]
                j = i
                while j > 0 and y[j - 1] > y[j] + 1e-12:
                    new_y = (w[j - 1] * y[j - 1] + w[j] * y[j]) / (w[j - 1] + w[j])
                    y[j - 1] = y[j] = new_y
                    w[j - 1] += w[j]
                    j -= 1
                i = j
        return y
