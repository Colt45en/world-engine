"""Classification metrics helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


@dataclass
class MetricReport:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confmat: np.ndarray
    labels: List[Any]
    per_class: Dict[Any, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": _as_float(self.accuracy),
            "precision_macro": _as_float(self.precision_macro),
            "recall_macro": _as_float(self.recall_macro),
            "f1_macro": _as_float(self.f1_macro),
            "precision_micro": _as_float(self.precision_micro),
            "recall_micro": _as_float(self.recall_micro),
            "f1_micro": _as_float(self.f1_micro),
            "precision_weighted": _as_float(self.precision_weighted),
            "recall_weighted": _as_float(self.recall_weighted),
            "f1_weighted": _as_float(self.f1_weighted),
            "confmat": self.confmat.astype(float).tolist(),
            "labels": list(self.labels),
            "per_class": {
                str(label): {
                    "precision": _as_float(values["precision"]),
                    "recall": _as_float(values["recall"]),
                    "f1": _as_float(values["f1"]),
                }
                for label, values in self.per_class.items()
            },
        }


def compute_classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    labels: Optional[List[Any]] = None,
) -> MetricReport:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    if labels is None:
        labels = sorted(list(set(y_true_arr) | set(y_pred_arr)), key=str)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    def prf(avg: str) -> Tuple[float, float, float]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average=avg,
            zero_division=0,
        )
        return float(p), float(r), float(f1)

    p_mac, r_mac, f1_mac = prf("macro")
    p_mic, r_mic, f1_mic = prf("micro")
    p_w, r_w, f1_w = prf("weighted")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    cls_p, cls_r, cls_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class = {
        str(lbl): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
        for lbl, p, r, f1 in zip(labels, cls_p, cls_r, cls_f1)
    }

    return MetricReport(
        accuracy=acc,
        precision_macro=p_mac,
        recall_macro=r_mac,
        f1_macro=f1_mac,
        precision_micro=p_mic,
        recall_micro=r_mic,
        f1_micro=f1_mic,
        precision_weighted=p_w,
        recall_weighted=r_w,
        f1_weighted=f1_w,
        confmat=cm,
        labels=list(labels),
        per_class=per_class,
    )


def _as_float(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
