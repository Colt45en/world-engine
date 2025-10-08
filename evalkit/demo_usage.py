"""Demonstration script tying together evalkit utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .metrics import compute_classification_metrics
from .ab_compare import compare_models
from .robustness import evaluate_with_stressors
from .learning_curve import learning_curve
from .scalability import measure_latency


def run_demo(seed: int = 13) -> Dict[str, Any]:
    """Return a dictionary summarizing metrics, comparison, robustness."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=seed, stratify=data.target
    )

    clf_a = RandomForestClassifier(random_state=seed, n_estimators=50)
    clf_b = RandomForestClassifier(random_state=seed + 1, n_estimators=75)

    clf_a.fit(X_train, y_train)
    clf_b.fit(X_train, y_train)

    metrics = compute_classification_metrics(y_test, clf_a.predict(X_test))
    comparison = compare_models(y_test, clf_a.predict(X_test), clf_b.predict(X_test))

    def fit_predict(X_subset: np.ndarray, y_subset: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
        model = RandomForestClassifier(random_state=seed, n_estimators=50)
        model.fit(X_subset, y_subset)
        return model.predict(X_eval)

    lc = learning_curve(fit_predict, X_train, y_train, sizes=[10, 30, 60, len(X_train)])

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return clf_a.predict(batch)

    latency = measure_latency(predict_fn, X_test[:32])

    def evaluator(X_batch: np.ndarray, y_batch: np.ndarray) -> Dict[str, Any]:
        y_pred = clf_a.predict(X_batch)
        return compute_classification_metrics(y_batch, y_pred).to_dict()

    stress = evaluate_with_stressors(
        evaluator,
        X_test,
        y_test,
        stressors=[
            {"name": "missing_10", "type": "missing", "kwargs": {"fraction": 0.1}},
            {"name": "outliers_2", "type": "outliers", "kwargs": {"fraction": 0.02}},
        ],
    )

    return {
        "metrics": metrics.to_dict(),
        "comparison": comparison,
        "learning_curve": lc,
        "latency": latency,
        "stress_tests": stress,
    }


def main() -> None:
    report = run_demo()
    report_path = Path("evalkit_demo_report.json")
    report_path.write_text(
        json_dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Demo report saved to {report_path.resolve()}")


def json_dumps(payload: Dict[str, Any], indent: int = 2) -> str:
    """Local helper to avoid importing json multiple times for typing."""
    import json

    return json.dumps(payload, indent=indent, default=_json_default)


def _json_default(obj: Any):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    main()
