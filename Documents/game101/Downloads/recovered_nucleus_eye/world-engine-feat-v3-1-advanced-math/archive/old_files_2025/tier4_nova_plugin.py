"""Tier-4 Nova add-in providing clustering, forecasting, and IDE NDJSON emission."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math
import time

try:  # NumPy is optional; degrade gracefully if unavailable
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore

T4_STREAM = Path("run.ndjson")
MEMO_FILE = Path("nova_history.json")


@dataclass
class T4Metrics:
    step: int
    timestamp: str
    n_clusters: int
    explained_variance: float
    inertia: float
    ewma_ratio: float
    trend_slope: float
    latent_health: float


@dataclass
class Tier4Nova:
    nova: Any
    agent: Optional[Any] = None
    history: List[T4Metrics] = field(default_factory=list)
    k: int = 8
    seed: int = 7
    max_iter: int = 20
    batch_size: int = 2048
    ewma_alpha: float = 0.2

    def __post_init__(self) -> None:
        if _np is not None:
            _np.random.seed(self.seed)
        if not hasattr(self.nova, "compression_ratios"):
            self.nova.compression_ratios = []  # type: ignore[attr-defined]
        self._load_history()

    def step(self, step_id: int) -> None:
        inertia, explained = self._cluster_memory()
        ewma_ratio, slope = self._forecast()
        latent_health = float(
            max(0.0, min(1.0, 0.5 * explained + 0.5 * math.tanh(ewma_ratio + 0.5 * slope)))
        )

        metrics = T4Metrics(
            step=step_id,
            timestamp=datetime.utcnow().isoformat(),
            n_clusters=self.k,
            explained_variance=explained,
            inertia=inertia,
            ewma_ratio=ewma_ratio,
            trend_slope=slope,
            latent_health=latent_health,
        )
        self.history.append(metrics)
        self._save_history()

        self._emit_event("UP", "NOVA/CLUSTER", {
            "step": step_id,
            "k": self.k,
            "inertia": inertia,
            "explained_variance": explained,
        })
        self._emit_event("ST", "NOVA/FORECAST", {
            "ewma_ratio": ewma_ratio,
            "trend_slope": slope,
        })
        self._emit_event("STATE", "NOVA/HEALTH", {
            "latent_health": latent_health,
        })

        if self.agent and hasattr(self.agent, "log_imprint"):
            try:
                self.agent.log_imprint(
                    event=f"Tier4Nova step {step_id}",
                    analysis=(
                        f"Clusters={self.k}, EV={explained:.3f}, "
                        f"EWMA={ewma_ratio:.3f}, slope={slope:.3f}"
                    ),
                    visibleInfra={"clusters": self.k, "explained_variance": explained},
                    unseenInfra={"inertia": inertia, "health": latent_health},
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _cluster_memory(self) -> tuple[float, float]:
        raw = getattr(self.nova, "original_data", None)
        if raw is None:
            raise ValueError("NovaIntelligence instance missing 'original_data'.")

        if _np is None:
            self.nova.compression_ratios.append(0.0)
            return 0.0, 0.0

        x = _np.asarray(raw, dtype=_np.float32).reshape(-1, 1)
        n = x.shape[0]
        if n == 0:
            return 0.0, 0.0

        k = int(max(2, min(self.k, min(256, n))))
        idx = _np.random.choice(n, size=k, replace=False)
        centroids = x[idx].copy()
        counts = _np.zeros(k, dtype=_np.int64)

        for _ in range(max(1, self.max_iter)):
            start = int(_np.random.randint(0, n))
            batch = x[start : start + self.batch_size]
            if batch.size == 0:
                batch = x
            distances = (batch - centroids.T) ** 2
            assign = _np.argmin(distances, axis=1)
            for j in range(k):
                mask = assign == j
                if _np.any(mask):
                    counts[j] += int(mask.sum())
                    lr = 1.0 / max(1, counts[j])
                    centroids[j] = (1.0 - lr) * centroids[j] + lr * batch[mask].mean()

        nearest = _np.argmin((x - centroids.T) ** 2, axis=1)
        d2_full = x - centroids[nearest]
        inertia = float(_np.sum(d2_full ** 2))
        total_var = float(_np.var(x) * n) + 1e-9
        explained = float(max(0.0, min(1.0, 1.0 - inertia / total_var)))

        self.nova.compression_ratios.append(explained)
        return inertia, explained

    def _forecast(self) -> tuple[float, float]:
        ratios = list(getattr(self.nova, "compression_ratios", []))
        if not ratios:
            return 0.0, 0.0

        ewma = ratios[0]
        for value in ratios[1:]:
            ewma = self.ewma_alpha * value + (1.0 - self.ewma_alpha) * ewma

        slope = 0.0
        if _np is not None and len(ratios) >= 3:
            x = _np.arange(len(ratios), dtype=_np.float32)
            y = _np.asarray(ratios, dtype=_np.float32)
            slope = float(_np.polyfit(x, y, 1)[0])

        omega = getattr(self.nova, "omega", None)
        if omega and hasattr(omega, "predict_future"):
            try:
                omega.predict_future(ratios)
            except Exception:
                pass

        return float(ewma), slope

    def _emit_event(self, op: str, scope: str, payload: Dict[str, Any]) -> None:
        event = {
            "t": time.time(),
            "kind": "T4_EVENT",
            "op": op,
            "scope": scope,
            "payload": payload,
        }
        try:
            with T4_STREAM.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _load_history(self) -> None:
        if not MEMO_FILE.exists():
            return
        try:
            data = json.loads(MEMO_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        for row in data.get("history", []):
            try:
                self.history.append(T4Metrics(**row))
            except TypeError:
                continue
        stored = data.get("compression_ratios")
        if isinstance(stored, list):
            self.nova.compression_ratios = stored

    def _save_history(self) -> None:
        data = {
            "history": [vars(metric) for metric in self.history[-500:]],
            "compression_ratios": list(getattr(self.nova, "compression_ratios", [])),
        }
        try:
            MEMO_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
