"""FastAPI service to expose the Fractal Intelligence Engine for the dashboard UI."""
from __future__ import annotations

import base64
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

# Ensure the services directory is importable when executed directly.
import sys

for candidate in (str(CURRENT_DIR), str(PARENT_DIR)):
    if candidate not in sys.path:
        sys.path.append(candidate)

from fractal_intelligence_engine import FractalIntelligenceEngine  # type: ignore  # noqa: E402


class StartRequest(BaseModel):
    max_iterations: Optional[int] = Field(default=None, ge=1, le=500)
    delay_seconds: Optional[float] = Field(default=None, ge=0.0, le=5.0)


class ForceOptimizationResponse(BaseModel):
    algorithm: str
    compression_ratio: Optional[float]
    compressed_size: Optional[int]
    knowledge_size: int


class CompressionTestResponse(BaseModel):
    original_size: int
    compressed_size: int
    ratio: float


class InjectPainRequest(BaseModel):
    text: Optional[str] = None
    severity: Optional[int] = Field(default=None, ge=1, le=10)


class EngineController:
    def __init__(self) -> None:
        self.engine = FractalIntelligenceEngine()
        self.running = False
        self.delay_seconds = 0.5
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------
    def start(self, *, max_iterations: Optional[int] = None, delay_seconds: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            if self.running:
                return self._snapshot()

            if self.engine.state.get("completed"):
                raise HTTPException(status_code=400, detail="Engine has completed. Reset before starting again.")

            if max_iterations is not None:
                self.engine.state["max_iterations"] = max_iterations
            if delay_seconds is not None:
                self.delay_seconds = delay_seconds

            self.running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return self._snapshot()

    def pause(self) -> Dict[str, Any]:
        self.running = False
        return self._snapshot()

    def reset(self) -> Dict[str, Any]:
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.1)
        self.engine.reset()
        self._logs.clear()
        return self._snapshot()

    def force_optimization(self) -> ForceOptimizationResponse:
        with self._lock:
            self.engine.nick.algorithm = random.choice(["zlib", "simple", "hybrid"])
            metrics = self.engine._build_metrics_snapshot(  # pylint: disable=protected-access
                insight=self.engine.state.get("last_metrics", {}).get("insight"),
                chaos_adjustment="Manual optimization trigger",
                compressed_size=len(self.engine.nick.compress(self.engine.state["knowledge"])),
                pain_summary=self.engine.get_pain_insights(),
                compressed_blob=None,
            )
            self._store_metrics(metrics, include_completed_flag=False)
            return ForceOptimizationResponse(
                algorithm=metrics["algorithm"],
                compression_ratio=metrics["compression_ratio"],
                compressed_size=metrics["compressed_size"],
                knowledge_size=metrics["knowledge_size"],
            )

    def test_compression(self) -> CompressionTestResponse:
        sample_text = [f"payload-{i}-{random.random()}" for i in range(500)]
        original_size = len("".join(sample_text).encode("utf-8"))
        compressed = self.engine.nick.compress({"sample": sample_text})
        compressed_size = len(compressed)
        ratio = 0.0 if original_size == 0 else 1 - (compressed_size / original_size)
        return CompressionTestResponse(
            original_size=original_size,
            compressed_size=compressed_size,
            ratio=ratio,
        )

    def inject_pain(self, *, text: Optional[str], severity: Optional[int]) -> bool:
        payload_text = text or "Manual dashboard injection"
        return self.engine.inject_pain_event(payload_text, severity=severity or random.randint(1, 10))

    def refresh_pain(self) -> Dict[str, Any]:
        summary = self.engine.get_pain_insights()
        return summary or {"clusters": [], "totalEvents": 0}

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        while True:
            if not self.running:
                break

            metrics = self.engine.advance()
            self._store_metrics(metrics)

            if metrics.get("completed"):
                self.running = False
                break

            time.sleep(self.delay_seconds)

    def _store_metrics(self, metrics: Dict[str, Any], *, include_completed_flag: bool = True) -> None:
        snapshot = metrics.copy()
        if include_completed_flag:
            snapshot["completed"] = bool(metrics.get("completed"))

        insight = snapshot.get("insight")
        chaos = snapshot.get("chaos_adjustment")
        log_entry = {
            "timestamp": snapshot.get("timestamp"),
            "iteration": snapshot.get("iteration"),
            "insight": insight,
            "chaos_adjustment": chaos,
        }

        with self._lock:
            self.engine.state["last_metrics"] = snapshot
            if insight:
                self._logs.append(log_entry)
            self._logs = self._logs[-250:]

    def _snapshot(self) -> Dict[str, Any]:
        with self._lock:
            metrics = self.engine.state.get("last_metrics")
            if not metrics:
                metrics = self.engine._build_metrics_snapshot(completed=self.engine.state.get("completed", False))  # pylint: disable=protected-access

            response = {
                "running": self.running,
                "completed": bool(self.engine.state.get("completed")),
                "maxIterations": self.engine.state.get("max_iterations"),
                "delaySeconds": self.delay_seconds,
                "metrics": self._serialize_metrics(metrics),
                "logs": list(self._logs),
            }
            return response

    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {k: v for k, v in metrics.items() if k != "compressed_blob"}
        blob = metrics.get("compressed_blob")
        if blob is not None:
            cleaned["compressed_blob_b64"] = base64.b64encode(blob).decode("ascii")
        return cleaned


controller = EngineController()

app = FastAPI(title="Fractal Intelligence Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "running": controller.running, "completed": controller.engine.state.get("completed")}


@app.get("/engine/state")
def get_state() -> Dict[str, Any]:
    return controller._snapshot()  # pylint: disable=protected-access


@app.post("/engine/start")
def start_engine(payload: StartRequest = Body(default_factory=StartRequest)) -> Dict[str, Any]:
    return controller.start(max_iterations=payload.max_iterations, delay_seconds=payload.delay_seconds)


@app.post("/engine/pause")
def pause_engine() -> Dict[str, Any]:
    return controller.pause()


@app.post("/engine/reset")
def reset_engine() -> Dict[str, Any]:
    return controller.reset()


@app.post("/engine/force-optimization", response_model=ForceOptimizationResponse)
def force_optimization() -> ForceOptimizationResponse:
    return controller.force_optimization()


@app.post("/engine/test-compression", response_model=CompressionTestResponse)
def test_compression() -> CompressionTestResponse:
    return controller.test_compression()


@app.post("/pain/inject")
def inject_pain(payload: InjectPainRequest = Body(default_factory=InjectPainRequest)) -> Dict[str, Any]:
    success = controller.inject_pain(text=payload.text, severity=payload.severity)
    if not success:
        raise HTTPException(status_code=502, detail="Failed to reach downstream pain API")
    return {"status": "accepted"}


@app.get("/pain/summary")
def pain_summary() -> Dict[str, Any]:
    return controller.refresh_pain()


@app.get("/engine/logs")
def engine_logs() -> Dict[str, Any]:
    snapshot = controller._snapshot()  # pylint: disable=protected-access
    return {"logs": snapshot["logs"]}


def run(host: str = "127.0.0.1", port: int = 8600) -> None:
    """Helper for launching with `python fractal_dashboard_service.py`."""
    uvicorn.run("fractal_dashboard_service:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
