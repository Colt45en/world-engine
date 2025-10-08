
"""Performance Metrics Module

Simple utility for recording performance metrics and uptime.
"""
import time
from typing import Dict, Any


class PerformanceMetrics:
    def __init__(self, start_time: float | None = None) -> None:
        self.start_time = start_time or time.time()
        self.metrics: Dict[str, Any] = {}

    def record_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_duration(self) -> float:
        """Return seconds since the metrics object was started."""
        return time.time() - self.start_time
