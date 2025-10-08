"""
Multi-Engine Integration Daemon - Unified access to all C++ engines
==================================================================

Provides Python bindings and NDJSON messaging for:
- LoggingEngine: Structured logging and event analysis
- TimekeepingEngine: Precise timing and scheduling
- PredictionEngine: Pattern recognition and forecasting
- StateManagementEngine: Complex state coordination
- PerformanceMonitorEngine: System metrics and optimization
- AssetResourceBridge: Asset management (existing)
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import queue
import traceback

try:
    # Import all C++ engine bindings
    from logging_engine import LoggingCore, LogLevel, LogFilter, LogMetrics
    from timekeeping_engine import TimekeepingCore, TimerType, ClockType
    from prediction_engine import PredictionCore, ForecastModel, PatternType
    from state_management import StateStore, ConflictResolution
    from performance_monitor import PerformanceMonitorCore
    from assets import AssetBridge  # Existing
except ImportError as exc:
    raise RuntimeError(
        "Engine modules not built. Run compilation for all engines."
    ) from exc

@dataclass
class EngineMessage:
    """Unified message format for all engines"""
    engine: str  # "logging", "timing", "prediction", "state", "performance", "assets"
    operation: str
    payload: Dict[str, Any]
    message_id: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class EngineResponse:
    """Unified response format"""
    engine: str
    operation: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    message_id: Optional[str] = None

class MultiEngineCore:
    """Core coordinator for all engines"""

    def __init__(self):
        # Initialize all engines
        self.logging_core = LoggingCore.instance()
        self.timing_core = TimekeepingCore.instance()
        self.prediction_core = PredictionCore.instance()
        self.state_store = StateStore()
        self.performance_monitor = PerformanceMonitorCore.instance()
        self.asset_bridge = AssetBridge(2048.0)  # 2GB memory limit

        # Message processing
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        self.worker_thread = None

        # Callbacks for real-time events
        self.event_callbacks = {
            'logging': [],
            'timing': [],
            'prediction': [],
            'state': [],
            'performance': [],
            'assets': []
        }

        self._setup_engine_callbacks()

    def _setup_engine_callbacks(self):
        """Setup callbacks from engines for real-time events"""
        # Logging engine callbacks
        def on_log_event(event):
            self.emit_event('logging', 'event', {
                'level': str(event.level),
                'category': event.category,
                'message': event.message,
                'timestamp': event.timestamp.isoformat()
            })

        # Timing engine callbacks
        def on_timer_fired(timer_id, name):
            self.emit_event('timing', 'timer_fired', {
                'timer_id': timer_id,
                'name': name,
                'timestamp': datetime.now().isoformat()
            })

        def on_task_executed(task_id, name):
            self.emit_event('timing', 'task_executed', {
                'task_id': task_id,
                'name': name,
                'timestamp': datetime.now().isoformat()
            })

        # Prediction engine callbacks
        def on_prediction_generated(series_name, prediction):
            self.emit_event('prediction', 'prediction_generated', {
                'series_name': series_name,
                'predicted_value': prediction.predicted_value,
                'confidence': prediction.confidence,
                'timestamp': datetime.now().isoformat()
            })

        def on_anomaly_detected(series_name, anomaly):
            self.emit_event('prediction', 'anomaly_detected', {
                'series_name': series_name,
                'anomaly_score': anomaly.anomaly_score,
                'anomaly_type': anomaly.anomaly_type,
                'timestamp': datetime.now().isoformat()
            })

        # Register callbacks with engines
        try:
            self.timing_core.on_timer_fired(on_timer_fired)
            self.timing_core.on_task_executed(on_task_executed)
            self.prediction_core.on_prediction_generated(on_prediction_generated)
            self.prediction_core.on_anomaly_detected(on_anomaly_detected)
        except Exception as e:
            # Some engines might not support callbacks yet
            self.logging_core.log(LogLevel.WARN, "setup", f"Could not setup callbacks: {e}")

    def emit_event(self, engine: str, event_type: str, data: Dict[str, Any]):
        """Emit real-time event to stdout"""
        event_message = {
            "type": "ENGINE_EVENT",
            "engine": engine,
            "event_type": event_type,
            "payload": data,
            "timestamp": datetime.now().isoformat()
        }
        sys.stdout.write(json.dumps(event_message) + "\n")
        sys.stdout.flush()

    def start(self):
        """Start the multi-engine daemon"""
        self.running = True

        # Start all engines
        self.timing_core.start()
        self.performance_monitor.start_all_monitoring()
        self.asset_bridge.start(30)  # 30 Hz

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        self.logging_core.log(LogLevel.INFO, "daemon", "Multi-engine daemon started")

    def stop(self):
        """Stop the multi-engine daemon"""
        self.running = False

        # Stop all engines
        self.timing_core.stop()
        self.performance_monitor.stop_all_monitoring()
        self.asset_bridge.stop()

        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

        self.logging_core.log(LogLevel.INFO, "daemon", "Multi-engine daemon stopped")

    def _worker_loop(self):
        """Main worker loop for processing messages"""
        while self.running:
            try:
                # Check for incoming messages
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    response = self._process_message(message)
                    self.response_queue.put(response)

                time.sleep(0.001)  # Small sleep to prevent busy waiting

            except Exception as e:
                self.logging_core.log(LogLevel.ERROR, "worker", f"Worker error: {e}")
                traceback.print_exc()

    def _process_message(self, message: EngineMessage) -> EngineResponse:
        """Process incoming message and route to appropriate engine"""
        try:
            if message.engine == "logging":
                return self._handle_logging_message(message)
            elif message.engine == "timing":
                return self._handle_timing_message(message)
            elif message.engine == "prediction":
                return self._handle_prediction_message(message)
            elif message.engine == "state":
                return self._handle_state_message(message)
            elif message.engine == "performance":
                return self._handle_performance_message(message)
            elif message.engine == "assets":
                return self._handle_assets_message(message)
            else:
                return EngineResponse(
                    engine=message.engine,
                    operation=message.operation,
                    success=False,
                    error=f"Unknown engine: {message.engine}",
                    message_id=message.message_id
                )
        except Exception as e:
            return EngineResponse(
                engine=message.engine,
                operation=message.operation,
                success=False,
                error=str(e),
                message_id=message.message_id
            )

    def _handle_logging_message(self, message: EngineMessage) -> EngineResponse:
        """Handle logging engine operations"""
        op = message.operation
        payload = message.payload

        if op == "log":
            level = LogLevel[payload.get("level", "INFO")]
            self.logging_core.log(
                level,
                payload.get("category", "default"),
                payload.get("message", ""),
                payload.get("file", ""),
                payload.get("line", 0),
                payload.get("metadata", {})
            )
            return EngineResponse(message.engine, op, True, message_id=message.message_id)

        elif op == "get_metrics":
            metrics = self.logging_core.get_metrics()
            return EngineResponse(
                message.engine, op, True,
                result={
                    "total_events": metrics.total_events,
                    "events_by_level": {str(k): v for k, v in metrics.events_by_level.items()},
                    "events_by_category": metrics.events_by_category
                },
                message_id=message.message_id
            )

        elif op == "search_events":
            query = payload.get("query", "")
            events = self.logging_core.search_events(query)
            return EngineResponse(
                message.engine, op, True,
                result=[{
                    "timestamp": event.timestamp.isoformat(),
                    "level": str(event.level),
                    "category": event.category,
                    "message": event.message
                } for event in events],
                message_id=message.message_id
            )

        return EngineResponse(message.engine, op, False, error="Unknown logging operation")

    def _handle_timing_message(self, message: EngineMessage) -> EngineResponse:
        """Handle timing engine operations"""
        op = message.operation
        payload = message.payload

        if op == "create_timer":
            timer_id = self.timing_core.create_timer(
                payload["name"],
                payload["interval_ns"],
                lambda: self.emit_event("timing", "timer_callback", {"name": payload["name"]}),
                TimerType[payload.get("type", "REPEATING")]
            )
            return EngineResponse(
                message.engine, op, True,
                result={"timer_id": timer_id},
                message_id=message.message_id
            )

        elif op == "schedule_task":
            task_id = self.timing_core.schedule_task_in(
                payload["name"],
                payload["delay_ns"],
                lambda: self.emit_event("timing", "task_callback", {"name": payload["name"]}),
                payload.get("priority", 0)
            )
            return EngineResponse(
                message.engine, op, True,
                result={"task_id": task_id},
                message_id=message.message_id
            )

        elif op == "get_metrics":
            metrics = self.timing_core.get_metrics()
            return EngineResponse(
                message.engine, op, True,
                result={
                    "uptime_ns": int(metrics.total_uptime.count()),
                    "total_tasks": metrics.total_tasks_executed,
                    "frame_rate": metrics.current_frame_rate
                },
                message_id=message.message_id
            )

        return EngineResponse(message.engine, op, False, error="Unknown timing operation")

    def _handle_prediction_message(self, message: EngineMessage) -> EngineResponse:
        """Handle prediction engine operations"""
        op = message.operation
        payload = message.payload

        if op == "create_series":
            self.prediction_core.create_time_series(payload["name"])
            return EngineResponse(message.engine, op, True, message_id=message.message_id)

        elif op == "add_data_point":
            # Convert payload to DataPoint structure
            data_point = {
                "timestamp": datetime.fromisoformat(payload["timestamp"]),
                "value": payload["value"],
                "category": payload.get("category", ""),
                "features": payload.get("features", {}),
                "metadata": payload.get("metadata", {})
            }
            self.prediction_core.add_data_point(payload["series_name"], data_point)
            return EngineResponse(message.engine, op, True, message_id=message.message_id)

        elif op == "get_prediction":
            prediction = self.prediction_core.get_latest_prediction(
                payload["series_name"],
                ForecastModel[payload.get("model", "ENSEMBLE")]
            )
            return EngineResponse(
                message.engine, op, True,
                result={
                    "predicted_value": prediction.predicted_value,
                    "confidence": prediction.confidence,
                    "lower_bound": prediction.lower_bound,
                    "upper_bound": prediction.upper_bound,
                    "model_used": prediction.model_used
                },
                message_id=message.message_id
            )

        elif op == "detect_anomalies":
            anomalies = self.prediction_core.get_recent_anomalies(payload["series_name"])
            return EngineResponse(
                message.engine, op, True,
                result=[{
                    "anomaly_score": anomaly.anomaly_score,
                    "anomaly_type": anomaly.anomaly_type,
                    "detection_method": anomaly.detection_method,
                    "expected_value": anomaly.expected_value
                } for anomaly in anomalies],
                message_id=message.message_id
            )

        return EngineResponse(message.engine, op, False, error="Unknown prediction operation")

    def _handle_state_message(self, message: EngineMessage) -> EngineResponse:
        """Handle state management operations"""
        op = message.operation
        payload = message.payload

        if op == "set_state":
            success = self.state_store.set_state(
                payload["id"],
                payload["value"],
                payload.get("source", "daemon")
            )
            return EngineResponse(
                message.engine, op, success,
                message_id=message.message_id
            )

        elif op == "get_state":
            value = self.state_store.get_state(payload["id"])
            return EngineResponse(
                message.engine, op, True,
                result={"value": value} if value else None,
                message_id=message.message_id
            )

        elif op == "get_state_history":
            history = self.state_store.get_state_history(payload["id"])
            return EngineResponse(
                message.engine, op, True,
                result=[{
                    "version": change.change_version,
                    "timestamp": change.change_time.isoformat(),
                    "change_type": str(change.change_type),
                    "actor": change.actor
                } for change in history],
                message_id=message.message_id
            )

        elif op == "create_snapshot":
            version = self.state_store.create_snapshot(
                payload.get("name", ""),
                payload.get("description", "")
            )
            return EngineResponse(
                message.engine, op, True,
                result={"snapshot_version": version},
                message_id=message.message_id
            )

        return EngineResponse(message.engine, op, False, error="Unknown state operation")

    def _handle_performance_message(self, message: EngineMessage) -> EngineResponse:
        """Handle performance monitoring operations"""
        op = message.operation
        payload = message.payload

        if op == "get_system_metrics":
            metrics = self.performance_monitor.get_system_monitor().get_current_metrics()
            return EngineResponse(
                message.engine, op, True,
                result={
                    "cpu_usage": metrics.cpu_usage_percent,
                    "memory_usage": metrics.memory_usage_percent,
                    "memory_total": metrics.memory_total_bytes,
                    "memory_used": metrics.memory_used_bytes,
                    "load_average": metrics.load_average_1min,
                    "collection_time": metrics.collection_time.isoformat()
                },
                message_id=message.message_id
            )

        elif op == "start_profiling":
            self.performance_monitor.get_profiler().start_profiling()
            return EngineResponse(message.engine, op, True, message_id=message.message_id)

        elif op == "stop_profiling":
            self.performance_monitor.get_profiler().stop_profiling()
            return EngineResponse(message.engine, op, True, message_id=message.message_id)

        elif op == "get_performance_report":
            report = self.performance_monitor.generate_comprehensive_report()
            return EngineResponse(
                message.engine, op, True,
                result={
                    "system_metrics": {
                        "cpu_usage": report.system_metrics.cpu_usage_percent,
                        "memory_usage": report.system_metrics.memory_usage_percent,
                    },
                    "app_metrics": {
                        "memory_rss": report.app_metrics.process_memory_rss_bytes,
                        "cpu_percent": report.app_metrics.process_cpu_percent,
                    },
                    "alerts": len(report.alerts),
                    "suggestions": len(report.suggestions),
                    "report_time": report.report_time.isoformat()
                },
                message_id=message.message_id
            )

        return EngineResponse(message.engine, op, False, error="Unknown performance operation")

    def _handle_assets_message(self, message: EngineMessage) -> EngineResponse:
        """Handle asset management operations (existing)"""
        # This uses the existing asset bridge logic
        op = message.operation
        payload = message.payload

        if op == "register_base_path":
            self.asset_bridge.register_base_path(payload["type"], payload["path"])
            return EngineResponse(message.engine, op, True, message_id=message.message_id)

        elif op == "request_asset":
            self.asset_bridge.request(
                payload["type"],
                payload["id"],
                payload.get("priority", 0),
                lambda t, i: self.emit_event("assets", "loaded", {"type": t, "id": i}),
                lambda t, i, r: self.emit_event("assets", "error", {"type": t, "id": i, "reason": r})
            )
            return EngineResponse(message.engine, op, True, message_id=message.message_id)

        return EngineResponse(message.engine, op, False, error="Unknown assets operation")

# Global daemon instance
daemon = MultiEngineCore()

def emit(message: Dict[str, Any]) -> None:
    """Emit message to stdout"""
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def main():
    """Main daemon loop"""
    daemon.start()

    try:
        # Process incoming messages from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                message_data = json.loads(line)
                message = EngineMessage(
                    engine=message_data["engine"],
                    operation=message_data["operation"],
                    payload=message_data.get("payload", {}),
                    message_id=message_data.get("message_id"),
                    timestamp=message_data.get("timestamp")
                )

                daemon.message_queue.put(message)

                # Get response
                try:
                    response = daemon.response_queue.get(timeout=5.0)
                    emit({
                        "type": "ENGINE_RESPONSE",
                        "engine": response.engine,
                        "operation": response.operation,
                        "success": response.success,
                        "result": response.result,
                        "error": response.error,
                        "message_id": response.message_id
                    })
                except queue.Empty:
                    emit({
                        "type": "ENGINE_RESPONSE",
                        "engine": message.engine,
                        "operation": message.operation,
                        "success": False,
                        "error": "Operation timeout",
                        "message_id": message.message_id
                    })

            except json.JSONDecodeError as exc:
                emit({
                    "type": "ENGINE_RESPONSE",
                    "success": False,
                    "error": f"Invalid JSON: {exc}"
                })
            except Exception as exc:
                emit({
                    "type": "ENGINE_RESPONSE",
                    "success": False,
                    "error": f"Processing error: {exc}"
                })

    finally:
        daemon.stop()

if __name__ == "__main__":
    main()
