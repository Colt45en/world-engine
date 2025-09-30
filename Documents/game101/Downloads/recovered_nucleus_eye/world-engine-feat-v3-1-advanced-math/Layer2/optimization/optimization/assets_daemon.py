"""Tier-4 asset daemon bridging the AssetResourceBridge pybind module with NDJSON messaging."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

try:
    from assets import AssetBridge
except ImportError as exc:  # pragma: no cover - optional build step may be missing
    raise RuntimeError(
        "assets module not built. Run `pip install -v -e .` inside assets_bridge after compiling the extension."
    ) from exc

bridge = AssetBridge(2048.0)
bridge.start(30)


def emit(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def ok_cb(type_: str, id_: str) -> None:
    emit({
        "type": "ASSET_EVENT",
        "kind": "loaded",
        "payload": {"type": type_, "id": id_}
    })


def err_cb(type_: str, id_: str, reason: str) -> None:
    emit({
        "type": "ASSET_EVENT",
        "kind": "error",
        "payload": {"type": type_, "id": id_, "reason": reason}
    })


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:  # pragma: no cover - guard invalid input
        err_cb("_", "_", f"invalid JSON: {exc}")
        continue

    payload = message.get("payload", {})
    msg_type = message.get("type")

    try:
        if msg_type == "ASSET_REGISTER":
            bridge.register_base_path(payload["type"], payload["basePath"])
        elif msg_type == "ASSET_MEMORY":
            # Future extension: adjust memory budget dynamically
            # Could expose a setter on the bridge if AssetResourceManager supports it.
            pass
        elif msg_type == "ASSET_PRELOAD":
            items = [(item["type"], item["id"]) for item in payload.get("items", [])]
            if items:
                bridge.preload(items)
        elif msg_type == "ASSET_REQUEST":
            bridge.request(
                payload["type"],
                payload["id"],
                payload.get("priority", 0),
                ok_cb,
                err_cb,
            )
    except Exception as exc:  # pragma: no cover - convert to event output
        err_cb(payload.get("type", "_"), payload.get("id", "_"), str(exc))
