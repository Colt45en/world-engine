#!/usr/bin/env python3
"""Knowledge Vault Telemetry Bridge

Periodically inspects the Knowledge Vault database and forwards notable changes
to the NEXUS vector network telemetry WebSocket so the control center can show
live knowledge activity.

Usage:
    python services/knowledge_vault_telemetry_bridge.py \
        --db ./knowledge_vault.db --endpoint ws://localhost:8701 --interval 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import websockets

logger = logging.getLogger("knowledge_vault_telemetry_bridge")

TelemetryDetails = Dict[str, str]


class VaultTelemetryBridge:
    def __init__(
        self,
        db_path: Path,
        endpoint: str = "ws://localhost:8701",
        poll_interval: float = 6.0,
    ) -> None:
        self.db_path = db_path
        self.endpoint = endpoint
        self.poll_interval = poll_interval
        self._last_entry_rowid: int = 0
        self._last_connection_rowid: int = 0

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def _emit(self, category: str, message: str, details: TelemetryDetails) -> None:
        payload = {
            "type": "telemetry_event_push",
            "event": {
                "category": category,
                "message": message,
                "details": details,
                "origin": "knowledge-vault"
            }
        }

        try:
            async with websockets.connect(self.endpoint, ping_interval=None) as websocket:
                await websocket.send(json.dumps(payload))
        except Exception as exc:  # pragma: no cover - bridge must be fault tolerant
            logger.debug("Telemetry emit failed: %s", exc)

    def _format_entry_details(self, row: sqlite3.Row) -> TelemetryDetails:
        return {
            "vault": row["source_system"] or "unknown",
            "status": row["category"],
            "payload": row["id"][:12],
            "emojiMemory": "🧠📡"
        }

    async def _broadcast_new_entries(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute(
            "SELECT rowid, id, source_system, category FROM knowledge_entries WHERE rowid > ? ORDER BY rowid ASC",
            (self._last_entry_rowid,)
        )
        rows = cursor.fetchall()

        if rows:
            self._last_entry_rowid = rows[-1]["rowid"]
            for row in rows:
                await self._emit(
                    "vault",
                    "Knowledge entry captured",
                    self._format_entry_details(row)
                )

    async def _broadcast_new_connections(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute(
            "SELECT rowid, connection_type, strength FROM knowledge_connections WHERE rowid > ? ORDER BY rowid ASC",
            (self._last_connection_rowid,)
        )
        rows = cursor.fetchall()

        if rows:
            self._last_connection_rowid = rows[-1]["rowid"]
            for row in rows:
                await self._emit(
                    "vault",
                    "Knowledge link forged",
                    {
                        "channel": row["connection_type"],
                        "magnitude": f"{row['strength']:.2f}",
                        "emojiMemory": "🧬🔗"
                    }
                )

    async def _broadcast_summary(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute(
            "SELECT COUNT(*), MAX(timestamp) FROM knowledge_entries"
        )
        total_entries, latest_ts = cursor.fetchone()

        cursor = connection.execute("SELECT COUNT(*) FROM knowledge_connections")
        total_links = cursor.fetchone()[0]

        await self._emit(
            "vault",
            "Vault telemetry snapshot",
            {
                "status": f"entries:{total_entries}",
                "channel": f"links:{total_links}",
                "emojiMemory": "📊🧠"
            }
        )

    async def initialize(self) -> None:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Knowledge vault database not found: {self.db_path}")

        with self._connect() as connection:
            cursor = connection.execute("SELECT COALESCE(MAX(rowid), 0) FROM knowledge_entries")
            self._last_entry_rowid = cursor.fetchone()[0]

            cursor = connection.execute("SELECT COALESCE(MAX(rowid), 0) FROM knowledge_connections")
            self._last_connection_rowid = cursor.fetchone()[0]

        logger.info(
            "Knowledge vault telemetry bridge primed (entries=%s, connections=%s)",
            self._last_entry_rowid,
            self._last_connection_rowid,
        )

    async def run(self) -> None:
        await self.initialize()

        while True:
            start_time = time.perf_counter()
            try:
                with self._connect() as connection:
                    await self._broadcast_new_entries(connection)
                    await self._broadcast_new_connections(connection)
                    await self._broadcast_summary(connection)
            except Exception as exc:  # pragma: no cover - bridge should keep running
                logger.warning("Vault telemetry polling error: %s", exc)

            elapsed = time.perf_counter() - start_time
            await asyncio.sleep(max(0.0, self.poll_interval - elapsed))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Vault telemetry bridge")
    parser.add_argument("--db", type=Path, default=Path("knowledge_vault.db"), help="Path to the knowledge vault SQLite database")
    parser.add_argument("--endpoint", type=str, default="ws://localhost:8701", help="Vector network telemetry WebSocket endpoint")
    parser.add_argument("--interval", type=float, default=6.0, help="Polling interval in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    bridge = VaultTelemetryBridge(db_path=args.db, endpoint=args.endpoint, poll_interval=args.interval)
    await bridge.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Knowledge vault telemetry bridge stopped")
