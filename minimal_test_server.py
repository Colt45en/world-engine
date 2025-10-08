#!/usr/bin/env python3
"""
MINIMAL WORKING SERVER - configurable via port_config.json (fallback 8701)
=============================
"""

import asyncio
import os
import websockets
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_client(websocket: websockets.WebSocketServerProtocol, path: str | None = None): # type: ignore
    logger.info("Client connected")
    try:
        await websocket.send(json.dumps({
            "type": "welcome",
            "message": "Minimal server working!",
            "port": 8701,
            "timestamp": datetime.now().isoformat()
        }))
        
        async for message in websocket:
            data = json.loads(message)
            response = {
                "type": "echo",
                "received": data,
                "server_time": datetime.now().isoformat(),
                "status": "working"
            }
            await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info("Client disconnected")

async def main():
    # Prefer central port_config.json when present
    try:
        import json as _json, os as _os
        cfg_path = _os.path.abspath(_os.path.join(os.path.dirname(__file__), '..', '..', '..', 'port_config.json'))
        if _os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as fh:
                _cfg = _json.load(fh)
                target_port = int(_cfg.get('ws_port', 8701))
        else:
            target_port = 8701
    except Exception:
        target_port = 8701

    logger.info(f"Starting minimal server on port {target_port}")
    async with websockets.serve(handle_client, "localhost", target_port):
        logger.info(f"Server running on ws://localhost:{target_port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
