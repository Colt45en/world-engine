#!/usr/bin/env python3
"""
Simple Consciousness WebSocket Server
===================================
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleConsciousnessServer:
    def __init__(self, port=8900):
        self.port = port
        self.clients = set()
        
    async def register_client(self, websocket, path=None):
        self.clients.add(websocket)
        logger.info(f"Client connected. Total: {len(self.clients)}")
        
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "message": "Connected to Simple Consciousness Server",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_status": "online"
            }
            await websocket.send(json.dumps(welcome))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"Message handling error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Remaining: {len(self.clients)}")
    
    async def handle_message(self, websocket, data):
        """Handle incoming messages"""
        msg_type = data.get("type", "unknown")
        
        if msg_type == "ping":
            response = {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}
        elif msg_type == "status":
            response = {
                "type": "status_response",
                "server": "Simple Consciousness Server",
                "clients": len(self.clients),
                "uptime": "running",
                "consciousness_level": 0.75
            }
        elif msg_type == "consciousness_query":
            response = {
                "type": "consciousness_response",
                "consciousness_data": {
                    "awareness_level": 0.8,
                    "cognitive_load": 0.6,
                    "emotional_state": "curious",
                    "processing_speed": "optimal"
                }
            }
        else:
            response = {"type": "echo", "received": data, "processed_at": datetime.now(timezone.utc).isoformat()}
        
        await websocket.send(json.dumps(response))
    
    async def start_server(self):
        logger.info(f"Starting Simple Consciousness Server on port {self.port}")
        async with websockets.serve(self.register_client, "localhost", self.port):
            logger.info(f"Server running on ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever

def main():
    server = SimpleConsciousnessServer(8900)
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped")

if __name__ == "__main__":
    main()
