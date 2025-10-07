#!/usr/bin/env python3
"""
🚀 WORKING PYTHON SERVER
========================

Simple working server that tests all dependencies
"""

import asyncio
import json
import logging
from datetime import datetime
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingServer:
    def __init__(self, port=9100):
        self.port = port
        self.clients = set()
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        self.clients.add(websocket)
        client_id = len(self.clients)
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "client_id": client_id,
                "server_port": self.port,
                "message": "🚀 Python Server Working!",
                "timestamp": datetime.now().isoformat(),
                "dependencies": self.check_dependencies()
            }
            await websocket.send(json.dumps(welcome, indent=2))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(data, client_id)
                    await websocket.send(json.dumps(response, indent=2))
                except json.JSONDecodeError:
                    error_response = {
                        "type": "error",
                        "message": "Invalid JSON received",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} disconnected")
    
    async def process_message(self, data, client_id):
        """Process incoming WebSocket messages"""
        message_type = data.get("type", "unknown")
        
        if message_type == "ping":
            return {
                "type": "pong",
                "client_id": client_id,
                "server_time": datetime.now().isoformat(),
                "message": "Server alive and working!"
            }
        elif message_type == "status":
            return {
                "type": "status_response",
                "client_id": client_id,
                "server_status": "operational",
                "active_clients": len(self.clients),
                "dependencies": self.check_dependencies(),
                "timestamp": datetime.now().isoformat()
            }
        elif message_type == "test_math":
            # Test numpy if available
            try:
                import numpy as np
                arr = np.array([1, 2, 3, 4, 5])
                return {
                    "type": "math_result",
                    "numpy_test": f"Array sum: {arr.sum()}",
                    "numpy_version": np.__version__,
                    "success": True
                }
            except ImportError:
                return {
                    "type": "math_result",
                    "message": "NumPy not available",
                    "success": False
                }
        else:
            return {
                "type": "echo",
                "client_id": client_id,
                "received": data,
                "timestamp": datetime.now().isoformat()
            }
    
    def check_dependencies(self):
        """Check which dependencies are available"""
        deps = {}
        test_modules = [
            'numpy', 'pandas', 'matplotlib', 'asyncio',
            'websockets', 'json', 'logging', 'psutil'
        ]
        
        for module in test_modules:
            try:
                __import__(module)
                deps[module] = "✅ Available"
            except ImportError:
                deps[module] = "❌ Missing"
        
        return deps
    
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            # Import websockets
            import websockets
            
            logger.info(f"Starting WorkingServer on port {self.port}")
            print(f"🚀 PYTHON SERVER STARTING")
            print(f"📡 Port: {self.port}")
            print(f"🌐 URL: ws://localhost:{self.port}")
            
            # Check dependencies
            deps = self.check_dependencies()
            print("\n📦 DEPENDENCIES:")
            for module, status in deps.items():
                print(f"   {module}: {status}")
            
            # Start server
            async with websockets.serve(self.handle_client, "localhost", self.port):
                print(f"\n✅ Server running on ws://localhost:{self.port}")
                print("Press Ctrl+C to stop")
                await asyncio.Future()  # Run forever
                
        except ImportError:
            print("❌ websockets module not available")
            print("Installing websockets...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "websockets"])
            print("Please restart the server")
        except OSError as e:
            print(f"❌ Port {self.port} error: {e}")
            print("Trying alternative port...")
            self.port += 1
            await self.start_server()

async def main():
    """Main function"""
    print("🔍 PYTHON SYSTEM TEST")
    print("=" * 30)
    
    server = WorkingServer()
    await server.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("Check dependencies and try again")