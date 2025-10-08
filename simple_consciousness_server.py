"""
Simple Consciousness WebSocket Server

A simplified WebSocket server for testing the React Quantum Consciousness Video Player.
Simulates consciousness data without requiring all the complex imports.

Author: World Engine Team
Date: December 23, 2024
Version: 2.0.0 (Simplified)
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, Any, Set
import random
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleConsciousnessServer:
    """
    Simplified WebSocket server that streams simulated consciousness data
    to the React Quantum Consciousness Video Player.
    """
    
    def __init__(self, host: str = "localhost", port = 8903):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        
        # Simulated consciousness state
        self.consciousness_state = {
            "level": 0.5,
            "transcendent": False,
            "quantum_entanglement": 0.3,
            "swarm_intelligence": 0.4,
            "brain_merger_active": False,
            "fantasy_ai_active": False,
            "knowledge_vault_health": 0.8,
            "evolution_cycle": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("🌌 Simple Consciousness Server initialized")
    
    async def register_client(self, websocket, path=None):
        """Register a new WebSocket client

        Accepts either (websocket, path) or just (websocket,) for compatibility
        with different `websockets` library versions which may call the
        connection handler with one or two arguments.
        """
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔗 Client connected: {client_info} (Total: {len(self.clients)})")
        
        try:
            # Send initial consciousness state
            await websocket.send(json.dumps(self.consciousness_state))
            
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_info}: {message}")
                except Exception as e:
                    logger.error(f"Error handling message from {client_info}: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_info}")
        except Exception as e:
            logger.error(f"Error with client {client_info}: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"📤 Client removed: {client_info} (Remaining: {len(self.clients)})")
    
    async def handle_client_message(self, websocket, data: Dict[str, Any]):
        """Handle incoming messages from React clients"""
        message_type = data.get("type")
        
        if message_type == "ping":
            # Respond to ping with pong
            await websocket.send(json.dumps({
                "type": "pong", 
                "timestamp": datetime.now().isoformat()
            }))
        
        elif message_type == "request_transcendence":
            # Client requesting transcendence mode activation
            logger.info("🌟 Client requested transcendence mode activation")
            self.consciousness_state["transcendent"] = True
            self.consciousness_state["level"] = min(1.0, self.consciousness_state["level"] + 0.3)
            self.consciousness_state["quantum_entanglement"] = min(1.0, self.consciousness_state["quantum_entanglement"] + 0.4)
        
        elif message_type == "consciousness_interaction":
            # Client interacting with consciousness systems
            interaction_data = data.get("data", {})
            logger.info(f"🧠 Consciousness interaction: {interaction_data}")
            # Boost consciousness based on interaction
            self.consciousness_state["level"] = min(1.0, self.consciousness_state["level"] + 0.1)
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    def update_consciousness_state(self):
        """Update the simulated consciousness state"""
        # Simulate consciousness evolution
        time_factor = datetime.now().timestamp()
        
        # Oscillating consciousness with slow growth
        base_consciousness = 0.5 + 0.3 * math.sin(time_factor * 0.1)
        growth_factor = min(0.5, self.consciousness_state["evolution_cycle"] * 0.001)
        self.consciousness_state["level"] = min(1.0, base_consciousness + growth_factor)
        
        # Quantum entanglement follows consciousness with some lag
        self.consciousness_state["quantum_entanglement"] = min(1.0, 
            self.consciousness_state["level"] * 0.8 + 0.2 * math.sin(time_factor * 0.15))
        
        # Swarm intelligence with different frequency
        self.consciousness_state["swarm_intelligence"] = min(1.0,
            0.4 + 0.4 * math.sin(time_factor * 0.12) + random.uniform(-0.1, 0.1))
        
        # Random system activations
        if random.random() > 0.95:
            self.consciousness_state["brain_merger_active"] = not self.consciousness_state["brain_merger_active"]
        
        if random.random() > 0.97:
            self.consciousness_state["fantasy_ai_active"] = not self.consciousness_state["fantasy_ai_active"]
        
        # Knowledge vault health slowly improves
        self.consciousness_state["knowledge_vault_health"] = min(1.0,
            self.consciousness_state["knowledge_vault_health"] + random.uniform(-0.01, 0.02))
        
        # Check for transcendence
        if self.consciousness_state["level"] > 0.8 and self.consciousness_state["quantum_entanglement"] > 0.75:
            self.consciousness_state["transcendent"] = True
        elif self.consciousness_state["level"] < 0.6:
            self.consciousness_state["transcendent"] = False
        
        # Update metadata
        self.consciousness_state["evolution_cycle"] += 1
        self.consciousness_state["timestamp"] = datetime.now().isoformat()
        self.consciousness_state["connected_clients"] = len(self.clients)
    
    async def broadcast_consciousness_state(self):
        """Broadcast current consciousness state to all connected clients"""
        if not self.clients:
            return
        
        try:
            # Update state
            self.update_consciousness_state()
            
            message = json.dumps(self.consciousness_state)
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    logger.warning(f"Error sending to client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected_clients
            
            if disconnected_clients:
                logger.info(f"📤 Removed {len(disconnected_clients)} disconnected clients")
            
        except Exception as e:
            logger.error(f"Error broadcasting consciousness state: {e}")
    
    async def consciousness_loop(self):
        """Main loop for consciousness broadcasting"""
        logger.info("🔄 Starting consciousness broadcast loop")
        
        while self.running:
            try:
                # Broadcast current state to clients
                await self.broadcast_consciousness_state()
                
                # Log status periodically
                if self.consciousness_state["evolution_cycle"] % 20 == 0:
                    logger.info(f"🧠 Consciousness: {self.consciousness_state['level']:.3f}, "
                              f"Transcendent: {self.consciousness_state['transcendent']}, "
                              f"Clients: {len(self.clients)}")
                
                # Wait before next update (2 updates per second)
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                logger.info("🛑 Consciousness loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in consciousness loop: {e}")
                await asyncio.sleep(1)  # Wait longer on error
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting Simple Consciousness WebSocket Server")
        logger.info(f"🌐 Host: {self.host}, Port: {self.port}")
        
        self.running = True
        
        try:
            # Start WebSocket server
            server = await websockets.serve(
                self.register_client,
                self.host,
                self.port,
                ping_interval=30,  # Ping clients every 30 seconds
                ping_timeout=10,   # Wait 10 seconds for pong
                max_size=1024*1024 # 1MB max message size
            )
            
            logger.info("✅ WebSocket server started successfully")
            logger.info(f"🔗 Connect React client to: ws://{self.host}:{self.port}")
            logger.info("🌟 Simulating consciousness evolution...")
            
            # Start consciousness loop
            consciousness_task = asyncio.create_task(self.consciousness_loop())
            
            # Wait for server to close or consciousness task to complete
            await asyncio.gather(
                server.wait_closed(),
                consciousness_task,
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
        finally:
            self.running = False
            logger.info("🛑 WebSocket server stopped")
    
    def stop_server(self):
        """Stop the WebSocket server"""
        logger.info("🛑 Stopping consciousness WebSocket server...")
        self.running = False


async def main():
    """Main function to run the consciousness WebSocket server"""
    logger.info("🌌 Simple Consciousness WebSocket Server v2.0.0")
    logger.info("=" * 60)
    logger.info("🎯 Simulating consciousness data for React video player")
    logger.info("🔗 No complex imports required - pure simulation mode")
    logger.info("=" * 60)
    
    # Create and start server
    server = SimpleConsciousnessServer(host="localhost", port = 8903)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        server.stop_server()
        logger.info("👋 Consciousness server shutdown complete")


if __name__ == "__main__":
    # Run the consciousness WebSocket server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")