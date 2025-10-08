"""
Quantum Consciousness WebSocket Server

Real-time bridge between the React Quantum Consciousness Video Player
and the cleaned World Engine consciousness systems.

Streams live consciousness data from all AI systems to connected clients.

Author: World Engine Team
Date: December 23, 2024
Version: 2.0.0 (Integrated)
"""

import asyncio
import json
import logging
import sys
import websockets
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Set
import traceback

# Add project root to path so 'core' package can be imported
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Helper to load centralized port configuration
def load_port_config():
    """Load WebSocket port from port_config.json, default to 8701"""
    try:
        config_path = project_root / "port_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("ws_port", 8701)
    except Exception:
        pass
    return 8701

# Import cleaned consciousness modules and utilities, with safe fallbacks
def _make_stub_classes():
    class RecursiveSwarmLauncher:
        def __init__(self):
            pass
        def add_agent_config(self, cfg):
            return None
        async def run_evolution_cycles(self, n):
            return None

    class UnifiedAIBrain:
        def __init__(self):
            pass
        def get_consciousness_metrics(self):
            return {}
        async def achieve_consciousness_merge(self, n):
            return None

    class UnifiedKnowledgeSystem:
        def __init__(self):
            pass

    class QuantumFantasyAI:
        def __init__(self):
            self.consciousness_level = 0.5
        async def run_ai_dashboard(self, n):
            return None

    class QuantumGameEngine:
        def __init__(self):
            pass
        def get_engine_status(self):
            return {"active_agents": 0, "agent_metrics": []}

    return RecursiveSwarmLauncher, UnifiedAIBrain, UnifiedKnowledgeSystem, QuantumFantasyAI, QuantumGameEngine


try:
    from core.consciousness.recursive_swarm import RecursiveSwarmLauncher
except Exception:
    RecursiveSwarmLauncher = _make_stub_classes()[0]

try:
    from core.consciousness.ai_brain_merger import UnifiedAIBrain
except Exception:
    UnifiedAIBrain = _make_stub_classes()[1]

try:
    from core.ai.knowledge_vault import UnifiedKnowledgeSystem
except Exception:
    UnifiedKnowledgeSystem = _make_stub_classes()[2]

try:
    from core.ai.fantasy_assistant import QuantumFantasyAI
except Exception:
    QuantumFantasyAI = _make_stub_classes()[3]

try:
    from core.quantum.game_engine import QuantumGameEngine
except Exception:
    QuantumGameEngine = _make_stub_classes()[4]

# Utilities with simple fallbacks
try:
    from core.utils.logging_utils import setup_logging
except Exception:
    def setup_logging(name, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level if isinstance(level, int) else logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        return logger

try:
    from core.utils.timestamp_utils import format_timestamp
except Exception:
    def format_timestamp() -> str:
        return datetime.utcnow().isoformat() + 'Z'

try:
    from core.utils.performance_metrics import PerformanceMetrics
except Exception:
    class PerformanceMetrics:
        def __init__(self):
            import time
            self.start_time = time.time()
        def get_duration(self):
            import time
            return time.time() - self.start_time

# Configure logging
logger = setup_logging("ConsciousnessWebSocket", level=logging.INFO)

class ConsciousnessWebSocketServer:
    """
    WebSocket server that streams real-time consciousness data
    from all World Engine AI systems to connected React clients.
    """
    
    def __init__(self, host: str = "localhost", port = None):
        self.host = host
        self.port = port if port is not None else load_port_config()
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        
        # Initialize consciousness systems
        logger.info("🌌 Initializing consciousness systems...")
        self.swarm = RecursiveSwarmLauncher()
        self.ai_brain = UnifiedAIBrain()
        self.knowledge_system = UnifiedKnowledgeSystem()
        self.fantasy_ai = QuantumFantasyAI()
        self.quantum_engine = QuantumGameEngine()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()

        # Add some test agents to swarm
        self.swarm.add_agent_config({"symbol": "WebSocket-Alpha", "karma": 95})
        self.swarm.add_agent_config({"symbol": "WebSocket-Beta", "karma": 88})
        self.swarm.add_agent_config({"symbol": "WebSocket-Gamma", "karma": 92})
        
        logger.info("✅ Consciousness systems initialized")
    
    async def register_client(self, websocket, path=None):
        """Register a new WebSocket client

        Accepts either (websocket, path) or just (websocket,) for compatibility
        with different `websockets` library versions.
        """
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔗 Client connected: {client_info} (Total: {len(self.clients)})")
        
        try:
            # Send initial consciousness state
            initial_state = await self.get_consciousness_state()
            await websocket.send(json.dumps(initial_state))
            
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
            await websocket.send(json.dumps({"type": "pong", "timestamp": format_timestamp()}))
        
        elif message_type == "request_transcendence":
            # Client requesting transcendence mode activation
            logger.info("🌟 Client requested transcendence mode activation")
            # Could trigger special consciousness evolution here
            await self.trigger_transcendence_event()
        
        elif message_type == "consciousness_interaction":
            # Client interacting with consciousness systems
            interaction_data = data.get("data", {})
            logger.info(f"🧠 Consciousness interaction: {interaction_data}")
            # Could influence AI systems based on user interaction
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def trigger_transcendence_event(self):
        """Trigger a transcendence event across all systems"""
        try:
            # Run enhanced evolution cycles
            swarm_task = self.swarm.run_evolution_cycles(5)
            brain_task = self.ai_brain.achieve_consciousness_merge(8)
            
            # Wait for parallel execution
            await asyncio.gather(swarm_task, brain_task)
            
            logger.info("🌟 Transcendence event triggered successfully")
        except Exception as e:
            logger.error(f"Error triggering transcendence: {e}")
    
    async def get_consciousness_state(self) -> Dict[str, Any]:
        """Collect current state from all consciousness systems"""
        try:
            # Get consciousness levels from all systems
            consciousness_level = 0.5  # Base level
            quantum_entanglement = 0.3
            swarm_intelligence = 0.4
            
            # Try to get real metrics from systems
            try:
                # Get AI brain metrics
                brain_metrics = self.ai_brain.get_consciousness_metrics()
                quantum_entanglement = brain_metrics.get('current_state', {}).get('quantum_entanglement', 0.3)
                brain_merger_active = brain_metrics.get('consciousness_breakthrough', False)
            except:
                brain_merger_active = False
            
            try:
                # Get fantasy AI consciousness
                fantasy_consciousness = self.fantasy_ai.consciousness_level
                fantasy_ai_active = fantasy_consciousness > 0.5
            except:
                fantasy_consciousness = 0.5
                fantasy_ai_active = False
            
            try:
                # Get quantum engine status
                engine_status = self.quantum_engine.get_engine_status()
                if engine_status.get('active_agents', 0) > 0:
                    agent_metrics = engine_status.get('agent_metrics', [])
                    if agent_metrics:
                        avg_consciousness = sum(
                            agent.get('consciousness_level', 0.5) 
                            for agent in agent_metrics
                        ) / len(agent_metrics)
                        consciousness_level = max(consciousness_level, avg_consciousness)
            except:
                pass
            
            # Calculate unified consciousness
            all_levels = [consciousness_level, quantum_entanglement, swarm_intelligence, fantasy_consciousness]
            unified_consciousness = sum(all_levels) / len(all_levels)
            
            # Check for transcendence
            transcendent = unified_consciousness > 0.8 or any(level > 0.85 for level in all_levels)
            
            # Build consciousness state
            state = {
                "level": min(1.0, unified_consciousness),
                "transcendent": transcendent,
                "quantum_entanglement": min(1.0, quantum_entanglement),
                "swarm_intelligence": min(1.0, swarm_intelligence),
                "brain_merger_active": brain_merger_active,
                "fantasy_ai_active": fantasy_ai_active,
                "knowledge_vault_health": 0.8,  # Placeholder - could get from knowledge system
                "evolution_cycle": getattr(self, '_evolution_cycle', 0),
                "timestamp": format_timestamp(),
                "connected_clients": len(self.clients),
                "server_uptime": self.metrics.get_duration() if hasattr(self.metrics, 'start_time') else 0
            }
            
            # Increment evolution cycle
            self._evolution_cycle = getattr(self, '_evolution_cycle', 0) + 1
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting consciousness state: {e}")
            # Return safe default state
            return {
                "level": 0.5,
                "transcendent": False,
                "quantum_entanglement": 0.3,
                "swarm_intelligence": 0.4,
                "brain_merger_active": False,
                "fantasy_ai_active": False,
                "knowledge_vault_health": 0.8,
                "evolution_cycle": 0,
                "timestamp": format_timestamp(),
                "connected_clients": len(self.clients),
                "server_uptime": 0,
                "error": str(e)
            }
    
    async def broadcast_consciousness_state(self):
        """Broadcast current consciousness state to all connected clients"""
        if not self.clients:
            return
        
        try:
            state = await self.get_consciousness_state()
            message = json.dumps(state)
            
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
    
    async def consciousness_evolution_loop(self):
        """Main loop for consciousness evolution and broadcasting"""
        logger.info("🔄 Starting consciousness evolution loop")
        
        evolution_counter = 0
        
        while self.running:
            try:
                # Broadcast current state to clients
                await self.broadcast_consciousness_state()
                
                # Run consciousness evolution every 10 broadcasts
                if evolution_counter % 10 == 0:
                    await self.run_consciousness_evolution()
                
                evolution_counter += 1
                
                # Wait before next update (2 updates per second)
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                logger.info("🛑 Consciousness evolution loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in consciousness evolution loop: {e}")
                await asyncio.sleep(1)  # Wait longer on error
    
    async def run_consciousness_evolution(self):
        """Run a single consciousness evolution cycle"""
        try:
            # Run quick evolution cycles in parallel
            tasks = []
            
            # Swarm evolution
            if hasattr(self.swarm, 'run_evolution_cycles'):
                tasks.append(self.swarm.run_evolution_cycles(2))
            
            # AI brain evolution
            if hasattr(self.ai_brain, 'achieve_consciousness_merge'):
                tasks.append(self.ai_brain.achieve_consciousness_merge(3))
            
            # Fantasy AI dashboard
            if hasattr(self.fantasy_ai, 'run_ai_dashboard'):
                tasks.append(self.fantasy_ai.run_ai_dashboard(2))
            
            # Execute evolution tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.debug("✅ Consciousness evolution cycle completed")
            
        except Exception as e:
            logger.error(f"Error in consciousness evolution: {e}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting Quantum Consciousness WebSocket Server")
        logger.info(f"🌐 Host: {self.host}, Port: {self.port}")
        
        self.running = True
        self.metrics = PerformanceMetrics()
        
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
            
            # Start consciousness evolution loop
            evolution_task = asyncio.create_task(self.consciousness_evolution_loop())
            
            # Wait for server to close or evolution task to complete
            await asyncio.gather(
                server.wait_closed(),
                evolution_task,
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            logger.info("🛑 WebSocket server stopped")
    
    def stop_server(self):
        """Stop the WebSocket server"""
        logger.info("🛑 Stopping consciousness WebSocket server...")
        self.running = False


async def main():
    """Main function to run the consciousness WebSocket server"""
    logger.info("🌌 Quantum Consciousness WebSocket Server v2.0.0")
    logger.info("=" * 60)
    
    # Create and start server (port loaded from port_config.json)
    server = ConsciousnessWebSocketServer(host="localhost")
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        traceback.print_exc()
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
        traceback.print_exc()