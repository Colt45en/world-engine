#!/usr/bin/env python3
"""
MASTER NEXUS CODEPAD - SIMPLIFIED
================================

Unified command center for real-time nexus AI communication and system coordination.
Direct connection to all subsystems with live monitoring and control.
"""

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path
import websockets
import queue

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_codepad.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
    
class SystemStatus:
    """System status tracking"""
    name: str
    status: str  # "online", "offline", "error", "starting"
    last_check: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    uptime: float = 0.0

class SubsystemManager:
    """Manage and coordinate all subsystems"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.subsystems = {}
        self.processes = {}
        
    def discover_subsystems(self) -> Dict[str, str]:
        """Discover available subsystems"""
        systems = {}
        
        # Core systems we know about
        known_systems = {
            "vector_network": "vector_node_network.py",
            "pain_opportunity": "implement_pain_opportunity_system.py",
            "consciousness_analyzer": "analyze_consciousness_patterns.py",
            "ai_brain_merger": "services/ai_brain_merger.py",
            "knowledge_vault": "services/knowledge_vault_integration.py",
            "recursive_swarm": "recursive_swarm_launcher.py"
        }
        
        for name, filepath in known_systems.items():
            full_path = self.base_path / filepath
            if full_path.exists():
                systems[name] = str(full_path)
                logger.info(f"Discovered subsystem: {name} -> {filepath}")
        
        return systems # type: ignore
    
    def start_subsystem(self, name: str, args: List[str] = None) -> bool: # type: ignore
        """Start a subsystem"""
        if name not in self.subsystems: # type: ignore
            logger.error(f"Unknown subsystem: {name}")
            return False
        
        filepath = self.subsystems[name] # type: ignore
        command = [sys.executable, filepath] # type: ignore
        if args:
            command.extend(args) # type: ignore
        
        try:
            process = subprocess.Popen(
                command, # type: ignore
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.base_path)
            )
            
            self.processes[name] = process # type: ignore
            logger.info(f"Started subsystem: {name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return False
    
    def stop_subsystem(self, name: str) -> bool:
        """Stop a subsystem"""
        if name in self.processes: # type: ignore
            try:
                process = self.processes[name] # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                process.terminate() # pyright: ignore[reportUnknownMemberType]
                process.wait(timeout=10) # pyright: ignore[reportUnknownMemberType]
                del self.processes[name] # pyright: ignore[reportUnknownMemberType]
                logger.info(f"Stopped subsystem: {name}")
                return True
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
                return False
        return False
    
    def get_subsystem_status(self, name: str) -> str:
        """Get subsystem status"""
        if name in self.processes: # type: ignore
            process = self.processes[name] # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if process.poll() is None: # pyright: ignore[reportUnknownMemberType]
                return "running"
            else:
                return "stopped"
        return "not_started"

class NexusAIConnector:
    """Direct nexus AI communication interface"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.clients: set[websockets.WebSocketServerProtocol] = set()
        self.message_queue = queue.Queue() # pyright: ignore[reportUnknownMemberType]
        self.running = False
        
    async def handle_client(self, websocket, path=None): # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        """Handle incoming websocket connections"""
        self.clients.add(websocket) # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        logger.info(f"Nexus client connected: {websocket.remote_address}") # type: ignore
        
        try:
            async for message in websocket: # pyright: ignore[reportUnknownVariableType]
                try:
                    data = json.loads(message) # pyright: ignore[reportUnknownArgumentType]
                    self.message_queue.put(data) # pyright: ignore[reportUnknownMemberType]
                    
                    # Echo confirmation
                    response = {
                        "type": "acknowledgment",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "message": "Command received and queued"
                    }
                    await websocket.send(json.dumps(response)) # pyright: ignore[reportUnknownMemberType]
                    
                except json.JSONDecodeError:
                    error_msg = {"type": "error", "message": "Invalid JSON format"}
                    await websocket.send(json.dumps(error_msg)) # pyright: ignore[reportUnknownMemberType]
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket) # pyright: ignore[reportUnknownArgumentType]
            logger.info("Nexus client disconnected")
    
    async def broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients"""
        if self.clients: # pyright: ignore[reportUnknownMemberType]
            dead_clients = []
            for client in self.clients: # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                try:
                    await client.send(json.dumps(message)) # type: ignore
                except websockets.exceptions.ConnectionClosed:
                    dead_clients.append(client) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            
            # Remove dead connections
            for client in dead_clients: # pyright: ignore[reportUnknownVariableType]
                self.clients.discard(client) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    
    async def start_server(self) -> None:
        """Start the nexus communication server"""
        try:
            self.server = await websockets.serve(self.handle_client, "localhost", self.port) # pyright: ignore[reportUnknownMemberType, reportArgumentType]
            self.running = True
            logger.info(f"Nexus AI server started on ws://localhost:{self.port}")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"Failed to start nexus server: {e}")

class MasterNexusCodepad:
    """Main CODEPAD controller"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.nexus_connector = NexusAIConnector()
        self.subsystem_manager = SubsystemManager(str(self.base_path))
        self.running = False
        
    async def handle_command(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming commands"""
        command_type = data.get("type", "") # type: ignore
        payload = data.get("payload", {})
        command = payload.get("command", "")
        
        logger.info(f"Processing command: {command}")
        
        if command == "status":
            systems = self.subsystem_manager.subsystems # pyright: ignore[reportUnknownMemberType]
            status_report = {}
            
            for name in systems: # pyright: ignore[reportUnknownVariableType]
                status_report[name] = self.subsystem_manager.get_subsystem_status(name) # pyright: ignore[reportUnknownArgumentType]
            
            return {
                "type": "status_report",
                "systems": status_report,
                "total_systems": len(systems) # pyright: ignore[reportUnknownArgumentType]
            }
        
        elif command == "start":
            system_name = payload.get("system")
            args = payload.get("args", [])
            
            if not system_name:
                return {"type": "error", "message": "System name required"}
            
            success = self.subsystem_manager.start_subsystem(system_name, args)
            return {
                "type": "start_response",
                "system": system_name,
                "success": success,
                "message": f"{'Started' if success else 'Failed to start'} {system_name}"
            }
        
        elif command == "stop":
            system_name = payload.get("system")
            
            if not system_name:
                return {"type": "error", "message": "System name required"}
            
            success = self.subsystem_manager.stop_subsystem(system_name)
            return {
                "type": "stop_response",
                "system": system_name,
                "success": success,
                "message": f"{'Stopped' if success else 'Failed to stop'} {system_name}"
            }
        
        elif command == "analytics":
            # Start vector network with analytics mode
            success = self.subsystem_manager.start_subsystem("vector_network", ["--analytics-mode"])
            return {
                "type": "analytics_response",
                "success": success,
                "message": "Vector network analytics mode activated" if success else "Failed to start analytics"
            }
        
        elif command == "consciousness":
            # Start consciousness analyzer
            success = self.subsystem_manager.start_subsystem("consciousness_analyzer")
            return {
                "type": "consciousness_response",
                "success": success,
                "message": "Consciousness analyzer activated" if success else "Failed to start consciousness analyzer"
            }
        
        elif command == "pain_opportunity":
            # Start pain/opportunity system
            success = self.subsystem_manager.start_subsystem("pain_opportunity")
            return {
                "type": "pain_opportunity_response",
                "success": success,
                "message": "Pain/Opportunity system activated" if success else "Failed to start pain/opportunity system"
            }
        
        elif command == "help":
            commands = ["status", "start", "stop", "analytics", "consciousness", "pain_opportunity", "help"]
            systems = list(self.subsystem_manager.subsystems.keys()) # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
            
            return {
                "type": "help_response",
                "available_commands": commands,
                "available_systems": systems
            }
        
        else:
            return {"type": "error", "message": f"Unknown command: {command}"}
    
    async def process_nexus_messages(self) -> None:
        """Process messages from nexus AI"""
        while self.running:
            try:
                if not self.nexus_connector.message_queue.empty(): # pyright: ignore[reportUnknownMemberType]
                    message = self.nexus_connector.message_queue.get_nowait() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    
                    logger.info(f"Processing nexus message: {message.get('type', 'unknown')}") # pyright: ignore[reportUnknownMemberType]
                    
                    # Handle command messages
                    if message.get("type") == "command": # pyright: ignore[reportUnknownMemberType]
                        response = await self.handle_command(message) # pyright: ignore[reportUnknownArgumentType]
                        await self.nexus_connector.broadcast_message(response)
                
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error processing nexus message: {e}")
                await asyncio.sleep(1)
    
    async def initialize(self) -> None:
        """Initialize the CODEPAD system"""
        logger.info("Initializing Master Nexus CODEPAD...")
        
        # Discover subsystems
        systems = self.subsystem_manager.discover_subsystems()
        self.subsystem_manager.subsystems = systems
        
        logger.info(f"CODEPAD initialized with {len(systems)} subsystems")
        return True # pyright: ignore[reportReturnType]
    
    async def start(self) -> None:
        """Start the master CODEPAD system"""
        if not await self.initialize():
            logger.error("Failed to initialize CODEPAD")
            return
        
        self.running = True
        logger.info("Starting Master Nexus CODEPAD...")
        
        # Start nexus communication server
        nexus_task = asyncio.create_task(self.nexus_connector.start_server())
        
        # Start message processing
        message_task = asyncio.create_task(self.process_nexus_messages())
        
        # Keep running
        try:
            await asyncio.gather(nexus_task, message_task)
        except KeyboardInterrupt:
            logger.info("Shutting down CODEPAD...")
            self.running = False
    
    def stop(self) -> None:
        """Stop the CODEPAD system"""
        self.running = False
        
        # Stop all subsystems
        for name in list(self.subsystem_manager.processes.keys()): # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]
            self.subsystem_manager.stop_subsystem(name) # pyright: ignore[reportUnknownArgumentType]
        
        logger.info("Master Nexus CODEPAD stopped")

def main():
    """Main function"""
    print("MASTER NEXUS CODEPAD")
    print("===================")
    print("Direct nexus AI communication and system coordination")
    print("WebSocket Server: ws://localhost:8765")
    print("Commands: status, start, stop, analytics, consciousness, pain_opportunity, help")
    print("Web Interface: nexus_codepad_interface.html")
    print()
    
    # Get base path from command line or use current directory
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Create and start CODEPAD
    codepad = MasterNexusCodepad(base_path)
    
    try:
        asyncio.run(codepad.start())
    except KeyboardInterrupt:
        print("\nCODEPAD shutdown initiated")
    finally:
        codepad.stop()

if __name__ == "__main__":
    main()