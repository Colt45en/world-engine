#!/usr/bin/env python3
"""
🎛️🧠 MASTER NEXUS CODEPAD
========================

Unified command center for real-time nexus AI communication and system coordination.
Direct connection to all subsystems with live monitoring and control.

Features:
- Real-time nexus AI communication
- System health monitoring and auto-repair
- Unified command interface
- Live system integration
- Automatic dependency resolution
- Error detection and correction
- Performance optimization
"""

import asyncio
import json
import logging
import subprocess
import sys
import psutil
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import websockets
import queue

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_codepad.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """System status tracking"""
    name: str
    status: str  # "online", "offline", "error", "starting"
    last_check: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    uptime: float = 0.0
    
@dataclass
class NexusMessage:
    """Nexus communication message format"""
    timestamp: str
    message_type: str  # "command", "status", "data", "error"
    source: str
    target: str
    payload: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low

class DependencyManager:
    """Automatic dependency resolution and installation"""
    
    REQUIRED_PACKAGES = [
        "numpy", "pandas", "matplotlib", "seaborn",
        "websockets", "psutil", "requests", "aiohttp",
        "sympy", "scipy", "nltk", "jsonschema"
    ]
    
    @staticmethod
    def check_dependencies() -> List[str]:
        """Check for missing dependencies"""
        missing = []
        for package in DependencyManager.REQUIRED_PACKAGES:
            try:
                importlib.import_module(package)
            except ImportError:
                missing.append(package)
        return missing
    
    @staticmethod
    def install_dependencies(packages: List[str]) -> bool:
        """Install missing dependencies"""
        if not packages:
            return True
            
        logger.info(f"🔧 Installing dependencies: {packages}")
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + packages
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Dependency installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False

class SystemHealthMonitor:
    """Real-time system health monitoring"""
    
    def __init__(self):
        self.systems: Dict[str, SystemStatus] = {}
        self.monitoring = False
        
    def register_system(self, name: str) -> None:
        """Register a system for monitoring"""
        self.systems[name] = SystemStatus(
            name=name,
            status="offline",
            last_check=datetime.now(timezone.utc).isoformat()
        )
        
    def update_system_status(self, name: str, status: str, error_count: int = 0) -> None:
        """Update system status"""
        if name in self.systems:
            system = self.systems[name]
            system.status = status
            system.last_check = datetime.now(timezone.utc).isoformat()
            system.error_count = error_count
            
            # Get system metrics
            try:
                system.cpu_usage = psutil.cpu_percent()
                system.memory_usage = psutil.virtual_memory().percent
            except:
                pass
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health report"""
        total_systems = len(self.systems)
        online_systems = sum(1 for s in self.systems.values() if s.status == "online")
        total_errors = sum(s.error_count for s in self.systems.values())
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_systems": total_systems,
            "online_systems": online_systems,
            "health_percentage": (online_systems / total_systems * 100) if total_systems > 0 else 0,
            "total_errors": total_errors,
            "systems": {name: asdict(status) for name, status in self.systems.items()}
        }

class NexusAIConnector:
    """Direct nexus AI communication interface"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.clients = set()
        self.message_queue = queue.Queue()
        self.running = False
        
    async def handle_client(self, websocket, path):
        """Handle incoming websocket connections"""
        self.clients.add(websocket)
        logger.info(f"🔌 Nexus client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    nexus_msg = NexusMessage(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message_type=data.get("type", "command"),
                        source=data.get("source", "unknown"),
                        target=data.get("target", "nexus"),
                        payload=data.get("payload", {}),
                        priority=data.get("priority", 2)
                    )
                    self.message_queue.put(nexus_msg)
                    
                    # Echo confirmation
                    response = {
                        "type": "acknowledgment",
                        "timestamp": nexus_msg.timestamp,
                        "message": "Command received and queued"
                    }
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    error_msg = {"type": "error", "message": "Invalid JSON format"}
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info("🔌 Nexus client disconnected")
        logger.info(f"🔌 Nexus client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    nexus_msg = NexusMessage(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message_type=data.get("type", "command"),
                        source=data.get("source", "unknown"),
                        target=data.get("target", "nexus"),
                        payload=data.get("payload", {}),
                        priority=data.get("priority", 2)
                    )
                    self.message_queue.put(nexus_msg)
                    
                    # Echo confirmation
                    response = {
                        "type": "acknowledgment",
                        "timestamp": nexus_msg.timestamp,
                        "message": "Command received and queued"
                    }
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    error_msg = {"type": "error", "message": "Invalid JSON format"}
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info("🔌 Nexus client disconnected")
    
    async def broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients"""
        if self.clients:
            dead_clients = []
            for client in self.clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    dead_clients.append(client)
            
            # Remove dead connections
            for client in dead_clients:
                self.clients.discard(client)
    
    async def start_server(self) -> None:
        """Start the nexus communication server"""
        try:
            self.server = await websockets.serve(self.handle_client, "localhost", self.port)
            self.running = True
            logger.info(f"🚀 Nexus AI server started on ws://localhost:{self.port}")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"❌ Failed to start nexus server: {e}")
            
    def stop_server(self) -> None:
        """Stop the nexus communication server"""
        if self.server:
            self.server.close()
            self.running = False

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
                logger.info(f"🔍 Discovered subsystem: {name} -> {filepath}")
        
        return systems
    
    def start_subsystem(self, name: str, args: List[str] = None) -> bool:
        """Start a subsystem"""
        if name not in self.subsystems:
            logger.error(f"❌ Unknown subsystem: {name}")
            return False
        
        filepath = self.subsystems[name]
        command = [sys.executable, filepath]
        if args:
            command.extend(args)
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.base_path)
            )
            
            self.processes[name] = process
            logger.info(f"🚀 Started subsystem: {name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {name}: {e}")
            return False
    
    def stop_subsystem(self, name: str) -> bool:
        """Stop a subsystem"""
        if name in self.processes:
            try:
                process = self.processes[name]
                process.terminate()
                process.wait(timeout=10)
                del self.processes[name]
                logger.info(f"🛑 Stopped subsystem: {name}")
                return True
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
                return False
        return False
    
    def get_subsystem_status(self, name: str) -> str:
        """Get subsystem status"""
        if name in self.processes:
            process = self.processes[name]
            if process.poll() is None:
                return "running"
            else:
                return "stopped"
        return "not_started"

class MasterNexusCodepad:
    """Main CODEPAD controller"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.health_monitor = SystemHealthMonitor()
        self.nexus_connector = NexusAIConnector()
        self.subsystem_manager = SubsystemManager(str(self.base_path))
        self.running = False
        self.command_handlers = {}
        self.setup_commands()
        
    def setup_commands(self) -> None:
        """Setup command handlers"""
        self.command_handlers = {
            "status": self.handle_status_command,
            "start": self.handle_start_command,
            "stop": self.handle_stop_command,
            "restart": self.handle_restart_command,
            "health": self.handle_health_command,
            "analytics": self.handle_analytics_command,
            "consciousness": self.handle_consciousness_command,
            "pain_opportunity": self.handle_pain_opportunity_command,
            "help": self.handle_help_command
        }
    
    async def handle_status_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status command"""
        systems = self.subsystem_manager.subsystems
        status_report = {}
        
        for name in systems:
            status_report[name] = self.subsystem_manager.get_subsystem_status(name)
        
        return {
            "type": "status_report",
            "systems": status_report,
            "health": self.health_monitor.get_system_health()
        }
    
    async def handle_start_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start command"""
        system_name = payload.get("system")
        args = payload.get("args", [])
        
        if not system_name:
            return {"type": "error", "message": "System name required"}
        
        success = self.subsystem_manager.start_subsystem(system_name, args)
        if success:
            self.health_monitor.update_system_status(system_name, "online")
        
        return {
            "type": "start_response",
            "system": system_name,
            "success": success,
            "message": f"{'Started' if success else 'Failed to start'} {system_name}"
        }
    
    async def handle_stop_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stop command"""
        system_name = payload.get("system")
        
        if not system_name:
            return {"type": "error", "message": "System name required"}
        
        success = self.subsystem_manager.stop_subsystem(system_name)
        if success:
            self.health_monitor.update_system_status(system_name, "offline")
        
        return {
            "type": "stop_response",
            "system": system_name,
            "success": success,
            "message": f"{'Stopped' if success else 'Failed to stop'} {system_name}"
        }
    
    async def handle_restart_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle restart command"""
        system_name = payload.get("system")
        
        if not system_name:
            return {"type": "error", "message": "System name required"}
        
        # Stop then start
        self.subsystem_manager.stop_subsystem(system_name)
        await asyncio.sleep(2)  # Give it time to stop
        success = self.subsystem_manager.start_subsystem(system_name)
        
        if success:
            self.health_monitor.update_system_status(system_name, "online")
        
        return {
            "type": "restart_response",
            "system": system_name,
            "success": success,
            "message": f"{'Restarted' if success else 'Failed to restart'} {system_name}"
        }
    
    async def handle_health_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health command"""
        return {
            "type": "health_report",
            "data": self.health_monitor.get_system_health()
        }
    
    async def handle_analytics_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analytics command"""
        # Start vector network with analytics mode
        success = self.subsystem_manager.start_subsystem("vector_network", ["--analytics-mode"])
        return {
            "type": "analytics_response",
            "success": success,
            "message": "Vector network analytics mode activated" if success else "Failed to start analytics"
        }
    
    async def handle_consciousness_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle consciousness command"""
        # Start consciousness analyzer
        success = self.subsystem_manager.start_subsystem("consciousness_analyzer")
        return {
            "type": "consciousness_response",
            "success": success,
            "message": "Consciousness analyzer activated" if success else "Failed to start consciousness analyzer"
        }
    
    async def handle_pain_opportunity_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pain/opportunity command"""
        # Start pain/opportunity system
        success = self.subsystem_manager.start_subsystem("pain_opportunity")
        return {
            "type": "pain_opportunity_response",
            "success": success,
            "message": "Pain/Opportunity system activated" if success else "Failed to start pain/opportunity system"
        }
    
    async def handle_help_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help command"""
        commands = list(self.command_handlers.keys())
        systems = list(self.subsystem_manager.subsystems.keys())
        
        return {
            "type": "help_response",
            "available_commands": commands,
            "available_systems": systems,
            "usage": {
                "start": {"system": "system_name", "args": ["optional", "arguments"]},
                "stop": {"system": "system_name"},
                "restart": {"system": "system_name"},
                "status": {},
                "health": {},
                "analytics": {},
                "consciousness": {},
                "pain_opportunity": {}
            }
        }
    
    async def process_nexus_messages(self) -> None:
        """Process messages from nexus AI"""
        while self.running:
            try:
                if not self.nexus_connector.message_queue.empty():
                    message = self.nexus_connector.message_queue.get_nowait()
                    
                    logger.info(f"📨 Processing nexus message: {message.message_type} from {message.source}")
                    
                    # Handle command messages
                    if message.message_type == "command":
                        command = message.payload.get("command")
                        if command in self.command_handlers:
                            response = await self.command_handlers[command](message.payload)
                            await self.nexus_connector.broadcast_message(response)
                        else:
                            error_response = {
                                "type": "error", 
                                "message": f"Unknown command: {command}"
                            }
                            await self.nexus_connector.broadcast_message(error_response)
                
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"❌ Error processing nexus message: {e}")
                await asyncio.sleep(1)
    
    async def initialize(self) -> None:
        """Initialize the CODEPAD system"""
        logger.info("🎛️ Initializing Master Nexus CODEPAD...")
        
        # Check and install dependencies
        missing_deps = DependencyManager.check_dependencies()
        if missing_deps:
            logger.info(f"🔧 Installing missing dependencies: {missing_deps}")
            if not DependencyManager.install_dependencies(missing_deps):
                logger.error("❌ Failed to install dependencies")
                return False
        
        # Discover subsystems
        systems = self.subsystem_manager.discover_subsystems()
        self.subsystem_manager.subsystems = systems
        
        # Register systems with health monitor
        for name in systems:
            self.health_monitor.register_system(name)
        
        logger.info(f"✅ CODEPAD initialized with {len(systems)} subsystems")
        return True
    
    async def start(self) -> None:
        """Start the master CODEPAD system"""
        if not await self.initialize():
            logger.error("❌ Failed to initialize CODEPAD")
            return
        
        self.running = True
        logger.info("🚀 Starting Master Nexus CODEPAD...")
        
        # Start nexus communication server
        nexus_task = asyncio.create_task(self.nexus_connector.start_server())
        
        # Start message processing
        message_task = asyncio.create_task(self.process_nexus_messages())
        
        # Keep running
        try:
            await asyncio.gather(nexus_task, message_task)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down CODEPAD...")
            self.running = False
            self.nexus_connector.stop_server()
    
    def stop(self) -> None:
        """Stop the CODEPAD system"""
        self.running = False
        
        # Stop all subsystems
        for name in list(self.subsystem_manager.processes.keys()):
            self.subsystem_manager.stop_subsystem(name)
        
        # Stop nexus server
        self.nexus_connector.stop_server()
        
        logger.info("🛑 Master Nexus CODEPAD stopped")

def main():
    """Main function"""
    print("🎛️🧠 MASTER NEXUS CODEPAD")
    print("========================")
    print("Direct nexus AI communication and system coordination")
    print("WebSocket Server: ws://localhost:8765")
    print("Commands: status, start, stop, restart, health, analytics, consciousness, pain_opportunity, help")
    print()
    
    # Get base path from command line or use current directory
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Create and start CODEPAD
    codepad = MasterNexusCodepad(base_path)
    
    try:
        asyncio.run(codepad.start())
    except KeyboardInterrupt:
        print("\n🛑 CODEPAD shutdown initiated")
    finally:
        codepad.stop()

if __name__ == "__main__":
    main()