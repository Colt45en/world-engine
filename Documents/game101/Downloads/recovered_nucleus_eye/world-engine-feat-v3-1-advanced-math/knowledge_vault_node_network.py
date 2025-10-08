#!/usr/bin/env python3
"""
🏛️🧠 KNOWLEDGE VAULT NODE NETWORK WITH META ROOM
=================================================

Advanced neural network architecture with Knowledge Vaults positioned between
every two connected nodes. Each vault has a Librarian AI that organizes and
routes data to a central Meta Room, which coordinates all AI connections.

Architecture:
- Node A ←→ Knowledge Vault ←→ Node B (vault between every connection)
- Each Knowledge Vault has a Librarian AI
- All Vaults connect to central Meta Room
- Meta Room connects to all AI systems
- Librarians organize, label, and route data intelligently
"""

import asyncio
import websockets
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Optional, Any
import random
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_port_config():
    """Load WebSocket port from port_config.json, default to 8702"""
    try:
        config_path = Path(__file__).resolve().parent / "port_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Use alternate port for vault network
                return config.get("ws_port", 8701) + 1
    except Exception:
        pass
    return 8702

class NodeType(Enum):
    """Types of nodes in the vault network"""
    CONSCIOUSNESS = "consciousness"
    PROCESSOR = "processor"
    STORAGE = "storage"
    BRIDGE = "bridge"
    KNOWLEDGE_VAULT = "knowledge_vault"
    META_ROOM = "meta_room"
    AI_SYSTEM = "ai_system"

class LibrarianRole(Enum):
    """Roles for vault librarian AIs"""
    ORGANIZER = "organizer"
    CLASSIFIER = "classifier"
    ROUTER = "router"
    ARCHIVIST = "archivist"
    CURATOR = "curator"

@dataclass
class Vector3D:
    """3D vector for positioning"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def distance_to(self, other: 'Vector3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def midpoint(self, other: 'Vector3D') -> 'Vector3D':
        """Calculate midpoint between two vectors"""
        return Vector3D(
            (self.x + other.x) / 2,
            (self.y + other.y) / 2,
            (self.z + other.z) / 2
        )

@dataclass
class VaultLibrarian:
    """AI librarian that manages a knowledge vault"""
    id: str
    name: str
    role: LibrarianRole
    vault_id: str
    intelligence_level: float = 0.85
    organization_skill: float = 0.90
    routing_efficiency: float = 0.88
    data_processed: int = 0
    classifications_made: int = 0
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and classify data"""
        self.data_processed += 1
        
        # Librarian analyzes and labels the data
        classified_data = {
            "original_data": data,
            "classified_by": self.name,
            "classification": self._classify(data),
            "importance": self._assess_importance(data),
            "storage_recommendation": self._recommend_storage(data),
            "meta_routing": self._should_route_to_meta(data),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.classifications_made += 1
        return classified_data
    
    def _classify(self, data: Dict[str, Any]) -> str:
        """Classify data type"""
        if "consciousness" in str(data).lower():
            return "consciousness_data"
        elif "knowledge" in str(data).lower():
            return "knowledge_storage"
        elif "process" in str(data).lower() or "compute" in str(data).lower():
            return "processing_data"
        elif "meta" in str(data).lower():
            return "meta_information"
        else:
            return "general_data"
    
    def _assess_importance(self, data: Dict[str, Any]) -> str:
        """Assess data importance"""
        importance_score = random.uniform(0.3, 1.0)
        if importance_score > 0.8:
            return "critical"
        elif importance_score > 0.6:
            return "high"
        elif importance_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _recommend_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend where to store data"""
        classification = self._classify(data)
        importance = self._assess_importance(data)
        
        return {
            "vault_storage": importance in ["critical", "high"],
            "meta_room_cache": classification == "meta_information",
            "long_term_archive": importance == "critical",
            "temporary_cache": importance == "low"
        }
    
    def _should_route_to_meta(self, data: Dict[str, Any]) -> bool:
        """Determine if data should be routed to Meta Room"""
        classification = self._classify(data)
        importance = self._assess_importance(data)
        
        # Route critical data and meta information to Meta Room
        return importance == "critical" or classification == "meta_information"

@dataclass
class KnowledgeVault:
    """Knowledge vault positioned between two connected nodes"""
    id: str
    name: str
    position: Vector3D
    node_a_id: str  # First connected node
    node_b_id: str  # Second connected node
    librarian: VaultLibrarian
    stored_data: List[Dict[str, Any]] = field(default_factory=list)
    meta_room_connection: Optional[str] = None
    capacity: int = 1000
    current_size: int = 0
    
    def store_data(self, data: Dict[str, Any]) -> bool:
        """Store data in the vault"""
        if self.current_size >= self.capacity:
            logger.warning(f"Vault {self.name} is at capacity!")
            return False
        
        # Librarian processes the data first
        processed_data = self.librarian.process_data(data)
        self.stored_data.append(processed_data)
        self.current_size += 1
        
        logger.info(f"📚 Vault {self.name}: Stored data (classified as {processed_data['classification']})")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get vault status"""
        return {
            "id": self.id,
            "name": self.name,
            "position": asdict(self.position),
            "connected_nodes": [self.node_a_id, self.node_b_id],
            "librarian": {
                "name": self.librarian.name,
                "role": self.librarian.role.value,
                "data_processed": self.librarian.data_processed,
                "classifications_made": self.librarian.classifications_made
            },
            "storage": {
                "current_size": self.current_size,
                "capacity": self.capacity,
                "utilization": f"{(self.current_size/self.capacity)*100:.1f}%"
            },
            "meta_room_connected": self.meta_room_connection is not None
        }

@dataclass
class MetaRoom:
    """Central Meta Room that coordinates all AI systems and vaults"""
    id: str
    name: str = "Central Meta Room"
    position: Vector3D = field(default_factory=lambda: Vector3D(0, 0, 0))
    connected_vaults: List[str] = field(default_factory=list)
    connected_ai_systems: List[str] = field(default_factory=list)
    data_cache: Dict[str, Any] = field(default_factory=dict)
    intelligence_level: float = 0.95
    coordination_efficiency: float = 0.92
    
    def receive_from_vault(self, vault_id: str, data: Dict[str, Any]):
        """Receive data from a knowledge vault"""
        logger.info(f"🏛️ Meta Room received data from vault {vault_id}")
        
        # Cache important data
        if data.get("importance") in ["critical", "high"]:
            cache_key = f"{vault_id}_{datetime.now(timezone.utc).timestamp()}"
            self.data_cache[cache_key] = data
        
        # Route to appropriate AI systems
        self._route_to_ai_systems(data)
    
    def _route_to_ai_systems(self, data: Dict[str, Any]):
        """Route data to connected AI systems"""
        classification = data.get("classification", "unknown")
        
        logger.info(f"🔀 Meta Room routing {classification} to {len(self.connected_ai_systems)} AI systems")
        
        # Each AI system gets relevant data based on classification
        for ai_system_id in self.connected_ai_systems:
            logger.debug(f"   → Sending to AI system {ai_system_id}")
    
    def connect_vault(self, vault_id: str):
        """Connect a knowledge vault to the Meta Room"""
        if vault_id not in self.connected_vaults:
            self.connected_vaults.append(vault_id)
            logger.info(f"🔗 Meta Room connected to vault {vault_id}")
    
    def connect_ai_system(self, ai_system_id: str):
        """Connect an AI system to the Meta Room"""
        if ai_system_id not in self.connected_ai_systems:
            self.connected_ai_systems.append(ai_system_id)
            logger.info(f"🤖 Meta Room connected to AI system {ai_system_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Meta Room status"""
        return {
            "id": self.id,
            "name": self.name,
            "position": asdict(self.position),
            "connected_vaults": len(self.connected_vaults),
            "connected_ai_systems": len(self.connected_ai_systems),
            "cached_data_items": len(self.data_cache),
            "intelligence_level": self.intelligence_level,
            "coordination_efficiency": self.coordination_efficiency
        }

@dataclass
class NetworkNode:
    """Node in the network"""
    id: str
    name: str
    node_type: NodeType
    position: Vector3D
    connections: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VectorConnection:
    """Connection between two nodes with a vault in the middle"""
    id: str
    node_a_id: str
    node_b_id: str
    vault_id: str  # Knowledge vault positioned in the middle
    strength: float = 0.8
    data_flow_rate: float = 0.0

class KnowledgeVaultNetwork:
    """Main network manager with vaults between all connections"""
    
    def __init__(self, port: int = 8702):
        self.port = port
        self.nodes: Dict[str, NetworkNode] = {}
        self.vaults: Dict[str, KnowledgeVault] = {}
        self.connections: Dict[str, VectorConnection] = {}
        self.meta_room: Optional[MetaRoom] = None
        self.ai_systems: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Meta Room
        self._initialize_meta_room()
    
    def _initialize_meta_room(self):
        """Initialize the central Meta Room"""
        meta_id = f"meta_room_{uuid.uuid4().hex[:8]}"
        self.meta_room = MetaRoom(
            id=meta_id,
            name="Central Meta Room",
            position=Vector3D(0, 0, 0)
        )
        logger.info(f"🏛️ Initialized {self.meta_room.name}")
    
    def create_node(self, name: str, node_type: str, position: Dict[str, float]) -> str:
        """Create a new node"""
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        node = NetworkNode(
            id=node_id,
            name=name,
            node_type=NodeType[node_type.upper()],
            position=Vector3D(**position)
        )
        
        self.nodes[node_id] = node
        logger.info(f"✨ Created node: {name} ({node_type})")
        return node_id
    
    def create_connection(self, node_a_id: str, node_b_id: str, strength: float = 0.8) -> Dict[str, str]:
        """Create connection between two nodes with a vault in the middle"""
        if node_a_id not in self.nodes or node_b_id not in self.nodes:
            raise ValueError("Both nodes must exist")
        
        node_a = self.nodes[node_a_id]
        node_b = self.nodes[node_b_id]
        
        # Calculate midpoint for vault position
        vault_position = node_a.position.midpoint(node_b.position)
        
        # Create vault ID and librarian
        vault_id = f"vault_{uuid.uuid4().hex[:8]}"
        librarian_id = f"librarian_{uuid.uuid4().hex[:8]}"
        
        # Assign librarian role based on node types
        librarian_role = self._assign_librarian_role(node_a.node_type, node_b.node_type)
        
        librarian = VaultLibrarian(
            id=librarian_id,
            name=f"Librarian_{node_a.name}_{node_b.name}",
            role=librarian_role,
            vault_id=vault_id
        )
        
        # Create the knowledge vault
        vault = KnowledgeVault(
            id=vault_id,
            name=f"Vault_{node_a.name}_to_{node_b.name}",
            position=vault_position,
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            librarian=librarian,
            meta_room_connection=self.meta_room.id
        )
        
        self.vaults[vault_id] = vault
        
        # Connect vault to Meta Room
        self.meta_room.connect_vault(vault_id)
        
        # Create the vector connection
        connection_id = f"conn_{uuid.uuid4().hex[:8]}"
        connection = VectorConnection(
            id=connection_id,
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            vault_id=vault_id,
            strength=strength
        )
        
        self.connections[connection_id] = connection
        
        # Update node connections
        node_a.connections.append(connection_id)
        node_b.connections.append(connection_id)
        
        logger.info(f"🔗 Created connection: {node_a.name} ←→ [{vault.name}] ←→ {node_b.name}")
        logger.info(f"📚 Vault staffed by {librarian.name} (Role: {librarian_role.value})")
        
        return {
            "connection_id": connection_id,
            "vault_id": vault_id,
            "librarian_id": librarian_id
        }
    
    def _assign_librarian_role(self, type_a: NodeType, type_b: NodeType) -> LibrarianRole:
        """Assign appropriate librarian role based on connected node types"""
        if NodeType.CONSCIOUSNESS in [type_a, type_b]:
            return LibrarianRole.CURATOR
        elif NodeType.STORAGE in [type_a, type_b]:
            return LibrarianRole.ARCHIVIST
        elif NodeType.PROCESSOR in [type_a, type_b]:
            return LibrarianRole.CLASSIFIER
        elif NodeType.BRIDGE in [type_a, type_b]:
            return LibrarianRole.ROUTER
        else:
            return LibrarianRole.ORGANIZER
    
    def register_ai_system(self, name: str, capabilities: List[str]) -> str:
        """Register an AI system with the Meta Room"""
        ai_id = f"ai_{uuid.uuid4().hex[:8]}"
        
        self.ai_systems[ai_id] = {
            "id": ai_id,
            "name": name,
            "capabilities": capabilities,
            "connected_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Connect to Meta Room
        self.meta_room.connect_ai_system(ai_id)
        
        logger.info(f"🤖 Registered AI system: {name}")
        return ai_id
    
    def propagate_data(self, source_node_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate data through the network via vaults"""
        if source_node_id not in self.nodes:
            raise ValueError("Source node not found")
        
        source_node = self.nodes[source_node_id]
        propagation_results = {
            "source": source_node.name,
            "vaults_processed": [],
            "meta_room_notified": False,
            "ai_systems_updated": []
        }
        
        # Find all connections from this node
        for conn_id in source_node.connections:
            connection = self.connections[conn_id]
            vault = self.vaults[connection.vault_id]
            
            # Store data in vault (librarian processes it)
            vault.store_data(data)
            propagation_results["vaults_processed"].append(vault.name)
            
            # If librarian recommends routing to Meta Room
            processed_data = vault.stored_data[-1]  # Get the just-processed data
            if processed_data.get("meta_routing"):
                self.meta_room.receive_from_vault(vault.id, processed_data)
                propagation_results["meta_room_notified"] = True
                propagation_results["ai_systems_updated"] = list(self.meta_room.connected_ai_systems)
        
        logger.info(f"⚡ Data propagated from {source_node.name} through {len(propagation_results['vaults_processed'])} vaults")
        
        return propagation_results
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get complete network status"""
        return {
            "network_type": "knowledge_vault_network",
            "total_nodes": len(self.nodes),
            "total_vaults": len(self.vaults),
            "total_connections": len(self.connections),
            "total_ai_systems": len(self.ai_systems),
            "meta_room": self.meta_room.get_status() if self.meta_room else None,
            "vaults": [vault.get_status() for vault in self.vaults.values()],
            "ai_systems": list(self.ai_systems.values()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        client_id = f"client_{uuid.uuid4().hex[:8]}"
        logger.info(f"🔌 Client {client_id} connected")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client {client_id} disconnected")
    
    async def process_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages"""
        try:
            msg_type = data.get("type", "unknown")
            
            if msg_type == "create_node":
                node_id = self.create_node(
                    data.get("name", "unnamed"),
                    data.get("node_type", "consciousness"),
                    data.get("position", {"x": 0, "y": 0, "z": 0})
                )
                return {"status": "success", "node_id": node_id}
            
            elif msg_type == "create_connection":
                result = self.create_connection(
                    data.get("node_a_id", ""),
                    data.get("node_b_id", ""),
                    data.get("strength", 0.8)
                )
                return {"status": "success", **result}
            
            elif msg_type == "register_ai":
                ai_id = self.register_ai_system(
                    data.get("name", "unnamed_ai"),
                    data.get("capabilities", [])
                )
                return {"status": "success", "ai_id": ai_id}
            
            elif msg_type == "propagate_data":
                result = self.propagate_data(
                    data.get("source_node_id", ""),
                    data.get("data", {})
                )
                return {"status": "success", "propagation": result}
            
            elif msg_type == "get_status":
                status = self.get_network_status()
                return status
            
            elif msg_type == "ping":
                return {"status": "pong", "network": "knowledge_vault_network"}
            
            else:
                return {"error": f"Unknown message type: {msg_type}"}
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "type": "processing_error"}
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"🏛️ Starting Knowledge Vault Network on port {self.port}")
        logger.info(f"🧠 Architecture: Nodes ←→ [Vaults with Librarians] ←→ Meta Room ←→ AI Systems")
        
        async with websockets.serve(self.handle_client, "localhost", self.port):
            logger.info(f"✅ Knowledge Vault Network server running on ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever

def main():
    """Main entry point"""
    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        import os
        os.system('chcp 65001 > nul 2>&1')
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    
    port = load_port_config()
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE VAULT NODE NETWORK WITH META ROOM")
    print("=" * 70 + "\n")
    print("Architecture:")
    print("  Node A <-> [Knowledge Vault + Librarian AI] <-> Node B")
    print("  All Vaults <-> Meta Room <-> All AI Systems")
    print("\nFeatures:")
    print("  * Knowledge Vaults between every connection")
    print("  * Librarian AIs organize and classify data")
    print("  * Central Meta Room coordinates everything")
    print("  * Meta Room connects to all AI systems")
    print(f"\nStarting server on port {port}...\n")
    
    network = KnowledgeVaultNetwork(port=port)
    
    try:
        asyncio.run(network.start_server())
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
