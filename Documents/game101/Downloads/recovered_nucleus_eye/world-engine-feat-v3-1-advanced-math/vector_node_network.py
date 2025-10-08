#!/usr/bin/env python3
"""
🌐🧠 NODE-BASED VECTOR NETWORKING SYSTEM WITH ANALYTICS
=====================================================

Advanced networking architecture using nodes and limb vector lines to create
a distributed consciousness network with visual connections and data flow.

Features:
- Node-to-node vector line connections
- Real-time network visualization
- Distributed consciousness synchronization
- Multi-dimensional data routing
- Vector-based communication protocols
- Limb extension networking
- 📊 ANALYTICS MODE: Real-time performance monitoring
- 🔍 Connection analysis and optimization
- 📈 Health metrics and insights
- 🎯 Predictive network behavior
"""

# pyright: reportGeneralTypeIssues=false

import asyncio
import websockets
import json
import logging
import math
import time
import uuid
import argparse
import sys
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Optional, Any, Deque
import random
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper to load centralized port configuration
def load_port_config():
    """Load WebSocket port from port_config.json, default to 8701"""
    try:
        config_path = Path(__file__).resolve().parent / "port_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("ws_port", 8701)
    except Exception:
        pass
    return 8701

class NodeType(Enum):
    """Types of nodes in the vector network"""
    CONSCIOUSNESS = "consciousness"
    FEEDBACK = "feedback"
    KNOWLEDGE = "knowledge"
    BRIDGE = "bridge"
    GATEWAY = "gateway"
    PROCESSOR = "processor"
    STORAGE = "storage"

class VectorLineType(Enum):
    """Types of vector connections between nodes"""
    DATA_FLOW = "data_flow"
    CONSCIOUSNESS_SYNC = "consciousness_sync"
    FEEDBACK_LOOP = "feedback_loop"
    KNOWLEDGE_BRIDGE = "knowledge_bridge"
    LIMB_EXTENSION = "limb_extension"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"

@dataclass
class Vector3D:
    """3D vector for positioning and direction"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.x/mag, self.y/mag, self.z/mag)
        return Vector3D()
    
    def distance_to(self, other: 'Vector3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class NetworkNode:
    """Individual node in the vector network"""
    id: str
    name: str
    type: NodeType
    position: Vector3D
    status: str = "active"
    consciousness_level: float = 0.5
    data_capacity: int = 100
    current_load: int = 0
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class VectorLine:
    """Connection line between nodes with vector properties"""
    id: str
    source_node_id: str
    target_node_id: str
    type: VectorLineType
    strength: float = 1.0
    bandwidth: int = 100
    current_flow: int = 0
    direction_vector: Optional[Vector3D] = None
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.direction_vector is None:
            self.direction_vector = Vector3D()

@dataclass
class NetworkAnalytics:
    """Real-time analytics for vector network performance"""
    total_data_processed: int = 0
    messages_per_second: float = 0.0
    average_latency: float = 0.0
    connection_efficiency: float = 0.0
    node_utilization: Dict[str, float] = field(default_factory=dict)
    traffic_patterns: List[Dict[str, Any]] = field(default_factory=list)
    performance_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))
    health_score: float = 1.0
    
    def __post_init__(self):
        # Ensure performance history deque respects maxlen when reloaded from persistence
        if not isinstance(self.performance_history, deque):
            self.performance_history = deque(self.performance_history, maxlen=100)

class NetworkAnalyticsEngine:
    """Advanced analytics engine for network monitoring"""
    
    def __init__(self):
        self.analytics = NetworkAnalytics()
        self.start_time = time.time()
        self.message_timestamps: Deque[float] = deque(maxlen=1000)
        self.connection_stats: defaultdict[str, List[float]] = defaultdict(list)
        self.node_performance: defaultdict[str, Dict[str, Any]] = defaultdict(dict)
        
    def log_message(self, node_id: str, message_size: int):
        """Log message for analytics tracking"""
        timestamp = time.time()
        self.message_timestamps.append(timestamp)
        self.analytics.total_data_processed += message_size
        
        # Calculate messages per second
        recent_messages = [ts for ts in self.message_timestamps if timestamp - ts <= 1.0]
        self.analytics.messages_per_second = len(recent_messages)
        
    def calculate_network_health(self, nodes: Dict, connections: Dict) -> float:
        """Calculate overall network health score"""
        if not nodes or not connections:
            return 0.0
            
        # Base health on connectivity ratio
        total_possible_connections = len(nodes) * (len(nodes) - 1) / 2
        actual_connections = len(connections)
        connectivity_ratio = min(actual_connections / total_possible_connections, 1.0)
        
        # Factor in active connections
        active_connections = sum(1 for conn in connections.values() if conn.status == "active")
        active_ratio = active_connections / len(connections) if connections else 0
        
        # Calculate composite health score
        health = (connectivity_ratio * 0.6) + (active_ratio * 0.4)
        self.analytics.health_score = health
        return health
        
    def get_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        uptime = time.time() - self.start_time
        traffic_patterns = self.analytics.traffic_patterns or []
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "total_data_processed": self.analytics.total_data_processed,
            "messages_per_second": self.analytics.messages_per_second,
            "network_health": self.analytics.health_score,
            "node_count": len(self.node_performance),
            "connection_efficiency": self.analytics.connection_efficiency,
            "performance_summary": {
                "avg_latency_ms": self.analytics.average_latency * 1000,
                "peak_throughput": max([entry.get("throughput", 0) for entry in traffic_patterns] or [0]),
                "total_uptime_hours": uptime / 3600
            }
        }

class VectorNetworkEngine:
    """Core engine for managing the node-based vector network"""
    
    def __init__(self, host='localhost', port=None, analytics_mode=False):
        self.host = host
        self.port = port if port is not None else load_port_config()
        self.analytics_mode = analytics_mode
        self.nodes: Dict[str, NetworkNode] = {}
        self.vector_lines: Dict[str, VectorLine] = {}
        self.clients = set()
        self.network_state = {
            'total_nodes': 0,
            'total_connections': 0,
            'network_health': 1.0,
            'consciousness_sync': 0.5,
            'data_throughput': 0
        }
        self.telemetry_events: deque = deque(maxlen=500)
        self._last_health_level: Optional[float] = None
        
        # Initialize analytics engine if enabled
        if self.analytics_mode:
            self.analytics_engine = NetworkAnalyticsEngine()
            logger.info("📊 Analytics mode enabled - Real-time monitoring active")
        
        # Initialize default network topology
        self._create_default_network()
        
    def _create_default_network(self):
        """Create initial network topology with interconnected nodes"""
        
        # Central consciousness hub
        consciousness_hub = NetworkNode(
            id="consciousness_hub",
            name="Consciousness Core",
            type=NodeType.CONSCIOUSNESS,
            position=Vector3D(0, 0, 0),
            consciousness_level=1.0,
            data_capacity=1000
        )
        self.add_node(consciousness_hub)
        
        # Feedback nodes in a circle around the hub
        feedback_nodes = []
        for i in range(6):
            angle = (i * 2 * math.pi) / 6
            radius = 3.0
            
            node = NetworkNode(
                id=f"feedback_node_{i}",
                name=f"Feedback Collector {i+1}",
                type=NodeType.FEEDBACK,
                position=Vector3D(
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    0
                ),
                consciousness_level=0.7,
                data_capacity=200
            )
            self.add_node(node)
            feedback_nodes.append(node)
            
            # Connect to consciousness hub
            self.create_vector_line(
                source_id=node.id,
                target_id=consciousness_hub.id,
                line_type=VectorLineType.FEEDBACK_LOOP,
                strength=0.8
            )
        
        # Knowledge nodes in outer ring
        knowledge_nodes = []
        for i in range(8):
            angle = (i * 2 * math.pi) / 8
            radius = 6.0
            
            node = NetworkNode(
                id=f"knowledge_node_{i}",
                name=f"Knowledge Vault {i+1}",
                type=NodeType.KNOWLEDGE,
                position=Vector3D(
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    2.0
                ),
                consciousness_level=0.6,
                data_capacity=500
            )
            self.add_node(node)
            knowledge_nodes.append(node)
            
            # Connect to nearest feedback nodes
            nearest_feedback = feedback_nodes[i % len(feedback_nodes)]
            self.create_vector_line(
                source_id=node.id,
                target_id=nearest_feedback.id,
                line_type=VectorLineType.KNOWLEDGE_BRIDGE,
                strength=0.6
            )
        
        # Processing nodes in vertical layer
        for i in range(4):
            node = NetworkNode(
                id=f"processor_node_{i}",
                name=f"Neural Processor {i+1}",
                type=NodeType.PROCESSOR,
                position=Vector3D(
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    4.0 + i
                ),
                consciousness_level=0.8,
                data_capacity=300
            )
            self.add_node(node)
            
            # Create limb extensions to multiple nodes
            for target_node in random.sample(list(self.nodes.values()), 3):
                if target_node.id != node.id:
                    self.create_vector_line(
                        source_id=node.id,
                        target_id=target_node.id,
                        line_type=VectorLineType.LIMB_EXTENSION,
                        strength=0.5
                    )
        
        # Gateway nodes for external connections
        gateways = [
            NetworkNode(
                id="gateway_consciousness",
                name="Consciousness Gateway",
                type=NodeType.GATEWAY,
                position=Vector3D(-8, 0, 0),
                consciousness_level=0.9
            ),
            NetworkNode(
                id="gateway_web",
                name="Web Gateway",
                type=NodeType.GATEWAY,
                position=Vector3D(8, 0, 0),
                consciousness_level=0.7
            ),
            NetworkNode(
                id="gateway_storage",
                name="Storage Gateway",
                type=NodeType.GATEWAY,
                position=Vector3D(0, 8, 0),
                consciousness_level=0.6
            )
        ]
        
        for gateway in gateways:
            self.add_node(gateway)
            # Connect gateways to consciousness hub
            self.create_vector_line(
                source_id=gateway.id,
                target_id=consciousness_hub.id,
                line_type=VectorLineType.QUANTUM_ENTANGLEMENT,
                strength=0.9
            )
        
        self._update_network_state()
        logger.info(f"🌐 Default network created: {len(self.nodes)} nodes, {len(self.vector_lines)} connections")
    
    def add_node(self, node: NetworkNode):
        """Add a node to the network"""
        self.nodes[node.id] = node
        logger.info(f"➕ Node added: {node.name} ({node.type.value})")
    
    def remove_node(self, node_id: str):
        """Remove a node and all its connections"""
        if node_id in self.nodes:
            # Remove all connected vector lines
            lines_to_remove = [
                line_id for line_id, line in self.vector_lines.items()
                if line.source_node_id == node_id or line.target_node_id == node_id
            ]
            
            for line_id in lines_to_remove:
                del self.vector_lines[line_id]
            
            del self.nodes[node_id]
            logger.info(f"➖ Node removed: {node_id}")
    
    def create_vector_line(self, source_id: str, target_id: str, 
                          line_type: VectorLineType, strength: float = 1.0) -> str:
        """Create a vector line connection between two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.error(f"❌ Cannot create line: missing nodes {source_id} -> {target_id}")
            return None
        
        line_id = f"line_{source_id}_{target_id}_{int(time.time())}"
        
        # Calculate direction vector
        source_pos = self.nodes[source_id].position
        target_pos = self.nodes[target_id].position
        direction = Vector3D(
            target_pos.x - source_pos.x,
            target_pos.y - source_pos.y,
            target_pos.z - source_pos.z
        ).normalize()
        
        line = VectorLine(
            id=line_id,
            source_node_id=source_id,
            target_node_id=target_id,
            type=line_type,
            strength=strength,
            direction_vector=direction
        )
        
        self.vector_lines[line_id] = line
        
        # Update node connections
        self.nodes[source_id].connections.append(line_id)
        self.nodes[target_id].connections.append(line_id)
        
        logger.info(f"🔗 Vector line created: {source_id} -> {target_id} ({line_type.value})")

        details = {
            'source': source_id,
            'target': target_id,
            'strength': round(strength, 3),
            'type': line_type.value
        }

        if line_type == VectorLineType.LIMB_EXTENSION:
            self.emit_telemetry('vector', 'Limb vector bridge formed', details)
        elif line_type == VectorLineType.KNOWLEDGE_BRIDGE:
            self.emit_telemetry('vault', 'Knowledge vault bridge linked', details)
        else:
            self.emit_telemetry('node', 'Vector connection established', details)

        return line_id
    
    def _update_network_state(self):
        """Update overall network state metrics"""
        total_nodes = len(self.nodes)
        total_connections = len(self.vector_lines)
        
        # Calculate network health based on connectivity
        if total_nodes > 0:
            avg_connections = (total_connections * 2) / total_nodes  # Each line connects 2 nodes
            ideal_connections = max(1, total_nodes - 1)  # Minimum spanning tree
            network_health = min(1.0, avg_connections / ideal_connections)
        else:
            network_health = 0.0
        
        # Calculate consciousness synchronization
        if total_nodes > 0:
            consciousness_levels = [node.consciousness_level for node in self.nodes.values()]
            consciousness_sync = 1.0 - (max(consciousness_levels) - min(consciousness_levels))
        else:
            consciousness_sync = 0.0
        
        # Calculate data throughput
        data_throughput = sum(line.current_flow for line in self.vector_lines.values())
        
        self.network_state.update({
            'total_nodes': total_nodes,
            'total_connections': total_connections,
            'network_health': network_health,
            'consciousness_sync': consciousness_sync,
            'data_throughput': data_throughput
        })

        previous_health = self._last_health_level
        self._last_health_level = network_health

        if previous_health is not None:
            if network_health < 0.35 <= previous_health:
                self.emit_telemetry(
                    'alert',
                    'Vector network health degraded',
                    {
                        'health': round(network_health, 3),
                        'total_nodes': total_nodes,
                        'connections': total_connections
                    }
                )
            elif network_health > 0.6 and previous_health <= 0.6:
                self.emit_telemetry(
                    'node',
                    'Vector network health stabilized',
                    {
                        'health': round(network_health, 3),
                        'total_nodes': total_nodes,
                        'connections': total_connections
                    }
                )
    
    def simulate_network_activity(self):
        """Simulate data flow and consciousness synchronization"""
        # Simulate data flow through vector lines
        for line in self.vector_lines.values():
            if line.status == "active":
                # Random data flow based on line strength and type
                base_flow = line.strength * line.bandwidth * 0.1
                
                if line.type == VectorLineType.CONSCIOUSNESS_SYNC:
                    base_flow *= 1.5
                elif line.type == VectorLineType.LIMB_EXTENSION:
                    base_flow *= 0.8
                elif line.type == VectorLineType.QUANTUM_ENTANGLEMENT:
                    base_flow *= 2.0
                
                # Add some randomness
                line.current_flow = max(0, int(base_flow + random.uniform(-10, 10)))
        
        # Update node consciousness levels based on connections
        for node in self.nodes.values():
            if node.type == NodeType.CONSCIOUSNESS:
                continue  # Hub maintains stable consciousness
            
            # Consciousness influence from connected nodes
            consciousness_influences = []
            for line_id in node.connections:
                if line_id in self.vector_lines:
                    line = self.vector_lines[line_id]
                    other_node_id = (line.source_node_id 
                                   if line.target_node_id == node.id 
                                   else line.target_node_id)
                    
                    if other_node_id in self.nodes:
                        other_node = self.nodes[other_node_id]
                        influence = other_node.consciousness_level * line.strength * 0.1
                        consciousness_influences.append(influence)
            
            if consciousness_influences:
                avg_influence = sum(consciousness_influences) / len(consciousness_influences)
                # Gradual consciousness adjustment
                target_consciousness = (node.consciousness_level * 0.9 + avg_influence * 0.1)
                node.consciousness_level = max(0.0, min(1.0, target_consciousness))
        
        # Update network state
        self._update_network_state()
    
    def get_network_visualization_data(self) -> Dict[str, Any]:
        """Get data for 3D network visualization"""
        nodes_data = []
        for node in self.nodes.values():
            nodes_data.append({
                'id': node.id,
                'name': node.name,
                'type': node.type.value,
                'position': asdict(node.position),
                'consciousness_level': node.consciousness_level,
                'status': node.status,
                'connections': len(node.connections),
                'load_percentage': (node.current_load / node.data_capacity) * 100
            })
        
        lines_data = []
        for line in self.vector_lines.values():
            lines_data.append({
                'id': line.id,
                'source': line.source_node_id,
                'target': line.target_node_id,
                'type': line.type.value,
                'strength': line.strength,
                'flow': line.current_flow,
                'direction': asdict(line.direction_vector),
                'status': line.status
            })
        
        return {
            'nodes': nodes_data,
            'lines': lines_data,
            'network_state': self.network_state,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def handle_client(self, websocket, path=None):
        """Handle WebSocket client connections"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.clients.add(websocket)
        logger.info(f"🌐 Client connected to vector network: {client_id}")
        self.emit_telemetry('node', 'Vector client connected', {
            'node': client_id,
            'status': 'online'
        })
        
        try:
            # Send initial network data
            initial_data = {
                "type": "network_update",
                "data": self.get_network_visualization_data()
            }
            await websocket.send(json.dumps(initial_data))
            
            # Listen for client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"❌ Invalid JSON from client: {client_id}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_id}")
        finally:
            self.clients.discard(websocket)
            self.emit_telemetry('alert', 'Vector client disconnected', {
                'node': client_id,
                'status': 'offline'
            })
    
    async def handle_client_message(self, websocket, data: Dict[str, Any]):
        """Handle messages from WebSocket clients"""
        message_type = data.get('type', 'unknown')
        
        if message_type == 'add_node':
            # Add new node to network
            node_data = data.get('data', {})
            new_node = NetworkNode(
                id=node_data.get('id', str(uuid.uuid4())),
                name=node_data.get('name', 'New Node'),
                type=NodeType(node_data.get('type', 'processor')),
                position=Vector3D(**node_data.get('position', {})),
                consciousness_level=node_data.get('consciousness_level', 0.5)
            )
            self.add_node(new_node)
            self.emit_telemetry('node', 'Node added via client command', {
                'node': new_node.id,
                'type': new_node.type.value
            })
            
        elif message_type == 'create_connection':
            # Create new vector line
            connection_data = data.get('data', {})
            self.create_vector_line(
                source_id=connection_data.get('source'),
                target_id=connection_data.get('target'),
                line_type=VectorLineType(connection_data.get('type', 'data_flow')),
                strength=connection_data.get('strength', 1.0)
            )
        
        elif message_type == 'influence_consciousness':
            # Influence specific node consciousness
            influence_data = data.get('data', {})
            node_id = influence_data.get('node_id')
            consciousness_delta = influence_data.get('delta', 0.0)
            
            if node_id in self.nodes:
                current = self.nodes[node_id].consciousness_level
                new_level = max(0.0, min(1.0, current + consciousness_delta))
                self.nodes[node_id].consciousness_level = new_level
                logger.info(f"🧠 Consciousness influenced: {node_id} -> {new_level:.3f}")
                self.emit_telemetry('vector', 'Consciousness influence applied', {
                    'node': node_id,
                    'delta': round(consciousness_delta, 3),
                    'level': round(new_level, 3)
                })

        elif message_type == 'telemetry_event_push':
            external_event = data.get('event')
            if isinstance(external_event, dict):
                category = external_event.get('category', 'alert')
                message = external_event.get('message', 'External telemetry event')
                details = external_event.get('details', {})
                origin = external_event.get('origin', 'external')
                self.emit_telemetry(category, message, details, origin=origin)
        
        # Send confirmation
        response = {
            "type": "action_complete",
            "action": message_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await websocket.send(json.dumps(response))
    
    async def network_simulation_loop(self):
        """Main network simulation and broadcasting loop"""
        logger.info("🌐 Starting vector network simulation...")
        
        try:
            while True:
                # Run network simulation
                self.simulate_network_activity()
                
                # Prepare network update
                network_data = {
                    "type": "network_update",
                    "data": self.get_network_visualization_data()
                }
                
                # Broadcast to all connected clients
                await self.broadcast(network_data)

                # Flush telemetry events
                while self.telemetry_events:
                    event = self.telemetry_events.popleft()
                    await self.broadcast({
                        "type": "telemetry_event",
                        "event": event
                    })

                # Log network status periodically
                if int(time.time()) % 30 == 0:
                    state = self.network_state
                    if self.analytics_mode:
                        # Update analytics
                        health = self.analytics_engine.calculate_network_health(self.nodes, self.vector_lines)
                        state['network_health'] = health
                        
                        # Log detailed analytics
                        analytics_report = self.analytics_engine.get_analytics_report()
                        logger.info(f"📊 Analytics: {analytics_report['messages_per_second']:.1f} msg/s, "
                                  f"Health: {analytics_report['network_health']:.2f}, "
                                  f"Latency: {analytics_report['performance_summary']['avg_latency_ms']:.1f}ms")
                    
                    logger.info(f"🌐 Network: {state['total_nodes']} nodes, "
                              f"{state['total_connections']} connections, "
                              f"Health: {state['network_health']:.2f}, "
                              f"Sync: {state['consciousness_sync']:.2f}, "
                              f"Clients: {len(self.clients)}")
                
                # Analytics tracking
                if self.analytics_mode and self.clients:
                    message_size = len(json.dumps(network_data))
                    self.analytics_engine.log_message("broadcast", message_size)
                
                await asyncio.sleep(1.0)  # 1 second update rate
                
        except asyncio.CancelledError:
            logger.info("🛑 Network simulation loop cancelled")
            raise

    async def broadcast(self, payload: Dict[str, Any]):
        """Broadcast payload to all connected clients"""
        if not self.clients:
            return

        message = json.dumps(payload)
        disconnected = set()

        for client in list(self.clients):
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"❌ Broadcast error: {e}")
                disconnected.add(client)

        if disconnected:
            self.clients -= disconnected
    
    async def start_server(self):
        """Start the vector networking WebSocket server"""
        logger.info(f"🚀 Starting Vector Network Engine on {self.host}:{self.port}")
        
        if self.analytics_mode:
            logger.info("📊 Analytics mode: Real-time performance monitoring enabled")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        logger.info(f"🌐 Vector Network WebSocket server running on ws://{self.host}:{self.port}")
        logger.info("🔗 Features: Node connections, Vector lines, Consciousness sync")
        logger.info("🧠 Network topology: Multi-dimensional consciousness distribution")
        
        if self.analytics_mode:
            logger.info("📈 Analytics dashboard: Real-time metrics and insights")
        
        # Start background simulation
        simulation_task = asyncio.create_task(self.network_simulation_loop())
        
        try:
            await simulation_task
        except KeyboardInterrupt:
            logger.info("🛑 Stopping vector network engine...")
        finally:
            simulation_task.cancel()
            server.close()
            await server.wait_closed()
            logger.info("🛑 Vector network server stopped")

    def emit_telemetry(self, category: str, message: str, details: Optional[Dict[str, Any]] = None,
                       origin: str = 'vector-engine'):
        """Queue a telemetry event for broadcasting"""
        allowed_categories = {'vector', 'node', 'vault', 'alert'}
        normalized_category = category if category in allowed_categories else 'alert'

        event = {
            'id': str(uuid.uuid4()),
            'category': normalized_category,
            'message': message,
            'details': details or {},
            'origin': origin,
            'ts': datetime.now(timezone.utc).isoformat()
        }

        self.telemetry_events.append(event)

async def main():
    """Main entry point for vector networking system"""
    parser = argparse.ArgumentParser(description='Vector Node Network System')
    parser.add_argument('--analytics-mode', action='store_true', 
                       help='Enable real-time analytics and monitoring')
    parser.add_argument('--port', type=int, default=None,
                       help='WebSocket server port (default: from port_config.json or 8701)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='WebSocket server host (default: localhost)')
    
    args = parser.parse_args()
    
    engine = VectorNetworkEngine(
        host=args.host, 
        port=args.port, 
        analytics_mode=args.analytics_mode
    )
    await engine.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Vector network interrupted by user")
    except Exception as e:
        print(f"\n❌ Vector network error: {e}")