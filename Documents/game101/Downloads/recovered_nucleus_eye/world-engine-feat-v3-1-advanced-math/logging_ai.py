#!/usr/bin/env python3
"""
📊 LOGGING AI - CENTRAL LOG COLLECTOR & PROCESSOR
=================================================

Acts as a specialized AI that receives logs from all independent nodes,
processes them, shapes them to fit different needs, and organizes them.

Each node is standalone and passes its logs like a baton.
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
from collections import defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [LOGGING-AI] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class LogCategory(Enum):
    """Log categories for organization"""
    SYSTEM = "system"
    NODE_ACTIVITY = "node_activity"
    CONNECTION = "connection"
    DATA_FLOW = "data_flow"
    AI_PROCESS = "ai_process"
    VAULT_OPERATION = "vault_operation"
    META_ROOM = "meta_room"
    SECURITY = "security"
    PERFORMANCE = "performance"

@dataclass
class LogEntry:
    """Individual log entry"""
    id: str
    timestamp: str
    source_node: str
    level: LogLevel
    category: LogCategory
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class LogShape:
    """Different shaped/formatted logs for different consumers"""
    name: str
    format_type: str  # "json", "text", "html", "dashboard"
    filter_level: LogLevel
    filter_categories: List[LogCategory]
    include_data: bool = True

class LoggingAI:
    """
    Central Logging AI that collects, processes, shapes, and distributes logs.
    Each node is independent and passes logs like a baton to this AI.
    """
    
    def __init__(self, port: int = 8703):
        self.port = port
        self.logs: List[LogEntry] = []
        self.log_index: Dict[str, List[LogEntry]] = defaultdict(list)
        self.connected_nodes: Dict[str, Dict[str, Any]] = {}
        self.log_shapes: Dict[str, LogShape] = {}
        
        # Initialize default log shapes
        self._initialize_shapes()
    
    def _initialize_shapes(self):
        """Create default log shapes for different consumers"""
        
        # Shape 1: Critical Alerts Only
        self.log_shapes["critical_alerts"] = LogShape(
            name="Critical Alerts",
            format_type="json",
            filter_level=LogLevel.CRITICAL,
            filter_categories=[cat for cat in LogCategory],
            include_data=True
        )
        
        # Shape 2: System Overview
        self.log_shapes["system_overview"] = LogShape(
            name="System Overview",
            format_type="dashboard",
            filter_level=LogLevel.INFO,
            filter_categories=[LogCategory.SYSTEM, LogCategory.NODE_ACTIVITY],
            include_data=False
        )
        
        # Shape 3: AI Process Monitoring
        self.log_shapes["ai_monitoring"] = LogShape(
            name="AI Process Monitor",
            format_type="json",
            filter_level=LogLevel.DEBUG,
            filter_categories=[LogCategory.AI_PROCESS, LogCategory.PERFORMANCE],
            include_data=True
        )
        
        # Shape 4: Security Audit
        self.log_shapes["security_audit"] = LogShape(
            name="Security Audit",
            format_type="text",
            filter_level=LogLevel.WARNING,
            filter_categories=[LogCategory.SECURITY, LogCategory.CONNECTION],
            include_data=True
        )
        
        logger.info(f"📊 Initialized {len(self.log_shapes)} log shapes")
    
    def receive_log(self, log_data: Dict[str, Any]) -> str:
        """
        Receive log from an independent node (baton passed)
        """
        # Create log entry
        log_entry = LogEntry(
            id=f"log_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_node=log_data.get("source", "unknown"),
            level=LogLevel[log_data.get("level", "INFO").upper()],
            category=LogCategory[log_data.get("category", "SYSTEM").upper()],
            message=log_data.get("message", ""),
            data=log_data.get("data", {}),
            tags=log_data.get("tags", [])
        )
        
        # Store log
        self.logs.append(log_entry)
        
        # Index by source
        self.log_index[log_entry.source_node].append(log_entry)
        
        logger.info(f"📨 Received log from {log_entry.source_node}: {log_entry.message[:50]}")
        
        return log_entry.id
    
    def shape_logs(self, shape_name: str) -> Any:
        """
        Shape logs according to a specific format/need
        """
        if shape_name not in self.log_shapes:
            return {"error": "Unknown log shape"}
        
        shape = self.log_shapes[shape_name]
        
        # Filter logs based on shape criteria
        filtered_logs = []
        for log in self.logs:
            # Check level
            level_match = self._level_priority(log.level) >= self._level_priority(shape.filter_level)
            
            # Check category
            category_match = log.category in shape.filter_categories
            
            if level_match and category_match:
                filtered_logs.append(log)
        
        # Format according to shape
        if shape.format_type == "json":
            return self._format_json(filtered_logs, shape.include_data)
        elif shape.format_type == "text":
            return self._format_text(filtered_logs, shape.include_data)
        elif shape.format_type == "dashboard":
            return self._format_dashboard(filtered_logs)
        elif shape.format_type == "html":
            return self._format_html(filtered_logs, shape.include_data)
        else:
            return filtered_logs
    
    def _level_priority(self, level: LogLevel) -> int:
        """Get numeric priority for log level"""
        priorities = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
        return priorities.get(level, 0)
    
    def _format_json(self, logs: List[LogEntry], include_data: bool) -> Dict[str, Any]:
        """Format logs as JSON"""
        formatted = []
        for log in logs:
            entry = {
                "id": log.id,
                "timestamp": log.timestamp,
                "source": log.source_node,
                "level": log.level.value,
                "category": log.category.value,
                "message": log.message,
                "tags": log.tags
            }
            if include_data:
                entry["data"] = log.data
            formatted.append(entry)
        
        return {
            "count": len(formatted),
            "logs": formatted
        }
    
    def _format_text(self, logs: List[LogEntry], include_data: bool) -> str:
        """Format logs as plain text"""
        lines = []
        for log in logs:
            line = f"[{log.timestamp}] {log.level.value.upper()} - {log.source_node}: {log.message}"
            if include_data and log.data:
                line += f" | Data: {json.dumps(log.data)}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_dashboard(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Format logs for dashboard display"""
        # Group by source
        by_source = defaultdict(int)
        by_level = defaultdict(int)
        by_category = defaultdict(int)
        
        for log in logs:
            by_source[log.source_node] += 1
            by_level[log.level.value] += 1
            by_category[log.category.value] += 1
        
        return {
            "total_logs": len(logs),
            "by_source": dict(by_source),
            "by_level": dict(by_level),
            "by_category": dict(by_category),
            "recent_logs": [
                {
                    "source": log.source_node,
                    "level": log.level.value,
                    "message": log.message[:100]
                }
                for log in logs[-10:]  # Last 10 logs
            ]
        }
    
    def _format_html(self, logs: List[LogEntry], include_data: bool) -> str:
        """Format logs as HTML"""
        html = "<html><body><h1>Log Report</h1><table border='1'>"
        html += "<tr><th>Time</th><th>Source</th><th>Level</th><th>Message</th></tr>"
        
        for log in logs:
            color = self._level_color(log.level)
            html += f"<tr><td>{log.timestamp}</td><td>{log.source_node}</td>"
            html += f"<td style='color:{color}'>{log.level.value.upper()}</td>"
            html += f"<td>{log.message}</td></tr>"
        
        html += "</table></body></html>"
        return html
    
    def _level_color(self, level: LogLevel) -> str:
        """Get color for log level"""
        colors = {
            LogLevel.DEBUG: "gray",
            LogLevel.INFO: "blue",
            LogLevel.WARNING: "orange",
            LogLevel.ERROR: "red",
            LogLevel.CRITICAL: "darkred"
        }
        return colors.get(level, "black")
    
    def get_node_logs(self, node_id: str) -> List[LogEntry]:
        """Get all logs from a specific node"""
        return self.log_index.get(node_id, [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "total_logs": len(self.logs),
            "total_nodes": len(self.connected_nodes),
            "logs_by_level": {
                level.value: sum(1 for log in self.logs if log.level == level)
                for level in LogLevel
            },
            "logs_by_category": {
                cat.value: sum(1 for log in self.logs if log.category == cat)
                for cat in LogCategory
            },
            "active_shapes": list(self.log_shapes.keys())
        }
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket connections from standalone nodes"""
        client_id = f"client_{uuid.uuid4().hex[:8]}"
        logger.info(f"🔌 Node connected: {client_id}")
        
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
            logger.info(f"🔌 Node disconnected: {client_id}")
    
    async def process_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages from nodes"""
        try:
            msg_type = data.get("type", "unknown")
            
            if msg_type == "log":
                # Node passing log baton
                log_id = self.receive_log(data)
                return {"status": "success", "log_id": log_id}
            
            elif msg_type == "get_shaped_logs":
                # Request logs in specific shape
                shape_name = data.get("shape", "system_overview")
                shaped_logs = self.shape_logs(shape_name)
                return {"status": "success", "shaped_logs": shaped_logs}
            
            elif msg_type == "get_node_logs":
                # Get logs from specific node
                node_id = data.get("node_id", "")
                logs = self.get_node_logs(node_id)
                return {
                    "status": "success",
                    "node_id": node_id,
                    "count": len(logs),
                    "logs": [asdict(log) for log in logs]
                }
            
            elif msg_type == "get_statistics":
                # Get logging statistics
                stats = self.get_statistics()
                return {"status": "success", "statistics": stats}
            
            elif msg_type == "register_node":
                # Node registering itself
                node_id = data.get("node_id", f"node_{uuid.uuid4().hex[:8]}")
                self.connected_nodes[node_id] = {
                    "id": node_id,
                    "name": data.get("name", "unnamed"),
                    "type": data.get("node_type", "unknown"),
                    "connected_at": datetime.now(timezone.utc).isoformat()
                }
                logger.info(f"📝 Registered node: {node_id}")
                return {"status": "success", "node_id": node_id}
            
            elif msg_type == "ping":
                return {"status": "pong", "ai": "logging_ai"}
            
            else:
                return {"error": f"Unknown message type: {msg_type}"}
                
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
            return {"error": str(e)}
    
    async def start_server(self):
        """Start the Logging AI WebSocket server"""
        logger.info(f"📊 Starting Logging AI on port {self.port}")
        logger.info("🏃 Each node runs standalone and passes logs like a baton")
        
        async with websockets.serve(self.handle_client, "localhost", self.port):
            logger.info(f"✅ Logging AI running on ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever

def main():
    """Main entry point"""
    print("\n" + "📊" * 30)
    print("LOGGING AI - CENTRAL LOG PROCESSOR")
    print("📊" * 30 + "\n")
    print("Standalone nodes pass their logs like a baton 🏃→📨→🤖")
    print("\nFeatures:")
    print("  📨 Receives logs from independent nodes")
    print("  🔄 Shapes logs to fit different needs")
    print("  📊 Organizes and indexes all logs")
    print("  🎯 Filters by level and category")
    print("  📈 Provides statistics and analytics\n")
    
    logging_ai = LoggingAI(port=8703)
    
    try:
        asyncio.run(logging_ai.start_server())
    except KeyboardInterrupt:
        print("\n\n🛑 Logging AI stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
