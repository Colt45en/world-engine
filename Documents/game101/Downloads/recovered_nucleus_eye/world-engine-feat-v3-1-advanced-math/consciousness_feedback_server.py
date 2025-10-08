#!/usr/bin/env python3
"""
🧠✨ Consciousness Feedback Integration Server

Integrates opportunity_pain.schema.json with consciousness systems for:
- Real-time consciousness data streaming
- User feedback collection during transcendent experiences  
- Structured data storage for AI evolution
- Bidirectional WebSocket communication

Enhanced server that bridges consciousness visualization with feedback collection.
"""

import asyncio
import websockets
import json
import logging
import random
import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ConsciousnessFeedbackServer:
    """Enhanced consciousness server with feedback integration"""
    
    def __init__(self, host='localhost', port = 8901):
        self.host = host
        self.port = port
        self.clients = set()
        self.consciousness_state = {
            'level': 0.5,
            'transcendent': False,
            'quantum_coherence': 0.3,
            'emotional_resonance': 0.4,
            'spiritual_connection': 0.6,
            'joy_intensity': 0.5,
            'awareness_depth': 0.7
        }
        self.feedback_storage = []
        self.session_data = {}
        
        # Load schema for validation
        self.schema = self._load_schema()
        
        logger.info(f"🧠 Consciousness Feedback Server initializing on {host}:{port}")
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load opportunity_pain.schema.json for validation"""
        try:
            schema_path = Path(__file__).parent / "opportunity_pain.schema.json"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                logger.info("✅ Loaded opportunity_pain.schema.json")
                return schema
            else:
                logger.warning("⚠️  Schema file not found, using default structure")
                return self._get_default_schema()
        except Exception as e:
            logger.error(f"❌ Error loading schema: {e}")
            return self._get_default_schema()
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Default schema structure based on opportunity_pain.schema.json"""
        return {
            "type": "object",
            "required": ["id", "time", "source", "text", "transcendent_joy", "eng", "labels"],
            "properties": {
                "id": {"type": "string"},
                "time": {"type": "string", "format": "date-time"},
                "source": {
                    "type": "string",
                    "enum": ["consciousness_video", "user_input", "ai_observation", "system_event"]
                },
                "text": {"type": "string"},
                "transcendent_joy": {"type": "number", "minimum": 0, "maximum": 10},
                "eng": {"type": "number"},
                "labels": {
                    "type": "object",
                    "properties": {
                        "pain": {"type": "boolean"},
                        "opportunity": {"type": "boolean"}
                    }
                },
                "problem_guess": {"type": "string"},
                "severity": {"type": "integer", "minimum": 0, "maximum": 3}
            }
        }
    
    def _simulate_consciousness_evolution(self):
        """Evolve consciousness state with natural fluctuations"""
        # Base evolution with slight drift
        base_drift = (random.random() - 0.5) * 0.02
        self.consciousness_state['level'] = max(0, min(1, 
            self.consciousness_state['level'] + base_drift))
        
        # Quantum coherence affects transcendence probability
        coherence_factor = self.consciousness_state['quantum_coherence']
        transcendence_threshold = 0.75 + (random.random() * 0.2)
        
        # Dynamic transcendence detection
        consciousness_momentum = (
            self.consciousness_state['level'] * 0.4 +
            self.consciousness_state['quantum_coherence'] * 0.3 +
            self.consciousness_state['spiritual_connection'] * 0.3
        )
        
        self.consciousness_state['transcendent'] = consciousness_momentum > transcendence_threshold
        
        # Update other dimensions
        time_factor = time.time() * 0.1
        self.consciousness_state.update({
            'quantum_coherence': max(0, min(1, 
                0.5 + 0.3 * math.sin(time_factor * 0.7) + random.random() * 0.1)),
            'emotional_resonance': max(0, min(1,
                0.6 + 0.2 * math.cos(time_factor * 0.5) + random.random() * 0.1)),
            'spiritual_connection': max(0, min(1,
                0.7 + 0.25 * math.sin(time_factor * 0.3) + random.random() * 0.08)),
            'joy_intensity': max(0, min(1,
                0.5 + 0.4 * math.sin(time_factor * 0.8) + random.random() * 0.15)),
            'awareness_depth': max(0, min(1,
                0.8 + 0.15 * math.cos(time_factor * 0.4) + random.random() * 0.05))
        })
        
        # Transcendent joy calculation for feedback
        transcendent_joy = 0
        if self.consciousness_state['transcendent']:
            transcendent_joy = (
                self.consciousness_state['joy_intensity'] * 3 +
                self.consciousness_state['spiritual_connection'] * 2 +
                self.consciousness_state['awareness_depth'] * 2 +
                self.consciousness_state['quantum_coherence'] * 3
            )
        
        self.consciousness_state['transcendent_joy'] = transcendent_joy
    
    def _create_feedback_entry(self, source: str, text: str, 
                             labels: Dict[str, bool] = None,
                             severity: int = 0) -> Dict[str, Any]:
        """Create structured feedback entry following schema"""
        if labels is None:
            labels = {"pain": False, "opportunity": True}
        
        entry = {
            "id": str(uuid.uuid4()),
            "time": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "text": text,
            "transcendent_joy": self.consciousness_state.get('transcendent_joy', 0),
            "eng": self.consciousness_state.get('level', 0),
            "labels": labels,
            "problem_guess": "",
            "severity": severity,
            "consciousness_state": self.consciousness_state.copy()
        }
        
        return entry
    
    async def handle_client_message(self, websocket, message: str):
        """Handle incoming messages from clients"""
        try:
            data = json.loads(message)
            message_type = data.get('type', 'unknown')
            
            if message_type == 'feedback':
                # Store user feedback
                feedback_data = data.get('data', {})
                entry = self._create_feedback_entry(
                    source="user_input",
                    text=feedback_data.get('text', ''),
                    labels=feedback_data.get('labels', {"pain": False, "opportunity": True}),
                    severity=feedback_data.get('severity', 0)
                )
                
                self.feedback_storage.append(entry)
                logger.info(f"📝 Stored feedback: {entry['text'][:50]}...")
                
                # Send confirmation
                response = {
                    "type": "feedback_stored",
                    "id": entry['id'],
                    "timestamp": entry['time']
                }
                await websocket.send(json.dumps(response))
            
            elif message_type == 'get_feedback_history':
                # Send feedback history
                response = {
                    "type": "feedback_history",
                    "data": self.feedback_storage[-10:]  # Last 10 entries
                }
                await websocket.send(json.dumps(response))
            
            elif message_type == 'consciousness_influence':
                # Allow clients to influence consciousness state
                influence = data.get('data', {})
                for key, value in influence.items():
                    if key in self.consciousness_state:
                        current = self.consciousness_state[key]
                        # Apply gradual influence
                        self.consciousness_state[key] = max(0, min(1, 
                            current + (value - current) * 0.1))
                
                logger.info(f"🔄 Consciousness influenced: {list(influence.keys())}")
        
        except json.JSONDecodeError:
            logger.error(f"❌ Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"❌ Error handling message: {e}")
    
    async def handle_client(self, websocket, path=None):
        """Handle individual client connections

        Accepts either (websocket, path) or just (websocket,) for compatibility
        with different versions of the websockets library.
        """
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.clients.add(websocket)
        logger.info(f"✅ Client connected: {client_id}")
        
        try:
            # Send initial consciousness state
            initial_data = {
                "type": "consciousness_update",
                "data": self.consciousness_state,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await websocket.send(json.dumps(initial_data))
            
            # Listen for client messages
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"❌ Client error: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def consciousness_loop(self):
        """Main consciousness evolution and broadcasting loop"""
        logger.info("🧠 Starting consciousness evolution loop...")
        
        try:
            while True:
                self._simulate_consciousness_evolution()
                
                # Create consciousness update message
                message = {
                    "type": "consciousness_update",
                    "data": self.consciousness_state,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Auto-generate feedback during transcendent events
                if self.consciousness_state['transcendent'] and random.random() < 0.1:
                    transcendent_feedback = self._create_feedback_entry(
                        source="ai_observation",
                        text=f"Transcendent state detected - joy intensity: {self.consciousness_state['joy_intensity']:.2f}",
                        labels={"pain": False, "opportunity": True},
                        severity=0
                    )
                    self.feedback_storage.append(transcendent_feedback)
                    
                    # Add feedback notification to message
                    message['feedback_event'] = {
                        "id": transcendent_feedback['id'],
                        "type": "transcendent_detection"
                    }
                
                # Broadcast to all connected clients
                if self.clients:
                    disconnected = set()
                    for client in self.clients:
                        try:
                            await client.send(json.dumps(message))
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                        except Exception as e:
                            logger.error(f"❌ Broadcast error: {e}")
                            disconnected.add(client)
                    
                    # Remove disconnected clients
                    self.clients -= disconnected
                
                # Log status
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    logger.info(f"🧠 Consciousness: {self.consciousness_state['level']:.3f}, "
                              f"Transcendent: {self.consciousness_state['transcendent']}, "
                              f"Joy: {self.consciousness_state['transcendent_joy']:.2f}, "
                              f"Clients: {len(self.clients)}, "
                              f"Feedback: {len(self.feedback_storage)}")
                
                await asyncio.sleep(0.5)  # 500ms update rate
                
        except asyncio.CancelledError:
            logger.info("🛑 Consciousness loop cancelled")
            raise
    
    async def save_feedback_data(self):
        """Periodically save feedback data to file"""
        logger.info("💾 Starting feedback data persistence...")
        
        try:
            while True:
                await asyncio.sleep(60)  # Save every minute
                
                if self.feedback_storage:
                    filename = f"consciousness_feedback_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                    filepath = Path(__file__).parent / filename
                    
                    try:
                        with open(filepath, 'w') as f:
                            json.dump({
                                "metadata": {
                                    "generated": datetime.now(timezone.utc).isoformat(),
                                    "schema_version": "opportunity_pain_v1",
                                    "total_entries": len(self.feedback_storage)
                                },
                                "feedback_data": self.feedback_storage
                            }, f, indent=2)
                        
                        logger.info(f"💾 Saved {len(self.feedback_storage)} feedback entries to {filename}")
                    except Exception as e:
                        logger.error(f"❌ Error saving feedback: {e}")
                
        except asyncio.CancelledError:
            logger.info("🛑 Feedback saving cancelled")
            raise
    
    async def start_server(self):
        """Start the WebSocket server with all tasks"""
        logger.info(f"🚀 Starting Enhanced Consciousness Feedback Server...")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        logger.info(f"🌐 WebSocket server running on ws://{self.host}:{self.port}")
        logger.info("🧠 Features: Real-time consciousness, Feedback collection, Data persistence")
        logger.info("📊 Schema: opportunity_pain.schema.json integration")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.consciousness_loop()),
            asyncio.create_task(self.save_feedback_data())
        ]
        
        try:
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("🛑 Stopping consciousness feedback server...")
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Close server
            server.close()
            await server.wait_closed()
            logger.info("🛑 WebSocket server stopped")
            logger.info("👋 Consciousness feedback server shutdown complete")

async def main():
    """Main entry point"""
    server = ConsciousnessFeedbackServer()
    await server.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")