#!/usr/bin/env python3
"""
NEXUS AI Bot Server - Intelligence Operations Center
===================================================

Flask server that hosts:
- Intelligence Operations Center (3D room with 12 panels)
- AI Bot communication systems
- Librarian network management
- WebSocket integration for real-time updates
- Meta-Base thought engine integration

Usage: python ai_bot_server.py
Access: http://localhost:8000
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import flask
from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import queue
import websocket
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIBotServer:
    """Main AI Bot Server with Intelligence Operations Center"""

    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'nexus-intelligence-operations'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Intelligence system state
        self.librarians = {
            'Math Librarian': {'status': 'offline', 'active_tasks': 0},
            'English Librarian': {'status': 'offline', 'active_tasks': 0},
            'Pattern Librarian': {'status': 'offline', 'active_tasks': 0},
            'Meta-Librarian': {'status': 'configured', 'active_tasks': 0}
        }

        self.memory_systems = {
            'NexusArchivist': {'status': 'ready', 'storage_mb': 0, 'entries': 0},
            'NovaSynapse': {'status': 'ready', 'compression_ratio': 0.6, 'history_size': 0},
            'CacheManager': {'status': 'ready', 'cache_size': 0, 'hit_rate': 0.0},
            'QuantumStates': {'status': 'coherent', 'entanglements': 0, 'branches': 0}
        }

        self.communication_log = []
        self.max_log_size = 100

        # WebSocket connection to existing tier4_ws_relay.js
        self.ws_client = None
        self.ws_thread = None

        self.setup_routes()
        self.setup_socketio_events()

    def setup_routes(self):
        """Set up Flask routes"""

        @self.app.route('/')
        def index():
            """Main Intelligence Operations Center"""
            return send_from_directory('../public', 'Untitled-16.html')

        @self.app.route('/intelligence-hub')
        def intelligence_hub():
            """Intelligence Hub panel"""
            return send_from_directory('../public', 'intelligence-hub.html')

        @self.app.route('/librarian-math')
        def librarian_math():
            """Math Librarian panel"""
            return send_from_directory('../public', 'librarian-math.html')

        @self.app.route('/nucleus-control-center')
        def nucleus_control_center():
            """Nucleus Control Center panel"""
            try:
                return send_from_directory('../src/automations/nucleus', 'nucleus_control_center.tsx')
            except:
                return self.create_nucleus_control_panel()

        @self.app.route('/meta-librarian-canvas')
        def meta_librarian_canvas():
            """Meta-Librarian Canvas panel"""
            return send_from_directory('../public', 'meta-librarian-canvas.html')

        # API endpoints for AI bot functionality
        @self.app.route('/api/status')
        def api_status():
            """Get system status"""
            return jsonify({
                'librarians': self.librarians,
                'memory_systems': self.memory_systems,
                'communication_log': self.communication_log[-10:],  # Last 10 messages
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/send-message', methods=['POST'])
        def api_send_message():
            """Send AI bot message"""
            data = request.get_json()
            message_type = data.get('type', 'query')
            message = data.get('message', '')
            source = data.get('source', 'AI Bot')

            # Add to communication log
            log_entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'ai_bot',
                'source': source,
                'message': f"{message_type}: {message}",
                'routing': self.get_routing_info(message_type)
            }

            self.add_to_communication_log(log_entry)

            # Emit to connected clients
            self.socketio.emit('new_message', log_entry)

            return jsonify({'status': 'sent', 'entry': log_entry})

        @self.app.route('/api/librarian-data', methods=['POST'])
        def api_librarian_data():
            """Process librarian data"""
            data = request.get_json()
            librarian = data.get('librarian', 'Unknown Librarian')
            data_type = data.get('data_type', 'pattern')
            data_content = data.get('data', {})

            # Add to communication log
            log_entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'librarian',
                'source': librarian,
                'message': f"{data_type}: {json.dumps(data_content)}",
                'routing': self.get_routing_info(data_type)
            }

            self.add_to_communication_log(log_entry)

            # Emit to connected clients
            self.socketio.emit('librarian_data', log_entry)

            return jsonify({'status': 'processed', 'entry': log_entry})

        @self.app.route('/api/activate-system', methods=['POST'])
        def api_activate_system():
            """Activate intelligence systems"""
            data = request.get_json()
            system = data.get('system', 'all')

            if system == 'all' or system == 'librarians':
                for librarian in self.librarians:
                    if librarian != 'Meta-Librarian':
                        self.librarians[librarian]['status'] = 'active'

            if system == 'all' or system == 'memory':
                for memory_sys in self.memory_systems:
                    if memory_sys != 'QuantumStates':
                        self.memory_systems[memory_sys]['status'] = 'active'

            # Start WebSocket connection if not already connected
            if not self.ws_client and system == 'all':
                self.start_websocket_connection()

            self.socketio.emit('system_activated', {
                'system': system,
                'timestamp': datetime.now().isoformat()
            })

            return jsonify({'status': 'activated', 'system': system})

        # Serve static files from various directories
        @self.app.route('/public/<path:filename>')
        def serve_public(filename):
            return send_from_directory('../public', filename)

        @self.app.route('/src/<path:filename>')
        def serve_src(filename):
            return send_from_directory('../src', filename)

        @self.app.route('/demos/<path:filename>')
        def serve_demos(filename):
            return send_from_directory('../demos', filename)

    def setup_socketio_events(self):
        """Set up SocketIO event handlers"""

        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            join_room('intelligence_ops')
            emit('system_status', {
                'librarians': self.librarians,
                'memory_systems': self.memory_systems,
                'communication_log': self.communication_log[-20:]
            })

        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
            leave_room('intelligence_ops')

        @self.socketio.on('trigger_nucleus_event')
        def handle_nucleus_event(data):
            event_type = data.get('event', 'VIBRATE')
            operator = self.get_nucleus_operator(event_type)

            log_entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': 'nucleus',
                'source': 'Nucleus Core',
                'message': f"Event: {event_type} → {operator}",
                'routing': f"{event_type}→{operator}"
            }

            self.add_to_communication_log(log_entry)
            emit('nucleus_event_processed', log_entry, room='intelligence_ops')

        @self.socketio.on('query_memory_system')
        def handle_memory_query(data):
            system = data.get('system', 'NexusArchivist')
            query = data.get('query', '')

            # Simulate memory query processing
            result = {
                'system': system,
                'query': query,
                'results': f"Mock results for: {query}",
                'timestamp': datetime.now().isoformat()
            }

            emit('memory_query_result', result)

    def get_routing_info(self, message_type):
        """Get routing information for message types"""
        routing_map = {
            'query': 'VIBRATE→ST',
            'learning': 'OPTIMIZATION→UP',
            'feedback': 'STATE→CV',
            'pattern': 'VIBRATE→ST',
            'classification': 'STATE→CV',
            'analysis': 'OPTIMIZATION→UP'
        }
        return routing_map.get(message_type, 'UNKNOWN')

    def get_nucleus_operator(self, event_type):
        """Get nucleus operator for event type"""
        operator_map = {
            'VIBRATE': 'ST',
            'OPTIMIZATION': 'UP',
            'STATE': 'CV',
            'SEED': 'RB'
        }
        return operator_map.get(event_type, 'ST')

    def add_to_communication_log(self, entry):
        """Add entry to communication log with size limit"""
        self.communication_log.append(entry)
        if len(self.communication_log) > self.max_log_size:
            self.communication_log = self.communication_log[-self.max_log_size:]

    def start_websocket_connection(self):
        """Start WebSocket connection to tier4_ws_relay.js"""
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    # Forward to connected clients
                    self.socketio.emit('ws_message', data, room='intelligence_ops')
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from WebSocket: {message}")

            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")
                self.ws_client = None

            def on_open(ws):
                logger.info("WebSocket connection opened")
                ws.send(json.dumps({
                    'type': 'register',
                    'client_type': 'ai_bot_server'
                }))

            # Determine websocket URL from central port_config.json (fallback to 8701)
            try:
                import json as _json, os as _os
                cfg_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', 'port_config.json'))
                if _os.path.exists(cfg_path):
                    with open(cfg_path, 'r', encoding='utf-8') as _fh:
                        _cfg = _json.load(_fh)
                        target_port = int(_cfg.get('ws_port', 8701))
                else:
                    target_port = 8701
            except Exception:
                target_port = 8701

            self.ws_client = websocket.WebSocketApp(
                f"ws://localhost:{target_port}",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )

            # Run WebSocket in separate thread
            self.ws_thread = threading.Thread(
                target=self.ws_client.run_forever,
                daemon=True
            )
            self.ws_thread.start()

        except Exception as e:
            logger.error(f"Failed to start WebSocket connection: {e}")

    def create_nucleus_control_panel(self):
        """Create a simple nucleus control panel if the TSX file isn't found"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nucleus Control Center</title>
            <style>
                body {
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    color: #e8fff6;
                    font-family: 'Segoe UI', system-ui;
                    padding: 20px;
                }
                .control-panel {
                    max-width: 800px;
                    margin: 0 auto;
                }
                .control-btn {
                    background: rgba(124, 252, 203, 0.2);
                    border: 1px solid #7cfccb;
                    color: #e8fff6;
                    padding: 10px 20px;
                    margin: 5px;
                    border-radius: 8px;
                    cursor: pointer;
                }
                .status-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .status-card {
                    background: rgba(255, 255, 255, 0.08);
                    border: 1px solid rgba(124, 252, 203, 0.3);
                    border-radius: 12px;
                    padding: 20px;
                }
            </style>
        </head>
        <body>
            <div class="control-panel">
                <h1>🧠 Nucleus Control Center</h1>
                <p>Core intelligence orchestration system</p>

                <div class="status-grid">
                    <div class="status-card">
                        <h3>Nucleus Events</h3>
                        <button class="control-btn" onclick="triggerEvent('VIBRATE')">VIBRATE → ST</button>
                        <button class="control-btn" onclick="triggerEvent('OPTIMIZATION')">OPTIMIZATION → UP</button>
                        <button class="control-btn" onclick="triggerEvent('STATE')">STATE → CV</button>
                        <button class="control-btn" onclick="triggerEvent('SEED')">SEED → RB</button>
                    </div>

                    <div class="status-card">
                        <h3>System Status</h3>
                        <div id="status">Loading...</div>
                    </div>
                </div>
            </div>

            <script>
                function triggerEvent(eventType) {
                    fetch('/api/send-message', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            type: 'nucleus',
                            message: `Nucleus event: ${eventType}`,
                            source: 'Nucleus Control'
                        })
                    });
                }

                function updateStatus() {
                    fetch('/api/status')
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById('status').innerHTML = `
                                <p>Librarians: ${Object.keys(data.librarians).length}</p>
                                <p>Memory Systems: ${Object.keys(data.memory_systems).length}</p>
                                <p>Recent Messages: ${data.communication_log.length}</p>
                            `;
                        });
                }

                updateStatus();
                setInterval(updateStatus, 5000);
            </script>
        </body>
        </html>
        """
        return html_template

    def run(self):
        """Start the AI Bot Server"""
        logger.info(f"🚀 Starting NEXUS AI Bot Server...")
        logger.info(f"🌐 Intelligence Operations Center: http://{self.host}:{self.port}")
        logger.info(f"📊 Intelligence Hub: http://{self.host}:{self.port}/intelligence-hub")
        logger.info(f"📊 Math Librarian: http://{self.host}:{self.port}/librarian-math")
        logger.info(f"🧠 Nucleus Control: http://{self.host}:{self.port}/nucleus-control-center")

        # Try to start WebSocket connection to existing relay
        threading.Timer(2.0, self.start_websocket_connection).start()

        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=True,
            use_reloader=False  # Disable reloader to avoid threading issues
        )

def main():
    """Main entry point"""
    print("🧠 NEXUS AI Bot Server - Intelligence Operations Center")
    print("=" * 60)

    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / 'public').exists():
        print("⚠️  Warning: 'public' directory not found.")
        print("   Make sure you're running from the project root directory.")
        print(f"   Current directory: {current_dir}")

    try:
        server = AIBotServer(host='localhost', port=8000)
        server.run()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
