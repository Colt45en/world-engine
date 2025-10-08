#!/usr/bin/env python3
"""
World Engine RAG Bridge Server
HTTP API bridge between Python RAG system and React frontend
"""

import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

# Import our RAG system
from world_engine_rag import WorldEngineRAG, RAGConfig

class RAGBridgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RAG bridge"""

    def __init__(self, *args, rag_system=None, **kwargs):
        self.rag_system = rag_system
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)

        if parsed_url.path == '/health':
            self._send_response(200, {"status": "healthy", "system": "World Engine RAG"})

        elif parsed_url.path == '/query':
            params = parse_qs(parsed_url.query)
            question = params.get('q', [''])[0]

            if not question:
                self._send_response(400, {"error": "Missing query parameter 'q'"})
                return

            try:
                response = self.rag_system.query(question)
                self._send_response(200, response)
            except Exception as e:
                self._send_response(500, {"error": str(e)})

        elif parsed_url.path == '/components':
            # List available components
            components = [
                "visual-bleedway", "nexus-room", "crypto-dashboard",
                "bridge-system", "quantum-thought", "sensory-framework"
            ]
            self._send_response(200, {"components": components})

        else:
            self._send_response(404, {"error": "Endpoint not found"})

    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self._send_response(400, {"error": "Invalid JSON"})
            return

        parsed_url = urlparse(self.path)

        if parsed_url.path == '/query':
            question = data.get('question', '')
            context = data.get('context', '')
            top_k = data.get('top_k', 5)

            if not question:
                self._send_response(400, {"error": "Missing 'question' field"})
                return

            try:
                if context:
                    response = self.rag_system.get_contextual_help(context, question)
                else:
                    response = self.rag_system.query(question, top_k)

                self._send_response(200, response)
            except Exception as e:
                self._send_response(500, {"error": str(e)})

        elif parsed_url.path == '/component-docs':
            component = data.get('component', '')

            if not component:
                self._send_response(400, {"error": "Missing 'component' field"})
                return

            try:
                response = self.rag_system.get_component_docs(component)
                self._send_response(200, response)
            except Exception as e:
                self._send_response(500, {"error": str(e)})

        else:
            self._send_response(404, {"error": "Endpoint not found"})

    def _send_response(self, status_code, data):
        """Send JSON response with CORS headers"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """Custom log message format"""
        print(f"[RAG Bridge] {self.address_string()} - {format % args}")

class RAGBridgeServer:
    """RAG Bridge Server with graceful shutdown"""

    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.rag_system = None
        self.server = None
        self.running = False

    def initialize_rag(self):
        """Initialize the RAG system"""
        print("Initializing World Engine RAG system...")

        config = RAGConfig(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            top_k=5
        )

        self.rag_system = WorldEngineRAG(config)

        # Try to load existing vector store, otherwise build new one
        if not self.rag_system.load_vector_store():
            print("Building new vector store...")
            if not self.rag_system.build_vector_store():
                print("❌ Failed to build vector store")
                return False

            # Save for future use
            self.rag_system.save_vector_store()

        print("✅ RAG system initialized successfully")
        return True

    def start(self):
        """Start the RAG bridge server"""
        if not self.initialize_rag():
            print("❌ Failed to initialize RAG system")
            return False

        # Create handler with RAG system
        handler = lambda *args, **kwargs: RAGBridgeHandler(*args, rag_system=self.rag_system, **kwargs)

        try:
            self.server = HTTPServer((self.host, self.port), handler)
            self.running = True

            print(f"🚀 World Engine RAG Bridge Server starting on http://{self.host}:{self.port}")
            print(f"📚 RAG system ready with vector store")
            print(f"🔗 API Endpoints:")
            print(f"   GET  /health - Health check")
            print(f"   GET  /query?q=<question> - Simple query")
            print(f"   POST /query - Advanced query with context")
            print(f"   POST /component-docs - Component documentation")
            print(f"   GET  /components - List available components")
            print(f"\n💡 Example usage:")
            print(f"   curl \"http://{self.host}:{self.port}/query?q=How+does+World+Engine+work?\"")
            print(f"\nPress Ctrl+C to stop the server")

            self.server.serve_forever()

        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"❌ Server error: {e}")
            return False

        return True

    def stop(self):
        """Stop the server gracefully"""
        if self.server and self.running:
            print("\n🛑 Shutting down RAG Bridge Server...")
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            print("✅ Server stopped")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="World Engine RAG Bridge Server")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8888, help="Server port (default: 8888)")
    parser.add_argument("--test", action="store_true", help="Run quick test and exit")

    args = parser.parse_args()

    if args.test:
        # Quick test mode
        print("=== RAG Bridge Test Mode ===")
        server = RAGBridgeServer()
        if server.initialize_rag():
            # Test a few queries
            test_queries = [
                "What is World Engine?",
                "How to use NEXUS Room?",
                "Crypto dashboard features"
            ]

            for query in test_queries:
                print(f"\n🔍 Testing: {query}")
                response = server.rag_system.query(query, top_k=1)
                if response["success"]:
                    result = response["results"][0]
                    print(f"📄 {result['source']}: {result['content'][:100]}...")
                else:
                    print(f"❌ Error: {response['error']}")

            print("\n✅ RAG system test completed successfully!")
            return True
        else:
            print("❌ RAG system test failed")
            return False

    # Start the server
    server = RAGBridgeServer(host=args.host, port=args.port)
    return server.start()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
