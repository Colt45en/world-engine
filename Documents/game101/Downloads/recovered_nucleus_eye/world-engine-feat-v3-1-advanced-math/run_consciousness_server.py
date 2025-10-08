#!/usr/bin/env python3
"""Runner to start ConsciousnessWebSocketServer on an alternate port for testing."""
import asyncio
from consciousness_websocket_server import ConsciousnessWebSocketServer

async def main():
    server = ConsciousnessWebSocketServer(host='localhost', port=9201)
    try:
        await server.start_server()
    except Exception as e:
        print('Runner error:', e)
    finally:
        server.stop_server()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Server interrupted by user')
