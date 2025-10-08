#!/usr/bin/env python3
"""Quick smoke test for consciousness/vector WebSocket servers"""

import asyncio
import websockets
import json

async def test_websocket_server(port=8701, test_name="WebSocket"):
    """Test a WebSocket server with a simple message"""
    uri = f"ws://localhost:{port}"
    print(f"\n🔍 Testing {test_name} on port {port}...")
    
    try:
        async with websockets.connect(uri, open_timeout=5) as websocket:
            print(f"✅ Connected to {uri}")
            
            # Send test message
            test_msg = {"type": "test", "message": "smoke test", "timestamp": "2025-10-07T13:32:00"}
            print(f"📤 Sending: {test_msg}")
            await websocket.send(json.dumps(test_msg))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f"📥 Received: {response[:200]}...")
            
            print(f"✅ {test_name} smoke test PASSED")
            return True
            
    except asyncio.TimeoutError:
        print(f"❌ Timeout connecting to {uri}")
        return False
    except ConnectionRefusedError:
        print(f"❌ Connection refused on {uri} - server not running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    print("=" * 60)
    print("🧪 WEBSOCKET SERVER SMOKE TESTS")
    print("=" * 60)
    
    # Test port 8701 (whichever server is running)
    result1 = await test_websocket_server(8701, "Port 8701 Server")
    
    print("\n" + "=" * 60)
    if result1:
        print("✅ ALL SMOKE TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
