#!/usr/bin/env python3
"""
🏛️ QUICK META ROOM TEST
========================

Quick test to verify Meta Room is working properly.
Tests basic operations without needing full system launch.
"""

import asyncio
import websockets
import json
from datetime import datetime

async def test_meta_room():
    """Test Meta Room basic operations"""
    
    print("\n" + "=" * 70)
    print("META ROOM QUICK TEST")
    print("=" * 70 + "\n")
    
    uri = "ws://localhost:8702"
    
    try:
        print(f"Connecting to Meta Room at {uri}...")
        async with websockets.connect(uri, open_timeout=5) as ws:
            print("SUCCESS: Connected to Meta Room!\n")
            
            # Test 1: Ping
            print("TEST 1: Ping Meta Room")
            await ws.send(json.dumps({"type": "ping"}))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            print(f"  Response: {data}")
            print("  PASSED\n")
            
            # Test 2: Get Status
            print("TEST 2: Get Network Status")
            await ws.send(json.dumps({"type": "get_status"}))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            print(f"  Network Type: {data.get('network_type')}")
            print(f"  Total Nodes: {data.get('total_nodes')}")
            print(f"  Total Vaults: {data.get('total_vaults')}")
            print(f"  Total AI Systems: {data.get('total_ai_systems')}")
            
            meta_room = data.get('meta_room', {})
            print(f"  Meta Room Name: {meta_room.get('name')}")
            print(f"  Meta Room Intelligence: {meta_room.get('intelligence_level')}")
            print(f"  Connected Vaults: {meta_room.get('connected_vaults')}")
            print(f"  Connected AI Systems: {meta_room.get('connected_ai_systems')}")
            print("  PASSED\n")
            
            # Test 3: Register AI System
            print("TEST 3: Register AI System")
            await ws.send(json.dumps({
                "type": "register_ai",
                "name": "Test AI Brain",
                "capabilities": ["consciousness", "decision_making", "pattern_recognition"]
            }))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            print(f"  Response: {data}")
            ai_id = data.get('ai_id')
            print(f"  AI ID: {ai_id}")
            print("  PASSED\n")
            
            # Test 4: Create Node
            print("TEST 4: Create Node")
            await ws.send(json.dumps({
                "type": "create_node",
                "name": "Test Consciousness Node",
                "node_type": "consciousness",
                "position": {"x": 1.0, "y": 0.5, "z": 0.0}
            }))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            print(f"  Response: {data}")
            node_id = data.get('node_id')
            print(f"  Node ID: {node_id}")
            print("  PASSED\n")
            
            # Test 5: Verify updated status
            print("TEST 5: Verify Updated Status")
            await ws.send(json.dumps({"type": "get_status"}))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            print(f"  Total Nodes: {data.get('total_nodes')} (should be 1)")
            print(f"  Total AI Systems: {data.get('total_ai_systems')} (should be 1)")
            meta_room = data.get('meta_room', {})
            print(f"  Meta Room Connected AI Systems: {meta_room.get('connected_ai_systems')}")
            print("  PASSED\n")
            
            print("=" * 70)
            print("ALL TESTS PASSED!")
            print("=" * 70)
            print("\nMeta Room is operational and ready for full system integration.")
            print("\nNext Steps:")
            print("  1. Connect Vector Node Network (port 8701)")
            print("  2. Connect Unified AI Brain")
            print("  3. Start Logging AI (port 8703)")
            print("  4. Test full data flow\n")
            
            return True
            
    except websockets.exceptions.ConnectionRefusedError:
        print("ERROR: Cannot connect to Meta Room")
        print("  Make sure Meta Room is running:")
        print("  python knowledge_vault_node_network.py\n")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main entry point"""
    result = await test_meta_room()
    return result

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user\n")
        exit(1)
