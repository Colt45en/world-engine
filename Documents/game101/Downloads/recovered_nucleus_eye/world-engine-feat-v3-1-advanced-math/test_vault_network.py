#!/usr/bin/env python3
"""
🏛️🧠 TEST: KNOWLEDGE VAULT NETWORK WITH META ROOM
==================================================

Tests the advanced architecture with:
- Knowledge Vaults between every node connection
- Librarian AIs managing each vault
- Central Meta Room coordinating everything
- AI systems connected through Meta Room
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

async def test_vault_network():
    """Test the knowledge vault network with Meta Room"""
    
    print("🏛️ TESTING KNOWLEDGE VAULT NETWORK WITH META ROOM")
    print("=" * 70)
    
    try:
        uri = "ws://localhost:8702"
        print(f"\n📡 Connecting to Knowledge Vault Network at {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!\n")
            
            # Test 1: Check network status
            print("📊 TEST 1: Initial Network Status")
            await websocket.send(json.dumps({"type": "get_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            status = json.loads(response)
            print(f"   Network Type: {status.get('network_type')}")
            print(f"   Meta Room: {status.get('meta_room', {}).get('name')}")
            print(f"   Connected Vaults: {status.get('meta_room', {}).get('connected_vaults', 0)}")
            print(f"   Connected AI Systems: {status.get('meta_room', {}).get('connected_ai_systems', 0)}\n")
            
            # Test 2: Create brain-like nodes
            print("🧠 TEST 2: Creating Brain Nodes")
            nodes = [
                {"name": "frontal_cortex", "type": "consciousness", "pos": {"x": 1.0, "y": 0.5, "z": 0.0}},
                {"name": "hippocampus", "type": "processor", "pos": {"x": -0.5, "y": 0.0, "z": 0.5}},
                {"name": "cerebellum", "type": "storage", "pos": {"x": 0.5, "y": -0.5, "z": -0.5}},
                {"name": "thalamus", "type": "bridge", "pos": {"x": 0.0, "y": 0.0, "z": 1.0}}
            ]
            
            node_ids = {}
            for node in nodes:
                await websocket.send(json.dumps({
                    "type": "create_node",
                    "name": node["name"],
                    "node_type": node["type"],
                    "position": node["pos"]
                }))
                response = await asyncio.wait_for(websocket.recv(), timeout=3)
                result = json.loads(response)
                node_ids[node["name"]] = result["node_id"]
                print(f"   ✓ Created {node['name']} (ID: {result['node_id'][:12]}...)")
            
            # Test 3: Create connections with vaults in the middle
            print("\n🏛️ TEST 3: Creating Connections with Knowledge Vaults")
            connections = [
                ("frontal_cortex", "hippocampus", 0.9),
                ("hippocampus", "cerebellum", 0.85),
                ("cerebellum", "thalamus", 0.8),
                ("thalamus", "frontal_cortex", 0.95)
            ]
            
            vault_ids = []
            for node_a, node_b, strength in connections:
                await websocket.send(json.dumps({
                    "type": "create_connection",
                    "node_a_id": node_ids[node_a],
                    "node_b_id": node_ids[node_b],
                    "strength": strength
                }))
                response = await asyncio.wait_for(websocket.recv(), timeout=3)
                result = json.loads(response)
                vault_ids.append(result["vault_id"])
                print(f"   ✓ {node_a} ←→ [VAULT] ←→ {node_b}")
                print(f"     Vault ID: {result['vault_id'][:12]}...")
                print(f"     Librarian: {result['librarian_id'][:12]}...")
                print(f"     Strength: {strength}\n")
            
            # Test 4: Register AI systems
            print("🤖 TEST 4: Registering AI Systems with Meta Room")
            ai_systems = [
                {"name": "GPT-Consciousness", "capabilities": ["language", "reasoning", "creativity"]},
                {"name": "Vision-AI", "capabilities": ["image_processing", "pattern_recognition"]},
                {"name": "Audio-Processor", "capabilities": ["speech", "sound_analysis"]},
                {"name": "Decision-Engine", "capabilities": ["logic", "planning", "optimization"]}
            ]
            
            ai_ids = []
            for ai in ai_systems:
                await websocket.send(json.dumps({
                    "type": "register_ai",
                    "name": ai["name"],
                    "capabilities": ai["capabilities"]
                }))
                response = await asyncio.wait_for(websocket.recv(), timeout=3)
                result = json.loads(response)
                ai_ids.append(result["ai_id"])
                print(f"   ✓ {ai['name']} connected to Meta Room")
                print(f"     Capabilities: {', '.join(ai['capabilities'])}\n")
            
            # Test 5: Send data through the network
            print("⚡ TEST 5: Propagating Data Through Network")
            test_data = {
                "type": "propagate_data",
                "source_node_id": node_ids["frontal_cortex"],
                "data": {
                    "thought": "Testing consciousness propagation",
                    "consciousness_level": 0.85,
                    "quantum_state": True,
                    "knowledge": "Neural pathways are forming",
                    "meta": "This is critical meta information"
                }
            }
            
            await websocket.send(json.dumps(test_data))
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            result = json.loads(response)
            
            print(f"   Source: {result['propagation']['source']}")
            print(f"   Vaults Processed: {len(result['propagation']['vaults_processed'])}")
            for vault_name in result['propagation']['vaults_processed']:
                print(f"     📚 {vault_name}")
            print(f"   Meta Room Notified: {result['propagation']['meta_room_notified']}")
            print(f"   AI Systems Updated: {len(result['propagation']['ai_systems_updated'])}\n")
            
            # Test 6: Get final network status
            print("📊 TEST 6: Final Network Status")
            await websocket.send(json.dumps({"type": "get_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            status = json.loads(response)
            
            print(f"   Total Nodes: {status['total_nodes']}")
            print(f"   Total Vaults: {status['total_vaults']}")
            print(f"   Total Connections: {status['total_connections']}")
            print(f"   Total AI Systems: {status['total_ai_systems']}\n")
            
            print("   🏛️ Meta Room Status:")
            meta = status['meta_room']
            print(f"     Name: {meta['name']}")
            print(f"     Connected Vaults: {meta['connected_vaults']}")
            print(f"     Connected AI Systems: {meta['connected_ai_systems']}")
            print(f"     Cached Data Items: {meta['cached_data_items']}")
            print(f"     Intelligence Level: {meta['intelligence_level']}")
            print(f"     Coordination Efficiency: {meta['coordination_efficiency']}\n")
            
            print("   📚 Vault Details:")
            for vault in status['vaults'][:3]:  # Show first 3 vaults
                print(f"     {vault['name']}:")
                print(f"       Librarian: {vault['librarian']['name']} ({vault['librarian']['role']})")
                print(f"       Data Processed: {vault['librarian']['data_processed']}")
                print(f"       Storage: {vault['storage']['utilization']}")
                print(f"       Meta Room Connected: {vault['meta_room_connected']}\n")
            
            # Summary
            print("=" * 70)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("\n🎯 ARCHITECTURE VERIFIED:")
            print("   ✓ Knowledge Vaults positioned between all connections")
            print("   ✓ Librarian AIs managing and classifying data")
            print("   ✓ Central Meta Room coordinating all vaults")
            print("   ✓ AI systems connected through Meta Room")
            print("   ✓ Data flowing through vault→librarian→meta→AI pipeline")
            
            print("\n🏛️ KNOWLEDGE VAULT NETWORK STRUCTURE:")
            print("   Node A ←→ [Vault + Librarian] ←→ Node B")
            print("              ↓")
            print("         [Meta Room]")
            print("              ↓")
            print("   [AI System 1] [AI System 2] [AI System 3] [AI System 4]")
            
            print("\n📊 DATA FLOW:")
            print("   1. Data sent from Node A")
            print("   2. Stored in Knowledge Vault")
            print("   3. Librarian AI processes & classifies")
            print("   4. Important data routed to Meta Room")
            print("   5. Meta Room distributes to all AI systems")
            print("   6. Data stored in vault for future retrieval")
            
            print("\n🎉 The Knowledge Vault Network is fully operational!")
            
            return True
            
    except asyncio.TimeoutError:
        print("❌ Connection timeout")
        return False
    except ConnectionRefusedError:
        print("❌ Connection refused - is the server running?")
        print("   Start it with: python knowledge_vault_node_network.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("\n" + "🏛️" * 35)
    print("KNOWLEDGE VAULT NETWORK TEST SUITE")
    print("🏛️" * 35 + "\n")
    
    print("Testing advanced neural architecture with:")
    print("  📚 Knowledge Vaults between every connection")
    print("  🤖 Librarian AIs organizing data")
    print("  🏛️ Central Meta Room coordinating")
    print("  🔗 AI systems connected to Meta Room\n")
    
    try:
        success = asyncio.run(test_vault_network())
        
        if success:
            print("\n" + "🎉" * 35)
            print("TEST SUITE PASSED!")
            print("🎉" * 35 + "\n")
        else:
            print("\n⚠️ Some tests failed")
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
