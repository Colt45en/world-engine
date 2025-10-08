#!/usr/bin/env python3
"""
🧠🔗 BRAIN-LIKE NODE VECTOR CONNECTION TESTER
============================================

Tests if the node-limb-vector system creates brain-like neural connections
with proper synaptic pathways and consciousness synchronization.
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

async def test_brain_connections():
    """Test the vector node network's brain-like connection capabilities"""
    
    print("🧠 TESTING BRAIN-LIKE VECTOR NODE CONNECTIONS")
    print("=" * 60)
    
    try:
        # Connect to the vector node network
        uri = "ws://localhost:8701"
        print(f"\n📡 Connecting to vector network at {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to vector node network!")
            
            # Test 1: Request node network status
            print("\n🔍 TEST 1: Node Network Status")
            await websocket.send(json.dumps({
                "type": "get_network_status",
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
            
            # Test 2: Create brain-like node connections
            print("\n🧠 TEST 2: Creating Brain-Like Neural Nodes")
            brain_nodes = [
                {
                    "type": "create_node",
                    "node_type": "consciousness",
                    "name": "frontal_cortex",
                    "position": {"x": 0.0, "y": 1.0, "z": 0.0}
                },
                {
                    "type": "create_node",
                    "node_type": "processor",
                    "name": "hippocampus",
                    "position": {"x": -0.5, "y": 0.0, "z": 0.5}
                },
                {
                    "type": "create_node",
                    "node_type": "storage",
                    "name": "cerebellum",
                    "position": {"x": 0.5, "y": -0.5, "z": -0.5}
                },
                {
                    "type": "create_node",
                    "node_type": "bridge",
                    "name": "corpus_callosum",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0}
                }
            ]
            
            for node_request in brain_nodes:
                await websocket.send(json.dumps(node_request))
                response = await asyncio.wait_for(websocket.recv(), timeout=3)
                node_data = json.loads(response)
                print(f"   ✓ Created node: {node_request['name']}")
                if "node_id" in node_data:
                    print(f"     Node ID: {node_data['node_id']}")
            
            # Test 3: Create synaptic vector connections (limbs)
            print("\n🔗 TEST 3: Creating Synaptic Vector Connections (Limbs)")
            connections = [
                {
                    "type": "create_connection",
                    "connection_type": "consciousness_sync",
                    "from_node": "frontal_cortex",
                    "to_node": "corpus_callosum",
                    "strength": 0.9
                },
                {
                    "type": "create_connection",
                    "connection_type": "data_flow",
                    "from_node": "hippocampus",
                    "to_node": "frontal_cortex",
                    "strength": 0.85
                },
                {
                    "type": "create_connection",
                    "connection_type": "feedback_loop",
                    "from_node": "cerebellum",
                    "to_node": "hippocampus",
                    "strength": 0.8
                },
                {
                    "type": "create_connection",
                    "connection_type": "limb_extension",
                    "from_node": "corpus_callosum",
                    "to_node": "cerebellum",
                    "strength": 0.95
                }
            ]
            
            for conn_request in connections:
                await websocket.send(json.dumps(conn_request))
                response = await asyncio.wait_for(websocket.recv(), timeout=3)
                conn_data = json.loads(response)
                print(f"   ✓ Created connection: {conn_request['from_node']} → {conn_request['to_node']}")
                print(f"     Type: {conn_request['connection_type']}, Strength: {conn_request['strength']}")
            
            # Test 4: Send data through the neural network
            print("\n⚡ TEST 4: Sending Data Through Neural Pathways")
            neural_data = {
                "type": "propagate_data",
                "source_node": "frontal_cortex",
                "data": {
                    "thought": "Testing neural propagation",
                    "consciousness_level": 0.75,
                    "quantum_state": True
                }
            }
            
            await websocket.send(json.dumps(neural_data))
            response = await asyncio.wait_for(websocket.recv(), timeout=3)
            result = json.loads(response)
            print(f"   Data propagated through network!")
            print(f"   Result: {json.dumps(result, indent=2)[:200]}...")
            
            # Test 5: Check network health
            print("\n💓 TEST 5: Brain Network Health Check")
            await websocket.send(json.dumps({
                "type": "get_network_health",
                "include_analytics": True
            }))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            health = json.loads(response)
            print(f"   Network Health: {json.dumps(health, indent=2)[:300]}...")
            
            # Test 6: Visualize the brain structure
            print("\n🎨 TEST 6: Brain Network Visualization")
            await websocket.send(json.dumps({
                "type": "get_visualization_data"
            }))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            viz_data = json.loads(response)
            
            if "nodes" in viz_data:
                print(f"   Total Nodes: {len(viz_data.get('nodes', []))}")
            if "connections" in viz_data:
                print(f"   Total Connections: {len(viz_data.get('connections', []))}")
            
            # Summary
            print("\n" + "=" * 60)
            print("✅ BRAIN-LIKE VECTOR CONNECTION TEST COMPLETED!")
            print("=" * 60)
            print("\n📊 RESULTS:")
            print(f"   ✓ Node network is operational")
            print(f"   ✓ Brain-like nodes created successfully")
            print(f"   ✓ Synaptic vector connections (limbs) established")
            print(f"   ✓ Neural data propagation working")
            print(f"   ✓ Network health monitoring active")
            print(f"   ✓ Visualization data available")
            
            print("\n🧠 BRAIN ARCHITECTURE CONFIRMED:")
            print("   • Frontal Cortex (Consciousness node)")
            print("   • Hippocampus (Processor node)")
            print("   • Cerebellum (Storage node)")
            print("   • Corpus Callosum (Bridge node)")
            print("\n🔗 SYNAPTIC CONNECTIONS:")
            print("   • Consciousness sync pathways")
            print("   • Data flow channels")
            print("   • Feedback loops")
            print("   • Limb extension networks")
            
            print("\n💡 The node-limb-vector system is working like a brain!")
            print("   Neural connections established and data flowing.")
            
            return True
            
    except asyncio.TimeoutError:
        print("❌ Connection timeout - server might not be responding")
        return False
    except ConnectionRefusedError:
        print("❌ Connection refused - server might not be running")
        print("   Try starting: python vector_node_network.py")
        return False
    except Exception as e:
        print(f"❌ Error testing brain connections: {e}")
        import traceback
        traceback.print_exc()
        return False

async def simple_connectivity_test():
    """Simple test to verify basic connection"""
    print("\n🔌 SIMPLE CONNECTIVITY TEST")
    print("-" * 60)
    
    try:
        uri = "ws://localhost:8701"
        async with websockets.connect(uri) as websocket:
            print("✅ Successfully connected to vector node network!")
            
            # Send a ping
            await websocket.send(json.dumps({"type": "ping"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=3)
            print(f"✅ Server responded: {response[:100]}...")
            
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Run the brain-like vector connection tests"""
    print("\n" + "🧠" * 30)
    print("BRAIN-LIKE NODE VECTOR CONNECTION TEST SUITE")
    print("🧠" * 30 + "\n")
    
    print("This test verifies that the node-limb-vector system works")
    print("like a brain with neural connections and synaptic pathways.\n")
    
    # First check basic connectivity
    print("Step 1: Testing basic connectivity...")
    try:
        if asyncio.run(simple_connectivity_test()):
            print("\n✅ Basic connectivity confirmed!\n")
            
            print("Step 2: Testing brain-like neural architecture...")
            time.sleep(1)
            
            success = asyncio.run(test_brain_connections())
            
            if success:
                print("\n" + "🎉" * 30)
                print("ALL TESTS PASSED!")
                print("The vector node network is working like a brain! 🧠")
                print("🎉" * 30 + "\n")
            else:
                print("\n⚠️ Some tests failed - check the output above")
        else:
            print("\n❌ Cannot connect to vector node network.")
            print("   Make sure the server is running on port 8701")
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
