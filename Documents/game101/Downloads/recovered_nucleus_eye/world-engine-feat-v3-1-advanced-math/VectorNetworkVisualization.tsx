import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';

// Custom hook for vector network connection
const useVectorNetwork = (wsUrl = 'ws://localhost:8766') => {
  const [networkData, setNetworkData] = useState<{
    nodes: NodeType[];
    lines: LineType[];
    network_state: {
      total_nodes: number;
      total_connections: number;
      network_health: number;
      consciousness_sync: number;
      data_throughput: number;
    };
  }>({
    nodes: [],
    lines: [],
    network_state: {
      total_nodes: 0,
      total_connections: 0,
      network_health: 0,
      consciousness_sync: 0,
      data_throughput: 0
    }
  });
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const addNode = useCallback((nodeData: any) => {
    sendMessage({
      type: 'add_node',
      data: nodeData
    });
  }, [sendMessage]);

  const createConnection = useCallback((connectionData: any) => {
    sendMessage({
      type: 'create_connection',
      data: connectionData
    });
  }, [sendMessage]);

  const influenceConsciousness = useCallback((nodeId: any, delta: any) => {
    sendMessage({
      type: 'influence_consciousness',
      data: {
        node_id: nodeId,
        delta: delta
      }
    });
  }, [sendMessage]);

  useEffect(() => {
    const connectWebSocket = () => {
      try {
        wsRef.current = new WebSocket(wsUrl);
        
        wsRef.current.onopen = () => {
          console.log('🌐 Connected to vector network server');
          setIsConnected(true);
        };
        
        wsRef.current.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            
            if (message.type === 'network_update') {
              setNetworkData(message.data);
            }
          } catch (error) {
            console.error('❌ Error parsing network message:', error);
          }
        };
        
        wsRef.current.onclose = () => {
          console.log('🔌 Disconnected from vector network server');
          setIsConnected(false);
          // Attempt reconnection after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };
        
        wsRef.current.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          setIsConnected(false);
        };
        
      } catch (error) {
        console.error('❌ WebSocket connection error:', error);
        setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [wsUrl]);

  return {
    networkData,
    isConnected,
    addNode,
    createConnection,
    influenceConsciousness
  };
};

// Network node component
type NetworkNodeProps = {
  node: {
    id: string;
    name: string;
    type: string;
    position: { x: number; y: number; z: number };
    consciousness_level: number;
    connections: number;
    load_percentage: number;
    status?: string;
  };
  onNodeClick?: (node: any) => void;
  onNodeHover?: (node: any) => void;
};

const NetworkNode: React.FC<NetworkNodeProps> = ({ node, onNodeClick, onNodeHover }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const [clicked, setClicked] = useState(false);

  useFrame((state, delta) => {
    if (meshRef.current) {
      // Gentle rotation based on consciousness level
      meshRef.current.rotation.y += delta * node.consciousness_level * 0.5;
      
      // Pulsing effect based on consciousness
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * node.consciousness_level * 0.2;
      meshRef.current.scale.setScalar(scale);
    }
  });

  // Node color based on type and consciousness
  const getNodeColor = () => {
    const baseColors = {
      consciousness: '#FFD700', // Gold
      feedback: '#00CED1',      // Dark Turquoise
      knowledge: '#9370DB',     // Medium Purple
      bridge: '#FF6347',        // Tomato
      gateway: '#32CD32',       // Lime Green
      processor: '#FF69B4',     // Hot Pink
      storage: '#87CEEB'        // Sky Blue
    };
    
    const baseColor = new THREE.Color(baseColors[node.type as keyof typeof baseColors] || '#FFFFFF');
    // Darken based on consciousness level
    const consciousness = node.consciousness_level;
    return baseColor.clone().multiplyScalar(0.5 + consciousness * 0.5);
  };

  // Node size based on connections and load
  const getNodeSize = () => {
    const baseSize = 0.5;
    const connectionFactor = Math.min(node.connections / 10, 1);
    const loadFactor = node.load_percentage / 100;
    return baseSize + connectionFactor * 0.3 + loadFactor * 0.2;
  };

  return (
    <group
      position={[node.position.x, node.position.y, node.position.z]}
      onClick={(e) => {
        e.stopPropagation();
        setClicked(!clicked);
        onNodeClick?.(node);
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        onNodeHover?.(node);
      }}
      onPointerOut={(e) => {
        e.stopPropagation();
        setHovered(false);
      }}
    >
      {/* Main node sphere */}
      <Sphere ref={meshRef} args={[getNodeSize(), 32, 32]}>
        <meshPhongMaterial
          color={getNodeColor()}
          transparent
          opacity={hovered ? 0.9 : 0.7}
          emissive={getNodeColor()}
          emissiveIntensity={clicked ? 0.3 : (hovered ? 0.2 : 0.1)}
        />
      </Sphere>
      
      {/* Node label */}
      <Text
        position={[0, getNodeSize() + 0.5, 0]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {node.name}
      </Text>
      
      {/* Consciousness level indicator */}
      <Text
        position={[0, -getNodeSize() - 0.3, 0]}
        fontSize={0.2}
        color={getNodeColor()}
        anchorX="center"
        anchorY="middle"
      >
        {`${(node.consciousness_level * 100).toFixed(0)}%`}
      </Text>
      
      {/* Connection count indicator */}
      {node.connections > 0 && (
        <Text
          position={[getNodeSize() + 0.3, 0, 0]}
          fontSize={0.15}
          color="cyan"
          anchorX="center"
          anchorY="middle"
        >
          {node.connections}
        </Text>
      )}
    </group>
  );
};

// Vector line component
type LineType = {
    id: string;
    source: string;
    target: string;
    type: string;
    strength: number;
    flow: number;
};

type NodeType = {
    id: string;
    name: string;
    type: string;
    position: { x: number; y: number; z: number };
    consciousness_level: number;
    connections: number;
    load_percentage: number;
    status?: string;
};

function VectorLine({ line, nodes }: { line: LineType; nodes: NodeType[] }) {
    const [animationOffset, setAnimationOffset] = useState(0);

    useFrame(() => {
        ;
    });

    // Find source and target nodes
    const sourceNode = nodes.find((n: { id: any; }) => n.id === line.source);
    const targetNode = nodes.find((n: { id: any; }) => n.id === line.target);

    if (!sourceNode || !targetNode) return null;

    // Line color based on type
    const getLineColor = () => {
        const colors = {
            data_flow: '#00BFFF', // Deep Sky Blue
            consciousness_sync: '#FFD700', // Gold
            feedback_loop: '#FF69B4', // Hot Pink
            knowledge_bridge: '#9370DB', // Medium Purple
            limb_extension: '#32CD32', // Lime Green
            quantum_entanglement: '#FF0000' // Red
        };
        return colors[line.type as keyof typeof colors] || '#FFFFFF';
    };

    // Calculate line points
    const sourcePos = [sourceNode.position.x, sourceNode.position.y, sourceNode.position.z];
    const targetPos = [targetNode.position.x, targetNode.position.y, targetNode.position.z];

    // Create animated line with flow visualization
    const points = [];
    const segments = 20;
    for (let i = 0; i <= segments; i++) {
        const t = i / segments;
        const x = sourcePos[0] + (targetPos[0] - sourcePos[0]) * t;
        const y = sourcePos[1] + (targetPos[1] - sourcePos[1]) * t;
        const z = sourcePos[2] + (targetPos[2] - sourcePos[2]) * t;
        points.push(new THREE.Vector3(x, y, z));
    }

    return (
        <group>
            {/* Main connection line */}
            <Line
                points={points}
                color={getLineColor()}
                lineWidth={2 + line.strength * 3}
                transparent
                opacity={0.6 + line.strength * 0.4} />

            {/* Animated flow particles */}
            {line.flow > 0 && (
                <Sphere
                    args={[0.1, 8, 8]}
                    position={[
                        sourcePos[0] + (targetPos[0] - sourcePos[0]) * animationOffset,
                        sourcePos[1] + (targetPos[1] - sourcePos[1]) * animationOffset,
                        sourcePos[2] + (targetPos[2] - sourcePos[2]) * animationOffset
                    ]}
                >
                    <meshBasicMaterial color={getLineColor()} />
                </Sphere>
            )}

            {/* Line type indicator at midpoint */}
            <Text
                position={[
                    (sourcePos[0] + targetPos[0]) / 2,
                    (sourcePos[1] + targetPos[1]) / 2 + 0.2,
                    (sourcePos[2] + targetPos[2]) / 2
                ]}
                fontSize={0.1}
                color={getLineColor()}
                anchorX="center"
                anchorY="middle"
            >
                {line.type.replace('_', ' ').toUpperCase()}
            </Text>
        </group>
    );
}

// Network stats HUD
type NetworkStatsHUDProps = {
  networkData: {
    nodes: NodeType[];
    lines: LineType[];
    network_state: {
      total_nodes: number;
      total_connections: number;
      network_health: number;
      consciousness_sync: number;
      data_throughput: number;
    };
  };
  isConnected: boolean;
};

const NetworkStatsHUD: React.FC<NetworkStatsHUDProps> = ({ networkData, isConnected }) => {
  const { network_state } = networkData;
  
  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      left: '20px',
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '20px',
      borderRadius: '10px',
      minWidth: '300px',
      zIndex: 1000
    }}>
      <h3 style={{ margin: '0 0 15px 0', color: '#00CED1' }}>
        🌐 Vector Network Status
      </h3>
      
      <div style={{ marginBottom: '10px' }}>
        <span style={{ color: isConnected ? '#32CD32' : '#FF0000' }}>
          {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </span>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '14px' }}>
        <div>
          <strong>Nodes:</strong> {network_state.total_nodes}
        </div>
        <div>
          <strong>Connections:</strong> {network_state.total_connections}
        </div>
        <div>
          <strong>Health:</strong> {(network_state.network_health * 100).toFixed(1)}%
        </div>
        <div>
          <strong>Sync:</strong> {(network_state.consciousness_sync * 100).toFixed(1)}%
        </div>
        <div style={{ gridColumn: '1 / -1' }}>
          <strong>Data Flow:</strong> {network_state.data_throughput} units/sec
        </div>
      </div>
      
      <div style={{ 
        marginTop: '15px',
        padding: '10px',
        background: 'rgba(0, 206, 209, 0.1)',
        borderRadius: '5px',
        fontSize: '12px'
      }}>
        <strong>Active Node Types:</strong><br/>
        {networkData.nodes && networkData.nodes.reduce((types: { [x: string]: any; }, node: { type: string | number; }) => {
          types[node.type] = (types[node.type] || 0) + 1;
          return types;
        }, {}) && Object.entries(
          networkData.nodes.reduce((types: { [x: string]: any; }, node: { type: string | number; }) => {
            types[node.type] = (types[node.type] || 0) + 1;
            return types;
          }, {})
        ).map(([type, count]) => (
          <div key={type}>• {type}: {count}</div>
        ))}
      </div>
    </div>
  );
};

// Node control panel
type NodeControlPanelProps = {
  selectedNode: NodeType | null;
  onInfluenceConsciousness: (nodeId: string, delta: number) => void;
};

const NodeControlPanel: React.FC<NodeControlPanelProps> = ({ selectedNode, onInfluenceConsciousness }) => {
  if (!selectedNode) return null;
  
  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      right: '20px',
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '20px',
      borderRadius: '10px',
      minWidth: '250px',
      zIndex: 1000
    }}>
      <h3 style={{ margin: '0 0 15px 0', color: '#FFD700' }}>
        🎯 Node Control
      </h3>
      
      <div style={{ marginBottom: '15px' }}>
        <strong>{selectedNode.name}</strong><br/>
        <span style={{ fontSize: '12px', opacity: 0.8 }}>
          Type: {selectedNode.type}<br/>
          ID: {selectedNode.id}
        </span>
      </div>
      
      <div style={{ marginBottom: '15px' }}>
        <div>Consciousness: {(selectedNode.consciousness_level * 100).toFixed(1)}%</div>
        <div>Connections: {selectedNode.connections}</div>
        <div>Load: {selectedNode.load_percentage?.toFixed(1) || 0}%</div>
        <div>Status: {selectedNode.status}</div>
      </div>
      
      <div style={{ marginBottom: '15px' }}>
        <strong>Influence Consciousness:</strong>
        <div style={{ display: 'flex', gap: '5px', marginTop: '5px' }}>
          <button
            onClick={() => onInfluenceConsciousness(selectedNode.id, 0.1)}
            style={{
              padding: '5px 10px',
              background: '#32CD32',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer'
            }}
          >
            +10%
          </button>
          <button
            onClick={() => onInfluenceConsciousness(selectedNode.id, 0.05)}
            style={{
              padding: '5px 10px',
              background: '#00CED1',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer'
            }}
          >
            +5%
          </button>
          <button
            onClick={() => onInfluenceConsciousness(selectedNode.id, -0.05)}
            style={{
              padding: '5px 10px',
              background: '#FF6347',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer'
            }}
          >
            -5%
          </button>
          <button
            onClick={() => onInfluenceConsciousness(selectedNode.id, -0.1)}
            style={{
              padding: '5px 10px',
              background: '#FF0000',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer'
            }}
          >
            -10%
          </button>
        </div>
      </div>
    </div>
  );
};

// Main vector network visualization component
const VectorNetworkVisualization = ({ wsUrl = 'ws://localhost:8766' }) => {
  const {
    networkData,
    isConnected,
    influenceConsciousness
  } = useVectorNetwork(wsUrl);
  
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  const handleNodeClick = (node: React.SetStateAction<null>) => {
    setSelectedNode(node);
  };

  const handleNodeHover = (node: React.SetStateAction<null>) => {
    setHoveredNode(node);
  };

  return (
    <div style={{ width: '100%', height: '100vh', position: 'relative', background: '#000011' }}>
      {/* 3D Network Visualization */}
      <Canvas camera={{ position: [15, 15, 15], fov: 60 }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        
        {/* Render nodes */}
        {networkData.nodes?.map(node => (
          <NetworkNode
            key={node.id}
            node={node}
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
          />
        ))}
        
        {/* Render vector lines */}
        {networkData.lines?.map(line => (
          <VectorLine
            key={line.id}
            line={line}
            nodes={networkData.nodes}
          />
        ))}
        
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      </Canvas>
      
      {/* Network Stats HUD */}
      <NetworkStatsHUD
        networkData={networkData}
        isConnected={isConnected}
      />
      
      {/* Node Control Panel */}
      <NodeControlPanel
        selectedNode={selectedNode}
        onInfluenceConsciousness={influenceConsciousness}
      />
      
      {/* Instructions */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        left: '20px',
        right: '20px',
        background: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '15px',
        borderRadius: '10px',
        zIndex: 1000,
        fontSize: '14px'
      }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#00CED1' }}>
          🎮 Vector Network Controls
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '10px' }}>
          <div>
            <strong>🖱️ Mouse Controls:</strong><br/>
            • Click nodes to select and control<br/>
            • Hover for node information<br/>
            • Drag to rotate view<br/>
            • Scroll to zoom
          </div>
          <div>
            <strong>🌐 Network Features:</strong><br/>
            • Real-time consciousness synchronization<br/>
            • Vector line data flow visualization<br/>
            • Multi-dimensional node networking<br/>
            • Limb extension connections
          </div>
          <div>
            <strong>🎯 Node Types:</strong><br/>
            • Consciousness (Gold) - Core hubs<br/>
            • Feedback (Turquoise) - Data collectors<br/>
            • Knowledge (Purple) - Information storage<br/>
            • Processor (Pink) - Neural processing
          </div>
        </div>
      </div>
    </div>
  );
};

export default VectorNetworkVisualization;