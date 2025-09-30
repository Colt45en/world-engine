import { Html } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import './wow_lite_tier4_bridge.css';

// Types for the Tier-4 system
interface Tier4State {
  x: number[];
  kappa: number;
  level: number;
  operator: string;
  meta: {
    transformation: string;
    timestamp: number;
  };
}

interface Tier4Operator {
  name: string;
  matrix: number[][];
  bias: number[];
  kappaDelta: number;
}

interface NDJSONEvent {
  type: string;
  ts: string;
  id?: string;
  role?: string;
  tag?: string;
  nucleus?: string;
  tier4_operator?: string;
  tier4_suggested_macro?: string;
  state?: string;
}

interface Tier4Room {
  toast: (message: string, type?: string) => void;
  applyOperator: (operatorName: string, previousState: Tier4State, newState: Tier4State) => void;
  publishEvent: (event: any) => void;
  getStateHistory: () => { size: number };
  updateMetrics: (metrics: any) => void;
  whenReady: (callback: () => void) => void;
  sessionId: string;
}

// WoW Lite Tier-4 Bridge Component
export const WoWLiteTier4Bridge: React.FC<{
  websocketUrl?: string;
  onStateChange?: (state: Tier4State) => void;
  onOperatorApplied?: (operator: string, state: Tier4State) => void;
  position?: [number, number, number];
}> = ({
  websocketUrl = 'ws://localhost:8080/ws',
  onStateChange,
  onOperatorApplied,
  position = [0, 0, 0]
}) => {
  const { scene, camera } = useThree();
  const bridgeRef = useRef<THREE.Group>(null);

  // State management
  const [currentState, setCurrentState] = useState<Tier4State>({
    x: [0.5, 0.5, 0.5, 0.5],
    kappa: 0.5,
    level: 1,
    operator: 'INIT',
    meta: {
      transformation: 'initialization',
      timestamp: Date.now()
    }
  });

  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [metrics, setMetrics] = useState({
    opsRate: '0/s',
    kappaDelta: '0.000',
    transitions: '0',
    memory: '0KB',
    chartValue: 0.5
  });

  // Tier-4 Operators (WoW Lite themed)
  const operators: Record<string, Tier4Operator> = {
    'ST': { // Strength
      name: 'Strength',
      matrix: [[1.1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      bias: [0.1, 0, 0, 0],
      kappaDelta: 0.05
    },
    'SL': { // Spell Power
      name: 'Spell Power',
      matrix: [[1, 0, 0, 0], [0, 1.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      bias: [0, 0.1, 0, 0],
      kappaDelta: 0.03
    },
    'CP': { // Critical Power
      name: 'Critical Power',
      matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1.15, 0], [0, 0, 0, 1]],
      bias: [0, 0, 0.15, 0],
      kappaDelta: 0.08
    },
    'CV': { // Convert
      name: 'Convert',
      matrix: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      bias: [0, 0, 0, 0],
      kappaDelta: -0.02
    },
    'PR': { // Protect
      name: 'Protect',
      matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.2]],
      bias: [0, 0, 0, 0.2],
      kappaDelta: 0.04
    },
    'RC': { // Recover
      name: 'Recover',
      matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.8]],
      bias: [0, 0, 0, -0.2],
      kappaDelta: 0.06
    },
    'TL': { // Teleport
      name: 'Teleport',
      matrix: [[0.9, 0, 0, 0], [0, 0.9, 0, 0], [0, 0, 0.9, 0], [0, 0, 0, 0.9]],
      bias: [-0.1, -0.1, -0.1, -0.1],
      kappaDelta: -0.1
    },
    'RB': { // Rebirth
      name: 'Rebirth',
      matrix: [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]],
      bias: [0.5, 0.5, 0.5, 0.5],
      kappaDelta: 0.2
    },
    'MD': { // Modify
      name: 'Modify',
      matrix: [[1.05, 0.05, 0, 0], [0.05, 1.05, 0.05, 0], [0, 0.05, 1.05, 0.05], [0, 0, 0.05, 1.05]],
      bias: [0, 0, 0, 0],
      kappaDelta: 0.01
    }
  };

  // Nucleus operator mapping
  const nucleusOperatorMap: Record<string, string> = {
    'VIBRATE': 'ST',
    'OPTIMIZATION': 'SL',
    'STATE': 'CP',
    'SEED': 'RB'
  };

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (ws) {
      ws.close();
    }

    try {
      const newWs = new WebSocket(websocketUrl);
      setWs(newWs);

      newWs.onopen = () => {
        setIsConnected(true);
        setReconnectAttempts(0);
        console.log('WoW Lite Tier-4 Bridge: WebSocket connected');

        // Send initialization message
        newWs.send(JSON.stringify({
          type: 'tier4_room_init',
          sessionId: 'wow_lite_session',
          timestamp: Date.now()
        }));
      };

      newWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('WoW Lite Tier-4 Bridge: Failed to parse WebSocket message', error);
        }
      };

      newWs.onclose = () => {
        setIsConnected(false);
        attemptReconnect();
      };

      newWs.onerror = (error) => {
        console.error('WoW Lite Tier-4 Bridge: WebSocket error', error);
      };

    } catch (error) {
      console.error('WoW Lite Tier-4 Bridge: Failed to create WebSocket connection', error);
      attemptReconnect();
    }
  }, [websocketUrl, ws]);

  const attemptReconnect = useCallback(() => {
    if (reconnectAttempts >= 5) {
      console.error('WoW Lite Tier-4 Bridge: Max reconnection attempts reached');
      return;
    }

    const delay = 1000 * Math.pow(2, reconnectAttempts);
    setTimeout(() => {
      setReconnectAttempts(prev => prev + 1);
      connectWebSocket();
    }, delay);
  }, [reconnectAttempts, connectWebSocket]);

  const sendWebSocketMessage = useCallback((message: any) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }, [ws]);

  // Message handling
  const handleWebSocketMessage = useCallback((data: NDJSONEvent) => {
    console.log('WoW Lite Tier-4 Bridge: Received WebSocket message', data);

    let operatorToApply: string | null = null;

    if (data.type === 'nucleus_exec' && data.role) {
      operatorToApply = nucleusOperatorMap[data.role];
    } else if (data.type === 'memory_store' && data.tag) {
      operatorToApply = nucleusOperatorMap[data.tag];
    } else if (data.tier4_operator) {
      operatorToApply = data.tier4_operator;
    }

    if (operatorToApply && operators[operatorToApply]) {
      setTimeout(() => {
        applyOperator(operatorToApply!, {
          source: 'nucleus_auto',
          triggerEvent: data.type,
          triggerId: data.id || data.nucleus || 'unknown'
        });
      }, 100);
    }
  }, [nucleusOperatorMap, operators]);

  // State transformation
  const transformState = useCallback((state: Tier4State, operator: Tier4Operator): Tier4State => {
    const newX = new Array(state.x.length);

    for (let i = 0; i < state.x.length; i++) {
      let sum = 0;
      for (let j = 0; j < state.x.length; j++) {
        const matrixValue = operator.matrix[i]?.[j] || (i === j ? 1 : 0);
        sum += matrixValue * state.x[j];
      }
      newX[i] = sum + (operator.bias[i] || 0);
    }

    const newKappa = Math.max(0, Math.min(1, state.kappa + operator.kappaDelta));

    let levelChange = 0;
    if (operator.name === 'Update') levelChange = 1;
    else if (operator.name === 'Rollback') levelChange = -1;
    else if (operator.name === 'Reset') levelChange = -state.level;

    return {
      x: newX,
      kappa: newKappa,
      level: Math.max(0, state.level + levelChange),
      operator: operator.name,
      meta: {
        transformation: operator.name,
        timestamp: Date.now()
      }
    };
  }, []);

  // Operator application
  const applyOperator = useCallback((operatorName: string, meta?: any) => {
    const operator = operators[operatorName];
    if (!operator) {
      console.error(`Unknown operator: ${operatorName}`);
      return;
    }

    const previousState = { ...currentState };
    const newState = transformState(currentState, operator);

    setCurrentState(newState);
    onStateChange?.(newState);
    onOperatorApplied?.(operatorName, newState);

    sendWebSocketMessage({
      type: 'tier4_operator_applied',
      operator: operatorName,
      previousState,
      newState,
      meta,
      timestamp: Date.now()
    });

    updateMetrics();

    console.log(`WoW Lite Tier-4 Bridge: Applied ${operatorName}`, { previousState, newState });
  }, [currentState, operators, transformState, onStateChange, onOperatorApplied, sendWebSocketMessage]);

  // Macro application (WoW Lite themed sequences)
  const applyMacro = useCallback((macroName: string) => {
    const macros: Record<string, string[]> = {
      'WARRIOR_RAGE': ['ST', 'CP', 'PR'],
      'MAGE_ARCANE': ['SL', 'CV', 'MD'],
      'PALADIN_HOLY': ['PR', 'RB', 'SL'],
      'DRUID_BALANCE': ['CV', 'TL', 'RC'],
      'HUNTER_BEAST': ['CP', 'ST', 'MD'],
      'ROGUE_COMBAT': ['CP', 'TL', 'ST'],
      'PRIEST_SHADOW': ['SL', 'MD', 'CV'],
      'SHAMAN_ELEMENTAL': ['SL', 'RB', 'PR'],
      'WARLOCK_DEMONOLOGY': ['MD', 'CV', 'SL'],
      'DEATH_KNIGHT_UNHOLY': ['RB', 'PR', 'CP']
    };

    const sequence = macros[macroName];
    if (!sequence) {
      console.error(`Unknown macro: ${macroName}`);
      return;
    }

    console.log(`Executing WoW Lite macro: ${macroName}`);

    sequence.forEach((op, index) => {
      setTimeout(() => {
        const operatorName = operators[op] ? op : 'ST';
        applyOperator(operatorName, {
          source: 'macro',
          macro: macroName,
          step: index + 1,
          total: sequence.length
        });
      }, index * 300);
    });
  }, [operators, applyOperator]);

  // Metrics update
  const updateMetrics = useCallback(() => {
    const newMetrics = {
      opsRate: '1.2/s',
      kappaDelta: currentState.kappa.toFixed(3),
      transitions: '42', // Mock value
      memory: '24KB', // Mock value
      chartValue: currentState.kappa
    };
    setMetrics(newMetrics);
  }, [currentState.kappa]);

  // Event listeners setup
  useEffect(() => {
    const handleApplyOperator = (event: CustomEvent) => {
      applyOperator(event.detail.operator);
    };

    const handleApplyMacro = (event: CustomEvent) => {
      applyMacro(event.detail.macro);
    };

    const handleLoadState = (event: CustomEvent) => {
      const { state } = event.detail;
      setCurrentState({ ...state });
      sendWebSocketMessage({
        type: 'tier4_state_loaded',
        state: currentState,
        timestamp: Date.now()
      });
    };

    window.addEventListener('apply-operator', handleApplyOperator as EventListener);
    window.addEventListener('apply-macro', handleApplyMacro as EventListener);
    window.addEventListener('tier4-load-state', handleLoadState as EventListener);

    return () => {
      window.removeEventListener('apply-operator', handleApplyOperator as EventListener);
      window.removeEventListener('apply-macro', handleApplyMacro as EventListener);
      window.removeEventListener('tier4-load-state', handleLoadState as EventListener);
    };
  }, [applyOperator, applyMacro, sendWebSocketMessage, currentState]);

  // WebSocket connection on mount
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [connectWebSocket, ws]);

  // Animation loop
  useFrame((state, delta) => {
    if (bridgeRef.current) {
      // Gentle floating animation
      bridgeRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime) * 0.1;

      // Rotate based on kappa value
      bridgeRef.current.rotation.y = currentState.kappa * Math.PI * 2;
    }
  });

  return (
    <group ref={bridgeRef} position={position}>
      {/* Main Bridge Structure */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[2, 0.5, 2]} />
        <meshStandardMaterial
          color={isConnected ? "#4ade80" : "#ef4444"}
          emissive={isConnected ? "#22c55e" : "#dc2626"}
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* State Visualization Spheres */}
      {currentState.x.map((value, index) => (
        <mesh key={index} position={[index * 0.8 - 1.2, 1, 0]}>
          <sphereGeometry args={[0.2]} />
          <meshStandardMaterial
            color={`hsl(${value * 360}, 70%, 50%)`}
            emissive={`hsl(${value * 360}, 70%, 30%)`}
            emissiveIntensity={0.3}
          />
        </mesh>
      ))}

      {/* Kappa Indicator */}
      <mesh position={[0, 1.5, 0]}>
        <cylinderGeometry args={[0.1, 0.1, currentState.kappa * 2]} />
        <meshStandardMaterial
          color="#fbbf24"
          emissive="#f59e0b"
          emissiveIntensity={0.4}
        />
      </mesh>

      {/* Connection Status Indicator */}
      <mesh position={[0, 2, 0]}>
        <sphereGeometry args={[0.15]} />
        <meshStandardMaterial
          color={isConnected ? "#10b981" : "#ef4444"}
          emissive={isConnected ? "#059669" : "#dc2626"}
          emissiveIntensity={isConnected ? 0.5 : 0.8}
        />
      </mesh>

      {/* UI Overlay */}
      <Html position={[0, 2.5, 0]} center>
        <div className="wow-lite-bridge-ui">
          <div className="wow-lite-bridge-header">
            🏰 WoW Lite Tier-4 Bridge
          </div>
          <div>Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}</div>
          <div>Kappa: {currentState.kappa.toFixed(3)}</div>
          <div>Level: {currentState.level}</div>
          <div>Operator: {currentState.operator}</div>
          <div className="wow-lite-bridge-metrics">
            Ops: {metrics.opsRate} | Mem: {metrics.memory}
          </div>
        </div>
      </Html>

      {/* Operator Buttons */}
      <Html position={[-2, 0, 0]}>
        <div className="wow-lite-operator-panel">
          {Object.keys(operators).map((opName) => (
            <button
              key={opName}
              onClick={() => applyOperator(opName)}
              className="wow-lite-operator-button"
            >
              {opName}
            </button>
          ))}
        </div>
      </Html>

      {/* Macro Buttons */}
      <Html position={[2, 0, 0]}>
        <div className="wow-lite-macro-panel">
          {['WARRIOR_RAGE', 'MAGE_ARCANE', 'PALADIN_HOLY', 'DRUID_BALANCE'].map((macro) => (
            <button
              key={macro}
              onClick={() => applyMacro(macro)}
              className="wow-lite-macro-button"
            >
              {macro.replace('_', ' ')}
            </button>
          ))}
        </div>
      </Html>
    </group>
  );
};

// Hook for using the WoW Lite Tier-4 Bridge
export const useWoWLiteTier4Bridge = () => {
  const [bridgeState, setBridgeState] = useState<Tier4State | null>(null);
  const [lastOperator, setLastOperator] = useState<string>('');

  const handleStateChange = useCallback((state: Tier4State) => {
    setBridgeState(state);
  }, []);

  const handleOperatorApplied = useCallback((operator: string, state: Tier4State) => {
    setLastOperator(operator);
    setBridgeState(state);
  }, []);

  return {
    bridgeState,
    lastOperator,
    handleStateChange,
    handleOperatorApplied
  };
};

export default WoWLiteTier4Bridge;
