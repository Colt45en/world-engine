// EngineRoom.tsx - React component wrapper for Tier-4 Engine Room integration

import React, { useEffect, useRef, useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { createTier4RoomBridge, Tier4RoomBridge } from './tier4_room_integration';
import { Tier4State } from './roomAdapter';

export interface EngineRoomProps {
  /** URL to the Engine Room HTML file */
  iframeUrl?: string;

  /** WebSocket URL for Tier-4 integration */
  websocketUrl?: string;

  /** Initial session ID */
  sessionId?: string;

  /** Room title */
  title?: string;

  /** Initial Tier-4 state */
  initialState?: Tier4State;

  /** Callback when an operator is applied */
  onOperatorApplied?: (operator: string, previousState: Tier4State, newState: Tier4State) => void;

  /** Callback when state is loaded from snapshot */
  onStateLoaded?: (state: Tier4State, cid: string) => void;

  /** Callback when room is ready */
  onRoomReady?: (bridge: Tier4RoomBridge) => void;

  /** Callback for WebSocket connection status changes */
  onConnectionStatus?: (connected: boolean) => void;

  /** Custom CSS styles for the iframe container */
  className?: string;

  /** Additional iframe styles */
  style?: React.CSSProperties;

  /** Enable debug logging */
  debug?: boolean;
}

export interface EngineRoomRef {
  /** Apply a Tier-4 operator manually */
  applyOperator: (operator: string, meta?: any) => void;

  /** Apply a Three Ides macro */
  applyMacro: (macro: string) => void;

  /** Get current Tier-4 state */
  getCurrentState: () => Tier4State;

  /** Set the current state */
  setState: (state: Tier4State) => void;

  /** Trigger a nucleus event for testing */
  triggerNucleusEvent: (role: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED') => void;

  /** Show a toast message in the room */
  toast: (message: string) => void;

  /** Add a custom panel to the room */
  addPanel: (config: {
    wall: 'front' | 'left' | 'right' | 'back' | 'floor' | 'ceil';
    x: number;
    y: number;
    w: number;
    h: number;
    title?: string;
    html?: string;
  }) => void;

  /** Get the bridge instance for advanced operations */
  getBridge: () => Tier4RoomBridge | null;

  /** Check if WebSocket is connected */
  isConnected: () => boolean;
}

const EngineRoom = forwardRef<EngineRoomRef, EngineRoomProps>(({
  iframeUrl = '/worldengine.html',
  websocketUrl = 'ws://localhost:9000',
  sessionId,
  title = 'Tier-4 Engine Room',
  initialState = { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 },
  onOperatorApplied,
  onStateLoaded,
  onRoomReady,
  onConnectionStatus,
  className = '',
  style = {},
  debug = false
}, ref) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const bridgeRef = useRef<Tier4RoomBridge | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const log = useCallback((message: string, ...args: any[]) => {
    if (debug) {
      console.log(`[EngineRoom] ${message}`, ...args);
    }
  }, [debug]);

  // Initialize the bridge when iframe loads
  useEffect(() => {
    const initializeBridge = async () => {
      if (!iframeRef.current) return;

      try {
        log('Initializing Tier-4 Room Bridge...');
        const bridge = await createTier4RoomBridge(iframeRef.current, websocketUrl);
        bridgeRef.current = bridge;

        // Set initial state if provided
        if (initialState) {
          bridge.setState(initialState);
        }

        // Setup event listeners
        setupBridgeListeners(bridge);

        setIsReady(true);
        setError(null);
        log('Bridge initialized successfully');

        if (onRoomReady) {
          onRoomReady(bridge);
        }

      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(errorMessage);
        log('Failed to initialize bridge:', err);
      }
    };

    // Wait for iframe to load
    const iframe = iframeRef.current;
    if (iframe) {
      if (iframe.contentDocument?.readyState === 'complete') {
        setTimeout(initializeBridge, 100);
      } else {
        iframe.addEventListener('load', () => {
          setTimeout(initializeBridge, 100);
        });
      }
    }

    return () => {
      if (bridgeRef.current) {
        bridgeRef.current.disconnect();
      }
    };
  }, [iframeUrl, websocketUrl, initialState, onRoomReady, log]);

  const setupBridgeListeners = useCallback((bridge: Tier4RoomBridge) => {
    // Monitor WebSocket connection status
    const checkConnection = () => {
      const connected = bridge.isWebSocketConnected();
      if (connected !== isConnected) {
        setIsConnected(connected);
        if (onConnectionStatus) {
          onConnectionStatus(connected);
        }
        log(`WebSocket connection ${connected ? 'established' : 'lost'}`);
      }
    };

    // Check connection status periodically
    const connectionInterval = setInterval(checkConnection, 1000);

    // Listen for operator applications
    const handleOperatorEvent = (event: Event) => {
      const customEvent = event as CustomEvent;
      if (customEvent.detail && onOperatorApplied) {
        const { operator, previousState, newState } = customEvent.detail;
        onOperatorApplied(operator, previousState, newState);
      }
    };

    // Listen for state load events
    const handleStateLoadEvent = (event: Event) => {
      const customEvent = event as CustomEvent;
      if (customEvent.detail && onStateLoaded) {
        const { state, cid } = customEvent.detail;
        onStateLoaded(state, cid);
      }
    };

    window.addEventListener('tier4-operator-applied', handleOperatorEvent);
    window.addEventListener('tier4-load-state', handleStateLoadEvent);

    return () => {
      clearInterval(connectionInterval);
      window.removeEventListener('tier4-operator-applied', handleOperatorEvent);
      window.removeEventListener('tier4-load-state', handleStateLoadEvent);
    };
  }, [isConnected, onConnectionStatus, onOperatorApplied, onStateLoaded, log]);

  // Expose methods through ref
  useImperativeHandle(ref, () => ({
    applyOperator: (operator: string, meta?: any) => {
      if (bridgeRef.current) {
        bridgeRef.current.applyOperator(operator, meta);
      } else {
        log('Bridge not ready, cannot apply operator:', operator);
      }
    },

    applyMacro: (macro: string) => {
      if (bridgeRef.current) {
        bridgeRef.current.applyMacro(macro);
      } else {
        log('Bridge not ready, cannot apply macro:', macro);
      }
    },

    getCurrentState: () => {
      return bridgeRef.current?.getCurrentState() || initialState;
    },

    setState: (state: Tier4State) => {
      if (bridgeRef.current) {
        bridgeRef.current.setState(state);
      }
    },

    triggerNucleusEvent: (role: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED') => {
      if (bridgeRef.current) {
        bridgeRef.current.triggerNucleusEvent(role);
      }
    },

    toast: (message: string) => {
      if (bridgeRef.current) {
        bridgeRef.current.getRoom().toast(message);
      }
    },

    addPanel: (config) => {
      if (bridgeRef.current) {
        bridgeRef.current.getRoom().addPanel(config);
      }
    },

    getBridge: () => bridgeRef.current,

    isConnected: () => isConnected
  }), [isConnected, initialState, log]);

  const defaultStyle: React.CSSProperties = {
    width: '100%',
    height: '100%',
    border: 'none',
    borderRadius: '8px',
    backgroundColor: '#0b0e14',
    ...style
  };

  if (error) {
    return (
      <div className={`engine-room-error ${className}`} style={style}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: '#ff6b6b',
          fontFamily: 'system-ui',
          fontSize: '14px',
          padding: '20px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '48px', marginBottom: '20px' }}>⚠️</div>
          <div style={{ fontWeight: 'bold', marginBottom: '10px' }}>
            Engine Room Error
          </div>
          <div style={{ opacity: 0.8 }}>
            {error}
          </div>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '20px',
              padding: '8px 16px',
              background: '#1a2332',
              border: '1px solid #2a3548',
              color: '#e6f0ff',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Reload
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`engine-room-container ${className}`} style={{ position: 'relative', ...style }}>
      <iframe
        ref={iframeRef}
        src={iframeUrl}
        title={title}
        style={defaultStyle}
        allow="microphone; camera; geolocation"
      />

      {/* Status indicators */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        display: 'flex',
        gap: '8px',
        fontSize: '11px',
        fontFamily: 'system-ui'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          padding: '4px 8px',
          background: isReady ? '#0d4f3c' : '#67060c',
          border: `1px solid ${isReady ? '#238636' : '#da3633'}`,
          borderRadius: '12px',
          color: isReady ? '#46d158' : '#f85149'
        }}>
          <div style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: 'currentColor'
          }} />
          Room {isReady ? 'Ready' : 'Loading'}
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          padding: '4px 8px',
          background: isConnected ? '#0d4f3c' : '#67060c',
          border: `1px solid ${isConnected ? '#238636' : '#da3633'}`,
          borderRadius: '12px',
          color: isConnected ? '#46d158' : '#f85149'
        }}>
          <div style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: 'currentColor'
          }} />
          WebSocket {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
    </div>
  );
});

EngineRoom.displayName = 'EngineRoom';

// Hook for easier usage in functional components
export function useEngineRoom(props: Omit<EngineRoomProps, 'children'>) {
  const roomRef = useRef<EngineRoomRef>(null);
  const [isReady, setIsReady] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentState, setCurrentState] = useState<Tier4State>(
    props.initialState || { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 }
  );

  const enhancedProps = {
    ...props,
    onRoomReady: useCallback((bridge: Tier4RoomBridge) => {
      setIsReady(true);
      props.onRoomReady?.(bridge);
    }, [props]),

    onConnectionStatus: useCallback((connected: boolean) => {
      setIsConnected(connected);
      props.onConnectionStatus?.(connected);
    }, [props]),

    onOperatorApplied: useCallback((operator: string, previousState: Tier4State, newState: Tier4State) => {
      setCurrentState(newState);
      props.onOperatorApplied?.(operator, previousState, newState);
    }, [props]),

    onStateLoaded: useCallback((state: Tier4State, cid: string) => {
      setCurrentState(state);
      props.onStateLoaded?.(state, cid);
    }, [props])
  };

  const applyOperator = useCallback((operator: string, meta?: any) => {
    roomRef.current?.applyOperator(operator, meta);
  }, []);

  const applyMacro = useCallback((macro: string) => {
    roomRef.current?.applyMacro(macro);
  }, []);

  const triggerNucleus = useCallback((role: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED') => {
    roomRef.current?.triggerNucleusEvent(role);
  }, []);

  const toast = useCallback((message: string) => {
    roomRef.current?.toast(message);
  }, []);

  const addPanel = useCallback((config: Parameters<EngineRoomRef['addPanel']>[0]) => {
    roomRef.current?.addPanel(config);
  }, []);

  return {
    // Component props
    roomRef,
    roomProps: enhancedProps,

    // State
    isReady,
    isConnected,
    currentState,

    // Actions
    applyOperator,
    applyMacro,
    triggerNucleus,
    toast,
    addPanel,

    // Advanced
    getBridge: () => roomRef.current?.getBridge() || null,
    setState: (state: Tier4State) => roomRef.current?.setState(state)
  };
}

export default EngineRoom;
