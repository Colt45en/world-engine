// Enhanced EngineRoom with Clock Debugging Integration
// Extends the existing EngineRoom.tsx with timing analysis and WebSocket clock synchronization

import React, { useEffect, useRef, useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { createTier4RoomBridge, Tier4RoomBridge } from '../demo/tier4_room_integration';
import { Tier4State } from './roomAdapter';

// Import our clock debugging utilities
const ClockDebugUtils = typeof window !== 'undefined' && window.ClockDebug ?
  window.ClockDebug :
  { ClockBugDetector: null, ClockDebugOverlay: null };

export interface EnhancedEngineRoomProps {
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

  /** Enable clock debugging */
  enableClockDebug?: boolean;

  /** BPM for clock synchronization */
  bpm?: number;

  /** Callback when an operator is applied */
  onOperatorApplied?: (operator: string, previousState: Tier4State, newState: Tier4State) => void;

  /** Callback when state is loaded from snapshot */
  onStateLoaded?: (state: Tier4State, cid: string) => void;

  /** Callback when room is ready */
  onRoomReady?: (bridge: Tier4RoomBridge) => void;

  /** Callback for WebSocket connection status changes */
  onConnectionStatus?: (connected: boolean) => void;

  /** Callback for clock events (beats, bars, phase updates) */
  onClockEvent?: (event: {
    type: 'beat' | 'bar' | 'phase';
    bar: number;
    beat: number;
    phase: number;
    bpm: number;
    timestamp: number;
  }) => void;

  /** Callback for clock anomalies/bugs detected */
  onClockAnomaly?: (anomaly: {
    type: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH';
    message: string;
    timestamp: number;
    value?: number;
  }) => void;

  /** Custom CSS styles for the iframe container */
  className?: string;

  /** Additional iframe styles */
  style?: React.CSSProperties;

  /** Enable debug logging */
  debug?: boolean;
}

export interface EnhancedEngineRoomRef {
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

  // Enhanced clock debugging methods
  /** Get clock debug statistics */
  getClockStats: () => any;

  /** Export clock debug data */
  exportClockData: () => any;

  /** Reset clock debugging */
  resetClockDebug: () => void;

  /** Show/hide clock debug overlay */
  toggleClockDebug: () => void;

  /** Send clock sync message over WebSocket */
  syncClock: (bpm?: number) => void;
}

const EnhancedEngineRoom = forwardRef<EnhancedEngineRoomRef, EnhancedEngineRoomProps>(({
  iframeUrl = '/worldengine.html',
  websocketUrl = 'ws://localhost:9000',
  sessionId,
  title = 'Enhanced Tier-4 Engine Room',
  initialState = { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 },
  enableClockDebug = false,
  bpm = 120,
  onOperatorApplied,
  onStateLoaded,
  onRoomReady,
  onConnectionStatus,
  onClockEvent,
  onClockAnomaly,
  className = '',
  style = {},
  debug = false
}, ref) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const bridgeRef = useRef<Tier4RoomBridge | null>(null);
  const clockDebugRef = useRef<any>(null);
  const clockOverlayRef = useRef<any>(null);
  const clockAnimationRef = useRef<number | null>(null);

  const [isReady, setIsReady] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clockState, setClockState] = useState({
    bar: 0,
    beat: 0,
    phase: 0,
    bpm: bpm,
    running: false,
    startTime: 0
  });

  const log = useCallback((message: string, ...args: any[]) => {
    if (debug) {
      console.log(`[EnhancedEngineRoom] ${message}`, ...args);
    }
  }, [debug]);

  // Initialize clock debugging
  useEffect(() => {
    if (enableClockDebug && ClockDebugUtils.ClockBugDetector) {
      clockDebugRef.current = new ClockDebugUtils.ClockBugDetector();

      if (iframeRef.current) {
        clockOverlayRef.current = new ClockDebugUtils.ClockDebugOverlay(document.body);
      }

      log('Clock debugging initialized');
    }

    return () => {
      if (clockAnimationRef.current) {
        cancelAnimationFrame(clockAnimationRef.current);
      }
    };
  }, [enableClockDebug, log]);

  // Clock tick simulation and monitoring
  const clockTick = useCallback((timestamp: number) => {
    if (!clockState.running) return;

    const elapsed = (timestamp - clockState.startTime) / 1000; // seconds
    const secPerBeat = 60 / clockState.bpm;
    const totalBeats = elapsed / secPerBeat;

    const newBar = Math.floor(totalBeats / 4); // 4 beats per bar
    const newBeat = Math.floor(totalBeats) % 4;
    const newPhase = totalBeats - Math.floor(totalBeats);

    const newClockState = {
      ...clockState,
      bar: newBar,
      beat: newBeat,
      phase: newPhase
    };

    // Detect beat/bar changes for events
    const beatChanged = newBeat !== clockState.beat || newBar !== clockState.bar;
    const barChanged = newBar !== clockState.bar;

    setClockState(newClockState);

    // Emit clock events
    if (onClockEvent) {
      if (beatChanged) {
        onClockEvent({
          type: 'beat',
          bar: newBar,
          beat: newBeat,
          phase: newPhase,
          bpm: clockState.bpm,
          timestamp: timestamp
        });
      }

      if (barChanged) {
        onClockEvent({
          type: 'bar',
          bar: newBar,
          beat: newBeat,
          phase: newPhase,
          bpm: clockState.bpm,
          timestamp: timestamp
        });
      }

      // Always emit phase updates
      onClockEvent({
        type: 'phase',
        bar: newBar,
        beat: newBeat,
        phase: newPhase,
        bpm: clockState.bpm,
        timestamp: timestamp
      });
    }

    // Run clock debugging analysis
    if (clockDebugRef.current && enableClockDebug) {
      const sample = clockDebugRef.current.analyzeClock(newClockState, timestamp / 1000);

      // Update debug overlay
      if (clockOverlayRef.current) {
        clockOverlayRef.current.update(newClockState, timestamp / 1000);
      }

      // Check for anomalies
      const bugs = clockDebugRef.current.detectBugs();
      bugs.forEach((bug: any) => {
        if (onClockAnomaly) {
          onClockAnomaly({
            type: bug.type,
            severity: bug.severity,
            message: bug.description,
            timestamp: timestamp,
            value: bug.value
          });
        }
      });
    }

    // Send clock sync over WebSocket
    if (bridgeRef.current && isConnected && beatChanged) {
      try {
        // Send clock sync message through bridge
        const syncMessage = {
          type: 'clock.sync',
          data: {
            bar: newBar,
            beat: newBeat,
            phase: newPhase,
            bpm: clockState.bpm,
            timestamp: timestamp
          }
        };

        // If bridge has WebSocket access, send the sync
        if (typeof bridgeRef.current.sendWebSocketMessage === 'function') {
          bridgeRef.current.sendWebSocketMessage(syncMessage);
        }
      } catch (err) {
        log('Failed to send clock sync:', err);
      }
    }

    clockAnimationRef.current = requestAnimationFrame(clockTick);
  }, [clockState, isConnected, enableClockDebug, onClockEvent, onClockAnomaly, log]);

  // Start/stop clock
  const startClock = useCallback((startBpm?: number) => {
    const newBpm = startBpm || bpm;
    setClockState(prev => ({
      ...prev,
      bpm: newBpm,
      running: true,
      startTime: performance.now(),
      bar: 0,
      beat: 0,
      phase: 0
    }));

    clockAnimationRef.current = requestAnimationFrame(clockTick);
    log(`Clock started at ${newBpm} BPM`);
  }, [bpm, clockTick, log]);

  const stopClock = useCallback(() => {
    setClockState(prev => ({
      ...prev,
      running: false,
      bar: 0,
      beat: 0,
      phase: 0
    }));

    if (clockAnimationRef.current) {
      cancelAnimationFrame(clockAnimationRef.current);
      clockAnimationRef.current = null;
    }

    log('Clock stopped');
  }, [log]);

  // Initialize the bridge when iframe loads (enhanced with clock sync)
  useEffect(() => {
    const initializeBridge = async () => {
      if (!iframeRef.current) return;

      try {
        log('Initializing Enhanced Tier-4 Room Bridge...');
        const bridge = await createTier4RoomBridge(iframeRef.current, websocketUrl);
        bridgeRef.current = bridge;

        // Set initial state if provided
        if (initialState) {
          bridge.setState(initialState);
        }

        // Setup event listeners with enhanced clock integration
        setupBridgeListeners(bridge);

        // Start clock if enabled
        if (enableClockDebug) {
          startClock();
        }

        setIsReady(true);
        setError(null);
        log('Enhanced bridge initialized successfully');

        if (onRoomReady) {
          onRoomReady(bridge);
        }

      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(errorMessage);
        log('Failed to initialize enhanced bridge:', err);
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
      stopClock();
      if (bridgeRef.current) {
        bridgeRef.current.disconnect();
      }
    };
  }, [iframeUrl, websocketUrl, initialState, enableClockDebug, onRoomReady, log, startClock, stopClock]);

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

        // Start/stop clock based on connection
        if (connected && enableClockDebug && !clockState.running) {
          startClock();
        } else if (!connected && clockState.running) {
          stopClock();
        }
      }
    };

    // Check connection status periodically
    const connectionInterval = setInterval(checkConnection, 1000);

    // Listen for operator applications (enhanced with timing)
    const handleOperatorEvent = (event: Event) => {
      const customEvent = event as CustomEvent;
      if (customEvent.detail && onOperatorApplied) {
        const { operator, previousState, newState } = customEvent.detail;

        // Add timing information to the event
        const enhancedDetail = {
          ...customEvent.detail,
          clockState: clockState,
          timestamp: performance.now()
        };

        onOperatorApplied(operator, previousState, newState);

        // Log to clock debug if enabled
        if (clockDebugRef.current) {
          log(`Operator '${operator}' applied at bar:${clockState.bar} beat:${clockState.beat}`);
        }
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
  }, [isConnected, onConnectionStatus, onOperatorApplied, onStateLoaded, log, clockState, enableClockDebug, startClock, stopClock]);

  // Expose enhanced methods through ref
  useImperativeHandle(ref, () => ({
    // Original methods
    applyOperator: (operator: string, meta?: any) => {
      if (bridgeRef.current) {
        // Add clock timing to meta
        const enhancedMeta = {
          ...meta,
          clockState: clockState,
          timestamp: performance.now()
        };
        bridgeRef.current.applyOperator(operator, enhancedMeta);
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

    isConnected: () => isConnected,

    // Enhanced clock debugging methods
    getClockStats: () => {
      return clockDebugRef.current?.getStats() || {};
    },

    exportClockData: () => {
      return clockDebugRef.current?.exportData() || {};
    },

    resetClockDebug: () => {
      if (clockDebugRef.current) {
        clockDebugRef.current.reset();
        log('Clock debug data reset');
      }
    },

    toggleClockDebug: () => {
      if (clockOverlayRef.current) {
        clockOverlayRef.current.toggle();
      }
    },

    syncClock: (syncBpm?: number) => {
      if (syncBpm) {
        setClockState(prev => ({ ...prev, bpm: syncBpm }));
      }

      if (bridgeRef.current && isConnected) {
        const syncMessage = {
          type: 'clock.sync',
          data: {
            ...clockState,
            bpm: syncBpm || clockState.bpm,
            timestamp: performance.now()
          }
        };

        try {
          if (typeof bridgeRef.current.sendWebSocketMessage === 'function') {
            bridgeRef.current.sendWebSocketMessage(syncMessage);
            log(`Clock synced at ${syncBpm || clockState.bpm} BPM`);
          }
        } catch (err) {
          log('Failed to sync clock:', err);
        }
      }
    }
  }), [isConnected, initialState, log, clockState]);

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
      <div className={`enhanced-engine-room-error ${className}`} style={style}>
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
            Enhanced Engine Room Error
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
    <div className={`enhanced-engine-room-container ${className}`} style={{ position: 'relative', ...style }}>
      <iframe
        ref={iframeRef}
        src={iframeUrl}
        title={title}
        style={defaultStyle}
        allow="microphone; camera; geolocation"
      />

      {/* Enhanced status indicators */}
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

        {enableClockDebug && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '4px 8px',
            background: clockState.running ? '#0d4f3c' : '#67060c',
            border: `1px solid ${clockState.running ? '#238636' : '#da3633'}`,
            borderRadius: '12px',
            color: clockState.running ? '#46d158' : '#f85149',
            cursor: 'pointer'
          }}
          onClick={() => clockOverlayRef.current?.toggle()}>
            <div style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: 'currentColor'
            }} />
            Clock {clockState.running ? `${clockState.bpm} BPM` : 'Stopped'}
          </div>
        )}
      </div>

      {/* Clock debug mini display */}
      {enableClockDebug && clockState.running && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          right: '10px',
          padding: '8px 12px',
          background: 'rgba(0, 0, 0, 0.8)',
          border: '1px solid #333',
          borderRadius: '8px',
          color: '#fff',
          fontFamily: 'monospace',
          fontSize: '12px',
          minWidth: '150px'
        }}>
          <div>Bar: {clockState.bar} | Beat: {clockState.beat}</div>
          <div>Phase: {clockState.phase.toFixed(3)} | BPM: {clockState.bpm}</div>
        </div>
      )}
    </div>
  );
});

EnhancedEngineRoom.displayName = 'EnhancedEngineRoom';

export default EnhancedEngineRoom;
