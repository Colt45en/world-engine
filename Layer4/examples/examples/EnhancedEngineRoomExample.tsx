// Example Usage of Enhanced Engine Room Component
// Demonstrates how to use the EnhancedEngineRoom with clock debugging

import React, { useState, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import EnhancedEngineRoom, { EnhancedEngineRoomRef } from '../components/EnhancedEngineRoom';
import { Tier4State } from '../components/roomAdapter';
import './EnhancedEngineRoomExample.css';

interface ExampleAppProps {
  websocketUrl?: string;
  enableDebug?: boolean;
}

const ExampleApp: React.FC<ExampleAppProps> = ({
  websocketUrl = 'ws://localhost:9000',
  enableDebug = true
}) => {
  const engineRef = useRef<EnhancedEngineRoomRef>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [clockStats, setClockStats] = useState<any>({});
  const [currentState, setCurrentState] = useState<Tier4State>({
    x: [0, 0.5, 0.4, 0.6],
    kappa: 0.6,
    level: 0
  });
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [errorLog, setErrorLog] = useState<string[]>([]);

  // Handle operator applications
  const handleOperatorApplied = useCallback((
    operator: string,
    previousState: Tier4State,
    newState: Tier4State
  ) => {
    try {
      console.log(`Operator ${operator} applied:`, { previousState, newState });
      setCurrentState(newState);

      // Log successful operation
      if (enableDebug) {
        setErrorLog(prev => [...prev.slice(-9), `✅ ${operator}: ${JSON.stringify(newState)}`]);
      }
    } catch (error) {
      const errorMsg = `❌ Error applying ${operator}: ${error}`;
      console.error(errorMsg);
      setErrorLog(prev => [...prev.slice(-9), errorMsg]);
    }
  }, [enableDebug]);

  // Enhanced error handling wrapper
  const withErrorHandling = useCallback(<T extends any[]>(
    fn: (...args: T) => void,
    operation: string
  ) => {
    return (...args: T) => {
      try {
        fn(...args);
      } catch (error) {
        const errorMsg = `❌ Error in ${operation}: ${error}`;
        console.error(errorMsg);
        setErrorLog(prev => [...prev.slice(-9), errorMsg]);
        if (engineRef.current) {
          engineRef.current.toast(`Error: ${operation} failed`);
        }
      }
    };
  }, []);

  // Recording functionality
  const toggleRecording = useCallback(() => {
    setIsRecording(prev => {
      const newState = !prev;
      if (newState) {
        // Start recording
        const startTime = Date.now();
        const interval = setInterval(() => {
          setRecordingDuration(Date.now() - startTime);
        }, 1000);

        // Store interval ID for cleanup
        (window as any).recordingInterval = interval;

        console.log('🔴 Recording started');
        if (engineRef.current) {
          engineRef.current.toast('Recording started');
        }
      } else {
        // Stop recording
        if ((window as any).recordingInterval) {
          clearInterval((window as any).recordingInterval);
          delete (window as any).recordingInterval;
        }
        setRecordingDuration(0);

        console.log('⏹️ Recording stopped');
        if (engineRef.current) {
          engineRef.current.toast('Recording stopped');
        }
      }
      return newState;
    });
  }, []);

  // Handle state loading
  const handleStateLoaded = useCallback((state: Tier4State, cid: string) => {
    console.log('State loaded:', { state, cid });
    setCurrentState(state);
  }, []);

  // Handle room ready
  const handleRoomReady = useCallback((bridge: any) => {
    console.log('Enhanced Engine Room is ready!', bridge);

    // Example: Apply an initial operator after room is ready
    setTimeout(() => {
      if (engineRef.current) {
        engineRef.current.applyOperator('UP', { initial: true });
      }
    }, 1000);
  }, []);

  // Handle connection status changes
  const handleConnectionStatus = useCallback((connected: boolean) => {
    console.log('Connection status:', connected);
    setIsConnected(connected);
  }, []);

  // Handle clock events
  const handleClockEvent = useCallback((event: {
    type: 'beat' | 'bar' | 'phase';
    bar: number;
    beat: number;
    phase: number;
    bpm: number;
    timestamp: number;
  }) => {
    if (event.type === 'beat') {
      console.log(`Beat ${event.beat} of Bar ${event.bar}`);
    }

    // Update clock stats periodically
    if (event.type === 'bar' && engineRef.current) {
      const stats = engineRef.current.getClockStats();
      setClockStats(stats);
    }
  }, []);

  // Handle clock anomalies
  const handleClockAnomaly = useCallback((anomaly: {
    type: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH';
    message: string;
    timestamp: number;
    value?: number;
  }) => {
    console.warn('Clock anomaly detected:', anomaly);

    // Show toast for high severity anomalies
    if (anomaly.severity === 'HIGH' && engineRef.current) {
      engineRef.current.toast(`Clock Anomaly: ${anomaly.message}`);
    }
  }, []);

  // Manual controls with error handling
  const applyOperator = withErrorHandling((operator: string) => {
    if (engineRef.current) {
      engineRef.current.applyOperator(operator);
    }
  }, 'Apply Operator');

  const applyMacro = withErrorHandling((macro: string) => {
    if (engineRef.current) {
      engineRef.current.applyMacro(macro);
    }
  }, 'Apply Macro');

  const triggerNucleusEvent = withErrorHandling((role: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED') => {
    if (engineRef.current) {
      engineRef.current.triggerNucleusEvent(role);
    }
  }, 'Trigger Nucleus Event');

  const showToast = withErrorHandling((message: string) => {
    if (engineRef.current) {
      engineRef.current.toast(message);
    }
  }, 'Show Toast');

  const exportClockData = withErrorHandling(() => {
    if (engineRef.current) {
      const data = engineRef.current.exportClockData();
      console.log('Clock debug data:', data);

      // Enhanced export with metadata
      const exportData = {
        timestamp: new Date().toISOString(),
        session: 'enhanced-engine-room-example',
        clockData: data,
        currentState,
        clockStats,
        isRecording,
        recordingDuration,
        errorLog
      };

      // Download as JSON file
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `enhanced-engine-room-debug-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);

      if (engineRef.current) {
        engineRef.current.toast('Clock data exported successfully');
      }
    }
  }, 'Export Clock Data');

  const syncClock = withErrorHandling((bpm?: number) => {
    if (engineRef.current) {
      engineRef.current.syncClock(bpm);
    }
  }, 'Sync Clock');

  const toggleClockDebug = withErrorHandling(() => {
    if (engineRef.current) {
      engineRef.current.toggleClockDebug();
    }
  }, 'Toggle Clock Debug');

  return (
    <div className="engine-room-demo-container">
      {/* Control Panel */}
      <div className="control-panel">
        <h3>Tier 4 Enhanced Engine Room Demo</h3>

        <div className="button-group">
          <button onClick={() => applyOperator('RB')}>RB</button>
          <button onClick={() => applyOperator('UP')}>UP</button>
          <button onClick={() => applyOperator('CV')}>CV</button>
          <button onClick={() => applyOperator('TL')}>TL</button>
          <button onClick={() => applyOperator('MV')}>MV</button>
          <button onClick={() => applyOperator('SC')}>SC</button>
        </div>

        <div className="button-group">
          <button onClick={() => applyMacro('RESET')}>Reset</button>
          <button onClick={() => applyMacro('CHAOS')}>Chaos</button>
          <button onClick={() => applyMacro('STABILIZE')}>Stabilize</button>
          <button onClick={() => applyMacro('EXPLORE')}>Explore</button>
        </div>

        <div className="button-group">
          <button onClick={() => triggerNucleusEvent('VIBRATE')}>Vibrate</button>
          <button onClick={() => triggerNucleusEvent('OPTIMIZATION')}>Optimize</button>
          <button onClick={() => triggerNucleusEvent('STATE')}>State</button>
          <button onClick={() => triggerNucleusEvent('SEED')}>Seed</button>
        </div>

        <div className="button-group">
          <button onClick={() => showToast('Hello from Tier 4!')}>Toast</button>
          <button onClick={toggleClockDebug}>Toggle Clock Debug</button>
          <button onClick={() => syncClock(140)}>Sync Clock (140 BPM)</button>
          <button onClick={exportClockData}>Export Debug Data</button>
        </div>

        <div className="button-group">
          <button
            onClick={toggleRecording}
            className={isRecording ? 'recording' : ''}
          >
            {isRecording ? '⏹️ Stop Recording' : '🔴 Start Recording'}
          </button>
          {isRecording && (
            <span className="recording-timer">
              {Math.floor(recordingDuration / 1000)}s
            </span>
          )}
        </div>

        <div className="status-panel">\n          <span className={isConnected ? 'connected' : 'disconnected'}>
            Connected: {isConnected ? '✅' : '❌'}
          </span>\n          <span>Level: {currentState.level}</span>\n          <span>Kappa: {currentState.kappa.toFixed(3)}</span>\n          <span>X: [{currentState.x.map(x => x.toFixed(2)).join(', ')}]</span>
          {isRecording && <span className="recording-indicator">🔴 REC</span>}
        </div>
      </div>

      {/* Enhanced Engine Room */}
      <div className="engine-room-container">
        <EnhancedEngineRoom
          ref={engineRef}
          iframeUrl="/demos/tier4_collaborative_demo.html"
          websocketUrl={websocketUrl}
          sessionId="example-session"
          title="Enhanced Tier-4 Engine Room Example"
          initialState={currentState}
          enableClockDebug={true}
          bpm={120}
          onOperatorApplied={handleOperatorApplied}
          onStateLoaded={handleStateLoaded}
          onRoomReady={handleRoomReady}
          onConnectionStatus={handleConnectionStatus}
          onClockEvent={handleClockEvent}
          onClockAnomaly={handleClockAnomaly}
          debug={enableDebug}
          className="example-engine-room"
        />
      </div>

      {/* Debug Panel */}
      {enableDebug && (
        <div className="debug-panel">
          <h4>Enhanced Debug Info</h4>
          <div>
            <strong>Current State:</strong> {JSON.stringify(currentState, null, 2)}
          </div>
          <div>
            <strong>Clock Stats:</strong> {JSON.stringify(clockStats, null, 2)}
          </div>
          {isRecording && (
            <div>
              <strong>Recording:</strong> {Math.floor(recordingDuration / 1000)}s active
            </div>
          )}
          <div>
            <strong>Recent Events:</strong>
            {errorLog.length > 0 ? (
              errorLog.slice(-5).map((log, index) => (
                <div key={index} className="event-log-item">
                  {log}
                </div>
              ))
            ) : (
              <div className="no-events">No events logged</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ExampleApp;

// Example standalone usage
export const createEnhancedEngineRoomDemo = (container: HTMLElement) => {
  // React and ReactDOM are already imported at the top of the file
  // so we can use them directly here

  createRoot(container).render(<ExampleApp />);
};

// Example with custom configuration
export const createCustomEngineRoom = (config: {
  container: HTMLElement;
  websocketUrl?: string;
  initialState?: Tier4State;
  enableClockDebug?: boolean;
  bpm?: number;
  onStateChange?: (state: Tier4State) => void;
}) => {
  const {
    container,
    websocketUrl = 'ws://localhost:9000',
    initialState = { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 },
    enableClockDebug = true,
    bpm = 120,
    onStateChange
  } = config;

  const App = () => {
    return (
      <EnhancedEngineRoom
        iframeUrl="/demos/tier4_collaborative_demo.html"
        websocketUrl={websocketUrl}
        initialState={initialState}
        enableClockDebug={enableClockDebug}
        bpm={bpm}
        onOperatorApplied={(op, prev, next) => {
          console.log(`Operator ${op}:`, prev, '->', next);
          onStateChange?.(next);
        }}
        style={{ width: '100%', height: '100%' }}
      />
    );
  };

  // React and ReactDOM are already imported at the top of the file
  // so we can use them directly here

  createRoot(container).render(<App />);
};
