// tier4_engine_room_demo.tsx - Complete demo showing Engine Room integration with Tier-4

import React, { useRef, useState, useEffect } from 'react';
import EngineRoom, { useEngineRoom, EngineRoomRef } from './EngineRoom';
import { Tier4State, Tier4RoomBridge } from './tier4_room_integration';

// CSS for the demo (since inline styles cause lint issues)
const demoStyles = `
  .demo-container {
    display: flex;
    height: 100vh;
    background: #0b0e14;
    color: #e6f0ff;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  .control-panel {
    width: 320px;
    background: #121824;
    border-right: 1px solid #2a3548;
    padding: 20px;
    overflow-y: auto;
  }

  .engine-room-panel {
    flex: 1;
    position: relative;
  }

  .demo-section {
    margin-bottom: 24px;
    padding: 16px;
    background: #1a2332;
    border-radius: 8px;
    border: 1px solid #2a3548;
  }

  .demo-section h3 {
    margin: 0 0 12px 0;
    color: #54f0b8;
    font-size: 16px;
  }

  .state-display {
    background: #0d1117;
    border: 1px solid #2a3548;
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
    font-family: 'Fira Code', monospace;
    font-size: 13px;
  }

  .vector-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 8px 0;
  }

  .vector-item {
    text-align: center;
    padding: 8px;
    background: #243344;
    border-radius: 4px;
    font-size: 12px;
  }

  .button-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 8px 0;
  }

  .demo-button {
    background: #243344;
    border: 1px solid #54f0b8;
    color: #e6f0ff;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
  }

  .demo-button:hover {
    background: #54f0b8;
    color: #0b0e14;
  }

  .demo-button.macro {
    border-color: #b366d9;
  }

  .demo-button.macro:hover {
    background: #b366d9;
  }

  .demo-button.nucleus {
    border-color: #ff9f43;
  }

  .demo-button.nucleus:hover {
    background: #ff9f43;
  }

  .status-indicators {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 16px;
    font-size: 11px;
    font-weight: 600;
  }

  .status-indicator.ready {
    background: #0d4f3c;
    border: 1px solid #238636;
    color: #46d158;
  }

  .status-indicator.disconnected {
    background: #67060c;
    border: 1px solid #da3633;
    color: #f85149;
  }

  .status-indicator.loading {
    background: #1a2332;
    border: 1px solid #54f0b8;
    color: #54f0b8;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
  }

  .log-container {
    background: #0a0f1c;
    border: 1px solid #2a3548;
    border-radius: 6px;
    padding: 12px;
    max-height: 200px;
    overflow-y: auto;
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    line-height: 1.4;
  }

  .log-entry {
    margin-bottom: 4px;
    padding: 2px 0;
  }

  .log-timestamp {
    color: #7cdcff;
    opacity: 0.8;
  }

  .log-operator {
    color: #54f0b8;
    font-weight: 600;
  }

  .log-state {
    color: #ff9f43;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin: 8px 0;
  }

  .metric-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 8px;
    background: #0d1117;
    border-radius: 4px;
    font-size: 12px;
  }

  .custom-panel-controls {
    margin-top: 12px;
  }

  .panel-input {
    background: #0d1117;
    border: 1px solid #2a3548;
    color: #e6f0ff;
    padding: 6px 8px;
    border-radius: 4px;
    width: 100%;
    margin: 4px 0;
    font-size: 12px;
  }

  .small-button {
    padding: 4px 8px;
    font-size: 11px;
    background: #1a2332;
    border: 1px solid #2a3548;
    color: #e6f0ff;
    border-radius: 4px;
    cursor: pointer;
    margin: 2px 4px 2px 0;
  }

  .small-button:hover {
    background: #243344;
  }
`;

interface LogEntry {
  timestamp: number;
  type: 'operator' | 'state' | 'connection' | 'nucleus';
  message: string;
  data?: any;
}

interface CommunicationEntry {
  timestamp: string;
  type: 'ai_bot' | 'librarian';
  source: string;
  message: string;
}

const Tier4EngineRoomDemo: React.FC = () => {
  const roomRef = useRef<EngineRoomRef>(null);
  const [currentState, setCurrentState] = useState<Tier4State>({
    x: [0, 0.5, 0.4, 0.6],
    kappa: 0.6,
    level: 0
  });
  const [isRoomReady, setIsRoomReady] = useState(false);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [bridge, setBridge] = useState<Tier4RoomBridge | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [communicationLog, setCommunicationLog] = useState<CommunicationEntry[]>([]);
  const [operationCount, setOperationCount] = useState(0);
  const [sessionStartTime] = useState(Date.now());

  // Add log entry
  const addLog = (type: LogEntry['type'], message: string, data?: any) => {
    const entry: LogEntry = {
      timestamp: Date.now(),
      type,
      message,
      data
    };
    setLogs(prev => [entry, ...prev.slice(0, 49)]); // Keep last 50 entries
  };

  // Handle operator applications
  const handleOperatorApplied = (operator: string, previousState: Tier4State, newState: Tier4State) => {
    setCurrentState(newState);
    setOperationCount(prev => prev + 1);
    addLog('operator', `Applied ${operator}`, { previousState, newState });
  };

  // Handle state loads
  const handleStateLoaded = (state: Tier4State, cid: string) => {
    setCurrentState(state);
    addLog('state', `Loaded state ${cid.slice(0, 12)}...`, state);
  };

  // Combined Nucleus Processing System
  const nucleusToOperatorMap = {
    'VIBRATE': 'ST',      // Stabilization
    'OPTIMIZATION': 'UP',  // Update/Progress
    'STATE': 'CV',        // Convergence
    'SEED': 'RB'          // Rollback
  } as const;

  const processNucleusEvent = (role: keyof typeof nucleusToOperatorMap, data?: any) => {
    if (!roomRef.current) return;

    // Get the corresponding Tier-4 operator
    const operator = nucleusToOperatorMap[role];

    // Log the nucleus event
    addLog('nucleus', `Processing nucleus: ${role} → ${operator}`, { role, operator, data });

    // Auto-apply the corresponding operator
    roomRef.current.applyOperator(operator, {
      source: 'nucleus_auto',
      nucleusRole: role,
      triggerId: data?.id || `nucleus_${Date.now()}`,
      ...data
    });

    // Also trigger the nucleus event for WebSocket relay
    roomRef.current.triggerNucleusEvent(role);
  };

  // Manual nucleus trigger (simplified interface)
  const triggerNucleus = (role: keyof typeof nucleusToOperatorMap) => {
    processNucleusEvent(role, { source: 'manual_trigger' });
  };

  // Handle external nucleus events (via custom event system)
  useEffect(() => {
    const handleExternalNucleusEvent = (event: CustomEvent) => {
      const { role, data } = event.detail;
      if (role && nucleusToOperatorMap[role as keyof typeof nucleusToOperatorMap]) {
        processNucleusEvent(role as keyof typeof nucleusToOperatorMap, {
          source: 'external_event',
          originalEvent: event.detail,
          ...data
        });
      }
    };

    // Listen for custom nucleus events
    window.addEventListener('nucleus-event', handleExternalNucleusEvent as EventListener);

    return () => {
      window.removeEventListener('nucleus-event', handleExternalNucleusEvent as EventListener);
    };
  }, [processNucleusEvent]);

  // Handle room ready
  const handleRoomReady = (bridgeInstance: Tier4RoomBridge) => {
    setIsRoomReady(true);
    setBridge(bridgeInstance);
    addLog('connection', 'Engine Room initialized and ready');
  };

  // Handle connection status
  const handleConnectionStatus = (connected: boolean) => {
    setIsWebSocketConnected(connected);
    addLog('connection', `WebSocket ${connected ? 'connected' : 'disconnected'}`);
  };

  // Manual operator application
  const applyOperator = (operator: string) => {
    if (roomRef.current) {
      roomRef.current.applyOperator(operator, { source: 'manual' });
    }
  };

  // Manual macro application
  const applyMacro = (macro: string) => {
    if (roomRef.current) {
      roomRef.current.applyMacro(macro);
      addLog('operator', `Started macro: ${macro}`);
    }
  };

  // Add custom panel
  const addCustomPanel = () => {
    if (roomRef.current) {
      roomRef.current.addPanel({
        wall: 'back',
        x: 100 + Math.random() * 200,
        y: 100 + Math.random() * 200,
        w: 300,
        h: 200,
        title: `Custom Panel ${Date.now()}`,
        html: `<!doctype html><meta charset="utf-8"><style>
          body{margin:0;background:#0b1418;color:#cfe;font:12px system-ui;padding:16px}
          .custom-content{text-align:center;padding:20px}
          .timestamp{color:#54f0b8;font-family:monospace}
        </style>
        <div class="custom-content">
          <h3>Custom Panel</h3>
          <p>Created at:</p>
          <div class="timestamp">${new Date().toLocaleString()}</div>
          <p>This panel was added dynamically from the control interface.</p>
        </div>`
      });
      addLog('state', 'Added custom panel to room');
    }
  };

  // Show toast message
  const showToast = () => {
    if (roomRef.current) {
      roomRef.current.toast(`Demo toast message at ${new Date().toLocaleTimeString()}`);
    }
  };

  // AI Bot Communication with Nucleus
  const sendAIBotMessage = (message: string, type: 'query' | 'learning' | 'feedback' = 'query') => {
    if (!roomRef.current) return;

    // Add to communication log
    setCommunicationLog(prev => [...prev.slice(-19), {
      timestamp: new Date().toLocaleTimeString(),
      type: 'ai_bot',
      source: 'AI Bot',
      message: `${type}: ${message}`
    }]);

    const aiMessage = {
      type: 'ai_bot_message',
      role: type === 'learning' ? 'OPTIMIZATION' : type === 'feedback' ? 'STATE' : 'VIBRATE',
      message,
      timestamp: Date.now(),
      source: 'ai_bot'
    };

    // Send to nucleus for processing
    processNucleusEvent(aiMessage.role as keyof typeof nucleusToOperatorMap, {
      source: 'ai_bot',
      message: aiMessage.message,
      messageType: type,
      originalMessage: aiMessage
    });

    addLog('nucleus', `AI Bot → Nucleus: ${type.toUpperCase()} - ${message.slice(0, 50)}...`);
  };

  // Librarian Data Processing and Transmission
  const sendLibrarianData = (librarian: string, dataType: 'pattern' | 'classification' | 'analysis', data: any) => {
    if (!roomRef.current) return;

    // Add to communication log
    setCommunicationLog(prev => [...prev.slice(-19), {
      timestamp: new Date().toLocaleTimeString(),
      type: 'librarian',
      source: librarian,
      message: `${dataType}: ${JSON.stringify(data)}`
    }]);

    const librarianMessage = {
      type: 'librarian_data',
      librarian,
      dataType,
      data,
      timestamp: Date.now()
    };

    // Determine nucleus pipeline based on data type
    const nucleusRole = dataType === 'pattern' ? 'VIBRATE' :
                       dataType === 'classification' ? 'STATE' : 'OPTIMIZATION';

    processNucleusEvent(nucleusRole as keyof typeof nucleusToOperatorMap, {
      source: 'librarian',
      librarian,
      dataType,
      data,
      originalMessage: librarianMessage
    });

    addLog('nucleus', `📚 ${librarian} → Nucleus: ${dataType} data processed`);
  };

  // WorldEngine unified integration
  useEffect(() => {
    // Initialize WorldEngine unified system if available
    if (typeof window !== 'undefined' && (window as any).WorldEngineUI) {
      const WorldEngineUI = (window as any).WorldEngineUI;

      // Set up nucleus event integration with unified system
      const handleUnifiedNucleusEvent = (event: any) => {
        if (event.type === 'nucleus_exec' && event.role) {
          processNucleusEvent(event.role as keyof typeof nucleusToOperatorMap, {
            source: 'worldengine_unified',
            originalEvent: event,
            ...event.data
          });
        }
      };

      // Listen for WorldEngine unified events
      if (WorldEngineUI.StudioBridge && typeof WorldEngineUI.StudioBridge.onBus === 'function') {
        WorldEngineUI.StudioBridge.onBus(handleUnifiedNucleusEvent);
        addLog('connection', 'WorldEngine unified system integrated');
      }

      return () => {
        // Cleanup if needed
        if (WorldEngineUI.StudioBridge && typeof WorldEngineUI.StudioBridge.offBus === 'function') {
          WorldEngineUI.StudioBridge.offBus(handleUnifiedNucleusEvent);
        }
      };
    }
  }, [processNucleusEvent]);

  // Calculate uptime
  const uptimeMinutes = Math.floor((Date.now() - sessionStartTime) / 60000);

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: demoStyles }} />
      <div className="demo-container">
        {/* Control Panel */}
        <div className="control-panel">
          <h2 style={{ margin: '0 0 20px 0', color: '#54f0b8' }}>🧠 Nucleus Control Center</h2>
          <div style={{ fontSize: '12px', marginBottom: '16px', opacity: 0.8, lineHeight: '1.4' }}>
            The Nucleus lives inside the Engine Room, orchestrating librarians and automation pipelines.
            It's the heartbeat connecting math and English algorithms to create intelligence.
          </div>

          {/* Nucleus Heartbeat Status */}
          <div className="demo-section">
            <h3>🧠 Nucleus Heartbeat</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              The central intelligence orchestrating all systems
            </div>
            <div className="status-indicators">
              <div className={`status-indicator ${isRoomReady ? 'ready' : 'loading'}`}>
                <div className="status-dot" />
                Nucleus {isRoomReady ? '💓 Active' : '⏳ Initializing'}
              </div>
            </div>
            <div className="status-indicators">
              <div className={`status-indicator ${isWebSocketConnected ? 'ready' : 'disconnected'}`}>
                <div className="status-dot" />
                Hub Connection {isWebSocketConnected ? '🔗 Connected' : '📡 Disconnected'}
              </div>
            </div>
            <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.7 }}>
              Pulse Rate: {operationCount}/min • Learning: Active • Teaching: Math+English
            </div>
          </div>

          {/* Librarian Data Management */}
          <div className="demo-section">
            <h3>📚 Librarian Network</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              Data librarians storing and labeling information for the Nucleus
            </div>
            <div className="metrics-grid">
              <div className="metric-item">
                <span>📊 Data Stores:</span>
                <span>{bridge?.getStateHistory().size || 0}</span>
              </div>
              <div className="metric-item">
                <span>🏷️ Labels:</span>
                <span>{logs.length}</span>
              </div>
              <div className="metric-item">
                <span>🔄 Processing:</span>
                <span>{operationCount}</span>
              </div>
              <div className="metric-item">
                <span>⏱️ Uptime:</span>
                <span>{uptimeMinutes}m</span>
              </div>
            </div>
            <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.7 }}>
              Librarians: Math Algorithm ✓ | English Algorithm ✓ | Pattern Recognition ✓
            </div>
          </div>

          {/* Nucleus Intelligence State */}
          <div className="demo-section">
            <h3>🧠 Nucleus Intelligence</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              Current learning state and neural pathways
            </div>
            <div className="state-display">
              <div>🎯 Intelligence Level: {currentState.level}</div>
              <div>🔮 Confidence (κ): {currentState.kappa.toFixed(3)}</div>
              <div>🧮 Neural Pathways [Persistence, Information, Goal, Context]:</div>
              <div className="vector-grid">
                {currentState.x.map((value, i) => (
                  <div key={i} className="vector-item">
                    {['🔒 P', '📡 I', '🎯 G', '🧠 C'][i]}: {value.toFixed(3)}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Mathematical Operations */}
          <div className="demo-section">
            <h3>🔢 Mathematical Engine</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              Core mathematical algorithms connected to the Nucleus
            </div>
            <div className="button-grid">
              {['ST', 'UP', 'PR', 'CV', 'RB', 'RS'].map(op => (
                <button
                  key={op}
                  className="demo-button"
                  onClick={() => applyOperator(op)}
                  disabled={!isRoomReady}
                  title={`Mathematical operation: ${op}`}
                >
                  🔢 {op}
                </button>
              ))}
            </div>
          </div>

          {/* Language Processing */}
          <div className="demo-section">
            <h3>📝 Language Engine</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              English language algorithms teaching the Nucleus
            </div>
            <div className="button-grid">
              {['IDE_A', 'IDE_B', 'MERGE_ABC'].map(macro => (
                <button
                  key={macro}
                  className="demo-button macro"
                  onClick={() => applyMacro(macro)}
                  disabled={!isRoomReady}
                  title={`Language macro: ${macro}`}
                >
                  📝 {macro}
                </button>
              ))}
            </div>
          </div>

          {/* AI Bot Communication */}
          <div className="demo-section">
            <h3>🤖 AI Bot Hub</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              AI Bot sends messages and queries to the Nucleus
            </div>
            <div className="button-grid">
              <button
                className="demo-button"
                onClick={() => sendAIBotMessage('Analyze current patterns', 'query')}
                disabled={!isRoomReady}
                style={{ borderColor: '#7cdcff' }}
              >
                🤖 Query
              </button>
              <button
                className="demo-button"
                onClick={() => sendAIBotMessage('Learning optimization needed', 'learning')}
                disabled={!isRoomReady}
                style={{ borderColor: '#7cdcff' }}
              >
                🧠 Learning
              </button>
              <button
                className="demo-button"
                onClick={() => sendAIBotMessage('Feedback on recent results', 'feedback')}
                disabled={!isRoomReady}
                style={{ borderColor: '#7cdcff' }}
              >
                💬 Feedback
              </button>
            </div>
          </div>

          {/* Librarian Data Processing */}
          <div className="demo-section">
            <h3>📚 Librarian Data Flow</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              Librarians process and send classified data to the Nucleus
            </div>
            <div className="button-grid">
              <button
                className="demo-button"
                onClick={() => sendLibrarianData('Math Librarian', 'pattern', { equations: 12, complexity: 'high' })}
                disabled={!isRoomReady}
                style={{ borderColor: '#ff9f43' }}
              >
                🔢 Math Data
              </button>
              <button
                className="demo-button"
                onClick={() => sendLibrarianData('English Librarian', 'classification', { words: 245, sentiment: 'positive' })}
                disabled={!isRoomReady}
                style={{ borderColor: '#ff9f43' }}
              >
                📝 Language Data
              </button>
              <button
                className="demo-button"
                onClick={() => sendLibrarianData('Pattern Librarian', 'analysis', { patterns: 8, confidence: 0.85 })}
                disabled={!isRoomReady}
                style={{ borderColor: '#ff9f43' }}
              >
                🔍 Pattern Data
              </button>
            </div>
            <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.7 }}>
              📚 Math Librarian: Mathematical patterns & equations<br/>
              📝 English Librarian: Language processing & sentiment<br/>
              🔍 Pattern Librarian: Recognition & analysis patterns
            </div>
          </div>

          {/* Communication Activity Log */}
          <div className="demo-section">
            <h3>📡 Communication Feed</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              Real-time communication flow between AI Bot, Librarians & Nucleus
            </div>
            <div style={{
              backgroundColor: 'rgba(0,0,0,0.3)',
              padding: '8px',
              borderRadius: '6px',
              minHeight: '80px',
              fontSize: '10px',
              fontFamily: 'monospace',
              maxHeight: '120px',
              overflowY: 'auto',
              border: '1px solid rgba(255,255,255,0.1)'
            }}>
              {communicationLog.length > 0 ? (
                communicationLog.slice(-8).map((entry, i) => (
                  <div key={i} style={{
                    marginBottom: '3px',
                    color: entry.type === 'ai_bot' ? '#7cdcff' : '#ff9f43'
                  }}>
                    <span style={{ opacity: 0.6 }}>[{entry.timestamp}]</span>
                    <span style={{ fontWeight: 'bold' }}> {entry.source}:</span> {entry.message}
                  </div>
                ))
              ) : (
                <div style={{ opacity: 0.5 }}>No communication activity yet...</div>
              )}
            </div>
          </div>

          <div className="demo-section">
            <h3>🌐 Intelligence Hub</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              Connection to the greater intelligence network
            </div>
            <div className="status-indicators">
              <div className={`status-indicator ${typeof window !== 'undefined' && (window as any).WorldEngineUI ? 'ready' : 'disconnected'}`}>
                <div className="status-dot" />
                Hub {typeof window !== 'undefined' && (window as any).WorldEngineUI ? '🌟 Online' : '📡 Offline'}
              </div>
            </div>
            {typeof window !== 'undefined' && (window as any).WorldEngineUI && (
              <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.7 }}>
                📡 Bridge: {(window as any).WorldEngineUI.StudioBridge ? '✓' : '✗'} |
                🏠 Room: {(window as any).WorldEngineUI.EngineRoom ? '✓' : '✗'} |
                🔄 Pipeline: {(window as any).WorldEngineUI.mountPipelineCanvas ? '✓' : '✗'}
              </div>
            )}
          </div>

          {/* Nucleus Automation Pipelines */}
          <div className="demo-section">
            <h3>🔄 Automation Pipelines</h3>
            <div style={{ fontSize: '11px', marginBottom: '12px', opacity: 0.8 }}>
              The Nucleus orchestrates these automation workflows
            </div>
            <div className="button-grid">
              {(Object.keys(nucleusToOperatorMap) as Array<keyof typeof nucleusToOperatorMap>).map(role => (
                <button
                  key={role}
                  className="demo-button nucleus"
                  onClick={() => triggerNucleus(role)}
                  disabled={!isRoomReady}
                  title={`Pipeline: ${role} executes ${nucleusToOperatorMap[role]} operation`}
                >
                  <div style={{ fontSize: '12px' }}>{role === 'VIBRATE' ? '🌊' : role === 'OPTIMIZATION' ? '⚡' : role === 'STATE' ? '🎯' : '🌱'}</div>
                  <div style={{ fontSize: '10px', marginTop: '2px' }}>{role}</div>
                  <div style={{ fontSize: '9px', opacity: 0.7 }}>→ {nucleusToOperatorMap[role]}</div>
                </button>
              ))}
            </div>
            <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.7 }}>
              🌊 VIBRATE→ST: Stabilize learning | ⚡ OPTIMIZATION→UP: Enhance algorithms<br/>
              🎯 STATE→CV: Converge understanding | 🌱 SEED→RB: Reset patterns
            </div>
          </div>

          {/* Room Controls */}
          <div className="demo-section">
            <h3>Room Controls</h3>
            <button className="small-button" onClick={addCustomPanel} disabled={!isRoomReady}>
              Add Custom Panel
            </button>
            <button className="small-button" onClick={showToast} disabled={!isRoomReady}>
              Show Toast
            </button>
            {typeof window !== 'undefined' && (window as any).WorldEngineUI && (
              <>
                <button className="small-button" onClick={() => triggerUnifiedFeature('synthetic_run')}>
                  Unified Synthetic
                </button>
                <button className="small-button" onClick={() => triggerUnifiedFeature('pipeline_canvas')}>
                  Pipeline Canvas
                </button>
              </>
            )}
          </div>

          {/* Metrics */}
          <div className="demo-section">
            <h3>Metrics</h3>
            <div className="metrics-grid">
              <div className="metric-item">
                <span>Operations:</span>
                <span>{operationCount}</span>
              </div>
              <div className="metric-item">
                <span>Uptime:</span>
                <span>{uptimeMinutes}m</span>
              </div>
              <div className="metric-item">
                <span>Log Entries:</span>
                <span>{logs.length}</span>
              </div>
              <div className="metric-item">
                <span>State History:</span>
                <span>{bridge?.getStateHistory().size || 0}</span>
              </div>
            </div>
          </div>

          {/* Activity Log */}
          <div className="demo-section">
            <h3>Activity Log</h3>
            <div className="log-container">
              {logs.map((log, i) => (
                <div key={i} className="log-entry">
                  <span className="log-timestamp">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  {' '}
                  <span className={`log-${log.type}`}>
                    [{log.type.toUpperCase()}]
                  </span>
                  {' '}
                  {log.message}
                </div>
              ))}
              {logs.length === 0 && (
                <div style={{ opacity: 0.6, fontStyle: 'italic' }}>
                  No activity yet. Try applying an operator!
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Engine Room Panel */}
        <div className="engine-room-panel">
          <EngineRoom
            ref={roomRef}
            iframeUrl="/worldengine.html"
            websocketUrl="ws://localhost:9000"
            title="Tier-4 Nucleus Engine Room"
            initialState={currentState}
            onOperatorApplied={handleOperatorApplied}
            onStateLoaded={handleStateLoaded}
            onRoomReady={handleRoomReady}
            onConnectionStatus={handleConnectionStatus}
            debug={true}
          />
        </div>
      </div>
    </>
  );
};

// Alternative hook-based usage example
export const HookBasedDemo: React.FC = () => {
  const {
    roomRef,
    roomProps,
    isReady,
    isConnected,
    currentState,
    applyOperator,
    applyMacro,
    triggerNucleus,
    toast
  } = useEngineRoom({
    iframeUrl: '/worldengine.html',
    websocketUrl: 'ws://localhost:9000',
    title: 'Hook-Based Engine Room',
    debug: true,
    onOperatorApplied: (op, prev, next) => {
      console.log(`Hook: Applied ${op}`, { prev, next });
    }
  });

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ width: '200px', padding: '20px', background: '#121824' }}>
        <h3>Hook-Based Controls</h3>
        <p>Ready: {isReady ? 'Yes' : 'No'}</p>
        <p>Connected: {isConnected ? 'Yes' : 'No'}</p>
        <p>κ: {currentState.kappa.toFixed(3)}</p>
        <button onClick={() => applyOperator('UP')}>Apply UP</button>
        <button onClick={() => triggerNucleus('VIBRATE')}>Trigger VIBRATE</button>
        <button onClick={() => toast('Hook toast!')}>Show Toast</button>
      </div>
      <div style={{ flex: 1 }}>
        <EngineRoom ref={roomRef} {...roomProps} />
      </div>
    </div>
  );
};

export default Tier4EngineRoomDemo;
