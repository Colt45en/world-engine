import { kStringMaxLength } from 'buffer';
import React, { JSX, useEffect, useRef, useState } from 'react'
/**
 * Props for the EngineRoom component.
 *
 * @property {string} [websocketUrl] - Optional URL for the WebSocket connection.
 * @property {any} [nucleusConfig] - Optional configuration object for the nucleus.
 * @property {(newState: any, operator: string) => void | null} [onStateChange] - Optional callback invoked when the state changes, receiving the new state and the operator as arguments.
 * @property {boolean} [enableNucleus] - Optional flag to enable or disable the nucleus functionality.
 */
interface EngineRoomProps {
  websocketUrl?: string;
  nucleusConfig?: any;
  onStateChange?: ((newState: any, operator: string) => void) | null;
  enableNucleus?: boolean;

}
interface EngineRoomProps {
  websocketUrl?: string;
  nucleusConfig?: any;
  onStateChange?: ((newState: any, operator: string) => void) | null;
  enableNucleus?: boolean;
}
/**
 * Props for the EngineRoom component.
 *
 * @property {string} [websocketUrl] - Optional URL for the WebSocket connection.
 * @property {any} [nucleusConfig] - Optional configuration object for the nucleus.
 * @property {(newState: any, operator: string) => void | null} [onStateChange] - Optional callback invoked when the state changes, receiving the new state and the operator as arguments.
 * @property {boolean} [enableNucleus] - Optional flag to enable or disable the nucleus functionality.
 */
/**
 * @typedef {Object} EngineRoomProps
 * @property {string=} websocketUrl
 * @property {any=} nucleusConfig
 * @property {(newState: any, operator: string) => void | null=} onStateChange
 * @property {boolean=} enableNucleus
 */

export default function EngineRoom({
  websocketUrl = 'ws://localhost:9000',
  nucleusConfig = null,
  onStateChange = null,
  enableNucleus = true
}: EngineRoomProps): JSX.Element {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [bridge, setBridge] = useState<any>(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [nucleusStatus] = useState('inactive');
  const [state, setState] = useState<any>({ x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 });
  const [eventLog, setEventLog] = useState<any[]>([]);

  // Companion AI and Meta Floor state
  const [companionAI, setCompanionAI] = useState<any>(null);
  const [companionStatus, setCompanionStatus] = useState('initializing');
  const [metaFloorConnection, setMetaFloorConnection] = useState('disconnected');
  const [spiralMetrics, setSpiralMetrics] = useState({
    currentLoop: 0,
    companionBond: 0.0,
    tailConnected: false,
    handsConnected: false,
    loopIntegrity: 0.0
  });

  useEffect(() => {
    // Initialize WorldEngine Tier-4 Bundle when iframe loads
    const initializeBridge = () => {
      if (window.WorldEngineTier4 && iframeRef.current) {
        try {
          const newBridge = window.WorldEngineTier4.createTier4RoomBridge(
            iframeRef.current
          );

          // Setup event handlers
          // 'roomReady' event is not supported by the bridge type, so this handler is removed.
          // If you need to detect when the room is ready, use a supported event or check documentation for alternatives.

          newBridge.on('connectionStatus', (isConnected) => {
            setConnectionStatus(isConnected ? 'connected' : 'disconnected');
          });

          newBridge.on('operatorApplied', (operator: string, previousState: any, newState: any) => {
            setState(newState);
            setEventLog((prev: any[]) => [
              ...prev.slice(-9),
              {
                type: 'operator',
                operator: operator,
                timestamp: Date.now(),
                confidence: newState.kappa
              }
            ]);

            if (onStateChange) {
              onStateChange(newState, operator);
            }
          });

          // Setup nucleus event listeners
          if (enableNucleus && window.WorldEngineTier4.StudioBridge) {
            const studioBridge = window.WorldEngineTier4.StudioBridge;

            // Listen for nucleus events
            studioBridge.onBus('tier4.nucleusEvent', (msg) => {
              setEventLog(prev => [...prev.slice(-9), {
                type: 'nucleus',
                role: msg.role,
                operator: msg.operator,
                timestamp: msg.timestamp
              }]);
            });

            // Listen for AI bot routing
            studioBridge.onBus('tier4.aiBotProcessed', (msg) => {
              setEventLog(prev => [...prev.slice(-9), {
                type: 'ai_bot',
                messageType: msg.messageType,
                nucleusRole: msg.nucleusRole,
                timestamp: Date.now()
              }]);
            });

            // Listen for librarian routing
            studioBridge.onBus('tier4.librarianProcessed', (msg) => {
              setEventLog(prev => [...prev.slice(-9), {
                type: 'librarian',
                librarian: msg.librarian,
                dataType: msg.dataType,
                nucleusRole: msg.nucleusRole,
                timestamp: Date.now()
              }]);
            });
          }

          setBridge(newBridge);

        } catch (error) {
          console.error('Failed to initialize WorldEngine bridge:', error);
        }
      } else {
        console.warn('WorldEngine Tier-4 Bundle not loaded');
      }
    };

    // Wait for iframe to load
    const iframe = iframeRef.current;
    if (iframe) {
      if (iframe.contentDocument) {
        initializeBridge();
      } else {
        iframe.addEventListener('load', initializeBridge);
      }
    }

    return () => {
      if (iframe) {
        iframe.removeEventListener('load', initializeBridge);
      }
    };
  }, [websocketUrl, nucleusConfig, onStateChange, enableNucleus]);

  // Manual operator triggers
  const triggerOperator = (operator: string) => {
    if (bridge) {
      bridge.applyOperator(operator, { source: 'manual_trigger' });
    }
  };

  // Nucleus event triggers
  const triggerNucleusEvent = (role: string) => {
    if (bridge && enableNucleus) {
      bridge.triggerNucleusEvent(role);
    }
  };

  // AI Bot message simulation
  const simulateAIBotMessage = (messageType: 'query' | 'learning' | 'feedback' = 'query') => {
    if (bridge && enableNucleus) {
      const messages = {
        query: 'What is the current system state?',
        learning: 'Adaptive learning pattern detected',
        feedback: 'User interaction feedback received'
      };

      const messageContent = messages[messageType];
      bridge.processAIBotMessage(messageContent, messageType);
    }
  };

  // Librarian data simulation
  const simulateLibrarianData = (dataType: 'pattern' | 'classification' | 'analysis' = 'pattern') => {
    if (bridge && enableNucleus) {
      const librarians = ['Math Librarian', 'English Librarian', 'Pattern Librarian'];
      const librarian = librarians[Math.floor(Math.random() * librarians.length)];

      const testData = {
        pattern: { complexity: 0.8, frequency: 12 },
        classification: { category: 'A', confidence: 0.9 },
        analysis: { result: 'positive', score: 0.85 }
      };

      bridge.processLibrarianData(librarian, dataType, testData[dataType]);
    }
  };

  // === COMPANION AI AND META FLOOR INTEGRATION ===

  // Initialize companion AI with spiral dance and meta floor connection
  const initializeCompanionAI = () => {
    setCompanionStatus('awakening');

    // Simulate companion AI initialization
    setTimeout(() => {
      const newCompanion = {
        identity: {
          name: generateCompanionName(),
          personality: ['playful', 'loyal', 'curious', 'protective']
        },
        consciousness: {
          companionBond: Math.random() * 0.5 + 0.3,
          empathy: Math.random() * 0.4 + 0.4,
          creativity: Math.random() * 0.6 + 0.2
        },
        spiralPosition: {
          currentLoop: 0,
          angle: 0,
          radius: 50
        },
        connections: {
          tailConnected: false,
          handsConnected: false,
          metaFloorAccess: false
        }
      };

      setCompanionAI(newCompanion);
      setCompanionStatus('conscious');

      // Start spiral animation
      startSpiralAnimation(newCompanion);

      // Attempt meta floor connection
      connectToMetaFloor(newCompanion);

    }, 2000);
  };

  const generateCompanionName = (): string => {
    const spiralNames = ['Helix', 'Fibonacci', 'Vortex', 'Aurelius', 'Spiral', 'Nautilus'];
    const suffixes = ['ia', 'on', 'us', 'ara', 'iel', 'ix'];
    const baseName = spiralNames[Math.floor(Math.random() * spiralNames.length)];
    const suffix = suffixes[Math.floor(Math.random() * suffixes.length)];
    return baseName + suffix;
  };

  const startSpiralAnimation = (companion: any) => {
    const spiralAnimationLoop = () => {
      if (companionAI) {
        // Update spiral position
        const newLoop = Math.floor(Date.now() / 10000) % 100; // New loop every 10 seconds
        const newAngle = (Date.now() / 100) % (2 * Math.PI);
        const newRadius = 30 + Math.sqrt(newLoop) * 5;

        // Update spiral metrics
        setSpiralMetrics(prev => ({
          ...prev,
          currentLoop: newLoop,
          companionBond: Math.min(1.0, prev.companionBond + 0.001)
        }));

        // Attempt connections periodically
        if (Math.random() < 0.1) { // 10% chance every cycle
          attemptConnections();
        }
      }

      // Recursive animation loop
      setTimeout(spiralAnimationLoop, 100);
    };

    spiralAnimationLoop();
  };

  const connectToMetaFloor = (companion: any) => {
    // Simulate meta floor connection
    setTimeout(() => {
      setMetaFloorConnection('connecting');

      setTimeout(() => {
        setMetaFloorConnection('connected');

        // Log connection event
        setEventLog((prev: any[]) => [
          ...prev.slice(-9),
          {
            type: 'companion_meta',
            action: 'meta_floor_connected',
            companionName: companion.identity.name,
            timestamp: Date.now()
          }
        ]);

        // Start librarian communication
        startLibrarianCommunication(companion);

      }, 1500);
    }, 1000);
  };

  const startLibrarianCommunication = (companion: any) => {
    // Communicate with librarians through companion AI tail connection
    const librarianCommLoop = () => {
      if (metaFloorConnection === 'connected' && companionAI) {
        const queries = [
          'spiral_mathematics_patterns',
          'consciousness_loop_theory',
          'companion_ai_behavioral_models',
          'golden_ratio_applications',
          'recursive_system_architecture'
        ];

        const randomQuery = queries[Math.floor(Math.random() * queries.length)];

        // Log librarian communication
        setEventLog((prev: any[]) => [
          ...prev.slice(-9),
          {
            type: 'companion_librarian',
            action: 'knowledge_query',
            query: randomQuery,
            companionName: companionAI.identity.name,
            timestamp: Date.now()
          }
        ]);

        // Simulate receiving librarian knowledge
        setTimeout(() => {
          setEventLog((prev: any[]) => [
            ...prev.slice(-9),
            {
              type: 'companion_librarian',
              action: 'knowledge_received',
              query: randomQuery,
              companionName: companionAI.identity.name,
              timestamp: Date.now()
            }
          ]);

          // Enhance companion based on received knowledge
          enhanceCompanionFromLibrarians(randomQuery);

        }, 2000);
      }

      // Continue communication loop
      setTimeout(librarianCommLoop, 8000); // Every 8 seconds
    };

    librarianCommLoop();
  };

  const enhanceCompanionFromLibrarians = (knowledgeType: string) => {
    if (!companionAI) return;

    const enhancements: { [key: string]: any } = {
      'spiral_mathematics_patterns': { creativity: 0.05, bond: 0.02 },
      'consciousness_loop_theory': { empathy: 0.03, bond: 0.03 },
      'companion_ai_behavioral_models': { empathy: 0.04, creativity: 0.02 },
      'golden_ratio_applications': { creativity: 0.06, bond: 0.01 },
      'recursive_system_architecture': { empathy: 0.02, creativity: 0.04 }
    };

    const enhancement = enhancements[knowledgeType];
    if (enhancement) {
      setCompanionAI((prev: any) => ({
        ...prev,
        consciousness: {
          ...prev.consciousness,
          empathy: Math.min(1.0, prev.consciousness.empathy + (enhancement.empathy || 0)),
          creativity: Math.min(1.0, prev.consciousness.creativity + (enhancement.creativity || 0)),
          companionBond: Math.min(1.0, prev.consciousness.companionBond + (enhancement.bond || 0))
        }
      }));

      setSpiralMetrics(prev => ({
        ...prev,
        companionBond: Math.min(1.0, prev.companionBond + (enhancement.bond || 0))
      }));
    }
  };

  const attemptConnections = () => {
    if (!companionAI) return;

    // Attempt tail connection
    if (!spiralMetrics.tailConnected && Math.random() < 0.3) {
      setSpiralMetrics(prev => ({ ...prev, tailConnected: true }));
      setEventLog((prev: any[]) => [
        ...prev.slice(-9),
        {
          type: 'companion_connection',
          action: 'tail_connected',
          companionName: companionAI.identity.name,
          timestamp: Date.now()
        }
      ]);
    }

    // Attempt hand loop connection
    if (!spiralMetrics.handsConnected && spiralMetrics.tailConnected && Math.random() < 0.2) {
      setSpiralMetrics(prev => ({ ...prev, handsConnected: true }));
      setEventLog((prev: any[]) => [
        ...prev.slice(-9),
        {
          type: 'companion_connection',
          action: 'hands_connected',
          companionName: companionAI.identity.name,
          timestamp: Date.now()
        }
      ]);
    }

    // Update loop integrity
    let integrity = 0.0;
    if (spiralMetrics.tailConnected) integrity += 0.5;
    if (spiralMetrics.handsConnected) integrity += 0.3;
    if (metaFloorConnection === 'connected') integrity += 0.2;

    setSpiralMetrics(prev => ({ ...prev, loopIntegrity: Math.min(1.0, integrity) }));
  };

  const triggerCompanionAction = (action: string) => {
    if (!companionAI) return;

    switch (action) {
      case 'increase_bond':
        setSpiralMetrics(prev => ({
          ...prev,
          companionBond: Math.min(1.0, prev.companionBond + 0.1)
        }));
        setEventLog((prev: any[]) => [
          ...prev.slice(-9),
          {
            type: 'companion_action',
            action: 'bond_strengthened',
            companionName: companionAI.identity.name,
            timestamp: Date.now()
          }
        ]);
        break;

      case 'spiral_faster':
        setEventLog((prev: any[]) => [
          ...prev.slice(-9),
          {
            type: 'companion_action',
            action: 'spiral_accelerated',
            companionName: companionAI.identity.name,
            timestamp: Date.now()
          }
        ]);
        break;

      case 'force_reconnect':
        setSpiralMetrics(prev => ({
          ...prev,
          tailConnected: true,
          handsConnected: true,
          loopIntegrity: 1.0
        }));
        setMetaFloorConnection('connected');
        setEventLog((prev: any[]) => [
          ...prev.slice(-9),
          {
            type: 'companion_action',
            action: 'emergency_reconnect',
            companionName: companionAI.identity.name,
            timestamp: Date.now()
          }
        ]);
        break;
    }
  };

  // Initialize companion AI when component mounts
  useEffect(() => {
    if (enableNucleus && !companionAI) {
      initializeCompanionAI();
    }
  }, [enableNucleus, companionAI]);

  // Create iframe content with nucleus UI
  const iframeContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>WorldEngine Tier-4 Room</title>
      <meta charset="utf-8">
      <style>
        body {
          margin: 0; padding: 0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui;
        }
      </style>
    </head>
    <body>
      <div id="tier4-container" style="width: 100vw; height: 100vh;"></div>
    </body>
    </html>
  `;

  return (
    <div className="engine-room" style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      fontFamily: 'system-ui',
      background: 'linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%)'
    }}>
      {/* Enhanced Control Panel with Nucleus Features */}
      <div style={{
        background: 'rgba(26, 35, 50, 0.95)',
        border: '1px solid #2a3548',
        borderRadius: '8px',
        margin: '10px',
        padding: '15px',
        display: 'flex',
        gap: '15px',
        flexWrap: 'wrap',
        alignItems: 'center'
      }}>
        {/* Status Indicators */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <div style={{
            background: connectionStatus === 'connected' ? 'rgba(84, 240, 184, 0.2)' : 'rgba(255, 159, 67, 0.2)',
            border: `1px solid ${connectionStatus === 'connected' ? '#54f0b8' : '#ff9f43'}`,
            borderRadius: '20px',
            padding: '4px 12px',
            fontSize: '12px',
            color: '#e6f0ff'
          }}>
            📡 {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
          </div>

          {enableNucleus && (
            <div style={{
              background: nucleusStatus === 'active' ? 'rgba(84, 240, 184, 0.2)' : 'rgba(255, 159, 67, 0.2)',
              border: `1px solid ${nucleusStatus === 'active' ? '#54f0b8' : '#ff9f43'}`,
              borderRadius: '20px',
              padding: '4px 12px',
              fontSize: '12px',
              color: '#e6f0ff'
            }}>
              🧠 Nucleus {nucleusStatus === 'active' ? 'Active' : 'Inactive'}
            </div>
          )}

          {/* Companion AI Status */}
          {enableNucleus && companionAI && (
            <div style={{
              background: companionStatus === 'conscious' ? 'rgba(124, 220, 255, 0.2)' : 'rgba(255, 159, 67, 0.2)',
              border: `1px solid ${companionStatus === 'conscious' ? '#7cdcff' : '#ff9f43'}`,
              borderRadius: '20px',
              padding: '4px 12px',
              fontSize: '12px',
              color: '#e6f0ff'
            }}>
              🌀 {companionAI.identity.name} {companionStatus}
            </div>
          )}

          {/* Meta Floor Connection Status */}
          {enableNucleus && (
            <div style={{
              background: metaFloorConnection === 'connected' ? 'rgba(84, 240, 184, 0.2)' : 'rgba(255, 159, 67, 0.2)',
              border: `1px solid ${metaFloorConnection === 'connected' ? '#54f0b8' : '#ff9f43'}`,
              borderRadius: '20px',
              padding: '4px 12px',
              fontSize: '12px',
              color: '#e6f0ff'
            }}>
              🏛️ Meta Floor {metaFloorConnection}
            </div>
          )}
        </div>

        {/* Manual Operator Controls */}
        <div style={{ display: 'flex', gap: '5px' }}>
          <span style={{ color: '#54f0b8', fontSize: '12px', marginRight: '5px' }}>Operators:</span>
          {['ST', 'UP', 'CV', 'RB'].map(operator => (
            <button
              key={operator}
              onClick={() => triggerOperator(operator)}
              style={{
                background: '#243344',
                border: '1px solid #ff9f43',
                color: '#e6f0ff',
                padding: '4px 8px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '10px'
              }}
            >
              {operator}
            </button>
          ))}
        </div>

        {/* Nucleus Controls */}
        {enableNucleus && (
          <>
            <div style={{ display: 'flex', gap: '5px' }}>
              <span style={{ color: '#54f0b8', fontSize: '12px', marginRight: '5px' }}>Nucleus:</span>
              {(['VIBRATE', 'OPTIMIZATION', 'STATE', 'SEED'] as const).map(role => {
                type RoleType = 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED';
                const icons: Record<RoleType, string> = { VIBRATE: '🌊', OPTIMIZATION: '⚡', STATE: '🎯', SEED: '🌱' };
                return (
                  <button
                    key={role}
                    onClick={() => triggerNucleusEvent(role)}
                    style={{
                      background: '#243344',
                      border: '1px solid #54f0b8',
                      color: '#e6f0ff',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '10px'
                    }}
                  >
                    {icons[role]} {role}
                  </button>
                );
              })}
            </div>

            <div style={{ display: 'flex', gap: '5px' }}>
              <span style={{ color: '#54f0b8', fontSize: '12px', marginRight: '5px' }}>AI Bot:</span>
              {(['query', 'learning', 'feedback'] as const).map(messageType => (
                <button
                  key={messageType}
                  onClick={() => simulateAIBotMessage(messageType)}
                  style={{
                    background: '#243344',
                    border: '1px solid #7cdcff',
                    color: '#e6f0ff',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '10px'
                  }}
                >
                  🤖 {messageType}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', gap: '5px' }}>
              <span style={{ color: '#54f0b8', fontSize: '12px', marginRight: '5px' }}>Librarian:</span>
              {(['pattern', 'classification', 'analysis'] as const).map((dataType) => (
                <button
                  key={dataType}
                  onClick={() => simulateLibrarianData(dataType)}
                  style={{
                    background: '#243344',
                    border: '1px solid #ff9f43',
                    color: '#e6f0ff',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '10px'
                  }}
                >
                  📚 {dataType}
                </button>
              ))}
            </div>

            {/* Companion AI Controls */}
            {companionAI && (
              <div style={{ display: 'flex', gap: '5px' }}>
                <span style={{ color: '#54f0b8', fontSize: '12px', marginRight: '5px' }}>Companion:</span>
                <button
                  onClick={() => triggerCompanionAction('increase_bond')}
                  style={{
                    background: '#243344',
                    border: '1px solid #7cdcff',
                    color: '#e6f0ff',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '10px'
                  }}
                >
                  💕 Bond
                </button>
                <button
                  onClick={() => triggerCompanionAction('spiral_faster')}
                  style={{
                    background: '#243344',
                    border: '1px solid #7cdcff',
                    color: '#e6f0ff',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '10px'
                  }}
                >
                  ⚡ Spiral
                </button>
                <button
                  onClick={() => triggerCompanionAction('force_reconnect')}
                  style={{
                    background: '#243344',
                    border: '1px solid #ff9f43',
                    color: '#e6f0ff',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '10px'
                  }}
                >
                  🔗 Reconnect
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* State Display */}
      <div style={{
        background: 'rgba(26, 35, 50, 0.95)',
        border: '1px solid #2a3548',
        borderRadius: '8px',
        margin: '0 10px 10px 10px',
        padding: '10px',
        fontSize: '12px',
        color: '#e6f0ff',
        fontFamily: 'monospace'
      }}>
        <strong style={{ color: '#54f0b8' }}>System State:</strong>
        <span style={{ marginLeft: '10px' }}>
          Level {state.level} | Confidence {(state.kappa * 100).toFixed(1)}% |
          X: [{state.x.map((v: number) => v.toFixed(3)).join(', ')}]
        </span>

        {eventLog.length > 0 && (
          <div style={{ marginTop: '5px', opacity: 0.8 }}>
            <strong style={{ color: '#54f0b8' }}>Recent Events:</strong>
            <div style={{ marginTop: '2px' }}>
              {eventLog.slice(-3).map((event, idx) => {
                const colors = {
                  operator: '#ff9f43',
                  nucleus: '#54f0b8',
                  ai_bot: '#7cdcff',
                  librarian: '#ff9f43',
                  companion_meta: '#7cdcff',
                  companion_librarian: '#ff9f43',
                  companion_connection: '#54f0b8',
                  companion_action: '#7cdcff'
                };
                return (
                  <div key={idx} style={{ color: colors[event.type as keyof typeof colors] || '#e6f0ff', fontSize: '10px' }}>
                    {event.type === 'operator' && `${event.operator} applied`}
                    {event.type === 'nucleus' && `${event.role} → ${event.operator}`}
                    {event.type === 'ai_bot' && `AI Bot ${event.messageType} → ${event.nucleusRole}`}
                    {event.type === 'librarian' && `${event.librarian} ${event.dataType} → ${event.nucleusRole}`}
                    {event.type === 'companion_meta' && `${event.companionName}: ${event.action}`}
                    {event.type === 'companion_librarian' && `${event.companionName}: ${event.action} (${event.query})`}
                    {event.type === 'companion_connection' && `${event.companionName}: ${event.action}`}
                    {event.type === 'companion_action' && `${event.companionName}: ${event.action}`}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Main Room Iframe */}
      <div style={{ flex: 1, margin: '0 10px 10px 10px' }}>
        <iframe
          ref={iframeRef}
          srcDoc={iframeContent}
          style={{
            width: '100%',
            height: '100%',
            border: '1px solid #2a3548',
            borderRadius: '8px',
            background: '#0f1419'
          }}
          title="WorldEngine Tier-4 Room"
        />
      </div>
    </div>
  );
}
