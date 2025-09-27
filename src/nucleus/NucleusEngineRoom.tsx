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
                const colors = { operator: '#ff9f43', nucleus: '#54f0b8', ai_bot: '#7cdcff', librarian: '#ff9f43' };
                return (
                  <div key={idx} style={{ color: colors[event.type] || '#e6f0ff', fontSize: '10px' }}>
                    {event.type === 'operator' && `${event.operator} applied`}
                    {event.type === 'nucleus' && `${event.role} → ${event.operator}`}
                    {event.type === 'ai_bot' && `AI Bot ${event.messageType} → ${event.nucleusRole}`}
                    {event.type === 'librarian' && `${event.librarian} ${event.dataType} → ${event.nucleusRole}`}
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
