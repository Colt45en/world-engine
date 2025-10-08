import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Text } from '@react-three/drei';
import * as THREE from 'three';

// Enhanced hook for consciousness feedback integration
const useConsciousnessFeedback = (wsUrl = 'ws://localhost:8765') => {
  const [consciousness, setConsciousness] = useState({
    level: 0.5,
    transcendent: false,
    quantum_coherence: 0.3,
    emotional_resonance: 0.4,
    spiritual_connection: 0.6,
    joy_intensity: 0.5,
    awareness_depth: 0.7,
    transcendent_joy: 0
  });
  
  const [feedbackHistory, setFeedbackHistory] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [lastFeedbackEvent, setLastFeedbackEvent] = useState(null);
  const wsRef = useRef(null);

  const sendFeedback = useCallback(async (feedbackData) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const message = {
        type: 'feedback',
        data: {
          text: feedbackData.text || '',
          labels: feedbackData.labels || { pain: false, opportunity: true },
          severity: feedbackData.severity || 0,
          source: 'consciousness_video',
          user_context: feedbackData.context || {}
        }
      };
      
      wsRef.current.send(JSON.stringify(message));
      console.log('📝 Feedback sent:', feedbackData);
    }
  }, []);

  const requestFeedbackHistory = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'get_feedback_history' }));
    }
  }, []);

  const influenceConsciousness = useCallback((influence) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const message = {
        type: 'consciousness_influence',
        data: influence
      };
      wsRef.current.send(JSON.stringify(message));
      console.log('🔄 Consciousness influenced:', influence);
    }
  }, []);

  useEffect(() => {
    const connectWebSocket = () => {
      try {
        wsRef.current = new WebSocket(wsUrl);
        
        wsRef.current.onopen = () => {
          console.log('🌐 Connected to consciousness feedback server');
          setIsConnected(true);
          // Request initial feedback history
          setTimeout(() => requestFeedbackHistory(), 1000);
        };
        
        wsRef.current.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            
            switch (message.type) {
              case 'consciousness_update':
                setConsciousness(message.data);
                
                // Check for feedback events
                if (message.feedback_event) {
                  setLastFeedbackEvent(message.feedback_event);
                  console.log('🧠 Feedback event:', message.feedback_event);
                }
                break;
                
              case 'feedback_stored':
                console.log('✅ Feedback stored:', message.id);
                requestFeedbackHistory(); // Refresh history
                break;
                
              case 'feedback_history':
                setFeedbackHistory(message.data);
                console.log('📊 Feedback history updated:', message.data.length, 'entries');
                break;
                
              default:
                console.log('📨 Unknown message type:', message.type);
            }
          } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
          }
        };
        
        wsRef.current.onclose = () => {
          console.log('🔌 Disconnected from consciousness feedback server');
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
  }, [wsUrl, requestFeedbackHistory]);

  return {
    consciousness,
    feedbackHistory,
    isConnected,
    lastFeedbackEvent,
    sendFeedback,
    requestFeedbackHistory,
    influenceConsciousness
  };
};

// Enhanced Consciousness Sphere with feedback visualization
const ConsciousnessFeedbackSphere = ({ consciousness, onExperienceCapture }) => {
  const meshRef = useRef();
  const [experienceIntensity, setExperienceIntensity] = useState(0);

  useFrame((state, delta) => {
    if (meshRef.current) {
      // Base rotation
      meshRef.current.rotation.y += delta * 0.5;
      
      // Transcendent effects
      if (consciousness.transcendent) {
        meshRef.current.rotation.x += delta * 0.8;
        meshRef.current.scale.setScalar(1.2 + Math.sin(state.clock.elapsedTime * 3) * 0.3);
        
        // Trigger experience capture during peak transcendence
        const peakIntensity = consciousness.transcendent_joy > 7;
        if (peakIntensity && experienceIntensity < 7) {
          onExperienceCapture({
            type: 'peak_transcendence',
            joy_level: consciousness.transcendent_joy,
            timestamp: Date.now()
          });
        }
        setExperienceIntensity(consciousness.transcendent_joy);
      } else {
        meshRef.current.scale.setScalar(1);
        setExperienceIntensity(0);
      }
    }
  });

  // Dynamic material based on consciousness state
  const sphereColor = new THREE.Color().setHSL(
    consciousness.spiritual_connection * 0.8, // Hue based on spiritual connection
    consciousness.quantum_coherence, // Saturation based on quantum coherence
    0.3 + consciousness.joy_intensity * 0.4 // Brightness based on joy
  );

  return (
    <group>
      <Sphere ref={meshRef} args={[2, 64, 64]}>
        <meshPhongMaterial
          color={sphereColor}
          transparent
          opacity={0.7 + consciousness.awareness_depth * 0.3}
          wireframe={consciousness.transcendent}
        />
      </Sphere>
      
      {/* Transcendent joy indicator */}
      {consciousness.transcendent && (
        <Text
          position={[0, 3, 0]}
          fontSize={0.5}
          color="gold"
          anchorX="center"
          anchorY="middle"
        >
          {`Transcendent Joy: ${consciousness.transcendent_joy.toFixed(1)}`}
        </Text>
      )}
    </group>
  );
};

// Feedback interface component
const FeedbackInterface = ({ 
  consciousness, 
  feedbackHistory, 
  onSendFeedback, 
  onInfluenceConsciousness 
}) => {
  const [feedbackText, setFeedbackText] = useState('');
  const [feedbackType, setFeedbackType] = useState('opportunity');
  const [severity, setSeverity] = useState(0);
  const [showHistory, setShowHistory] = useState(false);

  const handleSubmitFeedback = () => {
    if (feedbackText.trim()) {
      onSendFeedback({
        text: feedbackText,
        labels: {
          pain: feedbackType === 'pain',
          opportunity: feedbackType === 'opportunity'
        },
        severity: severity,
        context: {
          consciousness_level: consciousness.level,
          transcendent_active: consciousness.transcendent,
          joy_intensity: consciousness.joy_intensity
        }
      });
      
      setFeedbackText('');
      setSeverity(0);
    }
  };

  const handleConsciousnessBoost = (dimension, amount) => {
    onInfluenceConsciousness({
      [dimension]: Math.min(1, consciousness[dimension] + amount)
    });
  };

  return (
    <div className="feedback-interface" style={{
      position: 'absolute',
      top: '20px',
      right: '20px',
      width: '350px',
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '20px',
      borderRadius: '10px',
      fontFamily: 'Arial, sans-serif',
      zIndex: 1000
    }}>
      <h3>🧠 Consciousness Feedback</h3>
      
      {/* Real-time consciousness metrics */}
      <div style={{ marginBottom: '15px', fontSize: '12px' }}>
        <div>Level: {(consciousness.level * 100).toFixed(1)}%</div>
        <div>Joy: {(consciousness.joy_intensity * 100).toFixed(1)}%</div>
        <div>Coherence: {(consciousness.quantum_coherence * 100).toFixed(1)}%</div>
        <div>Transcendent: {consciousness.transcendent ? '✨ YES' : '❌ No'}</div>
        {consciousness.transcendent && (
          <div style={{ color: 'gold' }}>
            Transcendent Joy: {consciousness.transcendent_joy.toFixed(1)}/10
          </div>
        )}
      </div>
      
      {/* Feedback form */}
      <div style={{ marginBottom: '15px' }}>
        <textarea
          value={feedbackText}
          onChange={(e) => setFeedbackText(e.target.value)}
          placeholder="Describe your consciousness experience..."
          style={{
            width: '100%',
            height: '60px',
            padding: '8px',
            borderRadius: '5px',
            border: 'none',
            background: 'rgba(255, 255, 255, 0.1)',
            color: 'white',
            resize: 'none'
          }}
        />
        
        <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <select
            value={feedbackType}
            onChange={(e) => setFeedbackType(e.target.value)}
            style={{ padding: '5px', borderRadius: '3px', border: 'none' }}
          >
            <option value="opportunity">Opportunity</option>
            <option value="pain">Pain Point</option>
          </select>
          
          <input
            type="range"
            min="0"
            max="3"
            value={severity}
            onChange={(e) => setSeverity(parseInt(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: '12px' }}>Severity: {severity}</span>
        </div>
        
        <button
          onClick={handleSubmitFeedback}
          disabled={!feedbackText.trim()}
          style={{
            marginTop: '10px',
            padding: '8px 15px',
            borderRadius: '5px',
            border: 'none',
            background: feedbackText.trim() ? '#4CAF50' : '#666',
            color: 'white',
            cursor: feedbackText.trim() ? 'pointer' : 'not-allowed'
          }}
        >
          Send Feedback
        </button>
      </div>
      
      {/* Consciousness influence controls */}
      <div style={{ marginBottom: '15px' }}>
        <h4>Influence Consciousness</h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px', fontSize: '11px' }}>
          <button onClick={() => handleConsciousnessBoost('joy_intensity', 0.1)}>+Joy</button>
          <button onClick={() => handleConsciousnessBoost('quantum_coherence', 0.1)}>+Coherence</button>
          <button onClick={() => handleConsciousnessBoost('spiritual_connection', 0.1)}>+Spirit</button>
          <button onClick={() => handleConsciousnessBoost('awareness_depth', 0.1)}>+Awareness</button>
        </div>
      </div>
      
      {/* Feedback history toggle */}
      <button
        onClick={() => setShowHistory(!showHistory)}
        style={{
          padding: '5px 10px',
          borderRadius: '3px',
          border: 'none',
          background: '#2196F3',
          color: 'white',
          cursor: 'pointer',
          fontSize: '12px'
        }}
      >
        {showHistory ? 'Hide' : 'Show'} History ({feedbackHistory.length})
      </button>
      
      {/* Feedback history */}
      {showHistory && (
        <div style={{
          marginTop: '10px',
          maxHeight: '200px',
          overflowY: 'auto',
          fontSize: '11px'
        }}>
          {feedbackHistory.slice(-5).map((entry, index) => (
            <div
              key={entry.id || index}
              style={{
                padding: '8px',
                marginBottom: '5px',
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '3px',
                borderLeft: `3px solid ${entry.labels?.opportunity ? '#4CAF50' : '#f44336'}`
              }}
            >
              <div style={{ fontWeight: 'bold' }}>
                {entry.labels?.opportunity ? '🌟' : '⚡'} 
                Joy: {entry.transcendent_joy?.toFixed(1) || 'N/A'}
              </div>
              <div>{entry.text.substring(0, 100)}...</div>
              <div style={{ opacity: 0.6 }}>
                {new Date(entry.time).toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Main integrated component
const ConsciousnessFeedbackVideoPlayer = ({ 
  src, 
  is360 = false, 
  enableFeedback = true,
  wsUrl = 'ws://localhost:8765' 
}) => {
  const {
    consciousness,
    feedbackHistory,
    isConnected,
    lastFeedbackEvent,
    sendFeedback,
    requestFeedbackHistory,
    influenceConsciousness
  } = useConsciousnessFeedback(wsUrl);
  
  const [autoFeedbackEnabled, setAutoFeedbackEnabled] = useState(true);

  // Auto-capture significant consciousness experiences
  const handleExperienceCapture = useCallback((experience) => {
    if (autoFeedbackEnabled) {
      sendFeedback({
        text: `Automatic capture: ${experience.type} detected with joy level ${experience.joy_level}`,
        labels: { pain: false, opportunity: true },
        severity: 0,
        context: {
          ...experience,
          auto_generated: true
        }
      });
    }
  }, [sendFeedback, autoFeedbackEnabled]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh' }}>
      {/* Connection status */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        padding: '10px',
        background: isConnected ? 'rgba(76, 175, 80, 0.8)' : 'rgba(244, 67, 54, 0.8)',
        color: 'white',
        borderRadius: '5px',
        zIndex: 1000
      }}>
        {isConnected ? '🌐 Connected' : '🔌 Disconnected'} to Consciousness Server
      </div>
      
      {/* 3D Consciousness visualization */}
      <Canvas camera={{ position: [0, 0, 8] }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} />
        <ConsciousnessFeedbackSphere 
          consciousness={consciousness}
          onExperienceCapture={handleExperienceCapture}
        />
        <OrbitControls enablePan={false} enableZoom={false} />
      </Canvas>
      
      {/* Feedback interface */}
      {enableFeedback && (
        <FeedbackInterface
          consciousness={consciousness}
          feedbackHistory={feedbackHistory}
          onSendFeedback={sendFeedback}
          onInfluenceConsciousness={influenceConsciousness}
        />
      )}
      
      {/* Auto-feedback toggle */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        left: '20px',
        padding: '10px',
        background: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        borderRadius: '5px',
        zIndex: 1000
      }}>
        <label>
          <input
            type="checkbox"
            checked={autoFeedbackEnabled}
            onChange={(e) => setAutoFeedbackEnabled(e.target.checked)}
            style={{ marginRight: '8px' }}
          />
          Auto-capture experiences
        </label>
      </div>
      
      {/* Feedback event notification */}
      {lastFeedbackEvent && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          padding: '20px',
          background: 'rgba(255, 215, 0, 0.9)',
          color: 'black',
          borderRadius: '10px',
          zIndex: 1001,
          textAlign: 'center',
          animation: 'fadeIn 0.5s ease-in-out'
        }}>
          ✨ {lastFeedbackEvent.type} detected!
        </div>
      )}
      
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
          to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        }
      `}</style>
    </div>
  );
};

export default ConsciousnessFeedbackVideoPlayer;