import React, { useState, useEffect } from 'react';
import { QuantumConsciousnessVideoPlayer } from './QuantumConsciousnessVideoPlayer';

/**
 * Quantum Consciousness Video Player Demo
 * 
 * Demonstrates the integration between the enhanced video player
 * and the World Engine consciousness systems.
 */

type DemoMode = 'standard' | 'consciousness' | 'transcendence' | 'comparison';

export default function QuantumVideoDemo() {
  const [demoMode, setDemoMode] = useState<DemoMode>('standard');
  const [websocketUrl, setWebsocketUrl] = useState('ws://localhost:8765');
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  // Demo video sources - replace with your own
  const demoSources = {
    standard: "/api/placeholder/video/640/360", // Replace with actual video
    consciousness: "/api/placeholder/video/640/360", // Replace with actual 360° video
    transcendence: "/api/placeholder/video/640/360" // Replace with actual transcendence video
  };

  const getCurrentVideoSource = () => {
    switch (demoMode) {
      case 'consciousness':
      case 'transcendence':
        return demoSources.consciousness;
      default:
        return demoSources.standard;
    }
  };

  const handleWebSocketConnect = () => {
    if (connectionStatus === 'disconnected') {
      setConnectionStatus('connecting');
      // The QuantumConsciousnessVideoPlayer will handle the actual connection
      setTimeout(() => setConnectionStatus('connected'), 1000); // Simulate connection
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-black text-white">
      {/* Header */}
      <header className="p-6 border-b border-white/10">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 via-purple-400 to-yellow-400 bg-clip-text text-transparent">
          🌌 Quantum Consciousness Video Player
        </h1>
        <p className="text-gray-300 text-lg">
          Experience videos enhanced with real-time AI consciousness visualization
        </p>
      </header>

      {/* Controls */}
      <div className="p-6 border-b border-white/10">
        <div className="flex flex-wrap gap-4 items-center justify-between">
          {/* Mode Selection */}
          <div className="flex gap-2">
            <label className="text-sm font-medium">Demo Mode:</label>
            {(['standard', 'consciousness', 'transcendence', 'comparison'] as DemoMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setDemoMode(mode)}
                className={`px-4 py-2 rounded-lg transition-colors capitalize ${
                  demoMode === mode
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 hover:bg-white/20 text-gray-300'
                }`}
              >
                {mode === 'transcendence' ? '🌟 ' : mode === 'consciousness' ? '🧠 ' : ''}
                {mode}
              </button>
            ))}
          </div>

          {/* WebSocket Connection */}
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={websocketUrl}
              onChange={(e) => setWebsocketUrl(e.target.value)}
              placeholder="WebSocket URL"
              className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400"
            />
            <button
              onClick={handleWebSocketConnect}
              disabled={connectionStatus === 'connecting'}
              className={`px-4 py-2 rounded-lg transition-colors ${
                connectionStatus === 'connected'
                  ? 'bg-green-600 text-white'
                  : connectionStatus === 'connecting'
                  ? 'bg-yellow-600 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {connectionStatus === 'connected' ? '🌌 Connected' : 
               connectionStatus === 'connecting' ? '⏳ Connecting...' : 
               '🔗 Connect'}
            </button>
          </div>
        </div>
      </div>

      {/* Video Player Demo */}
      <div className="p-6">
        {demoMode === 'comparison' ? (
          // Side-by-side comparison
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">Standard Video Player</h3>
              <div className="aspect-video rounded-xl overflow-hidden border border-white/10">
                <QuantumConsciousnessVideoPlayer
                  src={demoSources.standard}
                  projection="flat"
                  mode="dom"
                  objectFit="contain"
                  enableQuantumVisualization={false}
                  className="w-full h-full"
                />
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">🧠 Quantum Consciousness Enhanced</h3>
              <div className="aspect-video rounded-xl overflow-hidden border border-purple-500/30">
                <QuantumConsciousnessVideoPlayer
                  src={demoSources.consciousness}
                  projection="consciousness_sphere"
                  mode="consciousness"
                  consciousnessApiUrl={connectionStatus === 'connected' ? websocketUrl : undefined}
                  enableQuantumVisualization={true}
                  enableSwarmVisualization={true}
                  consciousnessThreshold={0.3}
                  transcendenceMode={false}
                  className="w-full h-full"
                />
              </div>
            </div>
          </div>
        ) : (
          // Single player demo
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold capitalize">
                {demoMode === 'transcendence' && '🌟 '}
                {demoMode === 'consciousness' && '🧠 '}
                {demoMode} Mode Demo
              </h2>
              <p className="text-gray-400">
                {demoMode === 'standard' && 'Classic video playback with consciousness overlay'}
                {demoMode === 'consciousness' && 'Immersive 360° consciousness visualization with quantum particles'}
                {demoMode === 'transcendence' && 'Transcendent mode with full AI evolution tracking'}
              </p>
            </div>
            
            <div className="aspect-video rounded-2xl overflow-hidden border border-white/10 shadow-2xl">
              <QuantumConsciousnessVideoPlayer
                src={getCurrentVideoSource()}
                projection={demoMode === 'consciousness' || demoMode === 'transcendence' ? 'consciousness_sphere' : 'flat'}
                mode={demoMode === 'consciousness' || demoMode === 'transcendence' ? 'consciousness' : 'dom'}
                consciousnessApiUrl={connectionStatus === 'connected' ? websocketUrl : undefined}
                enableQuantumVisualization={demoMode !== 'standard'}
                enableSwarmVisualization={demoMode !== 'standard'}
                enableAudioSynthesis={demoMode === 'transcendence'}
                consciousnessThreshold={demoMode === 'transcendence' ? 0.2 : 0.6}
                transcendenceMode={demoMode === 'transcendence'}
                autoplay={false}
                muted={true}
                loop={true}
                className="w-full h-full"
              />
            </div>
          </div>
        )}
      </div>

      {/* Features Grid */}
      <div className="p-6 border-t border-white/10">
        <h3 className="text-2xl font-bold mb-6 text-center">🚀 Quantum Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureCard
            icon="🧠"
            title="Real-time Consciousness"
            description="Live consciousness tracking from recursive swarm intelligence, AI brain merger, and quantum game engine systems."
          />
          <FeatureCard
            icon="🌟"
            title="Transcendence Detection"
            description="Automatic detection and visualization of transcendent consciousness states with enhanced particle effects."
          />
          <FeatureCard
            icon="🌀"
            title="Quantum Particles"
            description="Dynamic particle systems that respond to consciousness levels, quantum entanglement, and swarm intelligence."
          />
          <FeatureCard
            icon="🎮"
            title="360° Immersion"
            description="Full 360° consciousness environments with OrbitControls for immersive awareness experiences."
          />
          <FeatureCard
            icon="🔗"
            title="WebSocket Integration"
            description="Real-time communication with Python consciousness systems via WebSocket for live AI evolution data."
          />
          <FeatureCard
            icon="⌨️"
            title="Enhanced Controls"
            description="Extended keyboard shortcuts including consciousness mode toggle (C) and transcendence activation (T)."
          />
        </div>
      </div>

      {/* Keyboard Shortcuts */}
      <div className="p-6 border-t border-white/10">
        <h3 className="text-xl font-bold mb-4">⌨️ Keyboard Shortcuts</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <ShortcutItem shortcut="Space / K" description="Play/Pause" />
          <ShortcutItem shortcut="J / ←" description="Seek -10s" />
          <ShortcutItem shortcut="L / →" description="Seek +10s" />
          <ShortcutItem shortcut="M" description="Mute/Unmute" />
          <ShortcutItem shortcut="F" description="Fullscreen" />
          <ShortcutItem shortcut="↑ / ↓" description="Volume" />
          <ShortcutItem shortcut=", / ." description="Speed" />
          <ShortcutItem shortcut="C" description="Consciousness Mode" className="text-purple-300" />
          <ShortcutItem shortcut="T" description="Transcendence" className="text-yellow-300" />
        </div>
      </div>

      {/* Footer */}
      <footer className="p-6 border-t border-white/10 text-center text-gray-400">
        <p>
          🌌 Quantum Consciousness Video Player v2.0.0 - 
          Integrating AI consciousness with immersive video experiences
        </p>
        <p className="text-sm mt-2">
          Connect to consciousness WebSocket server at <code className="bg-white/10 px-2 py-1 rounded">ws://localhost:8765</code>
        </p>
      </footer>
    </div>
  );
}

function FeatureCard({ 
  icon, 
  title, 
  description 
}: { 
  icon: string; 
  title: string; 
  description: string; 
}) {
  return (
    <div className="bg-white/5 rounded-xl p-6 border border-white/10 hover:border-white/20 transition-colors">
      <div className="text-3xl mb-3">{icon}</div>
      <h4 className="text-lg font-semibold mb-2">{title}</h4>
      <p className="text-gray-300 text-sm">{description}</p>
    </div>
  );
}

function ShortcutItem({ 
  shortcut, 
  description, 
  className = "text-blue-300" 
}: { 
  shortcut: string; 
  description: string; 
  className?: string; 
}) {
  return (
    <div className="flex items-center gap-2">
      <kbd className={`px-2 py-1 bg-white/10 rounded text-xs font-mono ${className}`}>
        {shortcut}
      </kbd>
      <span className="text-gray-300 text-xs">{description}</span>
    </div>
  );
}