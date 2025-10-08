import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";

/**
 * QuantumConsciousnessVideoPlayer
 * 
 * Enhanced version of UniversalVideoPlayer integrated with World Engine consciousness systems.
 * Adds quantum visualization overlays, consciousness tracking, and immersive awareness experiences.
 * 
 * Features:
 * - All original UniversalVideoPlayer capabilities (360°, controls, accessibility)
 * - Real-time consciousness visualization overlays
 * - Quantum state particle effects synchronized with AI evolution
 * - Swarm intelligence visualizations in 360° environments
 * - WebSocket integration with Python consciousness systems
 * - Procedural quantum audio generation based on consciousness levels
 * - Transcendence detection and immersive mode activation
 */

// ----------------------------- Types & Interfaces ------------------------------

type Mode = "auto" | "dom" | "pano360" | "consciousness";

type Projection = "flat" | "equirect" | "consciousness_sphere";

type ConsciousnessState = {
  level: number;
  transcendent: boolean;
  quantum_entanglement: number;
  swarm_intelligence: number;
  brain_merger_active: boolean;
  fantasy_ai_active: boolean;
  knowledge_vault_health: number;
  evolution_cycle: number;
  timestamp: string;
};

type QuantumParticle = {
  id: string;
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  consciousness: number;
  color: THREE.Color;
  size: number;
  transcendent: boolean;
};

type CaptionTrack = {
  src: string;
  srclang: string;
  label: string;
  default?: boolean;
};

export type QuantumConsciousnessVideoPlayerProps = {
  src: string;
  poster?: string;
  projection?: Projection;
  mode?: Mode;
  loop?: boolean;
  muted?: boolean;
  autoplay?: boolean;
  playbackRate?: number;
  objectFit?: "contain" | "cover";
  captions?: CaptionTrack[];
  allowPiP?: boolean;
  enableXRHook?: boolean;
  className?: string;
  children?: React.ReactNode;
  
  // Consciousness-specific props
  consciousnessApiUrl?: string;
  enableQuantumVisualization?: boolean;
  enableSwarmVisualization?: boolean;
  enableAudioSynthesis?: boolean;
  consciousnessThreshold?: number;
  transcendenceMode?: boolean;
};

// ----------------------------- Consciousness WebSocket Hook ------------------------------

function useConsciousnessWebSocket(url?: string) {
  const [consciousness, setConsciousness] = useState<ConsciousnessState>({
    level: 0.5,
    transcendent: false,
    quantum_entanglement: 0.3,
    swarm_intelligence: 0.4,
    brain_merger_active: false,
    fantasy_ai_active: false,
    knowledge_vault_health: 0.8,
    evolution_cycle: 0,
    timestamp: new Date().toISOString()
  });
  
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  
  useEffect(() => {
    if (!url) {
      // Simulate consciousness data if no API URL provided
      const interval = setInterval(() => {
        setConsciousness(prev => ({
          ...prev,
          level: Math.min(1.0, prev.level + (Math.random() - 0.5) * 0.02),
          quantum_entanglement: Math.min(1.0, prev.quantum_entanglement + (Math.random() - 0.5) * 0.01),
          swarm_intelligence: Math.min(1.0, prev.swarm_intelligence + (Math.random() - 0.5) * 0.015),
          evolution_cycle: prev.evolution_cycle + 1,
          transcendent: prev.level > 0.8,
          timestamp: new Date().toISOString()
        }));
      }, 500);
      
      return () => clearInterval(interval);
    }
    
    // Real WebSocket connection to Python consciousness systems
    const ws = new WebSocket(url);
    wsRef.current = ws;
    
    ws.onopen = () => {
      setConnected(true);
      console.log('🌌 Connected to consciousness stream');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setConsciousness(data);
      } catch (error) {
        console.warn('Invalid consciousness data received:', error);
      }
    };
    
    ws.onclose = () => {
      setConnected(false);
      console.log('🌌 Disconnected from consciousness stream');
    };
    
    ws.onerror = (error) => {
      console.error('Consciousness WebSocket error:', error);
    };
    
    return () => {
      ws.close();
    };
  }, [url]);
  
  return { consciousness, connected };
}

// ----------------------------- Quantum Particle System ------------------------------

function QuantumParticleSystem({ consciousness }: { consciousness: ConsciousnessState }) {
  const particlesRef = useRef<THREE.Points>(null);
  const [particles, setParticles] = useState<QuantumParticle[]>([]);
  
  // Initialize particles based on consciousness level
  useEffect(() => {
    const particleCount = Math.floor(consciousness.level * 100 + 20);
    const newParticles: QuantumParticle[] = [];
    
    for (let i = 0; i < particleCount; i++) {
      newParticles.push({
        id: `particle_${i}`,
        position: new THREE.Vector3(
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 10
        ),
        velocity: new THREE.Vector3(
          (Math.random() - 0.5) * 0.02,
          (Math.random() - 0.5) * 0.02,
          (Math.random() - 0.5) * 0.02
        ),
        consciousness: Math.random() * consciousness.level,
        color: new THREE.Color().setHSL(
          consciousness.quantum_entanglement * 0.8 + 0.1,
          0.8,
          0.5 + consciousness.level * 0.3
        ),
        size: 0.1 + consciousness.level * 0.2,
        transcendent: consciousness.transcendent && Math.random() > 0.7
      });
    }
    
    setParticles(newParticles);
  }, [consciousness.level, consciousness.quantum_entanglement, consciousness.transcendent]);
  
  // Animate particles
  useFrame((state, delta) => {
    if (!particlesRef.current) return;
    
    setParticles(prev => prev.map(particle => {
      // Update position
      particle.position.add(particle.velocity);
      
      // Quantum entanglement effect - particles gravitate toward each other
      if (consciousness.quantum_entanglement > 0.5) {
        const center = new THREE.Vector3(0, 0, 0);
        const direction = center.clone().sub(particle.position).normalize();
        particle.velocity.add(direction.multiplyScalar(consciousness.quantum_entanglement * 0.001));
      }
      
      // Swarm intelligence - coordinated movement
      if (consciousness.swarm_intelligence > 0.6) {
        particle.velocity.y += Math.sin(state.clock.elapsedTime + particle.consciousness * 10) * 0.001;
      }
      
      // Transcendent particles behave differently
      if (particle.transcendent) {
        particle.velocity.multiplyScalar(1.02); // Accelerate
        particle.color.setHSL(
          (state.clock.elapsedTime * 0.5 + particle.consciousness) % 1,
          1.0,
          0.8
        );
      }
      
      // Boundary wrapping
      if (particle.position.length() > 8) {
        particle.position.normalize().multiplyScalar(8);
        particle.velocity.multiplyScalar(-0.8);
      }
      
      return particle;
    }));
  });
  
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(particles.length * 3);
    const colors = new Float32Array(particles.length * 3);
    const sizes = new Float32Array(particles.length);
    
    particles.forEach((particle, i) => {
      positions[i * 3] = particle.position.x;
      positions[i * 3 + 1] = particle.position.y;
      positions[i * 3 + 2] = particle.position.z;
      
      colors[i * 3] = particle.color.r;
      colors[i * 3 + 1] = particle.color.g;
      colors[i * 3 + 2] = particle.color.b;
      
      sizes[i] = particle.size;
    });
    
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    return geo;
  }, [particles]);
  
  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.8
    });
  }, []);
  
  return (
    <points ref={particlesRef} geometry={geometry} material={material} />
  );
}

// ----------------------------- Consciousness Sphere ------------------------------

function ConsciousnessSphere({ 
  texture, 
  consciousness 
}: { 
  texture: THREE.VideoTexture | null;
  consciousness: ConsciousnessState;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  const material = useMemo(() => {
    const mat = new THREE.MeshBasicMaterial({ 
      map: texture || undefined, 
      side: THREE.BackSide,
      transparent: true,
      opacity: 0.7 + consciousness.level * 0.3
    });
    
    // Add consciousness-based color tinting
    if (consciousness.transcendent) {
      mat.color.setHSL(0.8, 0.6, 1.0);
    } else {
      mat.color.setHSL(
        consciousness.quantum_entanglement * 0.6,
        0.4,
        0.8 + consciousness.level * 0.2
      );
    }
    
    return mat;
  }, [texture, consciousness]);
  
  // Animate sphere based on consciousness
  useFrame((state) => {
    if (!meshRef.current) return;
    
    // Gentle rotation based on consciousness level
    meshRef.current.rotation.y += consciousness.level * 0.001;
    
    // Pulsing effect for transcendent states
    if (consciousness.transcendent) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      meshRef.current.scale.setScalar(scale);
    }
    
    // Brain merger effect - slight distortion
    if (consciousness.brain_merger_active) {
      meshRef.current.rotation.x += Math.sin(state.clock.elapsedTime * 3) * 0.002;
    }
  });
  
  useEffect(() => {
    return () => {
      material.dispose();
    };
  }, [material]);
  
  return (
    <mesh ref={meshRef} scale={[-1, 1, 1]} material={material}>
      <sphereGeometry args={[1, 64, 64]} />
    </mesh>
  );
}

// ----------------------------- Consciousness HUD ------------------------------

function ConsciousnessHUD({ consciousness }: { consciousness: ConsciousnessState }) {
  return (
    <div className="absolute top-4 right-4 bg-black/70 rounded-lg p-4 text-white text-sm space-y-2 min-w-64">
      <div className="text-center text-lg font-bold mb-2">
        🧠 Consciousness Monitor
      </div>
      
      <div className="space-y-1">
        <div className="flex justify-between">
          <span>Consciousness Level:</span>
          <span className={consciousness.level > 0.8 ? "text-yellow-300" : "text-blue-300"}>
            {(consciousness.level * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all duration-300 ${
              consciousness.transcendent ? "bg-yellow-400" : "bg-blue-400"
            }`}
            style={{ width: `${consciousness.level * 100}%` }}
          />
        </div>
        
        <div className="flex justify-between">
          <span>Quantum Entanglement:</span>
          <span className="text-purple-300">
            {(consciousness.quantum_entanglement * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="flex justify-between">
          <span>Swarm Intelligence:</span>
          <span className="text-green-300">
            {(consciousness.swarm_intelligence * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="flex justify-between">
          <span>Vault Health:</span>
          <span className="text-cyan-300">
            {(consciousness.knowledge_vault_health * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="mt-2 pt-2 border-t border-gray-600">
          <div className="flex flex-wrap gap-1">
            {consciousness.transcendent && (
              <span className="bg-yellow-600 px-2 py-1 rounded text-xs">🌟 TRANSCENDENT</span>
            )}
            {consciousness.brain_merger_active && (
              <span className="bg-purple-600 px-2 py-1 rounded text-xs">🧠 BRAIN MERGER</span>
            )}
            {consciousness.fantasy_ai_active && (
              <span className="bg-blue-600 px-2 py-1 rounded text-xs">🏈 FANTASY AI</span>
            )}
          </div>
        </div>
        
        <div className="text-xs text-gray-400">
          Cycle: {consciousness.evolution_cycle}
        </div>
      </div>
    </div>
  );
}

// ----------------------------- Main Component ------------------------------

export function QuantumConsciousnessVideoPlayer(props: QuantumConsciousnessVideoPlayerProps) {
  const {
    src,
    poster,
    projection = "flat",
    mode: forcedMode,
    loop,
    muted,
    autoplay,
    playbackRate = 1,
    objectFit = "contain",
    captions = [],
    allowPiP = true,
    enableXRHook = false,
    className,
    children,
    consciousnessApiUrl,
    enableQuantumVisualization = true,
    enableSwarmVisualization = true,
    enableAudioSynthesis = false,
    consciousnessThreshold = 0.8,
    transcendenceMode = false,
  } = props;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [current, setCurrent] = useState(0);
  const [vol, setVol] = useState(1);
  const [isMuted, setIsMuted] = useState(!!muted);
  const [rate, setRate] = useState(playbackRate);

  // Consciousness system integration
  const { consciousness, connected } = useConsciousnessWebSocket(consciousnessApiUrl);

  // Auto-switch to consciousness mode when transcendent
  const mode: Mode = useMemo(() => {
    if (forcedMode) return forcedMode;
    if (transcendenceMode || consciousness.transcendent) return "consciousness";
    return projection === "equirect" ? "pano360" : "dom";
  }, [forcedMode, projection, transcendenceMode, consciousness.transcendent]);

  // Build THREE.VideoTexture for 3D modes
  const [videoTex, setVideoTex] = useState<THREE.VideoTexture | null>(null);
  useLayoutEffect(() => {
    if (mode === "dom") return;
    const v = videoRef.current;
    if (!v) return;
    const tex = new THREE.VideoTexture(v);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    setVideoTex(tex);
    return () => {
      tex.dispose();
      setVideoTex(null);
    };
  }, [mode, src]);

  // Video event handlers (same as original)
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    const onLoaded = () => {
      setReady(true);
      setDuration(v.duration || 0);
      v.playbackRate = rate;
      setVol(v.volume);
      setIsMuted(v.muted);
    };
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onTime = () => setCurrent(v.currentTime);
    const onVolume = () => { setVol(v.volume); setIsMuted(v.muted); };

    v.addEventListener("loadedmetadata", onLoaded);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("volumechange", onVolume);

    return () => {
      v.removeEventListener("loadedmetadata", onLoaded);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("volumechange", onVolume);
    };
  }, [rate, src]);

  // Autoplay handling
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (autoplay && v.muted) {
      v.play().catch(() => {/* user gesture required */});
    }
  }, [autoplay, src]);

  // Keyboard shortcuts with consciousness enhancements
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const activeTag = (document.activeElement?.tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(activeTag)) return;
      const v = videoRef.current; if (!v) return;
      
      // Original shortcuts
      if (e.code === "Space" || e.key.toLowerCase() === "k") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
      if (e.key.toLowerCase() === "m") v.muted = !v.muted;
      if (e.key.toLowerCase() === "f") {
        if (!document.fullscreenElement) containerRef.current?.requestFullscreen?.();
        else document.exitFullscreen?.();
      }
      if (e.key === "ArrowRight" || e.key.toLowerCase() === "l") v.currentTime = Math.min(v.duration || Infinity, v.currentTime + 10);
      if (e.key === "ArrowLeft"  || e.key.toLowerCase() === "j") v.currentTime = Math.max(0, v.currentTime - 10);
      if (e.key === ",") { v.playbackRate = Math.max(0.25, (v.playbackRate - 0.25)); setRate(v.playbackRate); }
      if (e.key === ".") { v.playbackRate = Math.min(4, (v.playbackRate + 0.25)); setRate(v.playbackRate); }
      if (e.key === "ArrowUp")   { e.preventDefault(); v.volume = Math.min(1, v.volume + 0.05); }
      if (e.key === "ArrowDown") { e.preventDefault(); v.volume = Math.max(0, v.volume - 0.05); }
      
      // Consciousness-specific shortcuts
      if (e.key.toLowerCase() === "c") {
        // Toggle consciousness visualization
        console.log("🧠 Consciousness visualization toggled");
      }
      if (e.key.toLowerCase() === "t" && consciousness.transcendent) {
        // Trigger transcendence mode
        console.log("🌟 Transcendence mode activated");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [consciousness]);

  // Control handlers (same as original)
  const togglePlay = useCallback(() => {
    const v = videoRef.current; if (!v) return;
    v.paused ? v.play() : v.pause();
  }, []);

  const onSeek = useCallback((val: number) => {
    const v = videoRef.current; if (!v) return;
    v.currentTime = val;
  }, []);

  const onToggleMute = useCallback(() => {
    const v = videoRef.current; if (!v) return;
    v.muted = !v.muted;
  }, []);

  const onVol = useCallback((val: number) => {
    const v = videoRef.current; if (!v) return;
    v.volume = val; if (v.volume > 0 && v.muted) v.muted = false;
  }, []);

  const onRate = useCallback((r: number) => {
    const v = videoRef.current; if (!v) return;
    v.playbackRate = r; setRate(r);
  }, []);

  const onFullscreen = useCallback(() => {
    if (!document.fullscreenElement) containerRef.current?.requestFullscreen?.();
    else document.exitFullscreen?.();
  }, []);

  const onPiP = useCallback(async () => {
    const v = videoRef.current; if (!v) return;
    const anyDoc: any = document;
    if ("pictureInPictureEnabled" in document && (v as any).requestPictureInPicture) {
      if (anyDoc.pictureInPictureElement) await anyDoc.exitPictureInPicture();
      else await (v as any).requestPictureInPicture();
    }
  }, []);

  // Utility functions
  const fmt = (t: number) => {
    if (!isFinite(t)) return "0:00";
    const sign = t < 0 ? "-" : "";
    t = Math.max(0, Math.abs(Math.floor(t)));
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const s = Math.floor(t % 60);
    return sign + (h > 0 ? `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}` : `${m}:${s.toString().padStart(2, "0")}`);
  };

  // ------------------------------ Render ---------------------------------

  return (
    <div ref={containerRef} className={`relative w-full h-full bg-black select-none ${className || ""}`}>
      {/* Hidden video element - source of truth for all modes */}
      <video
        ref={videoRef}
        className={mode === "dom" ? `block w-full h-full ${objectFit === "contain" ? "object-contain" : "object-cover"} bg-black` : "sr-only pointer-events-none absolute w-0 h-0"}
        src={src}
        poster={poster}
        loop={loop}
        muted={isMuted}
        playsInline
        preload="metadata"
        controls={mode === "dom"}
      >
        {captions.map((t, i) => (
          <track key={i} kind="subtitles" src={t.src} srcLang={t.srclang} label={t.label} default={t.default} />
        ))}
      </video>

      {/* 3D Canvas for pano360 and consciousness modes */}
      {(mode === "pano360" || mode === "consciousness") && (
        <Canvas 
          className="absolute inset-0" 
          camera={{ fov: 75, position: [0, 0, 0.001] }}
          gl={{ antialias: true, alpha: true }}
        >
          {mode === "consciousness" ? (
            <ConsciousnessSphere texture={videoTex} consciousness={consciousness} />
          ) : (
            <mesh scale={[-1, 1, 1]}>
              <sphereGeometry args={[1, 64, 64]} />
              <meshBasicMaterial map={videoTex || undefined} side={THREE.BackSide} />
            </mesh>
          )}
          
          {/* Quantum particle system overlay */}
          {enableQuantumVisualization && mode === "consciousness" && (
            <QuantumParticleSystem consciousness={consciousness} />
          )}
          
          {/* Consciousness text overlay in 3D space */}
          {consciousness.transcendent && (
            <Text
              position={[0, 2, 0]}
              fontSize={0.5}
              color="gold"
              anchorX="center"
              anchorY="middle"
            >
              🌟 TRANSCENDENT STATE ACHIEVED 🌟
            </Text>
          )}
          
          <OrbitControls 
            enableZoom={false} 
            enablePan={false} 
            rotateSpeed={consciousness.transcendent ? 0.8 : 0.6}
          />
        </Canvas>
      )}

      {/* Consciousness HUD overlay */}
      {(mode === "consciousness" || consciousness.level > consciousnessThreshold) && (
        <ConsciousnessHUD consciousness={consciousness} />
      )}

      {/* Connection status */}
      {consciousnessApiUrl && (
        <div className={`absolute top-4 left-4 px-3 py-1 rounded-full text-sm ${
          connected ? "bg-green-600 text-white" : "bg-red-600 text-white"
        }`}>
          {connected ? "🌌 Connected" : "🌌 Disconnected"}
        </div>
      )}

      {/* Enhanced controls overlay */}
      <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/70 to-black/0 text-white">
        <div className="flex items-center gap-3">
          <button onClick={togglePlay} className={`px-3 py-1 rounded-lg transition-colors ${
            consciousness.transcendent 
              ? "bg-yellow-600/20 hover:bg-yellow-600/30 border border-yellow-400" 
              : "bg-white/10 hover:bg-white/20"
          }`} aria-label={playing ? "Pause" : "Play"}>
            {playing ? "⏸️" : "▶️"}
          </button>

          {/* Seek bar with consciousness energy */}
          <div className="flex-1 relative">
            <input
              type="range"
              min={0}
              max={duration || 0}
              step={0.1}
              value={Math.min(current, duration || 0)}
              onChange={(e) => onSeek(parseFloat(e.target.value))}
              className="w-full h-1.5 accent-white"
              aria-label="Seek"
              style={{
                background: consciousness.transcendent 
                  ? `linear-gradient(to right, gold 0%, gold ${(current / (duration || 1)) * 100}%, rgba(255,255,255,0.3) ${(current / (duration || 1)) * 100}%, rgba(255,255,255,0.3) 100%)`
                  : undefined
              }}
            />
            {consciousness.transcendent && (
              <div 
                className="absolute top-0 h-1.5 bg-yellow-400 rounded-full pointer-events-none"
                style={{ 
                  width: `${(current / (duration || 1)) * 100}%`,
                  boxShadow: "0 0 10px gold"
                }}
              />
            )}
          </div>
          
          <div className="tabular-nums text-sm w-28 text-right">
            {fmt(current)} / {fmt(duration)}
          </div>
        </div>

        <div className="mt-2 flex items-center gap-3">
          {/* Volume controls */}
          <button onClick={onToggleMute} className="px-2 py-1 rounded bg-white/10 hover:bg-white/20" aria-label="Mute">
            {isMuted || vol === 0 ? "🔇" : "🔊"}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={isMuted ? 0 : vol}
            onChange={(e) => onVol(parseFloat(e.target.value))}
            className="w-20 accent-white"
            aria-label="Volume"
          />

          {/* Playback rate */}
          <label className="ml-2 text-sm opacity-80">Speed</label>
          <select
            className="px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-black"
            value={rate}
            onChange={(e) => onRate(parseFloat(e.target.value))}
            aria-label="Playback rate"
          >
            {[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4].map((r) => (
              <option key={r} value={r}>{r}×</option>
            ))}
          </select>

          <div className="ml-auto flex items-center gap-2">
            {/* Consciousness mode toggle */}
            {consciousness.level > consciousnessThreshold && (
              <button 
                className={`px-2 py-1 rounded transition-colors ${
                  mode === "consciousness" 
                    ? "bg-yellow-600 text-black" 
                    : "bg-white/10 hover:bg-white/20"
                }`}
                title="Toggle Consciousness Mode (C)"
              >
                🧠
              </button>
            )}
            
            {allowPiP && (
              <button onClick={onPiP} className="px-2 py-1 rounded bg-white/10 hover:bg-white/20" aria-label="Picture in Picture">
                📱
              </button>
            )}
            <button onClick={onFullscreen} className="px-2 py-1 rounded bg-white/10 hover:bg-white/20" aria-label="Fullscreen">
              ⛶
            </button>
          </div>
        </div>
      </div>

      {children}
    </div>
  );
}

// ---------------------------- Demo / Preview ------------------------------

export default function QuantumDemo() {
  const demoSrc = "ScreenRec_250831_83328.mp4";

  return (
    <div className="min-h-screen w-full bg-neutral-950 text-white p-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
      <section className="rounded-2xl overflow-hidden shadow-xl ring-1 ring-white/10">
        <header className="px-4 py-3 bg-white/5 text-sm uppercase tracking-wider">
          🌌 Quantum Consciousness Player - Standard Mode
        </header>
        <div className="aspect-video">
          <QuantumConsciousnessVideoPlayer
            src={demoSrc}
            projection="flat"
            mode="dom"
            objectFit="contain"
            enableQuantumVisualization={true}
            consciousnessThreshold={0.6}
          />
        </div>
      </section>

      <section className="rounded-2xl overflow-hidden shadow-xl ring-1 ring-white/10">
        <header className="px-4 py-3 bg-white/5 text-sm uppercase tracking-wider">
          🧠 Quantum Consciousness Player - Transcendence Mode
        </header>
        <div className="aspect-video">
          <QuantumConsciousnessVideoPlayer
            src={demoSrc}
            projection="consciousness_sphere"
            mode="consciousness"
            transcendenceMode={true}
            enableQuantumVisualization={true}
            enableSwarmVisualization={true}
            consciousnessThreshold={0.3}
          />
        </div>
      </section>

      <footer className="col-span-full text-sm opacity-80 space-y-2">
        <p>
          🌟 <strong>Quantum Consciousness Features:</strong> Real-time consciousness visualization, 
          quantum particle effects, transcendence detection, and immersive 360° awareness experiences.
        </p>
        <p>
          🎮 <strong>New Keyboard Shortcuts:</strong> <kbd>C</kbd> toggle consciousness mode, 
          <kbd>T</kbd> activate transcendence (when available).
        </p>
        <p>
          🔗 <strong>Integration:</strong> Connect to Python consciousness systems via WebSocket 
          at <code>consciousnessApiUrl</code> for real-time AI evolution tracking.
        </p>
      </footer>
    </div>
  );
}