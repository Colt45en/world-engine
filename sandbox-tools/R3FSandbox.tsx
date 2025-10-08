import React, { Suspense, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, Stats, Html, Text } from '@react-three/drei';
import { Leva, useControls } from 'leva';
import * as THREE from 'three';

// Import all 13 sandbox tools
import SonicFX from './sonic-fx';
import MaskMesher from './mask-mesher';
import RBFSolver from './rbf-solver';
import Tesseract from './tesseract';
import SwarmCodex from './swarm-codex';
import GlyphConstellation from './glyph-constellation';
import NexusRuntime from './nexus-runtime';
import AvatarImpostor from './avatar-impostor';
import ContourExtrude from './contour-extrude';
import HyperBowl from './hyperbowl';
import MushroomCodex from './mushroom-codex';
import RaymarchingCinema from './raymarching-cinema';
import EasingStudio from './easing-studio';

// =============================================================================
// Shared R3F Sandbox Configuration
// =============================================================================

interface SandboxTool {
  id: string;
  name: string;
  component: React.ComponentType<any>;
  description: string;
  category: string;
  complexity: 1 | 2 | 3 | 4 | 5;
  requiresAudio?: boolean;
  requiresWebcam?: boolean;
  mathematical?: boolean;
  interactive?: boolean;
}

export const SANDBOX_TOOLS: Record<string, SandboxTool> = {
  'sonic-fx': {
    id: 'sonic-fx',
    name: 'Sonic FX',
    component: SonicFX,
    description: 'Cinematic overlay system with blur effects and temporal distortion',
    category: 'Visual Effects',
    complexity: 2,
    requiresAudio: true,
    interactive: true
  },
  'mask-mesher': {
    id: 'mask-mesher',
    name: 'Mask Mesher',
    component: MaskMesher,
    description: 'Silhouette-to-volume conversion using marching cubes',
    category: 'Geometry Processing',
    complexity: 4,
    requiresWebcam: true,
    mathematical: true,
    interactive: true
  },
  'rbf-solver': {
    id: 'rbf-solver',
    name: 'RBF Solver',
    component: RBFSolver,
    description: 'Radial basis function interpolation with real-time solving',
    category: 'Mathematics',
    complexity: 5,
    mathematical: true,
    interactive: true
  },
  'tesseract': {
    id: 'tesseract',
    name: 'Tesseract',
    component: Tesseract,
    description: '4D hypercube visualization with dimensional projection',
    category: 'Mathematics',
    complexity: 5,
    mathematical: true,
    interactive: true
  },
  'swarm-codex': {
    id: 'swarm-codex',
    name: 'Swarm Codex',
    component: SwarmCodex,
    description: 'AI behavior simulation with emergent patterns',
    category: 'AI Simulation',
    complexity: 4,
    interactive: true
  },
  'glyph-constellation': {
    id: 'glyph-constellation',
    name: 'Glyph Constellation',
    component: GlyphConstellation,
    description: 'Typography analysis with 3D character mapping',
    category: 'Data Visualization',
    complexity: 3,
    interactive: true
  },
  'nexus-runtime': {
    id: 'nexus-runtime',
    name: 'Nexus Runtime',
    component: NexusRuntime,
    description: 'System monitoring dashboard with 3D data flow',
    category: 'Data Visualization',
    complexity: 4,
    interactive: true
  },
  'avatar-impostor': {
    id: 'avatar-impostor',
    name: 'Avatar Impostor',
    component: AvatarImpostor,
    description: 'Multi-view avatar rendering with depth-aware fusion',
    category: 'Rendering',
    complexity: 4,
    requiresWebcam: true,
    mathematical: true,
    interactive: true
  },
  'contour-extrude': {
    id: 'contour-extrude',
    name: 'Contour Extrude',
    component: ContourExtrude,
    description: 'Image-to-3D contour extrusion with edge detection',
    category: 'Image Processing',
    complexity: 3,
    requiresWebcam: true,
    mathematical: true,
    interactive: true
  },
  'hyperbowl': {
    id: 'hyperbowl',
    name: 'HyperBowl',
    component: HyperBowl,
    description: 'Parametric surface visualization with complex mathematics',
    category: 'Mathematics',
    complexity: 5,
    mathematical: true,
    interactive: true
  },
  'mushroom-codex': {
    id: 'mushroom-codex',
    name: 'Mushroom Codex',
    component: MushroomCodex,
    description: 'Parametric mushroom generation with toon shading',
    category: 'Procedural Generation',
    complexity: 3,
    interactive: true
  },
  'raymarching-cinema': {
    id: 'raymarching-cinema',
    name: 'Raymarching Cinema',
    component: RaymarchingCinema,
    description: 'Real-time cinematic visual generation with advanced raymarching',
    category: 'Visual Effects',
    complexity: 5,
    requiresAudio: true,
    mathematical: true,
    interactive: true
  },
  'easing-studio': {
    id: 'easing-studio',
    name: 'Easing Studio',
    component: EasingStudio,
    description: 'Advanced 3D animation curve editor with mathematical precision',
    category: 'Animation Tools',
    complexity: 5,
    mathematical: true,
    interactive: true
  },
  'nexus-forge': {
    id: 'nexus-forge',
    name: 'Nexus Forge v2.1',
    component: React.lazy(() => import('./nexus-forge/NexusForge')),
    description: 'Advanced visual reality constructor with media integration and thematic environments',
    category: 'Reality Construction',
    complexity: 5,
    requiresWebcam: true,
    mathematical: true,
    interactive: true
  }
};

// =============================================================================
// Sandbox Environment Components
// =============================================================================

interface LoadingFallbackProps {
  toolName?: string;
}

const LoadingFallback: React.FC<LoadingFallbackProps> = ({ toolName }) => (
  <Html center>
    <div style={{
      padding: '20px 40px',
      background: 'rgba(0, 0, 0, 0.8)',
      borderRadius: '12px',
      border: '1px solid #00ff88',
      color: '#00ff88',
      fontFamily: 'monospace',
      fontSize: '16px',
      textAlign: 'center',
      backdropFilter: 'blur(10px)'
    }}>
      <div style={{ marginBottom: '10px', fontSize: '18px', fontWeight: 'bold' }}>
        {toolName ? `Loading ${toolName}...` : 'Initializing Sandbox...'}
      </div>
      <div style={{ opacity: 0.7 }}>
        Preparing R3F Environment
      </div>
    </div>
  </Html>
);

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class SandboxErrorBoundary extends React.Component<
  React.PropsWithChildren<{ toolName?: string }>,
  ErrorBoundaryState
> {
  constructor(props: React.PropsWithChildren<{ toolName?: string }>) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <Html center>
          <div style={{
            padding: '30px',
            background: 'rgba(255, 0, 0, 0.1)',
            borderRadius: '12px',
            border: '1px solid #ff4444',
            color: '#ff4444',
            fontFamily: 'monospace',
            fontSize: '14px',
            textAlign: 'center',
            maxWidth: '400px'
          }}>
            <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '15px' }}>
              ⚠ Tool Error
            </div>
            <div style={{ marginBottom: '10px' }}>
              {this.props.toolName || 'Sandbox tool'} encountered an error
            </div>
            <div style={{ opacity: 0.7, fontSize: '12px' }}>
              {this.state.error?.message}
            </div>
          </div>
        </Html>
      );
    }

    return this.props.children;
  }
}

// =============================================================================
// Main Sandbox Interface
// =============================================================================

interface SandboxProps {
  /** Tool ID to render, or null for tool selection */
  selectedTool?: string | null;
  /** Show FPS stats */
  showStats?: boolean;
  /** Show Leva controls */
  showControls?: boolean;
  /** Custom camera settings */
  cameraSettings?: {
    position?: [number, number, number];
    fov?: number;
  };
  /** Canvas configuration */
  canvasConfig?: {
    shadows?: boolean;
    antialias?: boolean;
    alpha?: boolean;
  };
  /** Environment settings */
  environment?: 'studio' | 'city' | 'forest' | 'sunset' | 'dawn' | 'night' | null;
  /** Callback when tool changes */
  onToolChange?: (toolId: string | null) => void;
}

const SandboxCanvas: React.FC<SandboxProps & { children?: React.ReactNode }> = ({
  selectedTool,
  showStats = false,
  showControls = true,
  cameraSettings = { position: [0, 0, 5], fov: 75 },
  canvasConfig = { shadows: true, antialias: true, alpha: false },
  environment = 'studio',
  onToolChange,
  children
}) => {
  const globalControls = useControls('Sandbox', {
    resetCamera: { value: false, label: 'Reset Camera' },
    showGrid: { value: false, label: 'Show Grid' },
    enableFog: { value: false, label: 'Enable Fog' }
  });

  // Handle camera reset
  useEffect(() => {
    if (globalControls.resetCamera) {
      // Reset handled by OrbitControls ref if needed
    }
  }, [globalControls.resetCamera]);

  return (
    <>
      <Canvas
        camera={{
          position: cameraSettings.position,
          fov: cameraSettings.fov
        }}
        shadows={canvasConfig.shadows}
        gl={{ antialias: canvasConfig.antialias, alpha: canvasConfig.alpha }}
        style={{ background: canvasConfig.alpha ? 'transparent' : '#000' }}
      >
        <Suspense fallback={<LoadingFallback />}>
          {/* Environment */}
          {environment && <Environment preset={environment} />}

          {/* Lighting */}
          <ambientLight intensity={0.2} />
          <directionalLight
            position={[10, 10, 5]}
            intensity={1}
            castShadow={canvasConfig.shadows}
            shadow-mapSize={[2048, 2048]}
          />
          <pointLight position={[10, 10, 10]} intensity={0.5} />

          {/* Global grid */}
          {globalControls.showGrid && (
            <gridHelper args={[20, 20, '#444', '#222']} />
          )}

          {/* Fog */}
          {globalControls.enableFog && (
            <fog attach="fog" args={['#000', 5, 50]} />
          )}

          {/* Controls */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            dampingFactor={0.05}
            enableDamping
          />

          {/* Selected tool or children */}
          {selectedTool && SANDBOX_TOOLS[selectedTool] ? (
            <SandboxErrorBoundary toolName={SANDBOX_TOOLS[selectedTool].name}>
              <SANDBOX_TOOLS[selectedTool].component />
            </SandboxErrorBoundary>
          ) : children}

          {/* Stats */}
          {showStats && <Stats />}
        </Suspense>
      </Canvas>

      {/* Leva controls */}
      {showControls && <Leva collapsed={false} />}
    </>
  );
};

// =============================================================================
// Tool Selector Component
// =============================================================================

interface ToolSelectorProps {
  onSelectTool: (toolId: string) => void;
  currentTool?: string | null;
}

const ToolSelector: React.FC<ToolSelectorProps> = ({ onSelectTool, currentTool }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  const categories = ['All', ...Array.from(new Set(Object.values(SANDBOX_TOOLS).map(tool => tool.category)))];

  const filteredTools = Object.values(SANDBOX_TOOLS).filter(tool => {
    const matchesSearch = tool.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tool.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || tool.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      left: '20px',
      width: '350px',
      background: 'rgba(0, 0, 0, 0.9)',
      borderRadius: '12px',
      border: '1px solid #00ff88',
      padding: '20px',
      fontFamily: 'monospace',
      fontSize: '14px',
      color: '#00ff88',
      zIndex: 1000,
      backdropFilter: 'blur(10px)'
    }}>
      <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', textAlign: 'center' }}>
        R3F Sandbox Tools ({Object.keys(SANDBOX_TOOLS).length})
      </h3>

      {/* Search */}
      <input
        type="text"
        placeholder="Search tools..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        style={{
          width: '100%',
          padding: '8px 12px',
          background: 'rgba(0, 255, 136, 0.1)',
          border: '1px solid #00ff88',
          borderRadius: '6px',
          color: '#00ff88',
          fontSize: '13px',
          marginBottom: '15px',
          fontFamily: 'inherit'
        }}
      />

      {/* Category Filter */}
      <select
        value={selectedCategory}
        onChange={(e) => setSelectedCategory(e.target.value)}
        style={{
          width: '100%',
          padding: '8px 12px',
          background: 'rgba(0, 255, 136, 0.1)',
          border: '1px solid #00ff88',
          borderRadius: '6px',
          color: '#00ff88',
          fontSize: '13px',
          marginBottom: '15px',
          fontFamily: 'inherit'
        }}
      >
        {categories.map(category => (
          <option key={category} value={category} style={{ background: '#000' }}>
            {category}
          </option>
        ))}
      </select>

      {/* Tool Grid */}
      <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
        {filteredTools.map(tool => (
          <div
            key={tool.id}
            onClick={() => onSelectTool(tool.id)}
            style={{
              padding: '12px',
              marginBottom: '8px',
              background: currentTool === tool.id ? 'rgba(0, 255, 136, 0.2)' : 'rgba(0, 255, 136, 0.05)',
              border: currentTool === tool.id ? '1px solid #00ff88' : '1px solid rgba(0, 255, 136, 0.3)',
              borderRadius: '8px',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              if (currentTool !== tool.id) {
                e.currentTarget.style.background = 'rgba(0, 255, 136, 0.1)';
              }
            }}
            onMouseLeave={(e) => {
              if (currentTool !== tool.id) {
                e.currentTarget.style.background = 'rgba(0, 255, 136, 0.05)';
              }
            }}
          >
            <div style={{ fontWeight: 'bold', marginBottom: '4px', fontSize: '13px' }}>
              {tool.name} {'★'.repeat(tool.complexity)}
            </div>
            <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '6px' }}>
              {tool.category}
              {tool.requiresAudio && ' • Audio'}
              {tool.requiresWebcam && ' • Camera'}
              {tool.mathematical && ' • Math'}
            </div>
            <div style={{ fontSize: '11px', opacity: 0.6, lineHeight: '1.3' }}>
              {tool.description}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// =============================================================================
// Main Sandbox Export
// =============================================================================

interface R3FSandboxProps extends SandboxProps {
  /** Show tool selector panel */
  showToolSelector?: boolean;
  /** Custom style for the container */
  style?: React.CSSProperties;
}

export const R3FSandbox: React.FC<R3FSandboxProps> = ({
  selectedTool,
  showToolSelector = true,
  onToolChange,
  style = { width: '100vw', height: '100vh' },
  ...canvasProps
}) => {
  const [currentTool, setCurrentTool] = useState<string | null>(selectedTool || null);

  const handleToolChange = (toolId: string | null) => {
    setCurrentTool(toolId);
    onToolChange?.(toolId);
  };

  return (
    <div style={style}>
      <SandboxCanvas selectedTool={currentTool} onToolChange={handleToolChange} {...canvasProps} />

      {showToolSelector && (
        <ToolSelector
          onSelectTool={handleToolChange}
          currentTool={currentTool}
        />
      )}

      {currentTool && (
        <button
          onClick={() => handleToolChange(null)}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'rgba(255, 0, 0, 0.8)',
            border: '1px solid #ff4444',
            color: 'white',
            padding: '10px 20px',
            borderRadius: '8px',
            cursor: 'pointer',
            fontFamily: 'monospace',
            fontSize: '13px',
            fontWeight: 'bold',
            zIndex: 1000
          }}
        >
          Close Tool
        </button>
      )}
    </div>
  );
};

// =============================================================================
// Utility Functions and Exports
// =============================================================================

export { SANDBOX_TOOLS };
export type { SandboxTool, SandboxProps };

// Helper function to get tools by category
export const getToolsByCategory = (category: string) => {
  if (category === 'All') return Object.values(SANDBOX_TOOLS);
  return Object.values(SANDBOX_TOOLS).filter(tool => tool.category === category);
};

// Helper function to get tool requirements
export const getToolRequirements = (toolId: string) => {
  const tool = SANDBOX_TOOLS[toolId];
  if (!tool) return null;

  return {
    audio: tool.requiresAudio || false,
    webcam: tool.requiresWebcam || false,
    mathematical: tool.mathematical || false,
    interactive: tool.interactive || false,
    complexity: tool.complexity
  };
};

// Helper function for permission requests
export const requestToolPermissions = async (toolId: string) => {
  const requirements = getToolRequirements(toolId);
  if (!requirements) return false;

  try {
    if (requirements.audio) {
      await navigator.mediaDevices.getUserMedia({ audio: true });
    }
    if (requirements.webcam) {
      await navigator.mediaDevices.getUserMedia({ video: true });
    }
    return true;
  } catch (error) {
    console.warn(`Permission denied for ${toolId}:`, error);
    return false;
  }
};

export default R3FSandbox;
