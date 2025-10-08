import React, { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

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
// Tool Registry
// =============================================================================

interface ToolInfo {
  name: string;
  component: React.ComponentType;
  description: string;
  complexity: number;
  category: string;
  mathematical: boolean;
  interactive: boolean;
}

const TOOL_REGISTRY: Record<string, ToolInfo> = {
  'sonic-fx': {
    name: 'Sonic FX',
    component: SonicFX,
    description: 'Cinematic overlay system with blur effects and temporal distortion',
    complexity: 2,
    category: 'Visual Effects',
    mathematical: false,
    interactive: true
  },
  'mask-mesher': {
    name: 'Mask Mesher',
    component: MaskMesher,
    description: 'Silhouette-to-volume conversion using marching cubes',
    complexity: 4,
    category: 'Geometry Processing',
    mathematical: true,
    interactive: true
  },
  'rbf-solver': {
    name: 'RBF Solver',
    component: RBFSolver,
    description: 'Mathematical PDE solver using radial basis functions',
    complexity: 5,
    category: 'Mathematics',
    mathematical: true,
    interactive: true
  },
  'tesseract': {
    name: 'Tesseract',
    component: Tesseract,
    description: '4D hypercube visualization with projection controls',
    complexity: 4,
    category: 'Mathematics',
    mathematical: true,
    interactive: true
  },
  'swarm-codex': {
    name: 'Swarm Codex',
    component: SwarmCodex,
    description: 'AI cultivation system with intelligence progression',
    complexity: 3,
    category: 'AI Simulation',
    mathematical: false,
    interactive: true
  },
  'glyph-constellation': {
    name: 'Glyph Constellation',
    component: GlyphConstellation,
    description: 'Cryptographic visualization with constellation mapping',
    complexity: 2,
    category: 'Data Visualization',
    mathematical: false,
    interactive: true
  },
  'nexus-runtime': {
    name: 'Nexus Runtime',
    component: NexusRuntime,
    description: 'Grid-based intelligence network monitoring',
    complexity: 3,
    category: 'Network Systems',
    mathematical: false,
    interactive: true
  },
  'avatar-impostor': {
    name: 'Avatar Impostor',
    component: AvatarImpostor,
    description: 'Multi-view avatar rendering with depth-aware fusion',
    complexity: 4,
    category: 'Rendering',
    mathematical: true,
    interactive: true
  },
  'contour-extrude': {
    name: 'Contour Extrude',
    component: ContourExtrude,
    description: 'Image-to-3D contour extrusion with edge detection',
    complexity: 3,
    category: 'Image Processing',
    mathematical: true,
    interactive: true
  },
  'hyperbowl': {
    name: 'HyperBowl',
    component: HyperBowl,
    description: 'Parametric surface visualization with complex mathematics',
    complexity: 5,
    category: 'Mathematics',
    mathematical: true,
    interactive: true
  },
  'mushroom-codex': {
    name: 'Mushroom Codex',
    component: MushroomCodex,
    description: 'Parametric mushroom generation with toon shading',
    complexity: 3,
    category: 'Procedural Generation',
    mathematical: false,
    interactive: true
  },
  'raymarching-cinema': {
    name: 'Raymarching Cinema',
    component: RaymarchingCinema,
    description: 'Real-time cinematic visual generation with advanced raymarching',
    complexity: 5,
    category: 'Visual Effects',
    mathematical: true,
    interactive: true
  },
  'easing-studio': {
    name: 'Easing Studio',
    component: EasingStudio,
    description: 'Advanced 3D animation curve editor with mathematical precision',
    complexity: 5,
    category: 'Animation Tools',
    mathematical: true,
    interactive: true
  }
};

const CATEGORIES = [
  'All',
  'Mathematics',
  'Visual Effects',
  'Geometry Processing',
  'AI Simulation',
  'Data Visualization',
  'Animation Tools',
  'Network Systems',
  'Rendering',
  'Image Processing',
  'Procedural Generation'
];

// =============================================================================
// Tool Showcase Component
// =============================================================================

export default function SandboxShowcase() {
  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [showMathematical, setShowMathematical] = useState<boolean | null>(null);

  // Filter tools based on selected category and mathematical complexity
  const filteredTools = Object.entries(TOOL_REGISTRY).filter(([key, tool]) => {
    if (selectedCategory !== 'All' && tool.category !== selectedCategory) {
      return false;
    }
    if (showMathematical !== null && tool.mathematical !== showMathematical) {
      return false;
    }
    return true;
  });

  // Get complexity distribution
  const complexityStats = Object.values(TOOL_REGISTRY).reduce((acc, tool) => {
    acc[tool.complexity] = (acc[tool.complexity] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);

  const SelectedComponent = selectedTool ? TOOL_REGISTRY[selectedTool]?.component : null;

  return (
    <div style={{
      width: '100%',
      height: '100vh',
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%)',
      color: '#fff',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      {selectedTool && SelectedComponent ? (
        // Full-screen tool view
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
          <SelectedComponent />

          {/* Exit button */}
          <button
            onClick={() => setSelectedTool(null)}
            style={{
              position: 'absolute',
              top: '20px',
              right: '20px',
              background: 'rgba(255, 0, 128, 0.2)',
              border: '1px solid #ff0080',
              color: '#ff0080',
              padding: '10px 20px',
              borderRadius: '8px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              fontSize: '14px',
              fontWeight: '600',
              zIndex: 1000,
              backdropFilter: 'blur(10px)'
            }}
          >
            ← Back to Showcase
          </button>
        </div>
      ) : (
        // Showcase gallery view
        <div style={{ padding: '40px', maxWidth: '1600px', margin: '0 auto' }}>
          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: '40px' }}>
            <h1 style={{
              fontSize: '3.5rem',
              margin: '0 0 10px 0',
              background: 'linear-gradient(45deg, #ff0080, #00ff80, #8000ff)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              fontWeight: 'bold'
            }}>
              SANDBOX TOOLS COLLECTION
            </h1>
            <p style={{
              fontSize: '1.2rem',
              opacity: 0.8,
              margin: '0 0 30px 0',
              color: '#64ffda'
            }}>
              13 Sophisticated React Three.js Tools for Advanced 3D Visualization
            </p>

            {/* Statistics */}
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '30px',
              marginTop: '20px',
              flexWrap: 'wrap'
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#00ff80' }}>12</div>
                <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>Total Tools</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#ff6b35' }}>
                  {Object.values(TOOL_REGISTRY).filter(t => t.mathematical).length}
                </div>
                <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>Mathematical</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#8000ff' }}>
                  {Object.values(complexityStats).reduce((a, b) => Math.max(a, b), 0)}
                </div>
                <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>Max Complexity</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#ff0080' }}>100%</div>
                <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>Complete</div>
              </div>
            </div>
          </div>

          {/* Filters */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '20px',
            marginBottom: '40px',
            flexWrap: 'wrap'
          }}>
            {/* Category filter */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', opacity: 0.8 }}>
                Category
              </label>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                style={{
                  padding: '8px 12px',
                  borderRadius: '6px',
                  background: 'rgba(255,255,255,0.1)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  color: '#fff',
                  fontSize: '0.9rem',
                  fontFamily: 'inherit'
                }}
              >
                {CATEGORIES.map(cat => (
                  <option key={cat} value={cat} style={{ background: '#1a1a1a' }}>
                    {cat}
                  </option>
                ))}
              </select>
            </div>

            {/* Mathematical filter */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', opacity: 0.8 }}>
                Type
              </label>
              <select
                value={showMathematical === null ? 'all' : showMathematical ? 'mathematical' : 'visual'}
                onChange={(e) => {
                  const val = e.target.value;
                  setShowMathematical(val === 'all' ? null : val === 'mathematical');
                }}
                style={{
                  padding: '8px 12px',
                  borderRadius: '6px',
                  background: 'rgba(255,255,255,0.1)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  color: '#fff',
                  fontSize: '0.9rem',
                  fontFamily: 'inherit'
                }}
              >
                <option value="all" style={{ background: '#1a1a1a' }}>All Types</option>
                <option value="mathematical" style={{ background: '#1a1a1a' }}>Mathematical</option>
                <option value="visual" style={{ background: '#1a1a1a' }}>Visual/Interactive</option>
              </select>
            </div>
          </div>

          {/* Tool Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: '24px',
            marginBottom: '40px'
          }}>
            {filteredTools.map(([key, tool]) => (
              <div
                key={key}
                onClick={() => setSelectedTool(key)}
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '12px',
                  padding: '24px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  backdropFilter: 'blur(10px)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.boxShadow = '0 8px 32px rgba(255,255,255,0.1)';
                  e.currentTarget.style.borderColor = '#00ff80';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)';
                }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: '12px'
                }}>
                  <h3 style={{
                    margin: 0,
                    fontSize: '1.3rem',
                    fontWeight: 'bold',
                    color: '#00ff80'
                  }}>
                    {tool.name}
                  </h3>
                  <div style={{
                    display: 'flex',
                    gap: '4px'
                  }}>
                    {Array.from({ length: tool.complexity }).map((_, i) => (
                      <span key={i} style={{ color: '#ff6b35', fontSize: '0.8rem' }}>⭐</span>
                    ))}
                  </div>
                </div>

                <div style={{
                  fontSize: '0.8rem',
                  padding: '4px 8px',
                  borderRadius: '12px',
                  background: tool.mathematical
                    ? 'rgba(255, 107, 53, 0.2)'
                    : 'rgba(100, 255, 218, 0.2)',
                  color: tool.mathematical ? '#ff6b35' : '#64ffda',
                  display: 'inline-block',
                  marginBottom: '12px',
                  fontWeight: '600'
                }}>
                  {tool.category}
                </div>

                <p style={{
                  margin: 0,
                  fontSize: '0.9rem',
                  opacity: 0.8,
                  lineHeight: 1.4
                }}>
                  {tool.description}
                </p>

                <div style={{
                  marginTop: '16px',
                  display: 'flex',
                  gap: '12px',
                  fontSize: '0.8rem',
                  opacity: 0.6
                }}>
                  {tool.mathematical && <span>🧮 Mathematical</span>}
                  {tool.interactive && <span>🎮 Interactive</span>}
                </div>
              </div>
            ))}
          </div>

          {/* Footer Info */}
          <div style={{
            textAlign: 'center',
            padding: '40px 20px',
            borderTop: '1px solid rgba(255,255,255,0.1)',
            opacity: 0.7
          }}>
            <p style={{ margin: '0 0 10px 0', fontSize: '0.9rem' }}>
              Complete collection of sophisticated React Three.js sandbox tools
            </p>
            <p style={{ margin: 0, fontSize: '0.8rem' }}>
              Click any tool to explore • Built with React Three Fiber • TypeScript Ready
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
