// Visual Bleedway Demo - Showcasing the complete production pipeline
import * as React from 'react';
import { VisualBleedway } from './VisualBleedway';
import { BUILTIN_PRESETS } from './SilhouetteProcessing';

export function VisualBleedwayDemo() {
  const [demoMode, setDemoMode] = React.useState<'full' | 'quick' | 'advanced'>('full');

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      {/* Demo Header */}
      <div className="relative z-50 p-6 bg-black/20 backdrop-blur border-b border-white/10">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2">
            Visual Bleedway Production Demo
          </h1>
          <p className="text-gray-300 mb-4">
            Complete PNG silhouette → 3D mesh → sensory-enhanced visualization pipeline
          </p>

          {/* Feature Highlights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="p-4 bg-white/5 rounded-lg backdrop-blur">
              <h3 className="text-emerald-400 font-semibold mb-2">🧠 Advanced Algorithms</h3>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• Otsu auto-thresholding</li>
                <li>• Morphological operations</li>
                <li>• 3D volume field generation</li>
                <li>• Marching cubes preparation</li>
              </ul>
            </div>

            <div className="p-4 bg-white/5 rounded-lg backdrop-blur">
              <h3 className="text-cyan-400 font-semibold mb-2">⚡ Production Features</h3>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• Real C++ algorithm ports</li>
                <li>• Dual thresholding support</li>
                <li>• Automatic side generation</li>
                <li>• GLB export capability</li>
              </ul>
            </div>

            <div className="p-4 bg-white/5 rounded-lg backdrop-blur">
              <h3 className="text-purple-400 font-semibold mb-2">🎨 Sensory Integration</h3>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• 6-channel sensory overlay</li>
                <li>• R3F visualization</li>
                <li>• Procedural moments</li>
                <li>• Environmental context</li>
              </ul>
            </div>
          </div>

          {/* Demo Mode Selector */}
          <div className="flex gap-2">
            <button
              onClick={() => setDemoMode('full')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                demoMode === 'full'
                  ? 'bg-emerald-500 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              Full Pipeline
            </button>
            <button
              onClick={() => setDemoMode('quick')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                demoMode === 'quick'
                  ? 'bg-cyan-500 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              Quick Test
            </button>
            <button
              onClick={() => setDemoMode('advanced')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                demoMode === 'advanced'
                  ? 'bg-purple-500 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              Advanced Controls
            </button>
          </div>
        </div>
      </div>

      {/* Instructions Panel */}
      <DemoInstructions mode={demoMode} />

      {/* Main Visual Bleedway Component */}
      <div className="relative">
        <VisualBleedway />

        {/* Demo Overlay Info */}
        {demoMode !== 'full' && (
          <div className="absolute top-4 right-4 max-w-sm p-4 bg-black/60 backdrop-blur rounded-lg text-white">
            <DemoModeInfo mode={demoMode} />
          </div>
        )}
      </div>
    </div>
  );
}

function DemoInstructions({ mode }: { mode: 'full' | 'quick' | 'advanced' }) {
  const instructions = {
    full: {
      title: "Complete Pipeline Workflow",
      steps: [
        "1. Drag & drop a PNG silhouette image (front view) into the left panel",
        "2. Optionally add a side silhouette for enhanced 3D accuracy",
        "3. Choose processing preset: Auto (Otsu), Baked presets, or High Detail",
        "4. Click 'Build Mesh' to generate 3D geometry using advanced algorithms",
        "5. Explore the 6-channel sensory overlay in the right panel",
        "6. Export your mesh as GLB for use in other applications"
      ]
    },
    quick: {
      title: "Quick Test Mode",
      steps: [
        "1. Drop any PNG silhouette for rapid prototyping",
        "2. Use 'Auto (Otsu)' preset for automatic thresholding",
        "3. Hit 'Build Mesh' - the system will auto-generate side view",
        "4. Instant 3D visualization with sensory context"
      ]
    },
    advanced: {
      title: "Advanced Algorithm Controls",
      steps: [
        "1. Load silhouettes and expand 'Advanced Controls' panel",
        "2. Experiment with dual thresholding (separate F/S values)",
        "3. Adjust morphological kernel size for edge refinement",
        "4. Modify 3D resolution, isosurface, and height parameters",
        "5. Fine-tune Laplacian smoothing and mesh decimation",
        "6. Test different presets vs custom parameter combinations"
      ]
    }
  };

  const current = instructions[mode];

  return (
    <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 backdrop-blur">
      <div className="max-w-6xl mx-auto p-6">
        <h2 className="text-xl font-semibold text-white mb-3">{current.title}</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {current.steps.map((step, i) => (
            <div key={i} className="p-3 bg-white/5 rounded-lg backdrop-blur">
              <p className="text-sm text-gray-200">{step}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DemoModeInfo({ mode }: { mode: 'full' | 'quick' | 'advanced' }) {
  const info = {
    quick: {
      icon: "⚡",
      title: "Quick Test Mode",
      description: "Optimized for rapid prototyping with automatic settings. Perfect for testing silhouettes quickly."
    },
    advanced: {
      icon: "🔬",
      title: "Advanced Controls",
      description: "Full algorithm parameter access. Experiment with thresholding, morphology, and mesh generation settings."
    }
  };

  if (mode === 'full') return null;

  const current = info[mode];

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">{current.icon}</span>
        <h3 className="font-semibold">{current.title}</h3>
      </div>
      <p className="text-sm opacity-80 mb-3">{current.description}</p>

      <div className="text-xs opacity-60">
        <p className="mb-1">Available Presets:</p>
        <div className="space-y-1">
          {Object.keys(BUILTIN_PRESETS).map(preset => (
            <div key={preset} className="flex items-center gap-2">
              <div className="w-1 h-1 bg-emerald-400 rounded-full" />
              <span>{preset}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Sample silhouette generator for testing (creates simple shapes)
export function generateTestSilhouette(type: 'human' | 'object' | 'complex' = 'human'): string {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  canvas.width = 256;
  canvas.height = 512;

  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = 'white';

  switch (type) {
    case 'human':
      // Simple human silhouette
      ctx.fillRect(110, 50, 36, 40);   // head
      ctx.fillRect(100, 90, 56, 120);  // torso
      ctx.fillRect(85, 210, 30, 150);  // left leg
      ctx.fillRect(141, 210, 30, 150); // right leg
      ctx.fillRect(70, 100, 30, 80);   // left arm
      ctx.fillRect(156, 100, 30, 80);  // right arm
      break;

    case 'object':
      // Simple bottle/vase shape
      ctx.fillRect(110, 50, 36, 30);   // neck
      ctx.fillRect(90, 80, 76, 280);   // body
      ctx.fillRect(100, 360, 56, 100); // base
      break;

    case 'complex':
      // More complex organic shape
      ctx.beginPath();
      ctx.ellipse(128, 100, 40, 60, 0, 0, Math.PI * 2);
      ctx.ellipse(128, 220, 60, 80, 0, 0, Math.PI * 2);
      ctx.ellipse(128, 360, 45, 70, 0, 0, Math.PI * 2);
      ctx.fill();
      break;
  }

  return canvas.toDataURL('image/png');
}
