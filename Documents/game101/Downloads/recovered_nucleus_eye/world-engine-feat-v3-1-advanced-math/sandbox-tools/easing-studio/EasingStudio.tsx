import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, Text, Line } from "@react-three/drei";
import { useControls, button, folder } from "leva";
import { mathEngine } from '../shared/utils';

// =============================================================================
// Types and Interfaces
// =============================================================================

interface EasingFunction {
  name: string;
  fn: (t: number) => number;
  category: 'basic' | 'advanced' | 'physics' | 'bounce' | 'elastic' | 'custom';
  description: string;
  parameters?: Record<string, { min: number; max: number; default: number; step?: number }>;
}

interface CurvePoint {
  t: number;
  value: number;
  position: THREE.Vector3;
}

interface AnimationPath {
  points: THREE.Vector3[];
  duration: number;
  easingFunction: string;
}

interface EasingPreset {
  name: string;
  type: 'cubic-bezier' | 'steps' | 'spring' | 'custom';
  value: number[] | string | any;
  category: string;
}

// =============================================================================
// Enhanced Easing Functions
// =============================================================================

const EASING_FUNCTIONS: Record<string, EasingFunction> = {
  // Basic CSS-compatible easings
  linear: {
    name: 'Linear',
    fn: (t: number) => t,
    category: 'basic',
    description: 'Constant speed throughout'
  },
  easeIn: {
    name: 'Ease In',
    fn: (t: number) => t * t * t,
    category: 'basic',
    description: 'Starts slow, ends fast'
  },
  easeOut: {
    name: 'Ease Out',
    fn: (t: number) => 1 - Math.pow(1 - t, 3),
    category: 'basic',
    description: 'Starts fast, ends slow'
  },
  easeInOut: {
    name: 'Ease In-Out',
    fn: (t: number) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
    category: 'basic',
    description: 'Slow start and end, fast middle'
  },

  // Advanced cubic easings
  easeInQuart: {
    name: 'Ease In Quart',
    fn: (t: number) => t * t * t * t,
    category: 'advanced',
    description: 'Strong acceleration'
  },
  easeOutQuart: {
    name: 'Ease Out Quart',
    fn: (t: number) => 1 - Math.pow(1 - t, 4),
    category: 'advanced',
    description: 'Strong deceleration'
  },
  easeInOutQuart: {
    name: 'Ease In-Out Quart',
    fn: (t: number) => t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2,
    category: 'advanced',
    description: 'Strong acceleration/deceleration'
  },

  // Physics-based easings
  easeInCirc: {
    name: 'Ease In Circular',
    fn: (t: number) => 1 - Math.sqrt(1 - Math.pow(t, 2)),
    category: 'physics',
    description: 'Circular motion entrance'
  },
  easeOutCirc: {
    name: 'Ease Out Circular',
    fn: (t: number) => Math.sqrt(1 - Math.pow(t - 1, 2)),
    category: 'physics',
    description: 'Circular motion exit'
  },
  easeInOutCirc: {
    name: 'Ease In-Out Circular',
    fn: (t: number) => t < 0.5
      ? (1 - Math.sqrt(1 - Math.pow(2 * t, 2))) / 2
      : (Math.sqrt(1 - Math.pow(-2 * t + 2, 2)) + 1) / 2,
    category: 'physics',
    description: 'Circular motion both ends'
  },

  // Spring physics
  spring: {
    name: 'Spring',
    fn: (t: number) => {
      const stiffness = 12;
      return 1 - (1 + stiffness * t) * Math.exp(-stiffness * t);
    },
    category: 'physics',
    description: 'Spring-damped motion',
    parameters: {
      stiffness: { min: 4, max: 24, default: 12, step: 0.5 }
    }
  },

  // Bounce easings
  bounceOut: {
    name: 'Bounce Out',
    fn: (t: number) => {
      if (t < 1 / 2.75) return 7.5625 * t * t;
      else if (t < 2 / 2.75) return 7.5625 * (t -= 1.5 / 2.75) * t + 0.75;
      else if (t < 2.5 / 2.75) return 7.5625 * (t -= 2.25 / 2.75) * t + 0.9375;
      else return 7.5625 * (t -= 2.625 / 2.75) * t + 0.984375;
    },
    category: 'bounce',
    description: 'Bouncing ball effect'
  },
  bounceIn: {
    name: 'Bounce In',
    fn: (t: number) => 1 - EASING_FUNCTIONS.bounceOut.fn(1 - t),
    category: 'bounce',
    description: 'Reverse bounce effect'
  },

  // Elastic easings
  elasticOut: {
    name: 'Elastic Out',
    fn: (t: number) => {
      const c4 = (2 * Math.PI) / 3;
      return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
    },
    category: 'elastic',
    description: 'Elastic oscillation'
  },
  elasticIn: {
    name: 'Elastic In',
    fn: (t: number) => {
      const c4 = (2 * Math.PI) / 3;
      return t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * c4);
    },
    category: 'elastic',
    description: 'Elastic windup'
  },

  // Back easings (overshoot)
  backOut: {
    name: 'Back Out',
    fn: (t: number) => {
      const c1 = 1.70158;
      const c3 = c1 + 1;
      return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    },
    category: 'advanced',
    description: 'Overshooting motion'
  },
  backIn: {
    name: 'Back In',
    fn: (t: number) => {
      const c1 = 1.70158;
      const c3 = c1 + 1;
      return c3 * t * t * t - c1 * t * t;
    },
    category: 'advanced',
    description: 'Reverse overshoot'
  }
};

const EASING_PRESETS: EasingPreset[] = [
  { name: 'Linear', type: 'custom', value: 'linear', category: 'Basic' },
  { name: 'Ease', type: 'cubic-bezier', value: [0.25, 0.1, 0.25, 1], category: 'CSS' },
  { name: 'Ease In', type: 'cubic-bezier', value: [0.42, 0, 1, 1], category: 'CSS' },
  { name: 'Ease Out', type: 'cubic-bezier', value: [0, 0, 0.58, 1], category: 'CSS' },
  { name: 'Ease In-Out', type: 'cubic-bezier', value: [0.42, 0, 0.58, 1], category: 'CSS' },
  { name: 'Material', type: 'cubic-bezier', value: [0.4, 0, 0.2, 1], category: 'Design System' },
  { name: 'iOS', type: 'cubic-bezier', value: [0.4, 0, 0.6, 1], category: 'Design System' },
  { name: 'Bounce', type: 'custom', value: 'bounceOut', category: 'Expressive' },
  { name: 'Elastic', type: 'custom', value: 'elasticOut', category: 'Expressive' },
  { name: 'Back', type: 'cubic-bezier', value: [0.34, 1.56, 0.64, 1], category: 'Expressive' },
  { name: 'Steps', type: 'steps', value: [8, 'end'], category: 'Discrete' }
];

// =============================================================================
// Utility Functions
// =============================================================================

const clamp = (n: number, min: number, max: number) => Math.min(Math.max(n, min), max);

const sampleEasingFunction = (easingFn: (t: number) => number, steps = 100): CurvePoint[] => {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps;
    const value = clamp(easingFn(t), -0.5, 1.5); // Allow slight overshoot for visualization
    return {
      t,
      value,
      position: new THREE.Vector3(t * 4 - 2, value * 2 - 1, 0)
    };
  });
};

const bezierFunction = ([x1, y1, x2, y2]: number[]) => {
  return (t: number) => {
    // Approximate cubic bezier using De Casteljau's algorithm
    const mt = 1 - t;
    const mt2 = mt * mt;
    const t2 = t * t;
    return 3 * mt2 * t * y1 + 3 * mt * t2 * y2 + t * t * t;
  };
};

const generateAnimationPath = (easingFn: (t: number) => number, pathType: string): THREE.Vector3[] => {
  const points: THREE.Vector3[] = [];
  const steps = 60;

  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const easedT = easingFn(t);

    let x, y, z;

    switch (pathType) {
      case 'linear':
        x = t * 4 - 2;
        y = 0;
        z = 0;
        break;
      case 'circular':
        x = Math.cos(easedT * Math.PI * 2) * 1.5;
        y = Math.sin(easedT * Math.PI * 2) * 1.5;
        z = 0;
        break;
      case 'spiral':
        x = Math.cos(easedT * Math.PI * 4) * (1.5 - easedT * 0.5);
        y = Math.sin(easedT * Math.PI * 4) * (1.5 - easedT * 0.5);
        z = easedT * 2 - 1;
        break;
      case 'wave':
        x = t * 4 - 2;
        y = Math.sin(easedT * Math.PI * 4) * 0.5;
        z = Math.cos(easedT * Math.PI * 2) * 0.3;
        break;
      default:
        x = t * 4 - 2;
        y = easedT * 2 - 1;
        z = 0;
    }

    points.push(new THREE.Vector3(x, y, z));
  }

  return points;
};

// =============================================================================
// 3D Curve Visualization Component
// =============================================================================

interface CurveVisualizationProps {
  curvePoints: CurvePoint[];
  showGrid: boolean;
  showPoints: boolean;
  curveColor: string;
  animated: boolean;
}

function CurveVisualization({
  curvePoints,
  showGrid,
  showPoints,
  curveColor,
  animated
}: CurveVisualizationProps) {
  const lineRef = useRef<THREE.Line>(null!);
  const pointsRef = useRef<THREE.Group>(null!);

  const linePoints = useMemo(() => {
    return curvePoints.map(point => point.position);
  }, [curvePoints]);

  useFrame((state) => {
    if (animated && lineRef.current) {
      lineRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <group>
      {/* Grid */}
      {showGrid && (
        <group>
          {/* X-axis grid lines */}
          {Array.from({ length: 9 }, (_, i) => {
            const x = (i - 4) * 0.5;
            return (
              <Line
                key={`grid-x-${i}`}
                points={[new THREE.Vector3(x, -1, 0), new THREE.Vector3(x, 1, 0)]}
                color="#333333"
                lineWidth={0.5}
                opacity={0.3}
              />
            );
          })}
          {/* Y-axis grid lines */}
          {Array.from({ length: 9 }, (_, i) => {
            const y = (i - 4) * 0.25;
            return (
              <Line
                key={`grid-y-${i}`}
                points={[new THREE.Vector3(-2, y, 0), new THREE.Vector3(2, y, 0)]}
                color="#333333"
                lineWidth={0.5}
                opacity={0.3}
              />
            );
          })}

          {/* Axes */}
          <Line
            points={[new THREE.Vector3(-2, 0, 0), new THREE.Vector3(2, 0, 0)]}
            color="#666666"
            lineWidth={2}
          />
          <Line
            points={[new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 1, 0)]}
            color="#666666"
            lineWidth={2}
          />
        </group>
      )}

      {/* Main curve */}
      <Line
        ref={lineRef}
        points={linePoints}
        color={curveColor}
        lineWidth={3}
      />

      {/* Sample points */}
      {showPoints && (
        <group ref={pointsRef}>
          {curvePoints.filter((_, i) => i % 5 === 0).map((point, index) => (
            <mesh key={index} position={point.position}>
              <sphereGeometry args={[0.02]} />
              <meshBasicMaterial color={curveColor} />
            </mesh>
          ))}
        </group>
      )}

      {/* Axis labels */}
      <Text
        position={[2.2, 0, 0]}
        fontSize={0.1}
        color="#999999"
        anchorX="left"
      >
        time (t)
      </Text>
      <Text
        position={[0, 1.2, 0]}
        fontSize={0.1}
        color="#999999"
        anchorX="center"
      >
        value
      </Text>
    </group>
  );
}

// =============================================================================
// Animation Path Visualization
// =============================================================================

interface PathAnimationProps {
  animationPath: THREE.Vector3[];
  isPlaying: boolean;
  duration: number;
  pathColor: string;
}

function PathAnimation({ animationPath, isPlaying, duration, pathColor }: PathAnimationProps) {
  const objectRef = useRef<THREE.Mesh>(null!);
  const pathRef = useRef<THREE.Line>(null!);
  const [progress, setProgress] = useState(0);

  useFrame((state) => {
    if (isPlaying && objectRef.current && animationPath.length > 0) {
      const elapsed = state.clock.elapsedTime % (duration / 1000);
      const t = elapsed / (duration / 1000);
      const index = Math.floor(t * (animationPath.length - 1));
      const nextIndex = Math.min(index + 1, animationPath.length - 1);
      const localT = (t * (animationPath.length - 1)) - index;

      if (animationPath[index] && animationPath[nextIndex]) {
        const position = new THREE.Vector3().lerpVectors(
          animationPath[index],
          animationPath[nextIndex],
          localT
        );
        objectRef.current.position.copy(position);
        setProgress(t);
      }
    }
  });

  return (
    <group>
      {/* Animation path */}
      <Line
        ref={pathRef}
        points={animationPath}
        color={pathColor}
        lineWidth={2}
        opacity={0.6}
        transparent
      />

      {/* Animated object */}
      <mesh ref={objectRef}>
        <sphereGeometry args={[0.05]} />
        <meshStandardMaterial
          color={pathColor}
          emissive={pathColor}
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* Progress indicator */}
      {isPlaying && (
        <Html position={[0, -2, 0]} center>
          <div style={{
            background: 'rgba(0,0,0,0.8)',
            color: '#fff',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            Progress: {Math.round(progress * 100)}%
          </div>
        </Html>
      )}
    </group>
  );
}

// =============================================================================
// Interactive Bezier Curve Editor
// =============================================================================

interface BezierEditorProps {
  bezierPoints: number[];
  onBezierChange: (points: number[]) => void;
}

function BezierEditor({ bezierPoints, onBezierChange }: BezierEditorProps) {
  const [dragging, setDragging] = useState<number | null>(null);
  const { camera, raycaster, mouse } = useThree();

  const controlPoints = useMemo(() => {
    const [x1, y1, x2, y2] = bezierPoints;
    return [
      new THREE.Vector3(x1 * 2 - 1, y1 * 2 - 1, 0),
      new THREE.Vector3(x2 * 2 - 1, y2 * 2 - 1, 0)
    ];
  }, [bezierPoints]);

  const handlePointerDown = useCallback((index: number) => {
    setDragging(index);
  }, []);

  const handlePointerMove = useCallback((event: any) => {
    if (dragging === null) return;

    raycaster.setFromCamera(mouse, camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const intersection = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, intersection);

    const newPoints = [...bezierPoints];
    const x = clamp((intersection.x + 1) / 2, -0.2, 1.2);
    const y = clamp((intersection.y + 1) / 2, -0.2, 1.2);

    if (dragging === 0) {
      newPoints[0] = x;
      newPoints[1] = y;
    } else {
      newPoints[2] = x;
      newPoints[3] = y;
    }

    onBezierChange(newPoints);
  }, [dragging, bezierPoints, onBezierChange, camera, raycaster, mouse]);

  const handlePointerUp = useCallback(() => {
    setDragging(null);
  }, []);

  useEffect(() => {
    const canvas = document.querySelector('canvas');
    if (canvas) {
      canvas.addEventListener('pointermove', handlePointerMove);
      canvas.addEventListener('pointerup', handlePointerUp);
      return () => {
        canvas.removeEventListener('pointermove', handlePointerMove);
        canvas.removeEventListener('pointerup', handlePointerUp);
      };
    }
  }, [handlePointerMove, handlePointerUp]);

  return (
    <group>
      {/* Control point connectors */}
      <Line
        points={[new THREE.Vector3(-1, -1, 0), controlPoints[0]]}
        color="#999999"
        lineWidth={1}
        dashed
      />
      <Line
        points={[new THREE.Vector3(1, 1, 0), controlPoints[1]]}
        color="#999999"
        lineWidth={1}
        dashed
      />

      {/* Control points */}
      {controlPoints.map((point, index) => (
        <mesh
          key={index}
          position={point}
          onPointerDown={() => handlePointerDown(index)}
        >
          <sphereGeometry args={[0.06]} />
          <meshBasicMaterial
            color={dragging === index ? "#ff6b35" : "#00ff80"}
          />
        </mesh>
      ))}

      {/* Corner points */}
      <mesh position={[-1, -1, 0]}>
        <sphereGeometry args={[0.04]} />
        <meshBasicMaterial color="#666666" />
      </mesh>
      <mesh position={[1, 1, 0]}>
        <sphereGeometry args={[0.04]} />
        <meshBasicMaterial color="#666666" />
      </mesh>
    </group>
  );
}

// =============================================================================
// Main Easing Studio Component
// =============================================================================

export default function EasingStudio() {
  // State management
  const [selectedEasing, setSelectedEasing] = useState('easeOut');
  const [selectedPreset, setSelectedPreset] = useState('Ease Out');
  const [customBezier, setCustomBezier] = useState([0, 0, 0.58, 1]);
  const [duration, setDuration] = useState(500);
  const [showGrid, setShowGrid] = useState(true);
  const [showPoints, setShowPoints] = useState(false);
  const [showBezierEditor, setShowBezierEditor] = useState(false);
  const [animationPath, setAnimationPath] = useState('curve');
  const [isPlaying, setIsPlaying] = useState(false);
  const [curveColor, setCurveColor] = useState('#00ff80');
  const [pathColor, setPathColor] = useState('#ff6b35');

  // Get current easing function
  const currentEasingFn = useMemo(() => {
    if (selectedPreset === 'Custom Bezier') {
      return bezierFunction(customBezier);
    }
    return EASING_FUNCTIONS[selectedEasing]?.fn || EASING_FUNCTIONS.linear.fn;
  }, [selectedEasing, selectedPreset, customBezier]);

  // Sample the easing curve
  const curvePoints = useMemo(() => {
    return sampleEasingFunction(currentEasingFn);
  }, [currentEasingFn]);

  // Generate animation path
  const animationPathPoints = useMemo(() => {
    return generateAnimationPath(currentEasingFn, animationPath);
  }, [currentEasingFn, animationPath]);

  // Generate CSS output
  const cssOutput = useMemo(() => {
    let timingFunction = 'ease-out';

    if (selectedPreset === 'Custom Bezier') {
      timingFunction = `cubic-bezier(${customBezier.map(n => n.toFixed(3)).join(', ')})`;
    } else if (selectedEasing === 'linear') {
      timingFunction = 'linear';
    } else {
      // Convert to closest CSS equivalent or custom cubic-bezier
      const preset = EASING_PRESETS.find(p => p.name === selectedPreset);
      if (preset) {
        if (preset.type === 'cubic-bezier') {
          const values = preset.value as number[];
          timingFunction = `cubic-bezier(${values.map(n => n.toFixed(3)).join(', ')})`;
        } else if (preset.type === 'steps') {
          const [steps, position] = preset.value as [number, string];
          timingFunction = `steps(${steps}, ${position})`;
        } else {
          timingFunction = preset.value as string;
        }
      }
    }

    return `:root {
  --duration-fast: 150ms;
  --duration-normal: ${duration}ms;
  --duration-slow: ${duration * 1.5}ms;
  --easing-primary: ${timingFunction};
}

.animated {
  transition: all var(--duration-normal) var(--easing-primary);
}

@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}`;
  }, [selectedPreset, selectedEasing, customBezier, duration]);

  // Leva controls
  const controls = useControls({
    'Easing Function': folder({
      preset: {
        value: selectedPreset,
        options: EASING_PRESETS.map(p => p.name),
        onChange: (value) => {
          setSelectedPreset(value);
          const preset = EASING_PRESETS.find(p => p.name === value);
          if (preset && preset.type === 'custom') {
            setSelectedEasing(preset.value as string);
          }
        }
      },
      customFunction: {
        value: selectedEasing,
        options: Object.keys(EASING_FUNCTIONS),
        onChange: setSelectedEasing
      }
    }),

    'Bezier Editor': folder({
      showEditor: {
        value: showBezierEditor,
        onChange: setShowBezierEditor
      },
      x1: {
        value: customBezier[0],
        min: -0.5,
        max: 1.5,
        step: 0.01,
        onChange: (value) => setCustomBezier(prev => [value, prev[1], prev[2], prev[3]])
      },
      y1: {
        value: customBezier[1],
        min: -0.5,
        max: 1.5,
        step: 0.01,
        onChange: (value) => setCustomBezier(prev => [prev[0], value, prev[2], prev[3]])
      },
      x2: {
        value: customBezier[2],
        min: -0.5,
        max: 1.5,
        step: 0.01,
        onChange: (value) => setCustomBezier(prev => [prev[0], prev[1], value, prev[3]])
      },
      y2: {
        value: customBezier[3],
        min: -0.5,
        max: 1.5,
        step: 0.01,
        onChange: (value) => setCustomBezier(prev => [prev[0], prev[1], prev[2], value])
      }
    }),

    'Animation': folder({
      duration: {
        value: duration,
        min: 100,
        max: 2000,
        step: 50,
        onChange: setDuration
      },
      pathType: {
        value: animationPath,
        options: ['curve', 'linear', 'circular', 'spiral', 'wave'],
        onChange: setAnimationPath
      },
      'Play Animation': button(() => {
        setIsPlaying(!isPlaying);
      })
    }),

    'Visualization': folder({
      showGrid: {
        value: showGrid,
        onChange: setShowGrid
      },
      showPoints: {
        value: showPoints,
        onChange: setShowPoints
      },
      curveColor: {
        value: curveColor,
        onChange: setCurveColor
      },
      pathColor: {
        value: pathColor,
        onChange: setPathColor
      }
    }),

    'Export': folder({
      'Copy CSS': button(() => {
        navigator.clipboard.writeText(cssOutput).then(() => {
          console.log('CSS copied to clipboard!');
        });
      }),
      'Save as JSON': button(() => {
        const data = {
          name: selectedPreset,
          easing: selectedEasing,
          bezier: customBezier,
          duration,
          timestamp: Date.now()
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `easing-${selectedPreset.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
      })
    })
  });

  return (
    <div style={{ width: '100%', height: '100vh', background: '#0a0a0a' }}>
      <Canvas
        camera={{ position: [4, 2, 4], fov: 60 }}
        style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%)' }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <pointLight position={[-2, 2, 2]} intensity={0.6} color="#64ffda" />

        {/* Main curve visualization */}
        <CurveVisualization
          curvePoints={curvePoints}
          showGrid={showGrid}
          showPoints={showPoints}
          curveColor={curveColor}
          animated={false}
        />

        {/* Interactive bezier editor */}
        {showBezierEditor && selectedPreset === 'Custom Bezier' && (
          <BezierEditor
            bezierPoints={customBezier}
            onBezierChange={setCustomBezier}
          />
        )}

        {/* Animation path visualization */}
        {animationPath !== 'curve' && (
          <group position={[0, 0, -3]}>
            <PathAnimation
              animationPath={animationPathPoints}
              isPlaying={isPlaying}
              duration={duration}
              pathColor={pathColor}
            />
          </group>
        )}

        {/* Function information overlay */}
        <Html position={[0, 2.5, 0]} center>
          <div style={{
            background: 'rgba(10, 10, 10, 0.9)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '16px',
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '14px',
            textAlign: 'center',
            minWidth: '300px'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '8px', color: curveColor }}>
              {EASING_FUNCTIONS[selectedEasing]?.name || selectedPreset}
            </div>
            <div style={{ opacity: 0.8, fontSize: '12px', marginBottom: '12px' }}>
              {EASING_FUNCTIONS[selectedEasing]?.description || 'Custom timing function'}
            </div>
            <div style={{
              background: 'rgba(255, 255, 255, 0.05)',
              padding: '8px',
              borderRadius: '6px',
              fontSize: '10px',
              fontFamily: 'monospace'
            }}>
              Duration: {duration}ms
            </div>
          </div>
        </Html>

        {/* CSS output overlay */}
        <Html position={[-3, -2, 0]} style={{ width: '400px' }}>
          <div style={{
            background: 'rgba(10, 10, 10, 0.95)',
            backdropFilter: 'blur(15px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '16px',
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '12px'
            }}>
              <span style={{ fontWeight: 'bold', color: '#64ffda' }}>CSS Output</span>
              <button
                onClick={() => navigator.clipboard.writeText(cssOutput)}
                style={{
                  background: 'rgba(100, 255, 218, 0.2)',
                  border: '1px solid #64ffda',
                  color: '#64ffda',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontSize: '10px',
                  cursor: 'pointer'
                }}
              >
                Copy
              </button>
            </div>
            <pre style={{
              overflow: 'auto',
              maxHeight: '200px',
              margin: 0,
              padding: '12px',
              background: 'rgba(0, 0, 0, 0.3)',
              borderRadius: '6px',
              lineHeight: 1.4
            }}>
              <code>{cssOutput}</code>
            </pre>
          </div>
        </Html>

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxPolarAngle={Math.PI * 0.8}
          minDistance={2}
          maxDistance={8}
        />
      </Canvas>
    </div>
  );
}
