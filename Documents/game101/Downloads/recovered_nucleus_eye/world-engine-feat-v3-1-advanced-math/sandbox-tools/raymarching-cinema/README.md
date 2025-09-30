# Raymarching Cinema - Advanced Procedural Visual Generator

## Overview

Raymarching Cinema is a sophisticated real-time visual generation system that combines mathematical raymarching techniques with audio-reactive parameters to create cinematic procedural content. Built with React Three.js and advanced GLSL shaders, it offers four distinct visual environments with full video recording capabilities.

## Features

### 🎬 Visual Environments
- **Fractal Dimension**: Infinite recursive box fractals with audio-reactive distortion
- **Infinite Tunnel**: Hypnotic tunnel structures with geometric detail layers
- **Galactic Spiral**: Swirling galaxy with stellar formation patterns and nebulae
- **Crystal Lattice**: Crystalline structures with geometric precision and fresnel effects

### 🎵 Audio Integration
- **Microphone Input**: Real-time audio analysis from user's microphone
- **Frequency Analysis**: FFT-based frequency response with mid-range focus
- **Beat Detection**: Audio level analysis for rhythmic visual response
- **Simulation Mode**: Procedural audio simulation when microphone unavailable

### 🎥 Recording System
- **High-Quality Recording**: WebM video capture with VP9 codec
- **Multiple Resolutions**: 720p, 1080p, and 4K recording options
- **Canvas Streaming**: Direct canvas-to-video stream capture
- **Auto-Download**: Automatic file download with timestamp naming

### 🎨 Visual Controls
- **Color Palettes**: 6 pre-designed color schemes (Cyberpunk, Cosmic, Fire & Ice, etc.)
- **Real-time Parameters**: Complexity, speed, distortion, zoom controls
- **Text Overlays**: Customizable title and subtitle text rendering
- **Interactive Controls**: Mouse interaction and camera positioning

## Mathematical Foundations

### Raymarching Algorithm
The core rendering technique uses raymarching (sphere tracing) to render implicit surfaces:

```glsl
float t = 0.0;
for (int i = 0; i < MAX_STEPS; i++) {
  vec3 pos = rayOrigin + rayDirection * t;
  float dist = sceneDistanceFunction(pos);

  if (dist < EPSILON) {
    // Surface hit - calculate lighting
    break;
  }

  t += dist * STEP_FACTOR;
  if (t > MAX_DISTANCE) break;
}
```

### Distance Field Functions
Each shader implements specialized signed distance functions:

#### Fractal Dimension
```glsl
float map(vec3 pos) {
  vec3 q = pos;
  float scale = 1.0;
  float dist = 1000.0;

  for (int i = 0; i < iterations; i++) {
    q = abs(q) - vec3(0.5 + audioModulation);
    q *= scaleStep + audioPulse;
    scale *= scaleStep;

    float box = sdBox(q, vec3(size)) / scale;
    dist = min(dist, box);
  }

  return dist;
}
```

#### Crystal Lattice
Uses octahedron and box combinations with space folding:
```glsl
float sdOctahedron(vec3 p, float s) {
  p = abs(p);
  float m = p.x + p.y + p.z - s;
  // Octahedron distance calculation...
}
```

### Audio Analysis Mathematics
Real-time FFT analysis for audio reactivity:

```typescript
// Frequency domain analysis
analyser.getByteFrequencyData(dataArray);

// Calculate RMS level
const rms = Math.sqrt(
  dataArray.reduce((sum, val) => sum + val * val, 0) / dataArray.length
) / 255;

// Mid-frequency response (200-2000 Hz range)
const midStart = Math.floor(dataArray.length * 0.2);
const midEnd = Math.floor(dataArray.length * 0.6);
const midFreqs = dataArray.slice(midStart, midEnd);
```

## Usage

### Basic Implementation

```tsx
import RaymarchingCinema from './RaymarchingCinema';

function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <RaymarchingCinema />
    </div>
  );
}
```

### Advanced Configuration

```tsx
// Custom shader parameters
const customControls = {
  complexity: 10,
  speed: 1.5,
  distortion: 45,
  zoom: 2.0,
  beatIntensity: 80,
  freqResponse: 70
};
```

## Shader Types

### 1. Fractal Dimension
**Mathematical Basis**: Iterated Function Systems (IFS)
- Recursive box folding transformations
- Audio-reactive scale factors
- Infinite detail at multiple scales
- Performance: 64 raymarching steps

**Key Features**:
- Box-folding operations: `q = abs(q) - offset`
- Dynamic rotation matrices with audio modulation
- Exponential scaling with `scaleStep = 1.2 + audioPulse`
- Color mixing based on iteration depth

### 2. Infinite Tunnel
**Mathematical Basis**: Cylindrical coordinate transformations
- Tube distance field with radial modulation
- Layered geometric details using harmonic series
- Perspective-correct depth illusion

**Key Features**:
- Base tunnel: `distance = length(p.xy) - radius`
- Harmonic detail layers: `detail += sin(p.z * freq + phase) / amplitude`
- Forward motion simulation through Z-translation
- Scanline post-processing effects

### 3. Galactic Spiral
**Mathematical Basis**: Polar coordinate systems and noise functions
- Spiral arm generation using parametric equations
- Fractal Brownian Motion (FBM) for stellar distribution
- Density falloff with exponential decay

**Key Features**:
- Spiral equation: `spiral = sin(theta * arms - radius * pitch + time)`
- Multi-octave noise: `fbm(p) = Σ(amplitude * noise(p * frequency))`
- Stellar formation: `stars = step(threshold, fbm(position))`
- Central bulge with gaussian distribution

### 4. Crystal Lattice
**Mathematical Basis**: Crystallographic symmetry operations
- Space group transformations
- Polyhedra distance functions (octahedron, box)
- Fresnel reflectance calculations

**Key Features**:
- Space folding: `p = abs(p) - offset` for crystal symmetry
- Alternating polyhedra: octahedron and box primitives
- Fresnel reflectance: `F = F0 + (1-F0) * (1-cosTheta)^5`
- Multiple light sources for crystal-like appearance

## Audio Integration

### Microphone Analysis
```typescript
const initAudioAnalysis = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext();
  const analyser = audioContext.createAnalyser();

  analyser.fftSize = 256;
  const dataArray = new Uint8Array(analyser.frequencyBinCount);

  // Real-time analysis loop
  const updateAudio = () => {
    analyser.getByteFrequencyData(dataArray);
    const audioLevel = dataArray.reduce((a, b) => a + b) / dataArray.length / 255;
    // Update shader uniforms...
  };
};
```

### Audio-Reactive Parameters
- **Beat Intensity**: Controls amplitude of geometric transformations
- **Frequency Response**: Modulates color and detail parameters
- **Real-time Modulation**: Direct mapping to shader uniforms
- **Fallback Simulation**: Sine wave generation when mic unavailable

## Color Palette System

### Pre-defined Palettes
```typescript
const PALETTES = {
  cyberpunk: { colorA: '#ff0080', colorB: '#00ffff', colorC: '#ff00ff' },
  cosmic: { colorA: '#4a0e4e', colorB: '#81689d', colorC: '#ffd700' },
  // ... additional palettes
};
```

### Dynamic Color Mixing
Shaders use three-color interpolation:
```glsl
vec3 baseColor = mix(colorA, colorB, spatialPattern);
baseColor = mix(baseColor, colorC, audioLevel);
```

## Recording System

### WebRTC Canvas Capture
```typescript
const startRecording = (duration: number, quality: string) => {
  const stream = canvas.captureStream(30); // 30 FPS
  const mediaRecorder = new MediaRecorder(stream, {
    mimeType: 'video/webm;codecs=vp9'
  });

  // Configure recording parameters based on quality
  const resolutions = {
    '720p': { width: 1280, height: 720 },
    '1080p': { width: 1920, height: 1080 },
    '4k': { width: 3840, height: 2160 }
  };
};
```

### Export Capabilities
- **Video Format**: WebM with VP9 codec
- **Frame Rate**: 30 FPS capture rate
- **Quality Options**: 720p, 1080p, 4K resolution
- **Duration Control**: 3-60 second recordings
- **Auto-naming**: Timestamp-based file naming

## Text Overlay System

### HTML/CSS Text Rendering
```tsx
const TextOverlay = ({ title, subtitle, visible }) => (
  <Html center distanceFactor={1}>
    <div style={{
      fontFamily: 'JetBrains Mono',
      background: 'linear-gradient(45deg, #ff0080, #00ff80, #8000ff)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent'
    }}>
      {title}
    </div>
  </Html>
);
```

## Performance Optimization

### Shader Optimization
- **Early Ray Termination**: Distance-based early exits
- **Adaptive Step Size**: `t += distance * stepFactor`
- **LOD System**: Complexity reduction at distance
- **Uniform Caching**: Minimize uniform updates

### JavaScript Optimization
```typescript
// Use RAF for smooth animation
const animate = () => {
  updateUniforms();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
};

// Efficient audio analysis
const analyzeAudio = useMemo(() => {
  // Heavy audio processing
}, [audioContext, analyser]);
```

## Integration with Sandbox Ecosystem

### Shared Utilities Usage
```typescript
import { mathEngine } from '../shared/utils';

// Use enhanced math functions
const interpolatedValue = mathEngine.smoothstep(0, 1, audioLevel);
const rotationMatrix = mathEngine.rotationMatrix(angle);
```

### TypeScript Integration
```typescript
interface RaymarchingControls {
  shader: keyof typeof SHADERS;
  complexity: number;
  audioReactive: boolean;
  palette: ColorPalette;
}
```

## API Reference

### Core Components

#### `RaymarchingCinema`
Main component with full functionality
```tsx
<RaymarchingCinema />
```

#### `RaymarchingMaterial`
Shader material component
```tsx
<RaymarchingMaterial
  shaderType="fractal"
  palette={palette}
  audioLevel={0.5}
  uniforms={shaderUniforms}
/>
```

### Hooks

#### `useAudioAnalysis(useMicrophone: boolean)`
Audio analysis hook with microphone support
```typescript
const { level, frequency, analyser } = useAudioAnalysis(true);
```

#### `useRecording(canvasRef: RefObject<HTMLCanvasElement>)`
Video recording hook
```typescript
const { startRecording, downloadVideo, isRecording } = useRecording(canvasRef);
```

## Advanced Features

### Custom Shader Development
Add new shaders to the SHADERS object:
```typescript
const customShader: ShaderDefinition = {
  name: "Custom Effect",
  description: "Your custom shader description",
  vertex: VERTEX_SHADER,
  fragment: `
    // Your custom fragment shader code
    uniform float time;
    // ... shader implementation
  `
};

SHADERS.custom = customShader;
```

### Audio Reactive Extensions
```typescript
// Custom audio processing
const processAudio = (dataArray: Uint8Array) => {
  // Bass analysis (0-200 Hz)
  const bassStart = 0;
  const bassEnd = Math.floor(dataArray.length * 0.1);
  const bass = dataArray.slice(bassStart, bassEnd);

  // Treble analysis (2000+ Hz)
  const trebleStart = Math.floor(dataArray.length * 0.6);
  const treble = dataArray.slice(trebleStart);

  return { bass: average(bass), treble: average(treble) };
};
```

## Troubleshooting

### Common Issues

#### Audio Not Working
- Check browser permissions for microphone access
- Ensure HTTPS context for getUserMedia API
- Verify audio context initialization

#### Performance Issues
- Reduce complexity parameter for better framerate
- Lower resolution for recording if needed
- Check GPU compatibility with WebGL shaders

#### Recording Problems
- Verify browser support for MediaRecorder API
- Check available codecs with `MediaRecorder.isTypeSupported()`
- Ensure sufficient disk space for video files

### Debug Tools
```typescript
// Shader compilation debugging
const checkShaderCompilation = (shader: WebGLShader, gl: WebGLRenderingContext) => {
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
  }
};

// Performance monitoring
const stats = new Stats();
document.body.appendChild(stats.dom);
```

## Browser Compatibility

### Required APIs
- **WebGL 2.0**: For advanced shader features
- **MediaRecorder API**: For video recording
- **getUserMedia**: For microphone access
- **Canvas.captureStream()**: For video capture

### Supported Browsers
- Chrome 60+ (full support)
- Firefox 70+ (full support)
- Safari 14+ (limited recording support)
- Edge 80+ (full support)

## Mathematical References

### Raymarching Theory
- "Ray Marching and Signed Distance Functions" - Íñigo Quílez
- "Modeling with Distance Functions" - Íñigo Quílez
- "Fractals in Real-Time Rendering" - various sources

### Audio Analysis
- "Real-time Audio Analysis" - Web Audio API documentation
- "FFT and Frequency Domain Analysis"
- "Audio Visualization Techniques"

---

Raymarching Cinema represents the cutting edge of real-time procedural visual generation, combining mathematical precision with artistic expression through advanced shader programming and audio-reactive systems. Perfect for VJ performances, music videos, and interactive installations.

**Total Shader Count**: 4 complete environments
**Mathematical Complexity**: Advanced raymarching and audio analysis
**Performance**: Optimized for 60 FPS real-time rendering
