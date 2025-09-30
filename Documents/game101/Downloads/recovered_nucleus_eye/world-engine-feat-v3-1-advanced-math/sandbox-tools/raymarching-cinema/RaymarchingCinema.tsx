import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree, extend } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { useControls, button, folder } from "leva";
import { mathEngine } from '../shared/utils';

// =============================================================================
// Types and Interfaces
// =============================================================================

interface ShaderUniforms {
  time: { value: number };
  resolution: { value: THREE.Vector2 };
  audioLevel: { value: number };
  complexity: { value: number };
  distortion: { value: number };
  zoom: { value: number };
  speed: { value: number };
  beatIntensity: { value: number };
  freqResponse: { value: number };
  colorA: { value: THREE.Color };
  colorB: { value: THREE.Color };
  colorC: { value: THREE.Color };
  mouse: { value: THREE.Vector2 };
}

interface ShaderDefinition {
  name: string;
  vertex: string;
  fragment: string;
  description: string;
}

interface ColorPalette {
  name: string;
  colorA: string;
  colorB: string;
  colorC: string;
}

interface RecordingState {
  isRecording: boolean;
  mediaRecorder: MediaRecorder | null;
  recordedChunks: Blob[];
  videoUrl: string | null;
  duration: number;
}

interface AudioAnalysisState {
  analyser: AnalyserNode | null;
  dataArray: Uint8Array | null;
  audioContext: AudioContext | null;
  source: MediaStreamAudioSourceNode | null;
  level: number;
  frequency: number;
}

// =============================================================================
// Color Palettes
// =============================================================================

const COLOR_PALETTES: ColorPalette[] = [
  {
    name: "Neon Cyberpunk",
    colorA: "#ff0080",
    colorB: "#00ffff",
    colorC: "#ff00ff"
  },
  {
    name: "Cosmic Deep",
    colorA: "#4a0e4e",
    colorB: "#81689d",
    colorC: "#ffd700"
  },
  {
    name: "Fire & Ice",
    colorA: "#ff4500",
    colorB: "#00bfff",
    colorC: "#ff69b4"
  },
  {
    name: "Ocean Dreams",
    colorA: "#006994",
    colorB: "#47b5ff",
    colorC: "#dceeff"
  },
  {
    name: "Synthwave",
    colorA: "#ff006e",
    colorB: "#8338ec",
    colorC: "#ffbe0b"
  },
  {
    name: "Matrix Green",
    colorA: "#001100",
    colorB: "#00ff41",
    colorC: "#ffffff"
  }
];

// =============================================================================
// Shader Definitions
// =============================================================================

const VERTEX_SHADER = `
  varying vec2 vUv;
  varying vec3 vPosition;
  void main() {
    vUv = uv;
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const SHADERS: Record<string, ShaderDefinition> = {
  fractal: {
    name: "Fractal Dimension",
    description: "Infinite recursive box fractals with audio-reactive distortion",
    vertex: VERTEX_SHADER,
    fragment: `
      uniform float time;
      uniform vec2 resolution;
      uniform float audioLevel;
      uniform float complexity;
      uniform float distortion;
      uniform float zoom;
      uniform float speed;
      uniform float beatIntensity;
      uniform float freqResponse;
      uniform vec3 colorA;
      uniform vec3 colorB;
      uniform vec3 colorC;
      uniform vec2 mouse;
      varying vec2 vUv;

      // Signed distance function for a box
      float sdBox(vec3 p, vec3 b) {
        vec3 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
      }

      // Rotation matrix around Y axis
      mat2 rot(float a) {
        float c = cos(a);
        float s = sin(a);
        return mat2(c, -s, s, c);
      }

      // Main distance field function
      float map(vec3 pos) {
        vec3 q = pos;
        float scale = 1.0;
        float dist = 1000.0;

        // Audio-reactive parameters
        float audioPulse = audioLevel * beatIntensity * 0.01;
        float freqMod = freqResponse * 0.01;

        for (int i = 0; i < 12; i++) {
          if (float(i) >= complexity) break;

          float t = time * speed * 0.3 + float(i) * 0.7;

          // Audio-reactive transformations
          q = abs(q) - vec3(0.5 + sin(t + audioPulse) * 0.3);
          q.xy *= rot(t * 0.2 + audioPulse);
          q.xz *= rot(t * 0.15 + freqMod);

          // Scale factor with audio reactivity
          float scaleStep = 1.2 + audioPulse * 0.3;
          q *= scaleStep;
          scale *= scaleStep;

          // Cycle through coordinate swizzling
          q = q.zxy;

          // Box size with frequency response
          float boxSize = 0.1 + sin(t * 2.0 + freqMod * 10.0) * 0.05;
          float box = sdBox(q, vec3(boxSize)) / scale;
          dist = min(dist, box);
        }

        return dist;
      }

      // Calculate surface normal using gradient
      vec3 calcNormal(vec3 pos) {
        vec2 epsilon = vec2(0.0005, 0.0);
        return normalize(vec3(
          map(pos + epsilon.xyy) - map(pos - epsilon.xyy),
          map(pos + epsilon.yxy) - map(pos - epsilon.yxy),
          map(pos + epsilon.yyx) - map(pos - epsilon.yyx)
        ));
      }

      // Soft shadows
      float calcShadow(vec3 ro, vec3 rd) {
        float res = 1.0;
        float t = 0.02;
        for (int i = 0; i < 32; i++) {
          float h = map(ro + rd * t);
          res = min(res, 8.0 * h / t);
          t += clamp(h, 0.02, 0.2);
          if (res < 0.005 || t > 2.5) break;
        }
        return clamp(res, 0.0, 1.0);
      }

      void main() {
        vec2 uv = (vUv - 0.5) * 2.0;
        uv.x *= resolution.x / resolution.y;

        // Camera setup with mouse interaction
        vec3 ro = vec3(0.0, 0.0, -2.5 / zoom);
        ro.xy += mouse * 0.5;
        vec3 rd = normalize(vec3(uv, 1.0));

        // Audio-reactive camera shake
        float shake = audioLevel * beatIntensity * 0.001;
        ro += sin(time * 20.0) * shake;

        // Distortion effects
        float distortAmount = distortion * 0.01;
        rd.xy += sin(time * speed + rd.z * 10.0) * distortAmount * audioLevel;

        float t = 0.0;
        vec3 col = vec3(0.0);
        vec3 glow = vec3(0.0);

        // Raymarching loop
        for (int i = 0; i < 64; i++) {
          vec3 pos = ro + rd * t;
          float d = map(pos);

          if (d < 0.001) {
            // Surface hit - calculate lighting
            vec3 normal = calcNormal(pos);
            vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
            float light = max(dot(normal, lightDir), 0.0);

            // Shadow calculation
            float shadow = calcShadow(pos + normal * 0.02, lightDir);
            light *= shadow;

            // Ambient occlusion
            float ao = 1.0 - float(i) / 64.0;

            // Color mixing based on position and audio
            vec3 baseCol = mix(colorA, colorB, sin(t * 0.1 + time) * 0.5 + 0.5);
            baseCol = mix(baseCol, colorC, audioLevel * 0.7);

            // Final color with lighting
            col = baseCol * light * ao;
            break;
          }

          // Accumulate glow from distance field
          glow += exp(-d * 8.0) * 0.02 * mix(colorA, colorC, audioLevel);

          t += d * 0.5;
          if (t > 10.0) break;
        }

        // Add glow effect
        col += glow * 0.5;

        // Tone mapping and gamma correction
        col = col / (col + vec3(1.0));
        col = pow(col, vec3(0.4545));

        // Vignette effect
        vec2 vigUv = vUv - 0.5;
        float vignette = 1.0 - dot(vigUv, vigUv) * 0.8;
        col *= vignette;

        gl_FragColor = vec4(col, 1.0);
      }
    `
  },

  tunnel: {
    name: "Infinite Tunnel",
    description: "Hypnotic tunnel with audio-reactive geometry",
    vertex: VERTEX_SHADER,
    fragment: `
      uniform float time;
      uniform vec2 resolution;
      uniform float audioLevel;
      uniform float complexity;
      uniform float distortion;
      uniform float zoom;
      uniform float speed;
      uniform float beatIntensity;
      uniform float freqResponse;
      uniform vec3 colorA;
      uniform vec3 colorB;
      uniform vec3 colorC;
      uniform vec2 mouse;
      varying vec2 vUv;

      float map(vec3 pos) {
        vec3 p = pos;

        // Audio-reactive tunnel parameters
        float audioPulse = audioLevel * beatIntensity * 0.02;
        float freqMod = freqResponse * 0.01;

        // Rotate the tunnel
        float angle = time * speed * 0.2 + audioPulse * 5.0;
        float c = cos(angle);
        float s = sin(angle);
        p.xy = mat2(c, -s, s, c) * p.xy;

        // Tunnel distance with audio modulation
        float tunnelRadius = 1.0 + sin(p.z * 0.2 + time * speed) * 0.2;
        tunnelRadius += audioPulse * 0.5;
        float tunnel = length(p.xy) - tunnelRadius;

        // Add geometric details
        float detail = complexity * 0.1;
        for (int i = 0; i < 8; i++) {
          if (float(i) >= complexity) break;

          float freq = pow(2.0, float(i)) * detail;
          float phase = time * speed * freq + freqMod * 10.0;
          tunnel += sin(p.z * freq + phase) * (0.1 / pow(2.0, float(i)));
        }

        return tunnel;
      }

      vec3 calcNormal(vec3 pos) {
        vec2 e = vec2(0.001, 0.0);
        return normalize(vec3(
          map(pos + e.xyy) - map(pos - e.xyy),
          map(pos + e.yxy) - map(pos - e.yxy),
          map(pos + e.yyx) - map(pos - e.yyx)
        ));
      }

      void main() {
        vec2 uv = (vUv - 0.5) * 2.0;
        uv.x *= resolution.x / resolution.y;

        // Camera moving through tunnel
        vec3 ro = vec3(0.0, 0.0, -time * speed * 2.0);
        ro.xy += mouse * 0.3;
        vec3 rd = normalize(vec3(uv, zoom));

        // Audio-reactive distortion
        float distortAmount = distortion * 0.01;
        rd.xy += sin(time * speed * 3.0 + rd.z * 5.0) * distortAmount * audioLevel;

        float t = 0.0;
        vec3 col = vec3(0.0);

        for (int i = 0; i < 64; i++) {
          vec3 pos = ro + rd * t;
          float d = map(pos);

          if (d < 0.001) {
            vec3 normal = calcNormal(pos);

            // Lighting based on tunnel position
            float light = abs(normal.z);

            // Color based on tunnel depth and audio
            float depth = fract(pos.z * 0.1);
            vec3 baseCol = mix(colorA, colorB, depth);
            baseCol = mix(baseCol, colorC, audioLevel);

            col = baseCol * light;
            break;
          }

          // Glow from tunnel walls
          col += exp(-d * 2.0) * 0.01 * colorC;

          t += d;
          if (t > 20.0) break;
        }

        // Add scanlines effect
        float scanlines = sin(vUv.y * resolution.y * 0.5) * 0.1 + 0.9;
        col *= scanlines;

        // Gamma correction
        col = pow(col, vec3(0.4545));

        gl_FragColor = vec4(col, 1.0);
      }
    `
  },

  galaxy: {
    name: "Galactic Spiral",
    description: "Swirling galaxy with stellar formation patterns",
    vertex: VERTEX_SHADER,
    fragment: `
      uniform float time;
      uniform vec2 resolution;
      uniform float audioLevel;
      uniform float complexity;
      uniform float distortion;
      uniform float zoom;
      uniform float speed;
      uniform float beatIntensity;
      uniform float freqResponse;
      uniform vec3 colorA;
      uniform vec3 colorB;
      uniform vec3 colorC;
      uniform vec2 mouse;
      varying vec2 vUv;

      // Hash function for noise
      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }

      // Noise function
      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
                   mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
      }

      // Fractal brownian motion
      float fbm(vec2 p, int octaves) {
        float value = 0.0;
        float amplitude = 0.5;
        float frequency = 1.0;

        for (int i = 0; i < 8; i++) {
          if (i >= octaves) break;
          value += amplitude * noise(p * frequency);
          frequency *= 2.0;
          amplitude *= 0.5;
        }
        return value;
      }

      void main() {
        vec2 uv = (vUv - 0.5) * 2.0;
        uv.x *= resolution.x / resolution.y;

        // Apply zoom and mouse interaction
        uv /= zoom;
        uv += mouse * 0.5;

        // Audio-reactive parameters
        float audioPulse = audioLevel * beatIntensity * 0.1;
        float freqMod = freqResponse * 0.01;

        // Galaxy rotation
        float angle = time * speed * 0.1 + audioPulse;
        float c = cos(angle);
        float s = sin(angle);
        uv = mat2(c, -s, s, c) * uv;

        // Convert to polar coordinates
        float r = length(uv);
        float theta = atan(uv.y, uv.x);

        // Galaxy spiral pattern
        float spiral = sin(theta * 2.0 - r * 3.0 + time * speed) * 0.5 + 0.5;
        spiral += sin(theta * 4.0 - r * 1.5 + time * speed * 0.7) * 0.3;

        // Density based on distance from center
        float density = exp(-r * 2.0) * (1.0 + audioPulse * 2.0);

        // Noise for stellar formation
        vec2 noisePos = uv * 5.0 + time * speed * 0.2;
        float stars = fbm(noisePos, int(complexity));

        // Distortion effects
        float distortAmount = distortion * 0.01;
        vec2 distortedUv = uv + sin(uv * 10.0 + time * speed) * distortAmount;
        float distortedStars = fbm(distortedUv * 3.0, int(complexity * 0.5));

        // Combine spiral and stellar patterns
        float galaxy = spiral * density * stars;
        galaxy += distortedStars * 0.3 * density;

        // Color mixing
        vec3 centerColor = mix(colorB, colorC, audioPulse);
        vec3 armColor = mix(colorA, colorB, sin(theta + time * speed) * 0.5 + 0.5);
        vec3 starColor = colorC;

        vec3 col = vec3(0.0);

        // Galaxy arms
        col += armColor * spiral * density * 0.8;

        // Central bulge
        float bulge = exp(-r * 8.0) * (1.0 + audioPulse);
        col += centerColor * bulge;

        // Individual stars
        float starField = step(0.85, stars) * density;
        col += starColor * starField * (1.0 + freqMod * 5.0);

        // Add nebula-like clouds
        float nebula = fbm(uv * 2.0 + time * speed * 0.1, int(complexity * 0.7));
        col += mix(colorA, colorB, nebula) * nebula * density * 0.3;

        // Enhance with audio reactivity
        col *= 1.0 + audioLevel * 0.5;

        // Add some cosmic dust
        float dust = noise(uv * 20.0 + time * speed * 0.05);
        col += dust * density * 0.1 * colorA;

        // Gamma correction and tone mapping
        col = col / (col + vec3(1.0));
        col = pow(col, vec3(0.4545));

        gl_FragColor = vec4(col, 1.0);
      }
    `
  },

  crystal: {
    name: "Crystal Lattice",
    description: "Crystalline structures with geometric precision",
    vertex: VERTEX_SHADER,
    fragment: `
      uniform float time;
      uniform vec2 resolution;
      uniform float audioLevel;
      uniform float complexity;
      uniform float distortion;
      uniform float zoom;
      uniform float speed;
      uniform float beatIntensity;
      uniform float freqResponse;
      uniform vec3 colorA;
      uniform vec3 colorB;
      uniform vec3 colorC;
      uniform vec2 mouse;
      varying vec2 vUv;

      // Rotation matrices
      mat3 rotX(float a) {
        float c = cos(a);
        float s = sin(a);
        return mat3(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
      }

      mat3 rotY(float a) {
        float c = cos(a);
        float s = sin(a);
        return mat3(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
      }

      mat3 rotZ(float a) {
        float c = cos(a);
        float s = sin(a);
        return mat3(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
      }

      // Signed distance functions
      float sdOctahedron(vec3 p, float s) {
        p = abs(p);
        float m = p.x + p.y + p.z - s;
        vec3 q;
        if (3.0 * p.x < m) q = p.xyz;
        else if (3.0 * p.y < m) q = p.yzx;
        else if (3.0 * p.z < m) q = p.zxy;
        else return m * 0.57735027;

        float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
        return length(vec3(q.x, q.y - s + k, q.z - k));
      }

      float sdBox(vec3 p, vec3 b) {
        vec3 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
      }

      // Crystal lattice function
      float map(vec3 pos) {
        vec3 p = pos;

        // Audio-reactive parameters
        float audioPulse = audioLevel * beatIntensity * 0.05;
        float freqMod = freqResponse * 0.02;

        // Rotation animation
        p *= rotX(time * speed * 0.2 + audioPulse);
        p *= rotY(time * speed * 0.15 + freqMod * 5.0);
        p *= rotZ(time * speed * 0.1);

        float scale = 1.0;
        float dist = 1000.0;

        // Fractal crystal structure
        for (int i = 0; i < 8; i++) {
          if (float(i) >= complexity) break;

          float t = time * speed * 0.3 + float(i) * 0.5;

          // Fold space for crystal symmetry
          p = abs(p) - vec3(0.3 + sin(t + audioPulse) * 0.1);

          // Crystal rotations
          p *= rotX(t * 0.7 + freqMod * 3.0);
          p *= rotY(t * 0.5);

          // Scale transformation
          float scaleStep = 1.3 + audioPulse * 0.2;
          p *= scaleStep;
          scale *= scaleStep;

          // Alternate between octahedron and box
          float crystal;
          if (mod(float(i), 2.0) < 1.0) {
            crystal = sdOctahedron(p, 0.5) / scale;
          } else {
            crystal = sdBox(p, vec3(0.3)) / scale;
          }

          dist = min(dist, crystal);

          // Coordinate permutation
          p = p.zxy;
        }

        return dist;
      }

      vec3 calcNormal(vec3 pos) {
        vec2 e = vec2(0.001, 0.0);
        return normalize(vec3(
          map(pos + e.xyy) - map(pos - e.xyy),
          map(pos + e.yxy) - map(pos - e.yxy),
          map(pos + e.yyx) - map(pos - e.yyx)
        ));
      }

      // Fresnel reflectance
      float fresnel(float cosTheta, float F0) {
        return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
      }

      void main() {
        vec2 uv = (vUv - 0.5) * 2.0;
        uv.x *= resolution.x / resolution.y;

        // Camera setup
        vec3 ro = vec3(0.0, 0.0, -3.0 / zoom);
        ro.xy += mouse * 0.5;
        vec3 rd = normalize(vec3(uv, 1.0));

        // Audio-reactive camera shake
        float shake = audioLevel * beatIntensity * 0.002;
        ro += sin(time * 50.0) * shake;

        // Distortion
        float distortAmount = distortion * 0.01;
        rd.xy += sin(time * speed * 2.0 + rd.z * 8.0) * distortAmount * audioLevel;

        float t = 0.0;
        vec3 col = vec3(0.0);
        vec3 glow = vec3(0.0);

        // Raymarching
        for (int i = 0; i < 64; i++) {
          vec3 pos = ro + rd * t;
          float d = map(pos);

          if (d < 0.001) {
            vec3 normal = calcNormal(pos);
            vec3 viewDir = -rd;

            // Multiple light sources
            vec3 lightDir1 = normalize(vec3(1.0, 1.0, -1.0));
            vec3 lightDir2 = normalize(vec3(-1.0, 0.5, 1.0));
            vec3 lightDir3 = normalize(vec3(0.0, -1.0, 0.5));

            float light1 = max(dot(normal, lightDir1), 0.0);
            float light2 = max(dot(normal, lightDir2), 0.0);
            float light3 = max(dot(normal, lightDir3), 0.0);

            // Fresnel effect for crystal-like appearance
            float F = fresnel(dot(normal, viewDir), 0.04);

            // Reflection
            vec3 reflectDir = reflect(rd, normal);
            float spec = pow(max(dot(reflectDir, lightDir1), 0.0), 16.0);

            // Color based on position and lighting
            vec3 baseCol = mix(colorA, colorB, sin(pos.x + pos.y + pos.z + time) * 0.5 + 0.5);
            baseCol = mix(baseCol, colorC, F);

            // Combine lighting
            float totalLight = light1 * 0.6 + light2 * 0.3 + light3 * 0.1;
            col = baseCol * totalLight + colorC * spec * 0.5;

            // Audio enhancement
            col *= 1.0 + audioLevel * 0.3;

            break;
          }

          // Crystal glow
          glow += exp(-d * 3.0) * 0.03 * mix(colorB, colorC, audioLevel);

          t += d * 0.7;
          if (t > 15.0) break;
        }

        // Add glow
        col += glow;

        // Crystal refraction-like effect
        vec2 refractUv = uv + sin(uv * 10.0 + time * speed) * 0.02 * audioLevel;
        float refract = length(refractUv) * 0.1;
        col += colorA * refract * 0.3;

        // Tone mapping
        col = col / (col + vec3(1.0));
        col = pow(col, vec3(0.4545));

        gl_FragColor = vec4(col, 1.0);
      }
    `
  }
};

// =============================================================================
// Audio Analysis Hook
// =============================================================================

function useAudioAnalysis(useMicrophone: boolean): AudioAnalysisState {
  const [audioState, setAudioState] = useState<AudioAnalysisState>({
    analyser: null,
    dataArray: null,
    audioContext: null,
    source: null,
    level: 0,
    frequency: 0
  });

  useEffect(() => {
    let animationId: number;

    const initAudio = async () => {
      if (!useMicrophone) {
        // Simulate audio with sine waves
        const simulate = () => {
          const time = Date.now() * 0.001;
          setAudioState(prev => ({
            ...prev,
            level: (Math.sin(time * 2) + Math.sin(time * 3.7) + Math.sin(time * 5.3)) / 3 * 0.5 + 0.5,
            frequency: Math.sin(time * 0.7) * 0.5 + 0.5
          }));
          animationId = requestAnimationFrame(simulate);
        };
        simulate();
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();

        analyser.fftSize = 256;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        source.connect(analyser);

        setAudioState({
          analyser,
          dataArray,
          audioContext,
          source,
          level: 0,
          frequency: 0
        });

        const updateAudio = () => {
          if (analyser && dataArray) {
            analyser.getByteFrequencyData(dataArray);

            // Calculate average level
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            const level = average / 255;

            // Calculate frequency response (focus on mid frequencies)
            const midStart = Math.floor(dataArray.length * 0.2);
            const midEnd = Math.floor(dataArray.length * 0.6);
            const midFreqs = dataArray.slice(midStart, midEnd);
            const frequency = midFreqs.reduce((a, b) => a + b) / midFreqs.length / 255;

            setAudioState(prev => ({ ...prev, level, frequency }));
          }
          animationId = requestAnimationFrame(updateAudio);
        };
        updateAudio();

      } catch (error) {
        console.warn('Microphone access denied, using simulation');
        // Fallback to simulation
        const simulate = () => {
          const time = Date.now() * 0.001;
          setAudioState(prev => ({
            ...prev,
            level: Math.random() * 0.8 + 0.2,
            frequency: Math.random() * 0.6 + 0.2
          }));
          animationId = requestAnimationFrame(simulate);
        };
        simulate();
      }
    };

    initAudio();

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
      if (audioState.audioContext) {
        audioState.audioContext.close();
      }
    };
  }, [useMicrophone]);

  return audioState;
}

// =============================================================================
// Raymarching Material Component
// =============================================================================

function RaymarchingMaterial({
  shaderType,
  palette,
  audioLevel,
  freqLevel,
  uniforms
}: {
  shaderType: string;
  palette: ColorPalette;
  audioLevel: number;
  freqLevel: number;
  uniforms: any;
}) {
  const materialRef = useRef<THREE.ShaderMaterial>(null!);
  const { viewport, mouse } = useThree();

  const shaderMaterial = useMemo(() => {
    const shader = SHADERS[shaderType];

    return new THREE.ShaderMaterial({
      vertexShader: shader.vertex,
      fragmentShader: shader.fragment,
      uniforms: {
        ...uniforms,
        resolution: { value: new THREE.Vector2(viewport.width * 100, viewport.height * 100) },
        colorA: { value: new THREE.Color(palette.colorA) },
        colorB: { value: new THREE.Color(palette.colorB) },
        colorC: { value: new THREE.Color(palette.colorC) },
        mouse: { value: new THREE.Vector2(0, 0) },
        audioLevel: { value: 0 },
        freqResponse: { value: 0 }
      }
    });
  }, [shaderType, palette]);

  useFrame((state) => {
    if (materialRef.current) {
      const material = materialRef.current;
      material.uniforms.time.value = state.clock.elapsedTime;
      material.uniforms.audioLevel.value = audioLevel;
      material.uniforms.freqResponse.value = freqLevel;
      material.uniforms.mouse.value.set(
        mouse.x * 0.5,
        mouse.y * 0.5
      );
      material.uniforms.resolution.value.set(
        viewport.width * 100,
        viewport.height * 100
      );
    }
  });

  return (
    <shaderMaterial
      ref={materialRef}
      attach="material"
      args={[shaderMaterial]}
    />
  );
}

// =============================================================================
// Recording Hook
// =============================================================================

function useRecording(canvasRef: React.RefObject<HTMLCanvasElement>) {
  const [recordingState, setRecordingState] = useState<RecordingState>({
    isRecording: false,
    mediaRecorder: null,
    recordedChunks: [],
    videoUrl: null,
    duration: 10
  });

  const startRecording = useCallback((duration: number, quality: string) => {
    if (!canvasRef.current || recordingState.isRecording) return;

    // Set recording resolution based on quality
    const resolutions = {
      '720p': { width: 1280, height: 720 },
      '1080p': { width: 1920, height: 1080 },
      '4k': { width: 3840, height: 2160 }
    };

    const res = resolutions[quality as keyof typeof resolutions] || resolutions['1080p'];

    try {
      const stream = canvasRef.current.captureStream(30);
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9'
      });

      const chunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const videoUrl = URL.createObjectURL(blob);

        setRecordingState(prev => ({
          ...prev,
          isRecording: false,
          mediaRecorder: null,
          recordedChunks: chunks,
          videoUrl
        }));
      };

      mediaRecorder.start();

      setRecordingState(prev => ({
        ...prev,
        isRecording: true,
        mediaRecorder,
        recordedChunks: [],
        duration
      }));

      // Stop recording after duration
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }, duration * 1000);

    } catch (error) {
      console.error('Failed to start recording:', error);
    }
  }, [canvasRef, recordingState.isRecording]);

  const downloadVideo = useCallback(() => {
    if (recordingState.videoUrl) {
      const a = document.createElement('a');
      a.href = recordingState.videoUrl;
      a.download = `raymarching-cinema-${Date.now()}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [recordingState.videoUrl]);

  return {
    ...recordingState,
    startRecording,
    downloadVideo
  };
}

// =============================================================================
// Text Overlay Component
// =============================================================================

interface TextOverlayProps {
  title: string;
  subtitle: string;
  visible: boolean;
}

function TextOverlay({ title, subtitle, visible }: TextOverlayProps) {
  if (!visible) return null;

  return (
    <Html
      center
      distanceFactor={1}
      style={{
        pointerEvents: 'none',
        userSelect: 'none'
      }}
    >
      <div style={{
        textAlign: 'center',
        fontFamily: 'JetBrains Mono, monospace',
        textShadow: '0 0 10px rgba(0,0,0,0.8)'
      }}>
        <div style={{
          fontSize: '3rem',
          fontWeight: 'bold',
          background: 'linear-gradient(45deg, #ff0080, #00ff80, #8000ff)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '0.5rem',
          textTransform: 'uppercase',
          letterSpacing: '0.2em'
        }}>
          {title}
        </div>
        <div style={{
          fontSize: '1.5rem',
          color: '#00ff80',
          fontWeight: '600',
          textTransform: 'uppercase',
          letterSpacing: '0.3em'
        }}>
          {subtitle}
        </div>
      </div>
    </Html>
  );
}

// =============================================================================
// Main Raymarching Cinema Component
// =============================================================================

export default function RaymarchingCinema() {
  const canvasRef = useRef<HTMLCanvasElement>(null!);

  // Control states
  const [currentShader, setCurrentShader] = useState('fractal');
  const [currentPalette, setCurrentPalette] = useState(COLOR_PALETTES[0]);
  const [useMicrophone, setUseMicrophone] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [titleText, setTitleText] = useState('RAYMARCHED');
  const [subtitleText, setSubtitleText] = useState('CINEMA');

  // Audio analysis
  const audioState = useAudioAnalysis(useMicrophone);

  // Recording functionality
  const recording = useRecording(canvasRef);

  // Shader uniforms
  const shaderUniforms = useMemo(() => ({
    time: { value: 0 },
    complexity: { value: 8 },
    distortion: { value: 30 },
    zoom: { value: 1 },
    speed: { value: 1 },
    beatIntensity: { value: 75 },
    freqResponse: { value: 60 }
  }), []);

  // Leva controls
  const controls = useControls({
    'Scene': folder({
      shader: {
        value: currentShader,
        options: Object.keys(SHADERS),
        onChange: (value) => setCurrentShader(value)
      },
      palette: {
        value: currentPalette.name,
        options: COLOR_PALETTES.map(p => p.name),
        onChange: (value) => {
          const palette = COLOR_PALETTES.find(p => p.name === value);
          if (palette) setCurrentPalette(palette);
        }
      }
    }),

    'Audio': folder({
      useMicrophone: {
        value: useMicrophone,
        onChange: setUseMicrophone
      },
      beatIntensity: {
        value: shaderUniforms.beatIntensity.value,
        min: 0,
        max: 100,
        onChange: (value) => { shaderUniforms.beatIntensity.value = value; }
      },
      freqResponse: {
        value: shaderUniforms.freqResponse.value,
        min: 0,
        max: 100,
        onChange: (value) => { shaderUniforms.freqResponse.value = value; }
      }
    }),

    'Visual': folder({
      complexity: {
        value: shaderUniforms.complexity.value,
        min: 1,
        max: 12,
        step: 1,
        onChange: (value) => { shaderUniforms.complexity.value = value; }
      },
      speed: {
        value: shaderUniforms.speed.value,
        min: 0.1,
        max: 3,
        step: 0.1,
        onChange: (value) => { shaderUniforms.speed.value = value; }
      },
      distortion: {
        value: shaderUniforms.distortion.value,
        min: 0,
        max: 100,
        onChange: (value) => { shaderUniforms.distortion.value = value; }
      },
      zoom: {
        value: shaderUniforms.zoom.value,
        min: 0.1,
        max: 5,
        step: 0.1,
        onChange: (value) => { shaderUniforms.zoom.value = value; }
      }
    }),

    'Text Overlay': folder({
      showOverlay: {
        value: showOverlay,
        onChange: setShowOverlay
      },
      title: {
        value: titleText,
        onChange: setTitleText
      },
      subtitle: {
        value: subtitleText,
        onChange: setSubtitleText
      }
    }),

    'Recording': folder({
      duration: {
        value: recording.duration,
        min: 3,
        max: 60,
        step: 1
      },
      'Record Video': button(() => {
        recording.startRecording(controls.duration, '1080p');
      }),
      'Download': button(() => {
        recording.downloadVideo();
      }, { disabled: !recording.videoUrl })
    }),

    'Audio Levels': folder({
      'Audio Level': {
        value: audioState.level,
        min: 0,
        max: 1,
        disabled: true
      },
      'Frequency': {
        value: audioState.frequency,
        min: 0,
        max: 1,
        disabled: true
      }
    })
  });

  return (
    <div style={{ width: '100%', height: '100vh', background: '#000' }}>
      <Canvas
        ref={canvasRef}
        camera={{ position: [0, 0, 1], fov: 60 }}
        dpr={window.devicePixelRatio}
      >
        <mesh scale={[2, 2, 1]}>
          <planeGeometry />
          <RaymarchingMaterial
            shaderType={currentShader}
            palette={currentPalette}
            audioLevel={audioState.level}
            freqLevel={audioState.frequency}
            uniforms={shaderUniforms}
          />
        </mesh>

        <TextOverlay
          title={titleText}
          subtitle={subtitleText}
          visible={showOverlay}
        />

        <OrbitControls enableZoom={false} enablePan={false} enableRotate={false} />
      </Canvas>

      {/* Recording indicator */}
      {recording.isRecording && (
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          background: '#ff0000',
          color: '#fff',
          padding: '8px 16px',
          borderRadius: '20px',
          fontSize: '14px',
          fontWeight: 'bold',
          fontFamily: 'JetBrains Mono, monospace',
          animation: 'pulse 1s infinite',
          zIndex: 1000
        }}>
          ● REC
        </div>
      )}

      {/* Video preview */}
      {recording.videoUrl && (
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          background: 'rgba(0,0,0,0.8)',
          padding: '10px',
          borderRadius: '8px',
          zIndex: 1000
        }}>
          <video
            src={recording.videoUrl}
            controls
            style={{
              width: '200px',
              height: 'auto',
              borderRadius: '4px'
            }}
          />
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
