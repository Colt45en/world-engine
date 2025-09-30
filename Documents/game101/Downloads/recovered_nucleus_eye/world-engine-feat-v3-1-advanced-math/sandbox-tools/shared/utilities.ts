import { useEffect, useRef, useState, useCallback } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// =============================================================================
// Audio Analysis Hook
// =============================================================================

interface AudioAnalysis {
    frequency: number[];
    volume: number;
    peak: number;
    bass: number;
    mid: number;
    treble: number;
    isPlaying: boolean;
}

export const useAudioAnalysis = (fftSize: number = 256) => {
    const [audioData, setAudioData] = useState<AudioAnalysis>({
        frequency: new Array(fftSize / 2).fill(0),
        volume: 0,
        peak: 0,
        bass: 0,
        mid: 0,
        treble: 0,
        isPlaying: false
    });

    const analyserRef = useRef<AnalyserNode | null>(null);
    const dataArrayRef = useRef<Uint8Array | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

    const initAudio = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);

            analyser.fftSize = fftSize;
            source.connect(analyser);

            audioContextRef.current = audioContext;
            analyserRef.current = analyser;
            sourceRef.current = source;
            dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

            return true;
        } catch (error) {
            console.warn('Audio access denied:', error);
            return false;
        }
    }, [fftSize]);

    useFrame(() => {
        if (analyserRef.current && dataArrayRef.current) {
            analyserRef.current.getByteFrequencyData(dataArrayRef.current);

            const frequency = Array.from(dataArrayRef.current);
            const volume = frequency.reduce((sum, val) => sum + val, 0) / frequency.length / 255;
            const peak = Math.max(...frequency) / 255;

            // Frequency band analysis
            const third = Math.floor(frequency.length / 3);
            const bass = frequency.slice(0, third).reduce((sum, val) => sum + val, 0) / third / 255;
            const mid = frequency.slice(third, third * 2).reduce((sum, val) => sum + val, 0) / third / 255;
            const treble = frequency.slice(third * 2).reduce((sum, val) => sum + val, 0) / (frequency.length - third * 2) / 255;

            setAudioData({
                frequency: frequency.map(val => val / 255),
                volume,
                peak,
                bass,
                mid,
                treble,
                isPlaying: volume > 0.01
            });
        }
    });

    useEffect(() => {
        return () => {
            if (audioContextRef.current) {
                audioContextRef.current.close();
            }
        };
    }, []);

    return { audioData, initAudio };
};

// =============================================================================
// Camera/Video Hook
// =============================================================================

interface VideoStream {
    videoElement: HTMLVideoElement | null;
    canvasElement: HTMLCanvasElement | null;
    isActive: boolean;
    dimensions: { width: number; height: number };
}

export const useVideoStream = () => {
    const [stream, setStream] = useState<VideoStream>({
        videoElement: null,
        canvasElement: null,
        isActive: false,
        dimensions: { width: 0, height: 0 }
    });

    const initVideo = useCallback(async (constraints: MediaStreamConstraints = { video: true }) => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');

            video.srcObject = mediaStream;
            video.autoplay = true;
            video.muted = true;

            video.addEventListener('loadedmetadata', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                setStream({
                    videoElement: video,
                    canvasElement: canvas,
                    isActive: true,
                    dimensions: { width: video.videoWidth, height: video.videoHeight }
                });
            });

            return true;
        } catch (error) {
            console.warn('Video access denied:', error);
            return false;
        }
    }, []);

    const stopVideo = useCallback(() => {
        if (stream.videoElement?.srcObject) {
            const tracks = (stream.videoElement.srcObject as MediaStream).getTracks();
            tracks.forEach(track => track.stop());

            setStream({
                videoElement: null,
                canvasElement: null,
                isActive: false,
                dimensions: { width: 0, height: 0 }
            });
        }
    }, [stream.videoElement]);

    const getImageData = useCallback(() => {
        if (stream.videoElement && stream.canvasElement) {
            const ctx = stream.canvasElement.getContext('2d');
            if (ctx) {
                ctx.drawImage(stream.videoElement, 0, 0);
                return ctx.getImageData(0, 0, stream.canvasElement.width, stream.canvasElement.height);
            }
        }
        return null;
    }, [stream]);

    useEffect(() => {
        return stopVideo;
    }, [stopVideo]);

    return { stream, initVideo, stopVideo, getImageData };
};

// =============================================================================
// Performance Monitoring Hook
// =============================================================================

interface PerformanceStats {
    fps: number;
    frameTime: number;
    memoryUsage: number;
    triangles: number;
    calls: number;
}

export const usePerformanceStats = () => {
    const [stats, setStats] = useState<PerformanceStats>({
        fps: 0,
        frameTime: 0,
        memoryUsage: 0,
        triangles: 0,
        calls: 0
    });

    const { gl, scene } = useThree();
    const frameTimeRef = useRef<number>(0);
    const lastTimeRef = useRef<number>(performance.now());

    useFrame(() => {
        const now = performance.now();
        const deltaTime = now - lastTimeRef.current;
        frameTimeRef.current = deltaTime;
        lastTimeRef.current = now;

        // Update stats less frequently to avoid performance impact
        if (Math.random() < 0.1) { // ~10% of frames
            const fps = Math.round(1000 / deltaTime);
            const memoryUsage = (performance as any).memory?.usedJSHeapSize / 1048576 || 0; // MB

            setStats({
                fps,
                frameTime: deltaTime,
                memoryUsage,
                triangles: gl.info.render.triangles,
                calls: gl.info.render.calls
            });
        }
    });

    return stats;
};

// =============================================================================
// Math Utilities
// =============================================================================

export const MathUtils = {
    // Easing functions
    easeInOut: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    easeInQuart: (t: number) => t * t * t * t,
    easeOutBounce: (t: number) => {
        if (t < 1 / 2.75) return 7.5625 * t * t;
        if (t < 2 / 2.75) return 7.5625 * (t -= 1.5 / 2.75) * t + 0.75;
        if (t < 2.5 / 2.75) return 7.5625 * (t -= 2.25 / 2.75) * t + 0.9375;
        return 7.5625 * (t -= 2.625 / 2.75) * t + 0.984375;
    },

    // Interpolation
    lerp: (a: number, b: number, t: number) => a + (b - a) * t,
    clamp: (value: number, min: number, max: number) => Math.max(min, Math.min(max, value)),
    normalize: (value: number, min: number, max: number) => (value - min) / (max - min),
    map: (value: number, inMin: number, inMax: number, outMin: number, outMax: number) =>
        outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin)),

    // Noise functions
    noise2D: (x: number, y: number) => {
        // Simple pseudo-noise implementation
        let n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
        return (n - Math.floor(n)) * 2 - 1;
    },

    // Vector utilities
    distance2D: (x1: number, y1: number, x2: number, y2: number) =>
        Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),

    distance3D: (p1: THREE.Vector3, p2: THREE.Vector3) => p1.distanceTo(p2),

    // Random utilities
    randomInRange: (min: number, max: number) => Math.random() * (max - min) + min,
    randomInt: (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min,
    randomChoice: <T>(array: T[]): T => array[Math.floor(Math.random() * array.length)],

    // Color utilities
    hslToRgb: (h: number, s: number, l: number): [number, number, number] => {
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
        const m = l - c / 2;

        let r = 0, g = 0, b = 0;
        if (h < 60) [r, g, b] = [c, x, 0];
        else if (h < 120) [r, g, b] = [x, c, 0];
        else if (h < 180) [r, g, b] = [0, c, x];
        else if (h < 240) [r, g, b] = [0, x, c];
        else if (h < 300) [r, g, b] = [x, 0, c];
        else[r, g, b] = [c, 0, x];

        return [r + m, g + m, b + m];
    },

    rgbToHex: (r: number, g: number, b: number) =>
        `#${Math.round(r * 255).toString(16).padStart(2, '0')}${Math.round(g * 255).toString(16).padStart(2, '0')}${Math.round(b * 255).toString(16).padStart(2, '0')}`
};

// =============================================================================
// Animation Hooks
// =============================================================================

export const useSpring = (target: number, stiffness: number = 0.1, damping: number = 0.8) => {
    const [current, setCurrent] = useState(target);
    const velocityRef = useRef(0);

    useFrame(() => {
        const force = (target - current) * stiffness;
        velocityRef.current += force;
        velocityRef.current *= damping;

        const newValue = current + velocityRef.current;
        if (Math.abs(newValue - current) > 0.001) {
            setCurrent(newValue);
        }
    });

    return current;
};

export const useTween = (duration: number, easing: (t: number) => number = MathUtils.easeInOut) => {
    const [progress, setProgress] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const startTimeRef = useRef<number>(0);

    const start = useCallback(() => {
        startTimeRef.current = performance.now();
        setIsPlaying(true);
        setProgress(0);
    }, []);

    const stop = useCallback(() => {
        setIsPlaying(false);
    }, []);

    useFrame(() => {
        if (isPlaying) {
            const elapsed = performance.now() - startTimeRef.current;
            const rawProgress = Math.min(elapsed / duration, 1);
            const easedProgress = easing(rawProgress);

            setProgress(easedProgress);

            if (rawProgress >= 1) {
                setIsPlaying(false);
            }
        }
    });

    return { progress, isPlaying, start, stop };
};

// =============================================================================
// Mesh Utilities
// =============================================================================

export const createParametricGeometry = (
    func: (u: number, v: number) => THREE.Vector3,
    slices: number = 32,
    stacks: number = 32
) => {
    const geometry = new THREE.ParametricGeometry(func, slices, stacks);
    geometry.computeVertexNormals();
    return geometry;
};

export const createNoiseGeometry = (
    width: number,
    height: number,
    widthSegments: number = 32,
    heightSegments: number = 32,
    noiseScale: number = 1,
    amplitude: number = 1
) => {
    const geometry = new THREE.PlaneGeometry(width, height, widthSegments, heightSegments);
    const positions = geometry.attributes.position.array as Float32Array;

    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const noise = MathUtils.noise2D(x * noiseScale, y * noiseScale);
        positions[i + 2] = noise * amplitude;
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
    return geometry;
};

// =============================================================================
// Texture Utilities
// =============================================================================

export const createDataTexture = (
    data: number[],
    width: number,
    height: number,
    format: THREE.PixelFormat = THREE.RGBAFormat
) => {
    const texture = new THREE.DataTexture(
        new Uint8Array(data.map(v => Math.floor(v * 255))),
        width,
        height,
        format
    );
    texture.needsUpdate = true;
    return texture;
};

export const createNoiseTexture = (width: number, height: number, scale: number = 1) => {
    const data = new Uint8Array(width * height * 4);

    for (let i = 0; i < width * height * 4; i += 4) {
        const x = (i / 4) % width;
        const y = Math.floor((i / 4) / width);
        const noise = (MathUtils.noise2D(x * scale / width, y * scale / height) + 1) / 2;

        data[i] = noise * 255;     // R
        data[i + 1] = noise * 255; // G
        data[i + 2] = noise * 255; // B
        data[i + 3] = 255;         // A
    }

    const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
    texture.needsUpdate = true;
    return texture;
};

// =============================================================================
// Storage Utilities
// =============================================================================

export const useLocalStorage = <T>(key: string, defaultValue: T) => {
    const [value, setValue] = useState<T>(() => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.warn(`Error reading localStorage key "${key}":`, error);
            return defaultValue;
        }
    });

    const setStoredValue = useCallback((newValue: T) => {
        try {
            setValue(newValue);
            localStorage.setItem(key, JSON.stringify(newValue));
        } catch (error) {
            console.warn(`Error setting localStorage key "${key}":`, error);
        }
    }, [key]);

    return [value, setStoredValue] as const;
};

// =============================================================================
// Export Utilities
// =============================================================================

export const exportToJson = (data: any, filename: string) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

export const copyToClipboard = async (text: string) => {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (error) {
        console.warn('Clipboard API failed, using fallback:', error);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        return true;
    }
};

// =============================================================================
// Time Utilities
// =============================================================================

export const useGameTime = (speed: number = 1) => {
    const [time, setTime] = useState(0);

    useFrame((_, delta) => {
        setTime(prevTime => prevTime + delta * speed);
    });

    return time;
};

export const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

// =============================================================================
// Validation Utilities
// =============================================================================

export const validateWebGL = () => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    return !!gl;
};

export const validateWebGL2 = () => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    return !!gl;
};

export const getWebGLCapabilities = () => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

    if (!gl) return null;

    return {
        version: gl.getParameter(gl.VERSION),
        renderer: gl.getParameter(gl.RENDERER),
        vendor: gl.getParameter(gl.VENDOR),
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
        maxFragmentUniforms: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS),
        maxVertexUniforms: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS)
    };
};
