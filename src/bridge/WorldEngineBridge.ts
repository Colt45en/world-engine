// Bridge system for sharing content between World Engine components
import * as React from 'react';
import * as THREE from 'three';

// Global content sharing system
class WorldEngineBridge {
    private static instance: WorldEngineBridge;
    private listeners: Map<string, Set<(data: any) => void>> = new Map();
    private storage: Map<string, any> = new Map();

    static getInstance(): WorldEngineBridge {
        if (!WorldEngineBridge.instance) {
            WorldEngineBridge.instance = new WorldEngineBridge();
        }
        return WorldEngineBridge.instance;
    }

    // Subscribe to data changes
    subscribe(key: string, callback: (data: any) => void) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, new Set());
        }
        this.listeners.get(key)!.add(callback);

        // Immediately call with current data if it exists
        if (this.storage.has(key)) {
            callback(this.storage.get(key));
        }

        // Return unsubscribe function
        return () => {
            this.listeners.get(key)?.delete(callback);
        };
    }

    // Publish data
    publish(key: string, data: any) {
        this.storage.set(key, data);
        const callbacks = this.listeners.get(key);
        if (callbacks) {
            callbacks.forEach(callback => callback(data));
        }
    }

    // Get current data
    get(key: string) {
        return this.storage.get(key);
    }

    // Check if data exists
    has(key: string) {
        return this.storage.has(key);
    }

    // Clear data
    clear(key: string) {
        this.storage.delete(key);
        const callbacks = this.listeners.get(key);
        if (callbacks) {
            callbacks.forEach(callback => callback(null));
        }
    }
}

export const bridge = WorldEngineBridge.getInstance();

// React hook for subscribing to bridge data
export function useBridgeData<T>(key: string): [T | null, (data: T) => void] {
    const [data, setData] = React.useState<T | null>(() => bridge.get(key) || null);

    React.useEffect(() => {
        const unsubscribe = bridge.subscribe(key, setData);
        return unsubscribe;
    }, [key]);

    const publishData = React.useCallback((newData: T) => {
        bridge.publish(key, newData);
    }, [key]);

    return [data, publishData];
}

// Specific hooks for common data types
export function useSharedMesh(): [THREE.Mesh | null, (mesh: THREE.Mesh | null) => void] {
    return useBridgeData<THREE.Mesh>('shared-mesh');
}

export function useSharedTexture(): [THREE.Texture | null, (texture: THREE.Texture | null) => void] {
    return useBridgeData<THREE.Texture>('shared-texture');
}

export function useSharedGeometry(): [THREE.BufferGeometry | null, (geometry: THREE.BufferGeometry | null) => void] {
    return useBridgeData<THREE.BufferGeometry>('shared-geometry');
}

export function useSharedGLB(): [ArrayBuffer | null, (glb: ArrayBuffer | null) => void] {
    return useBridgeData<ArrayBuffer>('shared-glb');
}

export function useCryptoData(): [CryptoData | null, (data: CryptoData | null) => void] {
    return useBridgeData<CryptoData>('crypto-data');
}

export function useSensoryMoment(): [SensoryMoment | null, (moment: SensoryMoment | null) => void] {
    return useBridgeData<SensoryMoment>('sensory-moment');
}

// Utility functions for working with 3D content
export const BridgeUtils = {
    // Convert Three.js mesh to data URL for iframe display
    meshToDataURL: (mesh: THREE.Mesh): string => {
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(512, 512);
        renderer.setClearColor(0x000000, 0);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
        camera.position.z = 5;

        // Add lighting
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const light = new THREE.DirectionalLight(0xffffff, 0.8);
        light.position.set(2, 2, 2);
        scene.add(light);

        // Center and scale mesh
        const box = new THREE.Box3().setFromObject(mesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2 / maxDim;

        mesh.position.copy(center.multiplyScalar(-1));
        mesh.scale.setScalar(scale);

        scene.add(mesh);
        renderer.render(scene, camera);

        const dataURL = renderer.domElement.toDataURL('image/png');

        renderer.dispose();

        return dataURL;
    },

    // Export mesh as GLB blob
    exportMeshAsGLB: async (mesh: THREE.Mesh): Promise<Blob> => {
        const { GLTFExporter } = await import('three/examples/jsm/exporters/GLTFExporter.js');

        const scene = new THREE.Scene();
        scene.add(mesh.clone());

        const exporter = new GLTFExporter();

        return new Promise((resolve, reject) => {
            exporter.parse(
                scene,
                (result: ArrayBuffer | object) => {
                    const blob = result instanceof ArrayBuffer
                        ? new Blob([result], { type: 'model/gltf-binary' })
                        : new Blob([JSON.stringify(result)], { type: 'application/json' });
                    resolve(blob);
                },
                (error) => reject(error),
                { binary: true }
            );
        });
    },

    // Create a data URL for a 3D model viewer HTML page
    createModelViewerHTML: (glbBlob: Blob): string => {
        const glbUrl = URL.createObjectURL(glbBlob);

        const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>3D Model Viewer</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    body { margin: 0; padding: 0; background: #0a0f17; font-family: ui-sans-serif, system-ui; }
    model-viewer { width: 100vw; height: 100vh; background-color: transparent; }
    .info { position: absolute; top: 10px; left: 10px; color: #64ffda; font-size: 12px; background: rgba(0,0,0,0.5); padding: 8px; border-radius: 4px; }
  </style>
</head>
<body>
  <model-viewer
    src="${glbUrl}"
    alt="Generated 3D Model"
    auto-rotate
    camera-controls
    environment-image="neutral"
    shadow-intensity="1"
    exposure="0.8">
  </model-viewer>
  <div class="info">
    Generated from Visual Bleedway<br>
    Use mouse to orbit • Scroll to zoom
  </div>
</body>
</html>`;

        return 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
    },

    // Create a simple HTML page displaying an image
    createImageViewerHTML: (imageUrl: string, title: string = 'Generated Content'): string => {
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
  <style>
    body { margin: 0; padding: 0; background: #0a0f17; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
    img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .overlay { position: absolute; top: 10px; left: 10px; color: #64ffda; font-size: 12px; background: rgba(0,0,0,0.5); padding: 8px; border-radius: 4px; font-family: ui-sans-serif, system-ui; }
  </style>
</head>
<body>
  <img src="${imageUrl}" alt="${title}" />
  <div class="overlay">${title}</div>
</body>
</html>`;

        return 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
    }
};

// Type definitions
export interface CryptoData {
    symbol: string;
    price: number;
    change24h: number;
    volume24h: number;
    candles: Array<{
        timestamp: number;
        open: number;
        high: number;
        low: number;
        close: number;
        volume: number;
    }>;
}

export interface SensoryMoment {
    id: string;
    label: string;
    perspective: string;
    details: Array<{
        channel: 'sight' | 'sound' | 'touch' | 'scent' | 'taste' | 'inner';
        description: string;
        strength: number;
    }>;
}
