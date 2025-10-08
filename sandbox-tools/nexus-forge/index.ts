import NexusForge from './NexusForge';

export default NexusForge;

// Export store and components for advanced usage
export { useStore } from './NexusForge';

// Export individual components for composition
export {
    Cube,
    EnhancedSpotlight,
    ParticleSystem,
    VideoScreen,
    SlideshowStrip,
    Scene,
    MediaBridges
} from './NexusForge';

// Export UI components for external use
export {
    MediaPanel,
    ThemeSelector,
    CubeControls,
    LightControls,
    SceneControls,
    MythicCodex
} from './NexusForge';

// Export theme system
export const THEME_MAP = {
    default: {
        bg: "#000000",
        fog: "#222222",
        title: "Neutral Void",
        ambience: "#222222",
        description: "The beginning and the end - pure potential in perfect balance."
    },
    Seeker: {
        bg: "#001122",
        fog: "#223344",
        title: "Karma Echoes",
        ambience: "#113355",
        description: "Those who search through the echoes of past actions find wisdom."
    },
    Weaver: {
        bg: "#331144",
        fog: "#442255",
        title: "Threading the Dao",
        ambience: "#662277",
        description: "Patterns interlace through the fabric of reality."
    },
    Newborn: {
        bg: "#0a0a0a",
        fog: "#1a1a1a",
        title: "Blank Origin",
        ambience: "#333333",
        description: "A clean slate of pure potential."
    },
    Ascendant: {
        bg: "#112233",
        fog: "#334455",
        title: "Celestial Ascension",
        ambience: "#446688",
        description: "Rising into realms of pure thought."
    },
    Twilight: {
        bg: "#221133",
        fog: "#332244",
        title: "Ethereal Twilight",
        ambience: "#553366",
        description: "The liminal space where mysteries reveal themselves."
    }
} as const;

// Export utility functions
export const NexusUtils = {
    // Create new cube with random properties
    createRandomCube: () => ({
        id: Date.now(),
        position: [
            (Math.random() - 0.5) * 5,
            (Math.random() - 0.5) * 3,
            (Math.random() - 0.5) * 5
        ],
        rotation: [0, 0, 0],
        tag: "Newborn",
        scale: 0.5 + Math.random() * 1.5,
        color: `hsl(${Math.random() * 360}, 70%, 60%)`
    }),

    // Generate particle positions in sphere
    generateSpherePositions: (count: number, radius: number = 10) => {
        const positions = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            const i3 = i * 3;
            const r = radius * Math.cbrt(Math.random()) + 1;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(Math.random() * 2 - 1);
            positions[i3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            positions[i3 + 2] = r * Math.cos(phi);
        }
        return positions;
    },

    // Convert file to texture
    fileToTexture: (file: File): Promise<THREE.Texture> => {
        return new Promise((resolve) => {
            const url = URL.createObjectURL(file);
            const loader = new THREE.TextureLoader();
            const texture = loader.load(url, () => {
                URL.revokeObjectURL(url);
                resolve(texture);
            });
        });
    },

    // Create video element from file
    createVideoElement: (file: File): HTMLVideoElement => {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.crossOrigin = 'anonymous';
        video.loop = true;
        video.muted = true;
        video.playsInline = true;
        video.autoplay = true;
        return video;
    },

    // Theme color utilities
    getThemeColors: (themeName: string) => {
        const theme = THEME_MAP[themeName as keyof typeof THEME_MAP] || THEME_MAP.default;
        return {
            background: theme.bg,
            fog: theme.fog,
            ambience: theme.ambience
        };
    },

    // Cube color presets
    CUBE_COLORS: [
        "#4080ff", // Blue
        "#ff40a0", // Pink
        "#40ff80", // Green
        "#ff8040", // Orange
        "#8040ff", // Purple
        "#40ffff"  // Cyan
    ],

    // Light color presets
    LIGHT_COLORS: [
        "#ffffff", // White
        "#ffeecc", // Warm white
        "#ccffee", // Cool white
        "#eeccff", // Purple tint
        "#ffccee", // Pink tint
        "#ccddff"  // Blue tint
    ]
};

// Export TypeScript interfaces
export interface CubeData {
    id: number;
    position: [number, number, number];
    rotation: [number, number, number];
    tag: string;
    scale: number;
    color: string;
}

export interface SpotlightConfig {
    enabled: boolean;
    position: [number, number, number];
    intensity: number;
    color: string;
}

export interface AmbientConfig {
    intensity: number;
    color: string;
}

export interface MediaState {
    imageTextures: THREE.Texture[];
    currentSlide: number;
    slideshowSpeed: number;
    slideshowActive: boolean;
    videoURL: string;
}

export interface ThemeData {
    bg: string;
    fog: string;
    title: string;
    ambience: string;
    description: string;
}

export interface NexusForgeConfig {
    initialCubes?: CubeData[];
    theme?: keyof typeof THEME_MAP;
    enableParticles?: boolean;
    showUI?: boolean;
    cameraPosition?: [number, number, number];
    autoRotate?: boolean;
}

// Create configured Nexus Forge instance
export const createConfiguredForge = (config: NexusForgeConfig) => {
    // This would return a pre-configured component
    // Implementation would depend on specific requirements
    return NexusForge;
};
