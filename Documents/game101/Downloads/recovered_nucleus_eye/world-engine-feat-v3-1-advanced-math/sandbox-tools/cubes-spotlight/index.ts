import CubesSpotlight from './CubesSpotlight';

export default CubesSpotlight;

// Export programmatic control functions
export { setSpotlightPosition, nudgeSpotlight, useStore } from './CubesSpotlight';

// Export type interfaces for external usage
export interface CubeData {
    id: number;
    position: [number, number, number];
    offset: number;
    color: string;
}

export interface SpotlightState {
    enabled: boolean;
    position: [number, number, number];
}

export interface StoreState {
    cubes: CubeData[];
    selected: number;
    animationSpeed: number;
    spotlight: SpotlightState;
    addCube: () => void;
    deleteSelected: () => void;
    setSelected: (index: number) => void;
    setAnimationSpeed: (speed: number) => void;
    toggleSpot: () => void;
    setSpotPos: (position: [number, number, number]) => void;
    setSpotPosDirect: (x: number, y: number, z: number) => void;
    nudgeSpot: (dx: number, dz: number) => void;
    moveCube: (index: number, position: [number, number, number]) => void;
}

// Utility functions for external integration
export const CubesSpotlightUtils = {
    // Create new cube with random properties
    createRandomCube: () => ({
        id: Date.now(),
        position: [
            (Math.random() - 0.5) * 10,
            0,
            (Math.random() - 0.5) * 10
        ] as [number, number, number],
        offset: Math.random() * Math.PI * 2,
        color: `hsl(${Math.random() * 360}, 70%, 70%)`
    }),

    // Generate spotlight presets
    spotlightPresets: {
        center: [0, 1, 0] as [number, number, number],
        frontLeft: [-3, 1, 3] as [number, number, number],
        frontRight: [3, 1, 3] as [number, number, number],
        backLeft: [-3, 1, -3] as [number, number, number],
        backRight: [3, 1, -3] as [number, number, number]
    },

    // Animation speed presets
    animationPresets: {
        static: 0,
        slow: 0.3,
        normal: 1,
        fast: 1.8,
        extreme: 2
    },

    // Color palette for cubes
    colorPalette: [
        "#b3c0ff", // Light blue
        "#b3ffe1", // Light green
        "#ffb3c0", // Light pink
        "#ffe1b3", // Light orange
        "#e1b3ff", // Light purple
        "#c0ffb3"  // Light lime
    ],

    // Validate position coordinates
    validatePosition: (pos: [number, number, number]) => {
        return pos.every(coord =>
            typeof coord === 'number' &&
            Number.isFinite(coord) &&
            Math.abs(coord) < 100
        );
    },

    // Calculate distance between two positions
    distance: (pos1: [number, number, number], pos2: [number, number, number]) => {
        const dx = pos1[0] - pos2[0];
        const dy = pos1[1] - pos2[1];
        const dz = pos1[2] - pos2[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    },

    // Generate grid positions for cube placement
    generateGridPositions: (size: number, spacing: number = 2) => {
        const positions: [number, number, number][] = [];
        const half = Math.floor(size / 2);

        for (let x = -half; x <= half; x++) {
            for (let z = -half; z <= half; z++) {
                positions.push([x * spacing, 0, z * spacing]);
            }
        }

        return positions;
    },

    // Create scene configuration
    createSceneConfig: (options: {
        cubeCount?: number;
        animationSpeed?: number;
        spotlightEnabled?: boolean;
        spotlightPosition?: [number, number, number];
        gridLayout?: boolean;
    }) => ({
        cubes: options.gridLayout
            ? CubesSpotlightUtils.generateGridPositions(options.cubeCount || 3).map((pos, i) => ({
                id: Date.now() + i,
                position: pos,
                offset: i,
                color: CubesSpotlightUtils.colorPalette[i % CubesSpotlightUtils.colorPalette.length]
            }))
            : Array.from({ length: options.cubeCount || 2 }, (_, i) =>
                CubesSpotlightUtils.createRandomCube()
            ),
        selected: -1,
        animationSpeed: options.animationSpeed || 1,
        spotlight: {
            enabled: options.spotlightEnabled || false,
            position: options.spotlightPosition || [0, 1, 0]
        }
    })
};

// Advanced control functions
export const CubesSpotlightController = {
    // Batch add multiple cubes
    addMultipleCubes: (count: number, layout: 'random' | 'grid' = 'random') => {
        const store = useStore.getState();
        const newCubes = layout === 'grid'
            ? CubesSpotlightUtils.generateGridPositions(count).map((pos, i) => ({
                id: Date.now() + i,
                position: pos,
                offset: store.cubes.length + i,
                color: CubesSpotlightUtils.colorPalette[i % CubesSpotlightUtils.colorPalette.length]
            }))
            : Array.from({ length: count }, () => CubesSpotlightUtils.createRandomCube());

        // This would require a batch add action in the store
        console.warn('Batch add requires store modification - use individual addCube() calls');
    },

    // Animate spotlight in a pattern
    animateSpotlight: (pattern: 'circle' | 'figure8' | 'square', radius: number = 3, duration: number = 5000) => {
        const store = useStore.getState();
        const startTime = Date.now();

        const animate = () => {
            const elapsed = (Date.now() - startTime) / duration;
            const progress = elapsed % 1; // 0-1 repeating

            let x: number, z: number;

            switch (pattern) {
                case 'circle':
                    const angle = progress * Math.PI * 2;
                    x = Math.cos(angle) * radius;
                    z = Math.sin(angle) * radius;
                    break;
                case 'figure8':
                    const t = progress * Math.PI * 2;
                    x = Math.sin(t) * radius;
                    z = Math.sin(t * 2) * radius / 2;
                    break;
                case 'square':
                    const side = Math.floor(progress * 4);
                    const sideProgress = (progress * 4) % 1;
                    switch (side) {
                        case 0: x = -radius + sideProgress * 2 * radius; z = -radius; break;
                        case 1: x = radius; z = -radius + sideProgress * 2 * radius; break;
                        case 2: x = radius - sideProgress * 2 * radius; z = radius; break;
                        case 3: x = -radius; z = radius - sideProgress * 2 * radius; break;
                        default: x = 0; z = 0;
                    }
                    break;
                default:
                    return;
            }

            store.setSpotPosDirect(x, 1, z);

            if (elapsed < 10) { // Run for 10 cycles
                requestAnimationFrame(animate);
            }
        };

        animate();
    },

    // Reset scene to default state
    resetScene: () => {
        const store = useStore.getState();
        // Reset would require store reset action
        console.warn('Scene reset requires store modification');
    }
};
