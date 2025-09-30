import R3FEditorFramework from './src/App';

export default R3FEditorFramework;

// Export all public APIs
export {
    useScene,
    setSpotlightPosition,
    nudgeSpotlight
} from './src/state/store';

export {
    StoreProvider,
    StoreUtils
} from './src/state/StoreProvider';

export type {
    Entity,
    CubeEntity,
    EntityId,
    Vec3
} from './src/state/store';

export type {
    Tool,
    CommandItem,
    TransformData
} from './src/tools/types';

// Export individual components for advanced usage
export { default as AppCanvas } from './src/AppCanvas';
export { default as SceneRoot } from './src/scene/SceneRoot';
export { ToolManager } from './src/tools/ToolManager';

// Export layer components
export { default as EntityLayer } from './src/scene/layers/EntityLayer';
export { default as LightLayer } from './src/scene/layers/LightLayer';
export { default as HelpersLayer } from './src/scene/layers/HelpersLayer';

// Export tool components
export { default as CubeScenePanel } from './src/tools/CubeScenePanel';
export { default as SceneGraphInspector } from './src/tools/SceneGraphInspector';
export { default as TransformTool } from './src/tools/TransformTool';
export { default as CommandPalette } from './src/tools/CommandPalette';

// Export hooks
export { useGroundPointer } from './src/scene/hooks/useGroundPointer';

// Configuration and utilities
export const R3FEditorConfig = {
    version: '1.0.0',
    name: 'R3F Editor Framework',

    // Default camera settings
    defaultCamera: {
        position: [0, 1.5, 6] as [number, number, number],
        fov: 50
    },

    // Default scene settings
    defaultScene: {
        background: '#0b0e12',
        fog: {
            color: '#0b0e12',
            near: 10,
            far: 25
        }
    },

    // Tool shortcuts
    shortcuts: {
        commandPalette: 'Ctrl+K',
        transformTool: 'Alt+T',
        sceneGraph: 'Alt+G',
        cubeScene: 'Alt+C',
        addCube: 'Ctrl+N',
        deleteSelected: 'Delete',
        toggleSpotlight: 'L',
        undo: 'Ctrl+Z',
        redo: 'Ctrl+Shift+Z'
    }
};

// Framework utilities
export const EditorUtils = {
    // Create new cube with default properties
    createCube: (overrides: Partial<CubeEntity> = {}) => ({
        id: `cube-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        kind: 'cube' as const,
        position: [0, 0, 0] as [number, number, number],
        rotation: [0, 0, 0] as [number, number, number],
        scale: [1, 1, 1] as [number, number, number],
        offset: 0,
        color: `hsl(${Math.random() * 360}, 70%, 70%)`,
        name: 'New Cube',
        visible: true,
        ...overrides
    }),

    // Validate entity data
    validateEntity: (entity: any): entity is Entity => {
        return entity &&
            typeof entity.id === 'string' &&
            ['cube', 'light', 'camera', 'empty'].includes(entity.kind) &&
            Array.isArray(entity.position) &&
            entity.position.length === 3 &&
            Array.isArray(entity.rotation) &&
            entity.rotation.length === 3 &&
            Array.isArray(entity.scale) &&
            entity.scale.length === 3;
    },

    // Convert degrees to radians
    degToRad: (degrees: number) => degrees * Math.PI / 180,

    // Convert radians to degrees
    radToDeg: (radians: number) => radians * 180 / Math.PI,

    // Generate unique entity ID
    generateEntityId: (prefix: string = 'entity') =>
        `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,

    // Color utilities
    randomColor: () => `hsl(${Math.random() * 360}, 70%, 70%)`,

    // Position utilities
    randomPosition: (range: number = 5): [number, number, number] => [
        (Math.random() - 0.5) * range * 2,
        0,
        (Math.random() - 0.5) * range * 2
    ],

    // Check if two Vec3 are equal
    vec3Equal: (a: [number, number, number], b: [number, number, number], tolerance = 0.001) => {
        return Math.abs(a[0] - b[0]) < tolerance &&
            Math.abs(a[1] - b[1]) < tolerance &&
            Math.abs(a[2] - b[2]) < tolerance;
    }
};

// Framework constants
export const EDITOR_CONSTANTS = {
    MAX_HISTORY_SIZE: 50,
    MIN_SCALE: 0.001,
    MAX_SCALE: 100,
    ANIMATION_SPEED_RANGE: [0, 2],
    SPOTLIGHT_HEIGHT: 1,
    GROUND_LEVEL: 0,
    GRID_SIZE: 20,
    GRID_DIVISIONS: 20
} as const;
