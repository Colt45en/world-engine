import { create } from 'zustand';
import { produceWithPatches, enablePatches } from 'immer';

enablePatches();

export type EntityId = string;
export type Vec3 = [number, number, number];

export interface CubeEntity {
    id: EntityId;
    kind: 'cube';
    position: Vec3;
    rotation: Vec3;
    scale: Vec3;
    offset: number;
    color: string;
    name?: string;
    visible?: boolean;
}

export interface Entity extends CubeEntity {
    // Union type for future entity types
    kind: 'cube' | 'light' | 'camera' | 'empty';
}

type PatchSet = { patches: any[]; inverse: any[] };

interface SceneState {
    // Core entities
    entities: Record<EntityId, Entity>;
    selected: EntityId[];

    // Cube scene specific
    animationSpeed: number;
    spotlight: { enabled: boolean; position: Vec3 };

    // History
    history: PatchSet[];
    future: PatchSet[];

    // Actions - Core
    addEntity: (e: Entity) => void;
    updateEntity: (id: EntityId, partial: Partial<Entity>) => void;
    updateEntities: (ids: EntityId[], partial: Partial<Entity>) => void;
    removeEntity: (id: EntityId) => void;
    selectOnly: (id?: EntityId) => void;

    // Actions - Cube Scene
    addCube: () => void;
    deleteSelected: () => void;
    setAnimationSpeed: (v: number) => void;
    toggleSpotlight: () => void;
    setSpotlightPosition: (position: Vec3) => void;
    setSpotlightPositionDirect: (x: number, y: number, z: number) => void;
    nudgeSpotlight: (dx: number, dz: number) => void;

    // History
    undo: () => void;
    redo: () => void;
    getSelectedEntities: () => Entity[];
}

export const useScene = create<SceneState>((set, get) => {
    const commit = (fn: (draft: SceneState) => void) => {
        const state = get();
        const [next, patches, inverse] = produceWithPatches(state, fn);
        set({
            ...next,
            history: [...state.history, { patches, inverse }],
            future: []
        });
    };

    const generateCubeId = () => `cube-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    return {
        // Initial state - original cubes
        entities: {
            'cube-1': {
                id: 'cube-1',
                kind: 'cube',
                position: [-1, 0, -1],
                rotation: [0, 0, 0],
                scale: [1, 1, 1],
                offset: 0,
                color: "#b3c0ff",
                name: "Cube 1",
                visible: true
            },
            'cube-2': {
                id: 'cube-2',
                kind: 'cube',
                position: [1, 0, -1],
                rotation: [0, 0, 0],
                scale: [1, 1, 1],
                offset: 1,
                color: "#b3ffe1",
                name: "Cube 2",
                visible: true
            }
        },
        selected: ['cube-1'],
        animationSpeed: 1,
        spotlight: { enabled: false, position: [0, 1, 0] },
        history: [],
        future: [],

        // Core entity actions
        addEntity: (e) => commit(d => {
            d.entities[e.id] = e;
        }),

        updateEntity: (id, partial) => commit(d => {
            if (d.entities[id]) {
                Object.assign(d.entities[id], partial);
            }
        }),

        updateEntities: (ids, partial) => commit(d => {
            ids.forEach(id => {
                if (d.entities[id]) {
                    Object.assign(d.entities[id], partial);
                }
            });
        }),

        removeEntity: (id) => commit(d => {
            delete d.entities[id];
            d.selected = d.selected.filter(s => s !== id);
        }),

        selectOnly: (id) => set({ selected: id ? [id] : [] }),

        // Cube scene actions
        addCube: () => commit(d => {
            const id = generateCubeId();
            const cubeCount = Object.values(d.entities).filter(e => e.kind === 'cube').length;

            d.entities[id] = {
                id,
                kind: 'cube',
                position: [0, 0, 0],
                rotation: [0, 0, 0],
                scale: [1, 1, 1],
                offset: cubeCount,
                color: `hsl(${(Math.random() * 360) | 0}, 70%, 70%)`,
                name: `Cube ${cubeCount + 1}`,
                visible: true
            };
        }),

        deleteSelected: () => commit(d => {
            d.selected.forEach(id => {
                delete d.entities[id];
            });
            d.selected = [];
        }),

        setAnimationSpeed: (speed) => set({ animationSpeed: speed }),

        toggleSpotlight: () => set(s => ({
            spotlight: { ...s.spotlight, enabled: !s.spotlight.enabled }
        })),

        setSpotlightPosition: (position) => set(s => ({
            spotlight: { ...s.spotlight, position }
        })),

        setSpotlightPositionDirect: (x, y, z) => set(s => ({
            spotlight: { ...s.spotlight, position: [x, y, z] }
        })),

        nudgeSpotlight: (dx, dz) => set(s => {
            const [x, y, z] = s.spotlight.position;
            return {
                spotlight: { ...s.spotlight, position: [x + dx, y, z + dz] }
            };
        }),

        // History actions
        undo: () => {
            const state = get();
            if (!state.history.length) return;

            const h = state.history[state.history.length - 1];
            const next = produceWithPatches(state, d => { }, h.inverse)[0];

            set({
                ...next,
                history: state.history.slice(0, -1),
                future: [h, ...state.future]
            });
        },

        redo: () => {
            const state = get();
            if (!state.future.length) return;

            const f = state.future[0];
            const next = produceWithPatches(state, d => { }, f.patches)[0];

            set({
                ...next,
                history: [...state.history, f],
                future: state.future.slice(1)
            });
        },

        getSelectedEntities: () => {
            const state = get();
            return state.selected.map(id => state.entities[id]).filter(Boolean);
        }
    };
});

// Export original programmatic controls
export const setSpotlightPosition = (x: number, y: number, z: number) =>
    useScene.getState().setSpotlightPositionDirect(x, y, z);

export const nudgeSpotlight = (dx: number, dz: number) =>
    useScene.getState().nudgeSpotlight(dx, dz);
