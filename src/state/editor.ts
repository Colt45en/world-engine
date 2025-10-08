import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import { enablePatches } from 'immer'

enablePatches()

export interface Entity {
    id: string
    type: 'cube' | 'sphere'
    position: [number, number, number]
    rotation: [number, number, number]
    scale: [number, number, number]
    color: string
    offset: number
}

interface HistoryEntry {
    state: EditorState
    action: string
    timestamp: number
}

interface EditorState {
    entities: Entity[]
    selectedEntityId: string | null
    animationSpeed: number
    camera: {
        position: [number, number, number]
        target: [number, number, number]
    }
}

interface EditorActions {
    // Entity management
    addEntity: (type: Entity['type']) => void
    deleteEntity: (id: string) => void
    selectEntity: (id: string | null) => void
    updateEntity: (id: string, updates: Partial<Entity>) => void
    clearScene: () => void

    // Animation
    setAnimationSpeed: (speed: number) => void

    // Camera
    setCamera: (position: [number, number, number], target: [number, number, number]) => void

    // History
    undo: () => void
    redo: () => void
    canUndo: boolean
    canRedo: boolean
}

type EditorStore = EditorState & EditorActions

const initialState: EditorState = {
    entities: [
        {
            id: 'cube-1',
            type: 'cube',
            position: [0, 0, 0],
            rotation: [0, 0, 0],
            scale: [1, 1, 1],
            color: '#ff6b6b',
            offset: 0
        }
    ],
    selectedEntityId: null,
    animationSpeed: 1,
    camera: {
        position: [5, 5, 5],
        target: [0, 0, 0]
    }
}

// History management
let history: HistoryEntry[] = []
let historyIndex = -1
const maxHistorySize = 50

const saveToHistory = (state: EditorState, action: string) => {
    // Remove any future history if we're not at the end
    if (historyIndex < history.length - 1) {
        history = history.slice(0, historyIndex + 1)
    }

    // Add new state to history
    history.push({
        state: JSON.parse(JSON.stringify(state)),
        action,
        timestamp: Date.now()
    })

    // Keep history within bounds
    if (history.length > maxHistorySize) {
        history = history.slice(-maxHistorySize)
    }

    historyIndex = history.length - 1
}

const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

const getRandomColor = () => {
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8']
    return colors[Math.floor(Math.random() * colors.length)]
}

export const useEditor = create<EditorStore>()(
    subscribeWithSelector(
        immer((set, get) => ({
            ...initialState,
            canUndo: false,
            canRedo: false,

            addEntity: (type) => {
                set((state) => {
                    const newEntity: Entity = {
                        id: `${type}-${generateId()}`,
                        type,
                        position: [
                            (Math.random() - 0.5) * 8,
                            Math.random() * 2,
                            (Math.random() - 0.5) * 8
                        ],
                        rotation: [0, 0, 0],
                        scale: [1, 1, 1],
                        color: getRandomColor(),
                        offset: Math.random() * Math.PI * 2
                    }
                    state.entities.push(newEntity)
                    state.selectedEntityId = newEntity.id
                })

                const currentState = get()
                saveToHistory(currentState, `Add ${type}`)
                set({ canUndo: true, canRedo: false })
            },

            deleteEntity: (id) => {
                set((state) => {
                    state.entities = state.entities.filter(e => e.id !== id)
                    if (state.selectedEntityId === id) {
                        state.selectedEntityId = null
                    }
                })

                const currentState = get()
                saveToHistory(currentState, 'Delete entity')
                set({ canUndo: true, canRedo: false })
            },

            selectEntity: (id) => {
                set((state) => {
                    state.selectedEntityId = id
                })
            },

            updateEntity: (id, updates) => {
                set((state) => {
                    const entity = state.entities.find(e => e.id === id)
                    if (entity) {
                        Object.assign(entity, updates)
                    }
                })

                const currentState = get()
                saveToHistory(currentState, 'Update entity')
                set({ canUndo: true, canRedo: false })
            },

            clearScene: () => {
                set((state) => {
                    state.entities = []
                    state.selectedEntityId = null
                })

                const currentState = get()
                saveToHistory(currentState, 'Clear scene')
                set({ canUndo: true, canRedo: false })
            },

            setAnimationSpeed: (speed) => {
                set((state) => {
                    state.animationSpeed = Math.max(0, Math.min(2, speed))
                })
            },

            setCamera: (position, target) => {
                set((state) => {
                    state.camera.position = position
                    state.camera.target = target
                })
            },

            undo: () => {
                if (historyIndex > 0) {
                    historyIndex--
                    const historyState = history[historyIndex].state
                    set((state) => {
                        Object.assign(state, historyState)
                        state.canUndo = historyIndex > 0
                        state.canRedo = historyIndex < history.length - 1
                    })
                }
            },

            redo: () => {
                if (historyIndex < history.length - 1) {
                    historyIndex++
                    const historyState = history[historyIndex].state
                    set((state) => {
                        Object.assign(state, historyState)
                        state.canUndo = historyIndex > 0
                        state.canRedo = historyIndex < history.length - 1
                    })
                }
            },
        }))
    )
)

// Initialize history with the initial state
saveToHistory(initialState, 'Initialize scene')
