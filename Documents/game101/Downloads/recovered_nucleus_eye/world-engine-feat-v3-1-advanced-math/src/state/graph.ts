import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import * as THREE from 'three'

// Types
export type NodeId = number
export type EdgeId = string

export interface GraphNode {
    id: NodeId
    label?: string
    mass?: number
    color?: string
    radius?: number
    pinned?: boolean
    community?: number
}

export interface GraphEdge {
    id: EdgeId
    source: NodeId
    target: NodeId
    weight?: number
    length?: number
    color?: string
}

export interface GraphMetrics {
    nodeCount: number
    edgeCount: number
    avgDegree: number
    maxDegree: number
    connected: boolean
    communities: number
}

export interface LayoutParams {
    // Spring constants
    edgeStiffness: number     // k in Hooke's law
    edgeLength: number        // L0 rest length
    repulsion: number         // kr repulsion strength
    damping: number           // velocity damping 0-1

    // Barnes-Hut octree
    theta: number             // accuracy vs speed tradeoff

    // Bounds & constraints
    bounds: number            // sphere radius, 0 = unbounded
    gravity: number           // gravity well strength
    gravityCenter: THREE.Vector3

    // Integration
    timeStep: number          // dt for simulation
    maxVelocity: number       // velocity clamping

    // Convergence
    minEnergyThreshold: number
    stabilizationSteps: number
}

export interface GraphState {
    // Topology
    nodes: Map<NodeId, GraphNode>
    edges: Map<EdgeId, GraphEdge>
    adjacency: Map<NodeId, Set<NodeId>>

    // Positions (instancing buffers)
    nodeCount: number
    positions: Float32Array    // xyz positions for rendering
    velocities: Float32Array   // internal to worker
    forces: Float32Array       // debug visualization

    // Edge geometry
    edgePositions: Float32Array // line segments
    edgeIndices: Uint32Array    // [src0,tgt0, src1,tgt1, ...]

    // Layout state
    layoutRunning: boolean
    layoutConverged: boolean
    layoutIteration: number
    layoutEnergy: number
    params: LayoutParams

    // Selection & interaction
    selectedNodes: Set<NodeId>
    selectedEdges: Set<EdgeId>
    hoveredNode: NodeId | null
    hoveredEdge: EdgeId | null
    pinnedNodes: Set<NodeId>

    // Visual settings
    nodeSize: number
    edgeWidth: number
    showLabels: boolean
    showCommunities: boolean
    lodLevel: number           // 0=full detail, 1+=simplified

    // Performance metrics
    lastFrameTime: number
    avgFrameTime: number
    renderNodeCount: number    // actual nodes rendered (LOD)

    // Actions - Topology
    addNode: (node: Omit<GraphNode, 'id'> & Partial<Pick<GraphNode, 'id'>>) => NodeId
    addEdge: (edge: Omit<GraphEdge, 'id'> & Partial<Pick<GraphEdge, 'id'>>) => EdgeId
    removeNode: (id: NodeId) => void
    removeEdge: (id: EdgeId) => void
    updateNode: (id: NodeId, updates: Partial<GraphNode>) => void
    updateEdge: (id: EdgeId, updates: Partial<GraphEdge>) => void

    // Actions - Positions
    setPositions: (buffer: Float32Array) => void
    setNodePosition: (id: NodeId, x: number, y: number, z: number) => void

    // Actions - Layout
    startLayout: () => void
    stopLayout: () => void
    resetLayout: () => void
    setLayoutParams: (params: Partial<LayoutParams>) => void
    setLayoutState: (running: boolean, converged: boolean, iteration: number, energy: number) => void

    // Actions - Selection
    selectNode: (id: NodeId, multi?: boolean) => void
    selectEdge: (id: EdgeId, multi?: boolean) => void
    clearSelection: () => void
    hoverNode: (id: NodeId | null) => void
    hoverEdge: (id: EdgeId | null) => void

    // Actions - Pinning
    pinNode: (id: NodeId, pinned?: boolean) => void
    unpinAll: () => void

    // Actions - Visual
    setNodeSize: (size: number) => void
    setEdgeWidth: (width: number) => void
    toggleLabels: () => void
    setLODLevel: (level: number) => void

    // Utilities
    getMetrics: () => GraphMetrics
    getNodeDegree: (id: NodeId) => number
    getNodeNeighbors: (id: NodeId) => NodeId[]
    getShortestPath: (from: NodeId, to: NodeId) => NodeId[]
    frameSelection: () => { center: THREE.Vector3; radius: number }

    // Data operations
    clear: () => void
    loadGraphData: (data: { nodes: GraphNode[]; edges: GraphEdge[] }) => void
    exportGraphData: () => { nodes: GraphNode[]; edges: GraphEdge[] }
}

// Default layout parameters
const DEFAULT_LAYOUT_PARAMS: LayoutParams = {
    edgeStiffness: 2.0,
    edgeLength: 1.1,
    repulsion: 0.8,
    damping: 0.9,
    theta: 0.5,
    bounds: 0, // unbounded
    gravity: 0,
    gravityCenter: new THREE.Vector3(0, 0, 0),
    timeStep: 1 / 60,
    maxVelocity: 10.0,
    minEnergyThreshold: 0.01,
    stabilizationSteps: 1000
}

let nextNodeId = 1
let nextEdgeId = 1

export const useGraph = create<GraphState>()(
    immer((set, get) => ({
        // State
        nodes: new Map(),
        edges: new Map(),
        adjacency: new Map(),

        nodeCount: 0,
        positions: new Float32Array(0),
        velocities: new Float32Array(0),
        forces: new Float32Array(0),

        edgePositions: new Float32Array(0),
        edgeIndices: new Uint32Array(0),

        layoutRunning: false,
        layoutConverged: false,
        layoutIteration: 0,
        layoutEnergy: Infinity,
        params: DEFAULT_LAYOUT_PARAMS,

        selectedNodes: new Set(),
        selectedEdges: new Set(),
        hoveredNode: null,
        hoveredEdge: null,
        pinnedNodes: new Set(),

        nodeSize: 0.04,
        edgeWidth: 0.002,
        showLabels: false,
        showCommunities: false,
        lodLevel: 0,

        lastFrameTime: 0,
        avgFrameTime: 16.67,
        renderNodeCount: 0,

        // Topology actions
        addNode: (nodeData) => {
            const id = nodeData.id ?? nextNodeId++
            const node: GraphNode = {
                id,
                label: `Node ${id}`,
                mass: 1.0,
                color: '#64b5f6',
                radius: 0.04,
                pinned: false,
                community: 0,
                ...nodeData
            }

            set(draft => {
                draft.nodes.set(id, node)
                draft.adjacency.set(id, new Set())

                // Resize buffers if needed
                const newCount = draft.nodes.size
                if (newCount > draft.nodeCount) {
                    const newSize = Math.max(newCount * 2, 64) // exponential growth
                    const oldPos = draft.positions
                    const oldVel = draft.velocities
                    const oldForce = draft.forces

                    draft.positions = new Float32Array(newSize * 3)
                    draft.velocities = new Float32Array(newSize * 3)
                    draft.forces = new Float32Array(newSize * 3)

                    // Copy existing data
                    if (oldPos.length > 0) {
                        draft.positions.set(oldPos.subarray(0, draft.nodeCount * 3))
                        draft.velocities.set(oldVel.subarray(0, draft.nodeCount * 3))
                        draft.forces.set(oldForce.subarray(0, draft.nodeCount * 3))
                    }

                    // Initialize new node at origin with small random offset
                    const idx = (newCount - 1) * 3
                    draft.positions[idx + 0] = (Math.random() - 0.5) * 0.1
                    draft.positions[idx + 1] = (Math.random() - 0.5) * 0.1
                    draft.positions[idx + 2] = (Math.random() - 0.5) * 0.1
                }

                draft.nodeCount = newCount
            })

            return id
        },

        addEdge: (edgeData) => {
            const id = edgeData.id ?? `edge_${nextEdgeId++}`
            const { source, target } = edgeData

            // Check nodes exist
            const state = get()
            if (!state.nodes.has(source) || !state.nodes.has(target)) {
                throw new Error(`Cannot add edge: nodes ${source} or ${target} do not exist`)
            }

            const edge: GraphEdge = {
                id,
                source,
                target,
                weight: 1.0,
                length: 1.0,
                color: '#90a4ae',
                ...edgeData
            }

            set(draft => {
                draft.edges.set(id, edge)

                // Update adjacency
                if (!draft.adjacency.has(source)) draft.adjacency.set(source, new Set())
                if (!draft.adjacency.has(target)) draft.adjacency.set(target, new Set())

                draft.adjacency.get(source)!.add(target)
                if (source !== target) { // avoid double self-loops
                    draft.adjacency.get(target)!.add(source)
                }
            })

            return id
        },

        removeNode: (id) => set(draft => {
            if (!draft.nodes.has(id)) return

            // Remove all incident edges
            const edgesToRemove: EdgeId[] = []
            for (const [edgeId, edge] of draft.edges) {
                if (edge.source === id || edge.target === id) {
                    edgesToRemove.push(edgeId)
                }
            }

            edgesToRemove.forEach(eid => draft.edges.delete(eid))

            // Update adjacency
            const neighbors = draft.adjacency.get(id) || new Set()
            for (const neighbor of neighbors) {
                draft.adjacency.get(neighbor)?.delete(id)
            }
            draft.adjacency.delete(id)

            // Remove node
            draft.nodes.delete(id)
            draft.selectedNodes.delete(id)
            draft.pinnedNodes.delete(id)
            if (draft.hoveredNode === id) draft.hoveredNode = null
        }),

        removeEdge: (id) => set(draft => {
            const edge = draft.edges.get(id)
            if (!edge) return

            const { source, target } = edge

            // Update adjacency
            draft.adjacency.get(source)?.delete(target)
            if (source !== target) {
                draft.adjacency.get(target)?.delete(source)
            }

            // Remove edge
            draft.edges.delete(id)
            draft.selectedEdges.delete(id)
            if (draft.hoveredEdge === id) draft.hoveredEdge = null
        }),

        updateNode: (id, updates) => set(draft => {
            const node = draft.nodes.get(id)
            if (node) {
                Object.assign(node, updates)
            }
        }),

        updateEdge: (id, updates) => set(draft => {
            const edge = draft.edges.get(id)
            if (edge) {
                Object.assign(edge, updates)
            }
        }),

        // Position actions
        setPositions: (buffer) => set(draft => {
            draft.positions = buffer.slice() // defensive copy
        }),

        setNodePosition: (id, x, y, z) => set(draft => {
            // Find node index
            let nodeIndex = 0
            for (const [nid, _] of draft.nodes) {
                if (nid === id) break
                nodeIndex++
            }

            if (nodeIndex < draft.nodeCount) {
                const idx = nodeIndex * 3
                draft.positions[idx + 0] = x
                draft.positions[idx + 1] = y
                draft.positions[idx + 2] = z
            }
        }),

        // Layout actions
        startLayout: () => set(draft => {
            draft.layoutRunning = true
            draft.layoutConverged = false
            draft.layoutIteration = 0
        }),

        stopLayout: () => set(draft => {
            draft.layoutRunning = false
        }),

        resetLayout: () => set(draft => {
            draft.layoutIteration = 0
            draft.layoutEnergy = Infinity
            draft.layoutConverged = false

            // Randomize positions
            for (let i = 0; i < draft.nodeCount * 3; i += 3) {
                draft.positions[i + 0] = (Math.random() - 0.5) * 4
                draft.positions[i + 1] = (Math.random() - 0.5) * 4
                draft.positions[i + 2] = (Math.random() - 0.5) * 4
            }

            // Zero velocities
            draft.velocities.fill(0)
        }),

        setLayoutParams: (params) => set(draft => {
            Object.assign(draft.params, params)
        }),

        setLayoutState: (running, converged, iteration, energy) => set(draft => {
            draft.layoutRunning = running
            draft.layoutConverged = converged
            draft.layoutIteration = iteration
            draft.layoutEnergy = energy
        }),

        // Selection actions
        selectNode: (id, multi = false) => set(draft => {
            if (!multi) draft.selectedNodes.clear()

            if (draft.selectedNodes.has(id)) {
                draft.selectedNodes.delete(id)
            } else {
                draft.selectedNodes.add(id)
            }
        }),

        selectEdge: (id, multi = false) => set(draft => {
            if (!multi) draft.selectedEdges.clear()

            if (draft.selectedEdges.has(id)) {
                draft.selectedEdges.delete(id)
            } else {
                draft.selectedEdges.add(id)
            }
        }),

        clearSelection: () => set(draft => {
            draft.selectedNodes.clear()
            draft.selectedEdges.clear()
        }),

        hoverNode: (id) => set(draft => {
            draft.hoveredNode = id
        }),

        hoverEdge: (id) => set(draft => {
            draft.hoveredEdge = id
        }),

        // Pinning actions
        pinNode: (id, pinned = true) => set(draft => {
            if (pinned) {
                draft.pinnedNodes.add(id)
            } else {
                draft.pinnedNodes.delete(id)
            }

            // Update node data
            const node = draft.nodes.get(id)
            if (node) {
                node.pinned = pinned
            }
        }),

        unpinAll: () => set(draft => {
            draft.pinnedNodes.clear()
            for (const node of draft.nodes.values()) {
                node.pinned = false
            }
        }),

        // Visual actions
        setNodeSize: (size) => set(draft => {
            draft.nodeSize = size
        }),

        setEdgeWidth: (width) => set(draft => {
            draft.edgeWidth = width
        }),

        toggleLabels: () => set(draft => {
            draft.showLabels = !draft.showLabels
        }),

        setLODLevel: (level) => set(draft => {
            draft.lodLevel = level
        }),

        // Utilities
        getMetrics: () => {
            const state = get()
            const nodeCount = state.nodes.size
            const edgeCount = state.edges.size

            let totalDegree = 0
            let maxDegree = 0
            for (const neighbors of state.adjacency.values()) {
                const degree = neighbors.size
                totalDegree += degree
                maxDegree = Math.max(maxDegree, degree)
            }

            return {
                nodeCount,
                edgeCount,
                avgDegree: nodeCount > 0 ? totalDegree / nodeCount : 0,
                maxDegree,
                connected: true, // TODO: implement connectivity check
                communities: 1   // TODO: implement community detection
            }
        },

        getNodeDegree: (id) => {
            const state = get()
            return state.adjacency.get(id)?.size ?? 0
        },

        getNodeNeighbors: (id) => {
            const state = get()
            return Array.from(state.adjacency.get(id) ?? [])
        },

        getShortestPath: (from, to) => {
            // TODO: implement BFS/Dijkstra
            return []
        },

        frameSelection: () => {
            const state = get()
            if (state.selectedNodes.size === 0) {
                return { center: new THREE.Vector3(), radius: 5 }
            }

            const box = new THREE.Box3()
            let nodeIndex = 0

            for (const [nodeId, _] of state.nodes) {
                if (state.selectedNodes.has(nodeId)) {
                    const idx = nodeIndex * 3
                    const pos = new THREE.Vector3(
                        state.positions[idx + 0],
                        state.positions[idx + 1],
                        state.positions[idx + 2]
                    )
                    box.expandByPoint(pos)
                }
                nodeIndex++
            }

            const center = new THREE.Vector3()
            box.getCenter(center)
            const radius = Math.max(box.getSize(new THREE.Vector3()).length() * 0.6, 1.0)

            return { center, radius }
        },

        // Data operations
        clear: () => set(draft => {
            draft.nodes.clear()
            draft.edges.clear()
            draft.adjacency.clear()
            draft.selectedNodes.clear()
            draft.selectedEdges.clear()
            draft.pinnedNodes.clear()
            draft.hoveredNode = null
            draft.hoveredEdge = null
            draft.nodeCount = 0
            draft.positions = new Float32Array(0)
            draft.velocities = new Float32Array(0)
            draft.forces = new Float32Array(0)
        }),

        loadGraphData: (data) => set(draft => {
            // Clear existing
            draft.nodes.clear()
            draft.edges.clear()
            draft.adjacency.clear()

            // Load nodes
            data.nodes.forEach(node => {
                draft.nodes.set(node.id, node)
                draft.adjacency.set(node.id, new Set())
                nextNodeId = Math.max(nextNodeId, node.id + 1)
            })

            // Resize buffers
            const nodeCount = data.nodes.length
            const bufferSize = Math.max(nodeCount * 2, 64)

            draft.positions = new Float32Array(bufferSize * 3)
            draft.velocities = new Float32Array(bufferSize * 3)
            draft.forces = new Float32Array(bufferSize * 3)
            draft.nodeCount = nodeCount

            // Initialize positions randomly
            for (let i = 0; i < nodeCount * 3; i += 3) {
                draft.positions[i + 0] = (Math.random() - 0.5) * 4
                draft.positions[i + 1] = (Math.random() - 0.5) * 4
                draft.positions[i + 2] = (Math.random() - 0.5) * 4
            }

            // Load edges
            data.edges.forEach(edge => {
                draft.edges.set(edge.id, edge)

                // Update adjacency
                draft.adjacency.get(edge.source)?.add(edge.target)
                if (edge.source !== edge.target) {
                    draft.adjacency.get(edge.target)?.add(edge.source)
                }

                const idNum = parseInt(edge.id.split('_')[1] || '0')
                nextEdgeId = Math.max(nextEdgeId, idNum + 1)
            })
        }),

        exportGraphData: () => {
            const state = get()
            return {
                nodes: Array.from(state.nodes.values()),
                edges: Array.from(state.edges.values())
            }
        }
    }))
)
