import { Command } from '../types'
import { useGraph } from '../../state/graph'
import { useCamera } from '../../hooks/useCamera'
import { useGraphPhysicsPresets } from '../../systems/GraphPhysicsSystem'

// Graph visualization commands
export const graphCommands: Command[] = [
    {
        id: 'graph.startLayout',
        label: 'Start Graph Layout',
        description: 'Begin physics simulation for graph layout',
        category: 'Graph',
        shortcut: 'G S',
        action: () => {
            useGraph.getState().startLayout()
        }
    },

    {
        id: 'graph.stopLayout',
        label: 'Stop Graph Layout',
        description: 'Pause physics simulation',
        category: 'Graph',
        shortcut: 'G T',
        action: () => {
            useGraph.getState().stopLayout()
        }
    },

    {
        id: 'graph.resetLayout',
        label: 'Reset Graph Layout',
        description: 'Randomize positions and restart layout',
        category: 'Graph',
        shortcut: 'G R',
        action: () => {
            const graph = useGraph.getState()
            graph.resetLayout()
            graph.startLayout()
        }
    },

    {
        id: 'graph.clearSelection',
        label: 'Clear Selection',
        description: 'Deselect all nodes and edges',
        category: 'Graph',
        shortcut: 'Escape',
        action: () => {
            useGraph.getState().clearSelection()
        }
    },

    {
        id: 'graph.selectAll',
        label: 'Select All Nodes',
        description: 'Select all nodes in the graph',
        category: 'Graph',
        shortcut: 'Ctrl+A',
        action: () => {
            const { nodes, selectNode } = useGraph.getState()
            for (const nodeId of nodes.keys()) {
                selectNode(nodeId, true)
            }
        }
    },

    {
        id: 'graph.frameSelection',
        label: 'Frame Selection',
        description: 'Focus camera on selected nodes',
        category: 'Graph',
        shortcut: 'F',
        action: () => {
            const { frameSelection } = useGraph.getState()
            const { center, radius } = frameSelection()

            // Use camera hook to animate to position
            const camera = useCamera()
            if (camera) {
                camera.animateTo(center, radius * 2)
            }
        }
    },

    {
        id: 'graph.frameAll',
        label: 'Frame All',
        description: 'Fit entire graph in view',
        category: 'Graph',
        shortcut: 'Shift+F',
        action: () => {
            const { nodes, positions, selectNode, frameSelection } = useGraph.getState()

            // Select all nodes temporarily
            for (const nodeId of nodes.keys()) {
                selectNode(nodeId, true)
            }

            // Frame them
            const { center, radius } = frameSelection()
            const camera = useCamera()
            if (camera) {
                camera.animateTo(center, radius * 2)
            }

            // Clear selection after a brief delay
            setTimeout(() => {
                useGraph.getState().clearSelection()
            }, 100)
        }
    },

    {
        id: 'graph.pinSelected',
        label: 'Pin Selected Nodes',
        description: 'Pin selected nodes in place',
        category: 'Graph',
        shortcut: 'P',
        action: () => {
            const { selectedNodes, pinNode } = useGraph.getState()
            for (const nodeId of selectedNodes) {
                pinNode(nodeId, true)
            }
        }
    },

    {
        id: 'graph.unpinSelected',
        label: 'Unpin Selected Nodes',
        description: 'Unpin selected nodes',
        category: 'Graph',
        shortcut: 'Shift+P',
        action: () => {
            const { selectedNodes, pinNode } = useGraph.getState()
            for (const nodeId of selectedNodes) {
                pinNode(nodeId, false)
            }
        }
    },

    {
        id: 'graph.unpinAll',
        label: 'Unpin All Nodes',
        description: 'Unpin all nodes in the graph',
        category: 'Graph',
        shortcut: 'Ctrl+Shift+P',
        action: () => {
            useGraph.getState().unpinAll()
        }
    },

    {
        id: 'graph.toggleLabels',
        label: 'Toggle Node Labels',
        description: 'Show/hide node labels',
        category: 'Graph',
        shortcut: 'L',
        action: () => {
            useGraph.getState().toggleLabels()
        }
    },

    {
        id: 'graph.increaseNodeSize',
        label: 'Increase Node Size',
        description: 'Make nodes larger',
        category: 'Graph',
        shortcut: 'Plus',
        action: () => {
            const { nodeSize, setNodeSize } = useGraph.getState()
            setNodeSize(Math.min(nodeSize * 1.2, 0.2))
        }
    },

    {
        id: 'graph.decreaseNodeSize',
        label: 'Decrease Node Size',
        description: 'Make nodes smaller',
        category: 'Graph',
        shortcut: 'Minus',
        action: () => {
            const { nodeSize, setNodeSize } = useGraph.getState()
            setNodeSize(Math.max(nodeSize / 1.2, 0.01))
        }
    },

    {
        id: 'graph.increaseEdgeWidth',
        label: 'Increase Edge Width',
        description: 'Make edges thicker',
        category: 'Graph',
        shortcut: 'Ctrl+Plus',
        action: () => {
            const { edgeWidth, setEdgeWidth } = useGraph.getState()
            setEdgeWidth(Math.min(edgeWidth * 1.2, 0.01))
        }
    },

    {
        id: 'graph.decreaseEdgeWidth',
        label: 'Decrease Edge Width',
        description: 'Make edges thinner',
        category: 'Graph',
        shortcut: 'Ctrl+Minus',
        action: () => {
            const { edgeWidth, setEdgeWidth } = useGraph.getState()
            setEdgeWidth(Math.max(edgeWidth / 1.2, 0.0001))
        }
    },

    {
        id: 'graph.increaseLOD',
        label: 'Increase Level of Detail',
        description: 'Show more distant nodes',
        category: 'Graph',
        shortcut: 'Ctrl+Shift+Plus',
        action: () => {
            const { lodLevel, setLODLevel } = useGraph.getState()
            setLODLevel(Math.max(lodLevel - 1, 0))
        }
    },

    {
        id: 'graph.decreaseLOD',
        label: 'Decrease Level of Detail',
        description: 'Hide distant nodes for performance',
        category: 'Graph',
        shortcut: 'Ctrl+Shift+Minus',
        action: () => {
            const { lodLevel, setLODLevel } = useGraph.getState()
            setLODLevel(lodLevel + 1)
        }
    }
]

// Physics preset commands
export const physicsPresetCommands: Command[] = [
    {
        id: 'graph.presetGentle',
        label: 'Apply Gentle Physics',
        description: 'Slow, stable layout with minimal motion',
        category: 'Graph Physics',
        action: () => {
            const { apply } = useGraphPhysicsPresets()
            apply('gentle')
        }
    },

    {
        id: 'graph.presetStandard',
        label: 'Apply Standard Physics',
        description: 'Balanced layout parameters',
        category: 'Graph Physics',
        action: () => {
            const { apply } = useGraphPhysicsPresets()
            apply('standard')
        }
    },

    {
        id: 'graph.presetAggressive',
        label: 'Apply Aggressive Physics',
        description: 'Fast layout with strong forces',
        category: 'Graph Physics',
        action: () => {
            const { apply } = useGraphPhysicsPresets()
            apply('aggressive')
        }
    },

    {
        id: 'graph.presetClustered',
        label: 'Apply Clustered Physics',
        description: 'Emphasize community structure',
        category: 'Graph Physics',
        action: () => {
            const { apply } = useGraphPhysicsPresets()
            apply('clustered')
        }
    },

    {
        id: 'graph.presetSpacious',
        label: 'Apply Spacious Physics',
        description: 'Spread nodes out with strong repulsion',
        category: 'Graph Physics',
        action: () => {
            const { apply } = useGraphPhysicsPresets()
            apply('spacious')
        }
    }
]

// Fine-grained physics parameter commands
export const physicsParameterCommands: Command[] = [
    {
        id: 'graph.increaseRepulsion',
        label: 'Increase Repulsion',
        description: 'Increase node repulsion force',
        category: 'Graph Physics',
        action: () => {
            const { params, setLayoutParams } = useGraph.getState()
            setLayoutParams({
                repulsion: Math.min(params.repulsion * 1.2, 5.0)
            })
        }
    },

    {
        id: 'graph.decreaseRepulsion',
        label: 'Decrease Repulsion',
        description: 'Decrease node repulsion force',
        category: 'Graph Physics',
        action: () => {
            const { params, setLayoutParams } = useGraph.getState()
            setLayoutParams({
                repulsion: Math.max(params.repulsion / 1.2, 0.1)
            })
        }
    },

    {
        id: 'graph.increaseStiffness',
        label: 'Increase Edge Stiffness',
        description: 'Make edges more rigid',
        category: 'Graph Physics',
        action: () => {
            const { params, setLayoutParams } = useGraph.getState()
            setLayoutParams({
                edgeStiffness: Math.min(params.edgeStiffness * 1.2, 10.0)
            })
        }
    },

    {
        id: 'graph.decreaseStiffness',
        label: 'Decrease Edge Stiffness',
        description: 'Make edges more flexible',
        category: 'Graph Physics',
        action: () => {
            const { params, setLayoutParams } = useGraph.getState()
            setLayoutParams({
                edgeStiffness: Math.max(params.edgeStiffness / 1.2, 0.1)
            })
        }
    },

    {
        id: 'graph.increaseDamping',
        label: 'Increase Damping',
        description: 'Reduce oscillations, slower convergence',
        category: 'Graph Physics',
        action: () => {
            const { params, setLayoutParams } = useGraph.getState()
            setLayoutParams({
                damping: Math.min(params.damping + 0.05, 0.99)
            })
        }
    },

    {
        id: 'graph.decreaseDamping',
        label: 'Decrease Damping',
        description: 'Allow more motion, faster convergence',
        category: 'Graph Physics',
        action: () => {
            const { params, setLayoutParams } = useGraph.getState()
            setLayoutParams({
                damping: Math.max(params.damping - 0.05, 0.1)
            })
        }
    },

    {
        id: 'graph.toggleBounds',
        label: 'Toggle Bounds Constraint',
        description: 'Enable/disable spherical boundary',
        category: 'Graph Physics',
        action: () => {
            const { params, setLayoutParams } = useGraph.getState()
            setLayoutParams({
                bounds: params.bounds > 0 ? 0 : 10
            })
        }
    },

    {
        id: 'graph.toggleGravity',
        label: 'Toggle Center Gravity',
        description: 'Enable/disable gravity well at origin',
        category: 'Graph Physics',
        action: () => {
            const { params, setLayoutParams } = useGraph.getState()
            setLayoutParams({
                gravity: params.gravity > 0 ? 0 : 0.1
            })
        }
    }
]

// Data commands
export const graphDataCommands: Command[] = [
    {
        id: 'graph.exportJSON',
        label: 'Export Graph as JSON',
        description: 'Save graph data to JSON file',
        category: 'Graph Data',
        action: () => {
            const { exportGraphData } = useGraph.getState()
            const data = exportGraphData()

            const blob = new Blob([JSON.stringify(data, null, 2)], {
                type: 'application/json'
            })
            const url = URL.createObjectURL(blob)

            const a = document.createElement('a')
            a.href = url
            a.download = `graph-${Date.now()}.json`
            a.click()

            URL.revokeObjectURL(url)
        }
    },

    {
        id: 'graph.importJSON',
        label: 'Import Graph from JSON',
        description: 'Load graph data from JSON file',
        category: 'Graph Data',
        action: () => {
            const input = document.createElement('input')
            input.type = 'file'
            input.accept = '.json'

            input.onchange = (e) => {
                const file = (e.target as HTMLInputElement).files?.[0]
                if (!file) return

                const reader = new FileReader()
                reader.onload = (e) => {
                    try {
                        const data = JSON.parse(e.target?.result as string)
                        useGraph.getState().loadGraphData(data)
                    } catch (error) {
                        console.error('Failed to parse JSON:', error)
                    }
                }
                reader.readAsText(file)
            }

            input.click()
        }
    },

    {
        id: 'graph.clear',
        label: 'Clear Graph',
        description: 'Remove all nodes and edges',
        category: 'Graph Data',
        action: () => {
            if (confirm('Are you sure you want to clear the entire graph?')) {
                useGraph.getState().clear()
            }
        }
    },

    {
        id: 'graph.generateRandom',
        label: 'Generate Random Graph',
        description: 'Create a random graph with 1000 nodes',
        category: 'Graph Data',
        action: () => {
            const { clear, addNode, addEdge } = useGraph.getState()

            // Clear existing graph
            clear()

            // Generate nodes
            const nodeCount = 1000
            const nodeIds: number[] = []

            for (let i = 0; i < nodeCount; i++) {
                const id = addNode({
                    label: `Node ${i}`,
                    community: Math.floor(i / 100), // 10 communities
                    radius: 0.02 + Math.random() * 0.04
                })
                nodeIds.push(id)
            }

            // Generate edges (random network)
            const edgeCount = nodeCount * 2 // Average degree = 4

            for (let i = 0; i < edgeCount; i++) {
                const source = nodeIds[Math.floor(Math.random() * nodeIds.length)]
                const target = nodeIds[Math.floor(Math.random() * nodeIds.length)]

                if (source !== target) {
                    addEdge({ source, target })
                }
            }

            // Start layout
            useGraph.getState().startLayout()
        }
    }
]

// Combine all graph commands
export const allGraphCommands = [
    ...graphCommands,
    ...physicsPresetCommands,
    ...physicsParameterCommands,
    ...graphDataCommands
]
