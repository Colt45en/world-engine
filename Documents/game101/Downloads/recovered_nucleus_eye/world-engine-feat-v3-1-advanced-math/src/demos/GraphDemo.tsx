import React, { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Stats } from '@react-three/drei'
import { useGraph } from '../state/graph'
import { useUI } from '../state/ui'
import { GraphPhysicsSystem, useGraphPhysicsStatus } from '../systems/GraphPhysicsSystem'
import { InteractiveGraphLayer } from '../layers/GraphLayer'

interface GraphDemoProps {
  selectedDemo?: 'small' | 'medium' | 'large' | 'communities' | 'scalefree'
  showStats?: boolean
  showGrid?: boolean
}

export function GraphDemo({
  selectedDemo = 'medium',
  showStats = true,
  showGrid = false
}: GraphDemoProps) {
  const { reducedMotion } = useUI()
  const {
    nodeCount,
    layoutRunning,
    layoutConverged,
    layoutIteration,
    layoutEnergy,
    selectedNodes,
    pinnedNodes,
    clear,
    addNode,
    addEdge,
    startLayout,
    getMetrics
  } = useGraph()

  const status = useGraphPhysicsStatus()
  const [isGenerating, setIsGenerating] = useState(false)

  // Generate different types of demo graphs
  const generateDemo = async (type: string) => {
    if (isGenerating) return
    setIsGenerating(true)

    clear()

    switch (type) {
      case 'small':
        await generateSmallGraph()
        break
      case 'medium':
        await generateMediumGraph()
        break
      case 'large':
        await generateLargeGraph()
        break
      case 'communities':
        await generateCommunityGraph()
        break
      case 'scalefree':
        await generateScaleFreeGraph()
        break
    }

    // Start layout after generation
    startLayout()
    setIsGenerating(false)
  }

  const generateSmallGraph = async () => {
    // Small graph: 100 nodes, ring + random edges
    const nodeIds: number[] = []

    // Create nodes in a rough circle
    for (let i = 0; i < 100; i++) {
      const angle = (i / 100) * Math.PI * 2
      const radius = 2 + Math.random() * 0.5
      const x = Math.cos(angle) * radius
      const y = Math.sin(angle) * radius
      const z = (Math.random() - 0.5) * 0.5

      const id = addNode({
        label: `Node ${i}`,
        community: Math.floor(i / 20),
        color: `hsl(${(i / 20) * 72}, 70%, 60%)`
      })

      // Set initial position
      useGraph.getState().setNodePosition(id, x, y, z)
      nodeIds.push(id)

      // Small delay for smooth generation
      if (i % 10 === 0) await new Promise(resolve => setTimeout(resolve, 1))
    }

    // Create ring connections
    for (let i = 0; i < nodeIds.length; i++) {
      const next = (i + 1) % nodeIds.length
      addEdge({ source: nodeIds[i], target: nodeIds[next] })
    }

    // Add some random edges
    for (let i = 0; i < 50; i++) {
      const a = nodeIds[Math.floor(Math.random() * nodeIds.length)]
      const b = nodeIds[Math.floor(Math.random() * nodeIds.length)]
      if (a !== b) {
        addEdge({ source: a, target: b })
      }
    }
  }

  const generateMediumGraph = async () => {
    // Medium graph: 1000 nodes, communities
    const nodeIds: number[] = []
    const communities = 5
    const nodesPerCommunity = 200

    for (let comm = 0; comm < communities; comm++) {
      const communityNodes: number[] = []

      // Generate nodes for this community
      for (let i = 0; i < nodesPerCommunity; i++) {
        const id = addNode({
          label: `C${comm}N${i}`,
          community: comm,
          color: `hsl(${comm * 72}, 70%, ${50 + Math.random() * 30}%)`,
          radius: 0.03 + Math.random() * 0.02
        })

        nodeIds.push(id)
        communityNodes.push(id)

        if (i % 25 === 0) await new Promise(resolve => setTimeout(resolve, 1))
      }

      // Dense connections within community
      for (let i = 0; i < communityNodes.length; i++) {
        const connections = 3 + Math.floor(Math.random() * 4)
        for (let j = 0; j < connections; j++) {
          const target = communityNodes[Math.floor(Math.random() * communityNodes.length)]
          if (communityNodes[i] !== target) {
            addEdge({
              source: communityNodes[i],
              target,
              weight: 0.8 + Math.random() * 0.4
            })
          }
        }
      }
    }

    // Sparse connections between communities
    for (let i = 0; i < 100; i++) {
      const a = nodeIds[Math.floor(Math.random() * nodeIds.length)]
      const b = nodeIds[Math.floor(Math.random() * nodeIds.length)]
      if (a !== b) {
        addEdge({
          source: a,
          target: b,
          weight: 0.3 + Math.random() * 0.3
        })
      }
    }
  }

  const generateLargeGraph = async () => {
    // Large graph: 5000 nodes, scale-free network
    const nodeIds: number[] = []
    const targetNodes = 5000

    // Start with a small connected component
    for (let i = 0; i < 10; i++) {
      const id = addNode({
        label: `Seed ${i}`,
        radius: 0.04 + Math.random() * 0.02,
        color: '#ff6b6b'
      })
      nodeIds.push(id)
    }

    // Connect initial nodes
    for (let i = 0; i < nodeIds.length; i++) {
      for (let j = i + 1; j < nodeIds.length; j++) {
        if (Math.random() < 0.3) {
          addEdge({ source: nodeIds[i], target: nodeIds[j] })
        }
      }
    }

    // Preferential attachment for scale-free property
    for (let i = 10; i < targetNodes; i++) {
      const id = addNode({
        label: `Node ${i}`,
        community: Math.floor(i / 500),
        color: `hsl(${(i / 500) * 36}, 60%, ${40 + Math.random() * 40}%)`,
        radius: 0.02 + Math.random() * 0.03
      })

      nodeIds.push(id)

      // Calculate node degrees for preferential attachment
      const degrees = new Map<number, number>()
      let totalDegree = 0

      for (const nodeId of nodeIds.slice(0, -1)) {
        const degree = useGraph.getState().getNodeDegree(nodeId) + 1 // +1 to avoid zero
        degrees.set(nodeId, degree)
        totalDegree += degree
      }

      // Connect to 2-5 existing nodes based on their degree
      const connections = 2 + Math.floor(Math.random() * 4)
      const connected = new Set<number>()

      for (let j = 0; j < connections && connected.size < nodeIds.length - 1; j++) {
        let target: number | null = null
        let attempts = 0

        while (target === null && attempts < 20) {
          const rand = Math.random() * totalDegree
          let sum = 0

          for (const [nodeId, degree] of degrees) {
            sum += degree
            if (sum >= rand && !connected.has(nodeId)) {
              target = nodeId
              break
            }
          }
          attempts++
        }

        if (target !== null) {
          addEdge({ source: id, target })
          connected.add(target)
        }
      }

      // Progress indicator
      if (i % 100 === 0) {
        console.log(`Generated ${i}/${targetNodes} nodes`)
        await new Promise(resolve => setTimeout(resolve, 1))
      }
    }
  }

  const generateCommunityGraph = async () => {
    // Community-focused graph with clear clustering
    const nodeIds: number[] = []
    const communities = 8
    const communitySize = 150

    for (let comm = 0; comm < communities; comm++) {
      const communityNodes: number[] = []
      const centerAngle = (comm / communities) * Math.PI * 2
      const centerRadius = 5
      const centerX = Math.cos(centerAngle) * centerRadius
      const centerY = Math.sin(centerAngle) * centerRadius

      // Generate community nodes around center
      for (let i = 0; i < communitySize; i++) {
        const angle = Math.random() * Math.PI * 2
        const radius = Math.random() * 1.5
        const x = centerX + Math.cos(angle) * radius
        const y = centerY + Math.sin(angle) * radius
        const z = (Math.random() - 0.5) * 0.8

        const id = addNode({
          label: `Comm${comm}-${i}`,
          community: comm,
          color: `hsl(${comm * 45}, 80%, ${50 + Math.random() * 30}%)`,
          radius: 0.025 + Math.random() * 0.025
        })

        // Set position near community center
        useGraph.getState().setNodePosition(id, x, y, z)
        nodeIds.push(id)
        communityNodes.push(id)

        if (i % 20 === 0) await new Promise(resolve => setTimeout(resolve, 1))
      }

      // Dense intra-community connections
      for (const nodeId of communityNodes) {
        const connections = 4 + Math.floor(Math.random() * 6)
        for (let i = 0; i < connections; i++) {
          const target = communityNodes[Math.floor(Math.random() * communityNodes.length)]
          if (nodeId !== target) {
            addEdge({ source: nodeId, target, weight: 0.9 })
          }
        }
      }
    }

    // Sparse inter-community connections
    for (let i = 0; i < 200; i++) {
      const a = nodeIds[Math.floor(Math.random() * nodeIds.length)]
      const b = nodeIds[Math.floor(Math.random() * nodeIds.length)]

      const nodeA = useGraph.getState().nodes.get(a)
      const nodeB = useGraph.getState().nodes.get(b)

      if (a !== b && nodeA?.community !== nodeB?.community) {
        addEdge({ source: a, target: b, weight: 0.2 })
      }
    }
  }

  const generateScaleFreeGraph = async () => {
    // Pure scale-free network using Barabási-Albert model
    const nodeIds: number[] = []
    const m = 3 // edges to add per new node

    // Start with complete graph of m+1 nodes
    for (let i = 0; i <= m; i++) {
      const id = addNode({
        label: `Hub ${i}`,
        color: '#4a90e2',
        radius: 0.05
      })
      nodeIds.push(id)
    }

    // Fully connect initial nodes
    for (let i = 0; i < nodeIds.length; i++) {
      for (let j = i + 1; j < nodeIds.length; j++) {
        addEdge({ source: nodeIds[i], target: nodeIds[j] })
      }
    }

    // Add remaining nodes with preferential attachment
    const targetSize = 3000

    for (let n = m + 1; n < targetSize; n++) {
      const id = addNode({
        label: `Node ${n}`,
        color: `hsl(${(n / 100) % 360}, 60%, 50%)`,
        radius: 0.02 + Math.random() * 0.02
      })

      nodeIds.push(id)

      // Calculate degrees for preferential attachment
      const degrees = nodeIds.slice(0, -1).map(nodeId =>
        useGraph.getState().getNodeDegree(nodeId) + 1
      )
      const totalDegree = degrees.reduce((sum, deg) => sum + deg, 0)

      // Connect to m existing nodes
      const targets = new Set<number>()

      while (targets.size < m && targets.size < nodeIds.length - 1) {
        const rand = Math.random() * totalDegree
        let sum = 0

        for (let i = 0; i < degrees.length; i++) {
          sum += degrees[i]
          if (sum >= rand && !targets.has(nodeIds[i])) {
            targets.add(nodeIds[i])
            addEdge({ source: id, target: nodeIds[i] })
            break
          }
        }

        // Fallback to prevent infinite loop
        if (targets.size === 0) {
          const randomTarget = nodeIds[Math.floor(Math.random() * (nodeIds.length - 1))]
          targets.add(randomTarget)
          addEdge({ source: id, target: randomTarget })
        }
      }

      if (n % 100 === 0) {
        console.log(`Scale-free: ${n}/${targetSize} nodes`)
        await new Promise(resolve => setTimeout(resolve, 1))
      }
    }
  }

  // Auto-generate demo on mount
  useEffect(() => {
    if (nodeCount === 0 && !isGenerating) {
      generateDemo(selectedDemo)
    }
  }, [selectedDemo])

  const metrics = getMetrics()

  return (
    <div className="graph-demo">
      {/* Control Panel */}
      <div className="demo-controls">
        <div className="control-row">
          <label>Demo Type:</label>
          <select
            value={selectedDemo}
            onChange={(e) => generateDemo(e.target.value)}
            disabled={isGenerating}
          >
            <option value="small">Small (100 nodes)</option>
            <option value="medium">Medium (1K nodes)</option>
            <option value="large">Large (5K nodes)</option>
            <option value="communities">Communities (1.2K nodes)</option>
            <option value="scalefree">Scale-Free (3K nodes)</option>
          </select>
          <button
            onClick={() => generateDemo(selectedDemo)}
            disabled={isGenerating}
          >
            {isGenerating ? '⏳ Generating...' : '🔄 Regenerate'}
          </button>
        </div>

        <div className="control-row">
          <div className="status-info">
            Nodes: {metrics.nodeCount.toLocaleString()} |
            Edges: {metrics.edgeCount.toLocaleString()} |
            Selected: {selectedNodes.size} |
            Pinned: {pinnedNodes.size}
          </div>
        </div>

        {layoutRunning && (
          <div className="control-row">
            <div className="physics-status">
              {status.running ? '🟢' : '⚫'} Layout: Iteration {layoutIteration.toLocaleString()},
              Energy: {layoutEnergy.toFixed(3)},
              {layoutConverged ? '✅ Converged' : '🔄 Running'}
            </div>
          </div>
        )}

        <div className="control-row">
          <small>
            💡 Tips: Click nodes to select, Shift+Click to pin/unpin,
            Ctrl+A to select all, F to frame selection, Escape to clear
          </small>
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="canvas-container">
        <Canvas
          camera={{
            position: [8, 8, 8],
            fov: 60
          }}
          dpr={reducedMotion ? 1 : [1, 2]}
          performance={{
            min: reducedMotion ? 0.2 : 0.5
          }}
        >
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[10, 10, 5]}
            intensity={0.6}
            castShadow
            shadow-mapSize={[1024, 1024]}
          />
          <pointLight position={[-10, -10, -5]} intensity={0.3} />

          {/* Controls */}
          <OrbitControls
            enableDamping
            dampingFactor={reducedMotion ? 0.1 : 0.05}
            rotateSpeed={reducedMotion ? 0.5 : 1}
            zoomSpeed={reducedMotion ? 0.5 : 1}
            panSpeed={reducedMotion ? 0.5 : 1}
            maxPolarAngle={Math.PI * 0.75}
            minDistance={2}
            maxDistance={50}
          />

          {/* Optional Grid */}
          {showGrid && (
            <Grid
              args={[20, 20]}
              cellSize={0.5}
              cellThickness={0.5}
              cellColor={'#6f6f6f'}
              sectionSize={5}
              sectionThickness={1}
              sectionColor={'#9d9d9d'}
              fadeDistance={25}
              fadeStrength={1}
            />
          )}

          {/* Graph Physics System */}
          <GraphPhysicsSystem />

          {/* Interactive Graph Layer */}
          <InteractiveGraphLayer />

          {/* Stats */}
          {showStats && <Stats />}
        </Canvas>
      </div>

      <style jsx>{`
        .graph-demo {
          width: 100%;
          height: 100vh;
          display: flex;
          flex-direction: column;
          background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
          color: white;
          font-family: 'SF Mono', 'Monaco', monospace;
        }

        .demo-controls {
          background: rgba(0, 0, 0, 0.8);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          padding: 12px 16px;
        }

        .control-row {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 8px;
          font-size: 12px;
        }

        .control-row:last-child {
          margin-bottom: 0;
        }

        .control-row label {
          color: rgba(255, 255, 255, 0.8);
          min-width: 80px;
        }

        .control-row select {
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: white;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
        }

        .control-row button {
          background: rgba(100, 181, 246, 0.8);
          border: none;
          color: white;
          padding: 6px 12px;
          border-radius: 4px;
          font-size: 11px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .control-row button:hover:not(:disabled) {
          background: rgba(100, 181, 246, 1);
        }

        .control-row button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .status-info, .physics-status {
          color: #64b5f6;
          font-size: 11px;
          font-family: monospace;
        }

        .control-row small {
          color: rgba(255, 255, 255, 0.6);
          font-size: 10px;
          font-style: italic;
        }

        .canvas-container {
          flex: 1;
          position: relative;
          background: radial-gradient(ellipse at center, #1a1a1a 0%, #0c0c0c 100%);
        }
      `}</style>
    </div>
  )
}

export default GraphDemo
