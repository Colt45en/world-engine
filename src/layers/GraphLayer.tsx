import React, { useRef, useMemo, useEffect, useState } from 'react'
import * as THREE from 'three'
import { useFrame, useThree } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import { useGraph } from '../state/graph'
import { useUI } from '../state/ui'

// Performance-optimized instanced node rendering
function GraphNodes() {
  const {
    nodeCount,
    positions,
    selectedNodes,
    hoveredNode,
    pinnedNodes,
    nodeSize,
    lodLevel,
    nodes
  } = useGraph()

  const meshRef = useRef<THREE.InstancedMesh>(null!)
  const { camera } = useThree()

  // Geometries and materials
  const geometry = useMemo(() => new THREE.SphereGeometry(1, 12, 8), [])

  const material = useMemo(() => new THREE.MeshStandardMaterial({
    metalness: 0.1,
    roughness: 0.7,
  }), [])

  // Color palette for different node states
  const colors = useMemo(() => ({
    default: new THREE.Color('#64b5f6'),
    selected: new THREE.Color('#ffeb3b'),
    hovered: new THREE.Color('#ff9800'),
    pinned: new THREE.Color('#e91e63'),
    community: [
      new THREE.Color('#f44336'),
      new THREE.Color('#9c27b0'),
      new THREE.Color('#3f51b5'),
      new THREE.Color('#00bcd4'),
      new THREE.Color('#4caf50'),
      new THREE.Color('#ff9800'),
      new THREE.Color('#795548'),
      new THREE.Color('#607d8b')
    ]
  }), [])

  // Update instance matrices and colors
  useEffect(() => {
    if (!meshRef.current || nodeCount === 0 || positions.length === 0) return

    const mesh = meshRef.current
    const matrix = new THREE.Matrix4()
    const color = new THREE.Color()

    // Create arrays for instance attributes
    const instanceColors = new Float32Array(nodeCount * 3)

    let nodeIndex = 0
    for (const [nodeId, node] of nodes) {
      if (nodeIndex >= nodeCount) break

      // Get position
      const x = positions[nodeIndex * 3]
      const y = positions[nodeIndex * 3 + 1]
      const z = positions[nodeIndex * 3 + 2]

      // Set scale based on node radius and global size
      const scale = (node.radius || 1) * nodeSize
      matrix.makeScale(scale, scale, scale)
      matrix.setPosition(x, y, z)
      mesh.setMatrixAt(nodeIndex, matrix)

      // Determine color based on state
      if (nodeId === hoveredNode) {
        color.copy(colors.hovered)
      } else if (selectedNodes.has(nodeId)) {
        color.copy(colors.selected)
      } else if (pinnedNodes.has(nodeId)) {
        color.copy(colors.pinned)
      } else if (node.community && node.community > 0) {
        const communityColor = colors.community[node.community % colors.community.length]
        color.copy(communityColor)
      } else if (node.color) {
        color.set(node.color)
      } else {
        color.copy(colors.default)
      }

      // Set instance color
      instanceColors[nodeIndex * 3] = color.r
      instanceColors[nodeIndex * 3 + 1] = color.g
      instanceColors[nodeIndex * 3 + 2] = color.b

      nodeIndex++
    }

    // Update instance matrix
    mesh.instanceMatrix.needsUpdate = true

    // Update instance colors
    if (mesh.geometry.attributes.instanceColor) {
      mesh.geometry.attributes.instanceColor.array.set(instanceColors)
      mesh.geometry.attributes.instanceColor.needsUpdate = true
    } else {
      mesh.geometry.setAttribute(
        'instanceColor',
        new THREE.InstancedBufferAttribute(instanceColors, 3)
      )
    }

    // Update material to use instance colors
    if (!material.vertexColors) {
      material.vertexColors = true
      material.needsUpdate = true
    }

  }, [positions, nodeCount, selectedNodes, hoveredNode, pinnedNodes, nodeSize, nodes, colors, material])

  // Performance optimizations based on camera distance
  const [culledCount, setCulledCount] = useState(nodeCount)

  useFrame(() => {
    if (!meshRef.current) return

    // Simple frustum culling and LOD
    const cameraPosition = camera.position
    let visibleCount = 0

    const tempVector = new THREE.Vector3()
    const tempMatrix = new THREE.Matrix4()

    for (let i = 0; i < nodeCount; i++) {
      const x = positions[i * 3]
      const y = positions[i * 3 + 1]
      const z = positions[i * 3 + 2]

      tempVector.set(x, y, z)
      const distance = cameraPosition.distanceTo(tempVector)

      // LOD - hide distant nodes based on level
      const lodThreshold = lodLevel * 10 + 20
      const visible = distance < lodThreshold

      if (!visible) {
        // Hide by scaling to zero
        tempMatrix.makeScale(0, 0, 0)
        tempMatrix.setPosition(x, y, z)
        meshRef.current.setMatrixAt(i, tempMatrix)
      }

      if (visible) visibleCount++
    }

    setCulledCount(visibleCount)
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  // Update render count in store
  useEffect(() => {
    useGraph.getState().renderNodeCount = culledCount
  }, [culledCount])

  if (nodeCount === 0) return null

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, nodeCount]}
      castShadow
      receiveShadow
      frustumCulled={false} // We handle culling manually
    >
      {/* Custom shader material for better performance could go here */}
    </instancedMesh>
  )
}

// Efficient edge rendering using line geometry
function GraphEdges() {
  const { edges, positions, nodes, edgeWidth, selectedEdges, hoveredEdge } = useGraph()
  const { reducedMotion } = useUI()

  const [edgeGeometry, setEdgeGeometry] = useState<{
    positions: Float32Array
    colors: Float32Array
    indices: Uint16Array
  } | null>(null)

  // Build edge geometry
  useEffect(() => {
    if (edges.size === 0 || positions.length === 0) {
      setEdgeGeometry(null)
      return
    }

    // Create node ID to array index mapping
    const nodeIdToIndex = new Map<number, number>()
    let arrayIndex = 0
    for (const [nodeId] of nodes) {
      nodeIdToIndex.set(nodeId, arrayIndex++)
    }

    // Allocate arrays
    const edgePositions = new Float32Array(edges.size * 6) // 2 points per edge, 3 coords per point
    const edgeColors = new Float32Array(edges.size * 6) // 2 colors per edge, 3 components per color
    const indices = new Uint16Array(edges.size * 2) // 2 indices per edge

    const defaultColor = new THREE.Color('#90a4ae')
    const selectedColor = new THREE.Color('#ffeb3b')
    const hoveredColor = new THREE.Color('#ff9800')

    let edgeIndex = 0
    let vertexIndex = 0

    for (const [edgeId, edge] of edges) {
      const sourceIndex = nodeIdToIndex.get(edge.source)
      const targetIndex = nodeIdToIndex.get(edge.target)

      if (sourceIndex === undefined || targetIndex === undefined) continue

      // Get positions
      const sx = positions[sourceIndex * 3]
      const sy = positions[sourceIndex * 3 + 1]
      const sz = positions[sourceIndex * 3 + 2]

      const tx = positions[targetIndex * 3]
      const ty = positions[targetIndex * 3 + 1]
      const tz = positions[targetIndex * 3 + 2]

      // Set positions
      edgePositions[vertexIndex * 3] = sx
      edgePositions[vertexIndex * 3 + 1] = sy
      edgePositions[vertexIndex * 3 + 2] = sz

      edgePositions[(vertexIndex + 1) * 3] = tx
      edgePositions[(vertexIndex + 1) * 3 + 1] = ty
      edgePositions[(vertexIndex + 1) * 3 + 2] = tz

      // Determine color
      let color = defaultColor
      if (edgeId === hoveredEdge) {
        color = hoveredColor
      } else if (selectedEdges.has(edgeId)) {
        color = selectedColor
      } else if (edge.color) {
        color = new THREE.Color(edge.color)
      }

      // Set colors
      edgeColors[vertexIndex * 3] = color.r
      edgeColors[vertexIndex * 3 + 1] = color.g
      edgeColors[vertexIndex * 3 + 2] = color.b

      edgeColors[(vertexIndex + 1) * 3] = color.r
      edgeColors[(vertexIndex + 1) * 3 + 1] = color.g
      edgeColors[(vertexIndex + 1) * 3 + 2] = color.b

      // Set indices
      indices[edgeIndex * 2] = vertexIndex
      indices[edgeIndex * 2 + 1] = vertexIndex + 1

      edgeIndex++
      vertexIndex += 2
    }

    setEdgeGeometry({
      positions: edgePositions,
      colors: edgeColors,
      indices: indices
    })

  }, [edges, positions, nodes, selectedEdges, hoveredEdge])

  if (!edgeGeometry) return null

  return (
    <lineSegments frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={edgeGeometry.positions}
          count={edgeGeometry.positions.length / 3}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          array={edgeGeometry.colors}
          count={edgeGeometry.colors.length / 3}
          itemSize={3}
        />
        <bufferAttribute
          attach="index"
          array={edgeGeometry.indices}
          count={edgeGeometry.indices.length}
          itemSize={1}
        />
      </bufferGeometry>
      <lineBasicMaterial
        vertexColors
        linewidth={edgeWidth * 1000} // Scale for visibility
        transparent
        opacity={reducedMotion ? 0.6 : 0.8}
      />
    </lineSegments>
  )
}

// Node labels (rendered conditionally for performance)
function GraphLabels() {
  const { nodes, positions, showLabels, hoveredNode, selectedNodes } = useGraph()
  const { camera } = useThree()

  if (!showLabels) return null

  const labelsToShow = useMemo(() => {
    const labels: Array<{ id: number; position: THREE.Vector3; text: string }> = []

    let nodeIndex = 0
    for (const [nodeId, node] of nodes) {
      // Only show labels for hovered, selected, or important nodes to avoid clutter
      const shouldShow = nodeId === hoveredNode ||
                        selectedNodes.has(nodeId) ||
                        (node.label && node.label.length < 10)

      if (shouldShow && node.label) {
        const x = positions[nodeIndex * 3]
        const y = positions[nodeIndex * 3 + 1]
        const z = positions[nodeIndex * 3 + 2]

        labels.push({
          id: nodeId,
          position: new THREE.Vector3(x, y + 0.1, z), // Offset above node
          text: node.label
        })
      }

      nodeIndex++
    }

    return labels
  }, [nodes, positions, hoveredNode, selectedNodes])

  return (
    <group>
      {labelsToShow.map(({ id, position, text }) => (
        <mesh key={id} position={position}>
          <planeGeometry args={[0.2, 0.05]} />
          <meshBasicMaterial color="white" transparent opacity={0.8} />
          {/* TODO: Add text rendering with troika-three-text or similar */}
        </mesh>
      ))}
    </group>
  )
}

// Main graph layer component
export function GraphLayer() {
  const { nodeCount } = useGraph()

  if (nodeCount === 0) return null

  return (
    <group name="graph-layer">
      <GraphEdges />
      <GraphNodes />
      <GraphLabels />
    </group>
  )
}

// Interactive graph layer with picking and selection
export function InteractiveGraphLayer() {
  const {
    selectNode,
    selectEdge,
    hoverNode,
    hoverEdge,
    clearSelection,
    pinNode,
    nodes,
    positions
  } = useGraph()

  const { camera, raycaster, mouse } = useThree()
  const [isShiftPressed, setIsShiftPressed] = useState(false)

  // Keyboard event handling
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Shift') setIsShiftPressed(true)
      if (e.key === 'Escape') clearSelection()
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Shift') setIsShiftPressed(false)
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [clearSelection])

  // Throttled mouse picking for performance
  const lastPickTime = useRef(0)
  const PICK_THROTTLE = 16 // ~60fps

  const handlePointerMove = (event: THREE.Event) => {
    const now = performance.now()
    if (now - lastPickTime.current < PICK_THROTTLE) return
    lastPickTime.current = now

    // Simple distance-based picking for performance
    const mousePos = new THREE.Vector2()
    mousePos.x = (event.clientX / window.innerWidth) * 2 - 1
    mousePos.y = -(event.clientY / window.innerHeight) * 2 + 1

    raycaster.setFromCamera(mousePos, camera)

    // Find closest node to ray
    let closestNodeId: number | null = null
    let closestDistance = Infinity

    let nodeIndex = 0
    for (const [nodeId] of nodes) {
      const x = positions[nodeIndex * 3]
      const y = positions[nodeIndex * 3 + 1]
      const z = positions[nodeIndex * 3 + 2]

      const nodePos = new THREE.Vector3(x, y, z)
      const distance = raycaster.ray.distanceToPoint(nodePos)

      if (distance < closestDistance && distance < 0.1) { // Within picking threshold
        closestDistance = distance
        closestNodeId = nodeId
      }

      nodeIndex++
    }

    hoverNode(closestNodeId)
  }

  const handleClick = (event: THREE.Event) => {
    if (event.button !== 0) return // Left click only

    // Use the currently hovered node from pointer move
    const hoveredNodeId = useGraph.getState().hoveredNode

    if (hoveredNodeId !== null) {
      if (isShiftPressed) {
        // Shift+click to pin/unpin
        const isPinned = useGraph.getState().pinnedNodes.has(hoveredNodeId)
        pinNode(hoveredNodeId, !isPinned)
      } else {
        // Regular click to select
        selectNode(hoveredNodeId, event.ctrlKey || event.metaKey)
      }
    } else if (!event.ctrlKey && !event.metaKey) {
      // Click on empty space - clear selection
      clearSelection()
    }
  }

  return (
    <group
      onPointerMove={handlePointerMove}
      onClick={handleClick}
    >
      <GraphLayer />

      {/* Invisible sphere for ray casting */}
      <mesh visible={false}>
        <sphereGeometry args={[100]} />
        <meshBasicMaterial />
      </mesh>
    </group>
  )
}

export default InteractiveGraphLayer
