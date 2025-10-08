import { useEffect, useRef, useState } from 'react'
import { useGraph, LayoutParams } from '../state/graph'
import { useUI } from '../state/ui'

interface WorkerMessage {
  type: 'init' | 'tick' | 'pin' | 'setParams' | 'reset' | 'terminate'
  positions?: Float32Array
  velocities?: Float32Array
  edges?: Uint32Array
  pinnedNodes?: number[]
  params?: Partial<LayoutParams>
  dt?: number
  nodeId?: number
  pinned?: boolean
}

interface WorkerResponse {
  type: 'ready' | 'positions' | 'state' | 'error'
  positions?: Float32Array
  velocities?: Float32Array
  forces?: Float32Array
  iteration?: number
  energy?: number
  converged?: boolean
  error?: string
}

export function GraphPhysicsSystem() {
  const {
    nodeCount,
    positions,
    edges,
    edgeIndices,
    pinnedNodes,
    layoutRunning,
    params,
    setPositions,
    setLayoutState
  } = useGraph()

  const { reducedMotion } = useUI()

  const workerRef = useRef<Worker | null>(null)
  const [workerReady, setWorkerReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Timing control
  const lastSimTime = useRef(0)
  const lastUITime = useRef(0)
  const rafId = useRef<number>()

  const SIM_RATE = 60 // Hz - fixed simulation rate
  const UI_RATE = reducedMotion ? 15 : 30 // Hz - throttled UI updates
  const SIM_DT = 1 / SIM_RATE
  const UI_DT = 1 / UI_RATE

  // Initialize worker
  useEffect(() => {
    if (typeof Worker === 'undefined') {
      setError('Web Workers not supported in this environment')
      return
    }

    try {
      // Create worker with the layout worker module
      const worker = new Worker(
        new URL('../workers/layoutWorker.ts', import.meta.url),
        { type: 'module' }
      )

      workerRef.current = worker
      setWorkerReady(false)
      setError(null)

      // Handle worker messages
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const response = event.data

        switch (response.type) {
          case 'ready':
            setWorkerReady(true)
            setError(null)
            break

          case 'state':
            // Update positions in store
            if (response.positions) {
              setPositions(response.positions)
            }

            // Update layout state
            if (typeof response.iteration === 'number' &&
                typeof response.energy === 'number' &&
                typeof response.converged === 'boolean') {
              setLayoutState(
                layoutRunning && !response.converged,
                response.converged,
                response.iteration,
                response.energy
              )
            }
            break

          case 'error':
            setError(response.error || 'Unknown worker error')
            setLayoutState(false, false, 0, Infinity)
            break
        }
      }

      worker.onerror = (error) => {
        console.error('Worker error:', error)
        setError(`Worker error: ${error.message}`)
        setWorkerReady(false)
      }

      return () => {
        if (rafId.current) {
          cancelAnimationFrame(rafId.current)
        }

        worker.postMessage({ type: 'terminate' } as WorkerMessage)
        worker.terminate()
        workerRef.current = null
        setWorkerReady(false)
      }
    } catch (error) {
      console.error('Failed to create worker:', error)
      setError(`Failed to create worker: ${error}`)
      setWorkerReady(false)
    }
  }, [])

  // Initialize worker with graph data when ready
  useEffect(() => {
    if (!workerReady || !workerRef.current || nodeCount === 0) return

    try {
      // Convert edges map to index pairs for worker
      const edgeArray = new Uint32Array(edges.size * 2)
      let edgeIndex = 0

      // Create node ID to array index mapping
      const nodeIdToIndex = new Map<number, number>()
      let arrayIndex = 0
      for (const [nodeId] of Array.from(useGraph.getState().nodes)) {
        nodeIdToIndex.set(nodeId, arrayIndex++)
      }

      // Convert edges to index pairs
      for (const edge of useGraph.getState().edges.values()) {
        const sourceIndex = nodeIdToIndex.get(edge.source)
        const targetIndex = nodeIdToIndex.get(edge.target)

        if (sourceIndex !== undefined && targetIndex !== undefined) {
          edgeArray[edgeIndex * 2] = sourceIndex
          edgeArray[edgeIndex * 2 + 1] = targetIndex
          edgeIndex++
        }
      }

      // Trim edge array to actual size
      const trimmedEdges = edgeArray.slice(0, edgeIndex * 2)

      // Convert pinned node IDs to indices
      const pinnedIndices: number[] = []
      for (const nodeId of pinnedNodes) {
        const index = nodeIdToIndex.get(nodeId)
        if (index !== undefined) {
          pinnedIndices.push(index)
        }
      }

      const message: WorkerMessage = {
        type: 'init',
        positions: positions.slice(),
        edges: trimmedEdges,
        pinnedNodes: pinnedIndices,
        params
      }

      workerRef.current.postMessage(message, [
        message.positions!.buffer,
        message.edges!.buffer
      ])

      setError(null)
    } catch (error) {
      console.error('Failed to initialize worker:', error)
      setError(`Failed to initialize worker: ${error}`)
    }
  }, [workerReady, nodeCount, edges.size, params])

  // Update pinned nodes
  useEffect(() => {
    if (!workerReady || !workerRef.current) return

    const nodeIdToIndex = new Map<number, number>()
    let arrayIndex = 0
    for (const [nodeId] of Array.from(useGraph.getState().nodes)) {
      nodeIdToIndex.set(nodeId, arrayIndex++)
    }

    for (const nodeId of pinnedNodes) {
      const index = nodeIdToIndex.get(nodeId)
      if (index !== undefined) {
        const message: WorkerMessage = {
          type: 'pin',
          nodeId: index,
          pinned: true
        }
        workerRef.current.postMessage(message)
      }
    }
  }, [pinnedNodes, workerReady])

  // Update layout parameters
  useEffect(() => {
    if (!workerReady || !workerRef.current) return

    const message: WorkerMessage = {
      type: 'setParams',
      params
    }
    workerRef.current.postMessage(message)
  }, [params, workerReady])

  // Main simulation loop
  useEffect(() => {
    if (!layoutRunning || !workerReady || !workerRef.current) return

    let accumulatedTime = 0
    let uiAccumulatedTime = 0
    let lastTime = performance.now()

    const tick = () => {
      const now = performance.now()
      const deltaTime = (now - lastTime) / 1000 // Convert to seconds
      lastTime = now

      accumulatedTime += deltaTime
      uiAccumulatedTime += deltaTime

      // Fixed timestep simulation - catch up if behind
      let simSteps = 0
      while (accumulatedTime >= SIM_DT && simSteps < 5) { // Max 5 catch-up steps
        if (workerRef.current) {
          const message: WorkerMessage = {
            type: 'tick',
            dt: SIM_DT
          }
          workerRef.current.postMessage(message)
        }

        accumulatedTime -= SIM_DT
        simSteps++
      }

      // Reset if we're too far behind (prevents spiral of death)
      if (accumulatedTime > SIM_DT * 10) {
        accumulatedTime = 0
      }

      // UI updates are throttled but don't need to be caught up
      if (uiAccumulatedTime >= UI_DT) {
        uiAccumulatedTime = 0
        // UI update happens automatically when worker sends positions
      }

      if (layoutRunning) {
        rafId.current = requestAnimationFrame(tick)
      }
    }

    rafId.current = requestAnimationFrame(tick)

    return () => {
      if (rafId.current) {
        cancelAnimationFrame(rafId.current)
      }
    }
  }, [layoutRunning, workerReady])

  // Expose system status for debugging
  useEffect(() => {
    if (error) {
      console.error('GraphPhysicsSystem error:', error)
    }
  }, [error])

  return null
}

// Hook for accessing physics system status
export function useGraphPhysicsStatus() {
  const {
    layoutRunning,
    layoutConverged,
    layoutIteration,
    layoutEnergy,
    nodeCount,
    renderNodeCount
  } = useGraph()

  const [workerReady, setWorkerReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  return {
    // Status
    running: layoutRunning,
    converged: layoutConverged,
    ready: workerReady,
    error,

    // Metrics
    iteration: layoutIteration,
    energy: layoutEnergy,
    nodeCount,
    renderNodeCount,

    // Performance
    efficiency: renderNodeCount > 0 ? (nodeCount / renderNodeCount) : 1,
    fps: 60, // TODO: calculate actual FPS
  }
}

// Utility hook for physics presets
export function useGraphPhysicsPresets() {
  const { setLayoutParams } = useGraph()

  const presets = {
    gentle: {
      edgeStiffness: 1.0,
      repulsion: 0.5,
      damping: 0.95,
      theta: 0.8
    },

    standard: {
      edgeStiffness: 2.0,
      repulsion: 0.8,
      damping: 0.9,
      theta: 0.5
    },

    aggressive: {
      edgeStiffness: 3.0,
      repulsion: 1.2,
      damping: 0.85,
      theta: 0.3
    },

    clustered: {
      edgeStiffness: 0.8,
      repulsion: 0.3,
      damping: 0.95,
      theta: 0.6,
      gravity: 0.1
    },

    spacious: {
      edgeStiffness: 1.5,
      repulsion: 2.0,
      damping: 0.9,
      theta: 0.4
    }
  }

  return {
    presets,
    apply: (presetName: keyof typeof presets) => {
      setLayoutParams(presets[presetName])
    }
  }
}
