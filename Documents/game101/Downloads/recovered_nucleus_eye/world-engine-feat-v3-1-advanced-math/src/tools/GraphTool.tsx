import React, { useState } from 'react'
import { Tool } from '../types'
import { useGraph } from '../../state/graph'
import { useGraphPhysicsStatus, useGraphPhysicsPresets } from '../../systems/GraphPhysicsSystem'

interface GraphToolProps {
  onClose: () => void
}

function GraphTool({ onClose }: GraphToolProps) {
  const {
    nodeCount,
    edges,
    layoutRunning,
    layoutConverged,
    layoutIteration,
    layoutEnergy,
    selectedNodes,
    selectedEdges,
    pinnedNodes,
    nodeSize,
    edgeWidth,
    showLabels,
    lodLevel,
    params,
    startLayout,
    stopLayout,
    resetLayout,
    setLayoutParams,
    pinNode,
    unpinAll,
    setNodeSize,
    setEdgeWidth,
    toggleLabels,
    setLODLevel,
    getMetrics,
    frameSelection,
    clear,
    exportGraphData,
    loadGraphData
  } = useGraph()

  const status = useGraphPhysicsStatus()
  const presets = useGraphPhysicsPresets()

  const [activeTab, setActiveTab] = useState<'layout' | 'physics' | 'visual' | 'data'>('layout')

  const metrics = getMetrics()

  const handleExport = () => {
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

  const handleImport = () => {
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
          loadGraphData(data)
        } catch (error) {
          console.error('Failed to parse JSON:', error)
        }
      }
      reader.readAsText(file)
    }

    input.click()
  }

  const generateRandomGraph = () => {
    // Generate a random graph
    clear()

    const nodeCount = 5000
    const nodeIds: number[] = []

    // Add nodes
    for (let i = 0; i < nodeCount; i++) {
      const id = useGraph.getState().addNode({
        label: `Node ${i}`,
        community: Math.floor(i / 500), // 10 communities
        radius: 0.02 + Math.random() * 0.04,
        color: `hsl(${(i / 500) * 36}, 70%, 60%)`
      })
      nodeIds.push(id)
    }

    // Add edges (scale-free network using preferential attachment)
    const edgeCount = nodeCount * 1.5 // Average degree ≈ 3

    for (let i = 0; i < edgeCount; i++) {
      const source = nodeIds[Math.floor(Math.random() * nodeIds.length)]
      let target = nodeIds[Math.floor(Math.random() * nodeIds.length)]

      // Avoid self-loops
      if (source === target) {
        target = nodeIds[(nodeIds.indexOf(source) + 1) % nodeIds.length]
      }

      useGraph.getState().addEdge({
        source,
        target,
        weight: 0.5 + Math.random() * 1.0
      })
    }

    // Start layout
    startLayout()
  }

  return (
    <div className="tool-panel">
      <div className="tool-header">
        <h3>Graph Visualization</h3>
        <button onClick={onClose} className="close-btn">×</button>
      </div>

      {/* Tab Navigation */}
      <div className="tab-nav">
        {(['layout', 'physics', 'visual', 'data'] as const).map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <div className="tool-content">
        {/* Layout Tab */}
        {activeTab === 'layout' && (
          <div className="section">
            <h4>Layout Control</h4>

            <div className="control-group">
              <div className="status-grid">
                <div>Nodes: {metrics.nodeCount.toLocaleString()}</div>
                <div>Edges: {metrics.edgeCount.toLocaleString()}</div>
                <div>Selected: {selectedNodes.size + selectedEdges.size}</div>
                <div>Pinned: {pinnedNodes.size}</div>
              </div>
            </div>

            <div className="control-group">
              <div className="button-row">
                <button
                  onClick={layoutRunning ? stopLayout : startLayout}
                  className={`btn ${layoutRunning ? 'danger' : 'primary'}`}
                >
                  {layoutRunning ? '⏸ Pause' : '▶ Start'} Layout
                </button>
                <button onClick={resetLayout} className="btn secondary">
                  🔄 Reset
                </button>
              </div>
            </div>

            {layoutRunning && (
              <div className="control-group">
                <div className="status-grid">
                  <div>Status: {status.running ? '🟢 Running' : '⚫ Stopped'}</div>
                  <div>Iteration: {layoutIteration.toLocaleString()}</div>
                  <div>Energy: {layoutEnergy.toFixed(3)}</div>
                  <div>Converged: {layoutConverged ? '✅' : '❌'}</div>
                </div>
              </div>
            )}

            <div className="control-group">
              <label>Selection Actions:</label>
              <div className="button-row">
                <button
                  onClick={() => {
                    for (const nodeId of selectedNodes) {
                      pinNode(nodeId, true)
                    }
                  }}
                  disabled={selectedNodes.size === 0}
                  className="btn secondary small"
                >
                  📌 Pin Selected
                </button>
                <button
                  onClick={() => {
                    for (const nodeId of selectedNodes) {
                      pinNode(nodeId, false)
                    }
                  }}
                  disabled={selectedNodes.size === 0}
                  className="btn secondary small"
                >
                  📍 Unpin Selected
                </button>
              </div>
              <button onClick={unpinAll} className="btn secondary small">
                🔓 Unpin All
              </button>
            </div>

            <div className="control-group">
              <button
                onClick={() => {
                  const { center, radius } = frameSelection()
                  // TODO: Use camera hook to animate to position
                  console.log('Frame selection:', center, radius)
                }}
                disabled={selectedNodes.size === 0}
                className="btn secondary"
              >
                🎯 Frame Selection
              </button>
            </div>
          </div>
        )}

        {/* Physics Tab */}
        {activeTab === 'physics' && (
          <div className="section">
            <h4>Physics Parameters</h4>

            <div className="control-group">
              <label>Presets:</label>
              <div className="preset-grid">
                {Object.keys(presets.presets).map((presetName) => (
                  <button
                    key={presetName}
                    onClick={() => presets.apply(presetName as any)}
                    className="btn secondary small"
                  >
                    {presetName}
                  </button>
                ))}
              </div>
            </div>

            <div className="control-group">
              <label>
                Edge Stiffness: {params.edgeStiffness.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                value={params.edgeStiffness}
                onChange={(e) =>
                  setLayoutParams({ edgeStiffness: parseFloat(e.target.value) })
                }
              />
            </div>

            <div className="control-group">
              <label>
                Repulsion: {params.repulsion.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.1"
                max="3.0"
                step="0.1"
                value={params.repulsion}
                onChange={(e) =>
                  setLayoutParams({ repulsion: parseFloat(e.target.value) })
                }
              />
            </div>

            <div className="control-group">
              <label>
                Damping: {params.damping.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="0.99"
                step="0.01"
                value={params.damping}
                onChange={(e) =>
                  setLayoutParams({ damping: parseFloat(e.target.value) })
                }
              />
            </div>

            <div className="control-group">
              <label>
                Edge Length: {params.edgeLength.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.5"
                max="3.0"
                step="0.1"
                value={params.edgeLength}
                onChange={(e) =>
                  setLayoutParams({ edgeLength: parseFloat(e.target.value) })
                }
              />
            </div>

            <div className="control-group">
              <label>
                Bounds: {params.bounds === 0 ? 'Off' : params.bounds.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="20"
                step="1"
                value={params.bounds}
                onChange={(e) =>
                  setLayoutParams({ bounds: parseFloat(e.target.value) })
                }
              />
            </div>

            <div className="control-group">
              <label>
                Center Gravity: {params.gravity.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.01"
                value={params.gravity}
                onChange={(e) =>
                  setLayoutParams({ gravity: parseFloat(e.target.value) })
                }
              />
            </div>
          </div>
        )}

        {/* Visual Tab */}
        {activeTab === 'visual' && (
          <div className="section">
            <h4>Visual Settings</h4>

            <div className="control-group">
              <label>
                Node Size: {nodeSize.toFixed(3)}
              </label>
              <input
                type="range"
                min="0.01"
                max="0.2"
                step="0.005"
                value={nodeSize}
                onChange={(e) => setNodeSize(parseFloat(e.target.value))}
              />
            </div>

            <div className="control-group">
              <label>
                Edge Width: {edgeWidth.toFixed(4)}
              </label>
              <input
                type="range"
                min="0.0001"
                max="0.01"
                step="0.0002"
                value={edgeWidth}
                onChange={(e) => setEdgeWidth(parseFloat(e.target.value))}
              />
            </div>

            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={showLabels}
                  onChange={toggleLabels}
                />
                Show Labels
              </label>
            </div>

            <div className="control-group">
              <label>
                LOD Level: {lodLevel} {lodLevel === 0 ? '(Full Detail)' : '(Simplified)'}
              </label>
              <input
                type="range"
                min="0"
                max="5"
                step="1"
                value={lodLevel}
                onChange={(e) => setLODLevel(parseInt(e.target.value))}
              />
              <small>Higher values hide distant nodes for performance</small>
            </div>

            <div className="control-group">
              <div className="status-grid">
                <div>Rendered Nodes: {status.renderNodeCount}</div>
                <div>Total Nodes: {metrics.nodeCount}</div>
                <div>Performance: {(status.efficiency * 100).toFixed(1)}%</div>
                <div>FPS: {status.fps}</div>
              </div>
            </div>
          </div>
        )}

        {/* Data Tab */}
        {activeTab === 'data' && (
          <div className="section">
            <h4>Graph Data</h4>

            <div className="control-group">
              <div className="button-row">
                <button onClick={handleExport} className="btn secondary">
                  📁 Export JSON
                </button>
                <button onClick={handleImport} className="btn secondary">
                  📂 Import JSON
                </button>
              </div>
            </div>

            <div className="control-group">
              <button onClick={generateRandomGraph} className="btn primary">
                🎲 Generate Random Graph (5k nodes)
              </button>
              <small>Creates a scale-free network with communities</small>
            </div>

            <div className="control-group">
              <button
                onClick={() => {
                  if (confirm('Are you sure you want to clear the entire graph?')) {
                    clear()
                  }
                }}
                className="btn danger"
              >
                🗑 Clear Graph
              </button>
            </div>

            <div className="control-group">
              <h5>Statistics</h5>
              <div className="stats-grid">
                <div>Nodes: {metrics.nodeCount.toLocaleString()}</div>
                <div>Edges: {metrics.edgeCount.toLocaleString()}</div>
                <div>Avg Degree: {metrics.avgDegree.toFixed(1)}</div>
                <div>Max Degree: {metrics.maxDegree}</div>
                <div>Communities: {metrics.communities}</div>
                <div>Connected: {metrics.connected ? '✅' : '❌'}</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .tool-panel {
          width: 340px;
          background: rgba(30, 30, 30, 0.95);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 0;
          color: white;
          font-family: 'SF Mono', 'Monaco', monospace;
          font-size: 12px;
          max-height: 80vh;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .tool-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(0, 0, 0, 0.3);
        }

        .tool-header h3 {
          margin: 0;
          font-size: 14px;
          font-weight: 600;
        }

        .close-btn {
          background: none;
          border: none;
          color: rgba(255, 255, 255, 0.7);
          font-size: 18px;
          cursor: pointer;
          padding: 0;
          width: 20px;
          height: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .close-btn:hover {
          color: white;
        }

        .tab-nav {
          display: flex;
          background: rgba(0, 0, 0, 0.2);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .tab {
          flex: 1;
          background: none;
          border: none;
          color: rgba(255, 255, 255, 0.6);
          padding: 8px 12px;
          font-size: 11px;
          cursor: pointer;
          border-bottom: 2px solid transparent;
        }

        .tab.active {
          color: white;
          border-bottom-color: #64b5f6;
          background: rgba(100, 181, 246, 0.1);
        }

        .tool-content {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
        }

        .section h4 {
          margin: 0 0 16px 0;
          font-size: 13px;
          color: #64b5f6;
          font-weight: 600;
        }

        .control-group {
          margin-bottom: 16px;
        }

        .control-group label {
          display: block;
          margin-bottom: 4px;
          font-size: 11px;
          color: rgba(255, 255, 255, 0.8);
        }

        .control-group input[type="range"] {
          width: 100%;
          margin: 4px 0;
        }

        .control-group input[type="checkbox"] {
          margin-right: 8px;
        }

        .button-row {
          display: flex;
          gap: 8px;
          margin-bottom: 8px;
        }

        .btn {
          padding: 6px 12px;
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 4px;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          font-size: 11px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .btn.primary {
          background: #64b5f6;
          border-color: #64b5f6;
        }

        .btn.secondary {
          background: rgba(255, 255, 255, 0.1);
        }

        .btn.danger {
          background: #f44336;
          border-color: #f44336;
        }

        .btn.small {
          padding: 4px 8px;
          font-size: 10px;
        }

        .btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .status-grid, .stats-grid, .preset-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          font-size: 10px;
          background: rgba(0, 0, 0, 0.3);
          padding: 8px;
          border-radius: 4px;
        }

        .preset-grid {
          grid-template-columns: 1fr 1fr 1fr;
        }

        .status-grid div, .stats-grid div {
          padding: 2px;
        }

        small {
          font-size: 9px;
          color: rgba(255, 255, 255, 0.6);
          display: block;
          margin-top: 4px;
        }
      `}</style>
    </div>
  )
}

export const graphTool: Tool = {
  id: 'graph',
  name: 'Graph Visualization',
  description: 'Physics-driven graph layout and visualization controls',
  category: 'Visualization',
  icon: '🕸️',
  shortcut: 'G',
  component: GraphTool
}
