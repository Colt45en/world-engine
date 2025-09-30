import * as React from 'react'
import AppCanvas from './AppCanvas'
import { ToolManager } from './tools/ToolManager'
import { Dashboard } from './Dashboard'
import { VisualBleedway } from './visual/VisualBleedway'
import { NexusRoom } from './spatial/NexusRoom'
import { CryptoDashboard } from './financial/CryptoDashboard'
import { CrashSafeRAGChat, useCrashSafeRAG } from './ai/CrashSafeRAG'
import { NexusWidgetDashboard } from './widgets/NexusWidgetManager'
import { useUI } from './state/ui'

export default function App() {
  const { dyslexiaMode, reducedMotion, uiScale } = useUI()
  const { query, isReady, trainingState, startTraining, stopTraining } = useCrashSafeRAG()
  const [currentRoute, setCurrentRoute] = React.useState('')

  React.useEffect(() => {
    const handleHashChange = () => {
      setCurrentRoute(window.location.hash.slice(1))
    }

    handleHashChange() // Initial load
    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  // Route-specific rendering
  const renderMainContent = () => {
    switch (currentRoute) {
      case 'free-mode':
        // Free mode with full toolset
        return (
          <div className="grid grid-cols-4 h-screen">
            <div className="col-span-3">
              <AppCanvas />
            </div>
            <div className="p-4 bg-black/80 overflow-y-auto">
              <ToolManager />
            </div>
          </div>
        )

      case 'sandbox-360':
        // 360 Camera prototype (dev-only)
        return <Dashboard />

      case 'visual-bleedway':
        // Visual Bleedway: mask → mesh → sensory overlay
        return <VisualBleedway />

      case 'nexus-room':
        // NEXUS 3D Iframe Room: immersive multi-panel environment
        return <NexusRoom />

      case 'crypto-dashboard':
        // Advanced crypto trading dashboard with 3D visualization
        return <CryptoDashboard />

      case 'nexus-widgets':
        // Nexus Widget Dashboard: Connect to all available widgets
        return <NexusWidgetDashboard bridge={{ query, isReady, trainingState, startTraining, stopTraining }} />

      default:
        // Default dashboard
        return <Dashboard />
    }
  }

  return (
    <div
      className={[
        dyslexiaMode ? 'dyslexia-mode' : '',
        reducedMotion ? 'reduced-motion' : '',
      ].filter(Boolean).join(' ')}
      style={{ ['--ui-scale' as any]: uiScale }}
    >
      {renderMainContent()}
      <CrashSafeRAGChat />
    </div>
  )
}
