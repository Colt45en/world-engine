import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ChevronDown, ChevronRight } from 'lucide-react'
import AccessibilityTool from './AccessibilityTool'
import SensoryTool from './SensoryTool'
import CommandPalette from './CommandPalette'
import { createBaseMoment } from '../sensory/types'

interface Tool {
  id: string
  label: string
  enable: () => void
  disable: () => void
  Panel: React.ComponentType
  isActive?: boolean
}

// Simple placeholder tools for demonstration
function EntityTool() {
  return (
    <Card className="bg-gray-900 text-white border-gray-800">
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Entities</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        <Button size="sm" className="w-full">Add Cube</Button>
        <Button size="sm" className="w-full">Add Sphere</Button>
        <Button size="sm" variant="outline" className="w-full">Clear All</Button>
      </CardContent>
    </Card>
  )
}

function PropertiesTool() {
  return (
    <Card className="bg-gray-900 text-white border-gray-800">
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Properties</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        <div>
          <label className="block mb-1">Position</label>
          <div className="grid grid-cols-3 gap-1">
            <input type="number" placeholder="X" className="bg-gray-800 p-1 rounded text-xs" />
            <input type="number" placeholder="Y" className="bg-gray-800 p-1 rounded text-xs" />
            <input type="number" placeholder="Z" className="bg-gray-800 p-1 rounded text-xs" />
          </div>
        </div>
        <div>
          <label className="block mb-1">Color</label>
          <input type="color" className="w-full h-8 rounded" />
        </div>
      </CardContent>
    </Card>
  )
}

function SceneTool() {
  return (
    <Card className="bg-gray-900 text-white border-gray-800">
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Scene</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        <div className="space-y-1">
          <div className="flex justify-between">
            <span>Animation Speed</span>
            <Badge variant="outline">1.0x</Badge>
          </div>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            defaultValue="1"
            className="w-full"
          />
        </div>
        <Button size="sm" variant="outline" className="w-full">Reset Camera</Button>
        <Button size="sm" variant="outline" className="w-full">Export Scene</Button>
      </CardContent>
    </Card>
  )
}

// Sensory tool wrapper component
function SensoryToolWrapper() {
  const [sensoryMoment, setSensoryMoment] = useState(createBaseMoment())
  const [isActive, setIsActive] = useState(false)

  return (
    <SensoryTool
      moment={sensoryMoment}
      onMomentChange={setSensoryMoment}
      isActive={isActive}
      onActiveChange={setIsActive}
    />
  )
}

const builtInTools: Tool[] = [
  {
    id: 'entities',
    label: 'Entities',
    enable: () => {},
    disable: () => {},
    Panel: EntityTool,
  },
  {
    id: 'properties',
    label: 'Properties',
    enable: () => {},
    disable: () => {},
    Panel: PropertiesTool,
  },
  {
    id: 'scene',
    label: 'Scene',
    enable: () => {},
    disable: () => {},
    Panel: SceneTool,
  },
  {
    id: 'sensory',
    label: 'Sensory System',
    enable: () => {},
    disable: () => {},
    Panel: SensoryToolWrapper,
  },
  {
    id: 'accessibility',
    label: 'Accessibility',
    enable: () => {},
    disable: () => {},
    Panel: AccessibilityTool,
  },
]

export function ToolManager() {
  const [activePanels, setActivePanels] = useState<string[]>([
    'entities',
    'sensory',
    'accessibility'
  ])
  const [showCommandPalette, setShowCommandPalette] = useState(false)

  const togglePanel = (toolId: string) => {
    setActivePanels(prev =>
      prev.includes(toolId)
        ? prev.filter(id => id !== toolId)
        : [...prev, toolId]
    )
  }

  return (
    <div className="space-y-3">
      {/* Header with Command Palette trigger */}
      <Card className="bg-gray-900 text-white border-gray-800">
        <CardContent className="p-3">
          <Button
            onClick={() => setShowCommandPalette(true)}
            className="w-full text-sm justify-start"
            variant="outline"
          >
            <span className="mr-2">⌘</span>
            Command Palette
            <Badge variant="secondary" className="ml-auto text-xs">
              Ctrl+K
            </Badge>
          </Button>
        </CardContent>
      </Card>

      {/* Tool Panel Toggles */}
      <Card className="bg-gray-900 text-white border-gray-800">
        <CardHeader className="py-2">
          <CardTitle className="text-sm">Tools</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 p-3 pt-0">
          {builtInTools.map(tool => {
            const isActive = activePanels.includes(tool.id)
            return (
              <Button
                key={tool.id}
                onClick={() => togglePanel(tool.id)}
                variant="ghost"
                className="w-full justify-start text-xs h-7 px-2"
              >
                {isActive ? (
                  <ChevronDown className="w-3 h-3 mr-2" />
                ) : (
                  <ChevronRight className="w-3 h-3 mr-2" />
                )}
                {tool.label}
              </Button>
            )
          })}
        </CardContent>
      </Card>

      {/* Active Tool Panels */}
      {builtInTools
        .filter(tool => activePanels.includes(tool.id))
        .map(tool => {
          const PanelComponent = tool.Panel
          return <PanelComponent key={tool.id} />
        })}

      {/* Command Palette */}
      {showCommandPalette && <CommandPalette />}
    </div>
  )
}

export default ToolManager
