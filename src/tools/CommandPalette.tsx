import React, { useState, useEffect, useMemo } from 'react'
import { useEditor } from '../state/editor'
import { useUI } from '../state/ui'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'

export interface Command {
  id: string
  title: string
  run: () => void
  shortcut?: string[]
  group?: string
  description?: string
}

export function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)

  const {
    dyslexiaMode,
    reducedMotion,
    uiScale,
    toggleReaderMode,
    toggleReducedMotion,
    bumpUiScale,
    setUiScale
  } = useUI()

  const {
    entities,
    selectedEntityId,
    addEntity,
    deleteEntity,
    clearScene,
    undo,
    redo,
    canUndo,
    canRedo
  } = useEditor()

  const commands: Command[] = useMemo(() => [
    // Entity Management
    {
      id: 'entity.add.cube',
      title: 'Add Cube',
      run: () => addEntity('cube'),
      shortcut: ['C'],
      group: 'Entities',
      description: 'Add a new cube to the scene'
    },
    {
      id: 'entity.add.sphere',
      title: 'Add Sphere',
      run: () => addEntity('sphere'),
      shortcut: ['S'],
      group: 'Entities',
      description: 'Add a new sphere to the scene'
    },
    {
      id: 'entity.delete',
      title: 'Delete Selected Entity',
      run: () => selectedEntityId && deleteEntity(selectedEntityId),
      shortcut: ['Delete'],
      group: 'Entities',
      description: 'Delete the currently selected entity'
    },
    {
      id: 'scene.clear',
      title: 'Clear Scene',
      run: clearScene,
      shortcut: ['Ctrl', 'Shift', 'N'],
      group: 'Scene',
      description: 'Remove all entities from the scene'
    },

    // History
    {
      id: 'edit.undo',
      title: 'Undo',
      run: undo,
      shortcut: ['Ctrl', 'Z'],
      group: 'Edit',
      description: 'Undo the last action'
    },
    {
      id: 'edit.redo',
      title: 'Redo',
      run: redo,
      shortcut: ['Ctrl', 'Y'],
      group: 'Edit',
      description: 'Redo the last undone action'
    },

    // Accessibility Commands
    {
      id: 'ui.readerMode.toggle',
      title: `Reader Mode: ${dyslexiaMode ? 'Disable' : 'Enable'}`,
      run: toggleReaderMode,
      shortcut: ['R'],
      group: 'Accessibility',
      description: 'Toggle dyslexia-friendly fonts and enhanced focus indicators'
    },
    {
      id: 'ui.reducedMotion.toggle',
      title: `Reduce Motion: ${reducedMotion ? 'Disable' : 'Enable'}`,
      run: toggleReducedMotion,
      shortcut: ['M'],
      group: 'Accessibility',
      description: 'Reduce or disable UI animations and scene motion'
    },
    {
      id: 'ui.font.increase',
      title: 'UI Font: Increase',
      run: () => bumpUiScale(+0.05),
      shortcut: ['+'],
      group: 'Accessibility',
      description: 'Increase UI font size (5% increment)'
    },
    {
      id: 'ui.font.decrease',
      title: 'UI Font: Decrease',
      run: () => bumpUiScale(-0.05),
      shortcut: ['-'],
      group: 'Accessibility',
      description: 'Decrease UI font size (5% decrement)'
    },
    {
      id: 'ui.font.reset',
      title: 'UI Font: Reset',
      run: () => setUiScale(1),
      shortcut: ['0'],
      group: 'Accessibility',
      description: 'Reset UI font size to 100%'
    },

    // View Controls
    {
      id: 'view.focusSelected',
      title: 'Focus Selected',
      run: () => {}, // TODO: Implement camera focus
      shortcut: ['F'],
      group: 'View',
      description: 'Focus camera on selected entity'
    },
    {
      id: 'view.resetCamera',
      title: 'Reset Camera',
      run: () => {}, // TODO: Implement camera reset
      shortcut: ['Home'],
      group: 'View',
      description: 'Reset camera to default position'
    },

  ], [
    dyslexiaMode,
    reducedMotion,
    uiScale,
    toggleReaderMode,
    toggleReducedMotion,
    bumpUiScale,
    setUiScale,
    addEntity,
    deleteEntity,
    selectedEntityId,
    clearScene,
    undo,
    redo,
    canUndo,
    canRedo
  ])

  // Filter commands based on search
  const filteredCommands = useMemo(() => {
    if (!search) return commands
    const searchLower = search.toLowerCase()
    return commands.filter(cmd =>
      cmd.title.toLowerCase().includes(searchLower) ||
      cmd.group?.toLowerCase().includes(searchLower) ||
      cmd.description?.toLowerCase().includes(searchLower)
    )
  }, [commands, search])

  // Group filtered commands
  const groupedCommands = useMemo(() => {
    const groups: Record<string, Command[]> = {}
    filteredCommands.forEach(cmd => {
      const group = cmd.group || 'Other'
      if (!groups[group]) groups[group] = []
      groups[group].push(cmd)
    })
    return groups
  }, [filteredCommands])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Open/close palette
      if (e.key === 'k' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        setIsOpen(prev => !prev)
        return
      }

      if (!isOpen) {
        // Global shortcuts when palette is closed
        commands.forEach(cmd => {
          if (cmd.shortcut && matchesShortcut(e, cmd.shortcut)) {
            e.preventDefault()
            cmd.run()
          }
        })
        return
      }

      // Palette navigation
      if (e.key === 'Escape') {
        setIsOpen(false)
        setSearch('')
        setSelectedIndex(0)
      } else if (e.key === 'Enter') {
        e.preventDefault()
        const allCommands = Object.values(groupedCommands).flat()
        if (allCommands[selectedIndex]) {
          allCommands[selectedIndex].run()
          setIsOpen(false)
          setSearch('')
          setSelectedIndex(0)
        }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        const totalCommands = Object.values(groupedCommands).flat().length
        setSelectedIndex(prev => Math.min(prev + 1, totalCommands - 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex(prev => Math.max(prev - 1, 0))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, groupedCommands, selectedIndex, commands])

  const matchesShortcut = (event: KeyboardEvent, shortcut: string[]): boolean => {
    const modifiers = {
      'Ctrl': event.ctrlKey,
      'Alt': event.altKey,
      'Shift': event.shiftKey,
      'Meta': event.metaKey
    }

    return shortcut.every(key => {
      if (key in modifiers) {
        return modifiers[key as keyof typeof modifiers]
      }
      return event.key.toLowerCase() === key.toLowerCase()
    })
  }

  if (!isOpen) return null

  let commandIndex = 0

  return (
    <div className="fixed inset-0 bg-black/50 flex items-start justify-center pt-32 z-50">
      <Card className="w-full max-w-2xl bg-gray-900 border-gray-700">
        <div className="p-4">
          <Input
            placeholder="Type a command or search..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setSelectedIndex(0)
            }}
            className="bg-gray-800 border-gray-600 text-white"
            autoFocus
          />
        </div>

        <div className="max-h-96 overflow-y-auto">
          {Object.entries(groupedCommands).map(([groupName, groupCommands]) => (
            <div key={groupName} className="px-4 pb-2">
              <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 px-2">
                {groupName}
              </div>
              {groupCommands.map((command) => {
                const isSelected = commandIndex === selectedIndex
                const currentIndex = commandIndex++

                return (
                  <div
                    key={command.id}
                    className={`
                      px-3 py-2 rounded cursor-pointer flex items-center justify-between
                      ${isSelected ? 'bg-blue-600 text-white' : 'hover:bg-gray-700 text-gray-200'}
                    `}
                    onClick={() => {
                      command.run()
                      setIsOpen(false)
                      setSearch('')
                      setSelectedIndex(0)
                    }}
                  >
                    <div className="flex-1">
                      <div className="font-medium">{command.title}</div>
                      {command.description && (
                        <div className="text-xs text-gray-400 mt-1">
                          {command.description}
                        </div>
                      )}
                    </div>
                    {command.shortcut && (
                      <div className="flex gap-1">
                        {command.shortcut.map((key, i) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {key}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ))}

          {filteredCommands.length === 0 && (
            <div className="px-4 py-8 text-center text-gray-400">
              No commands found for "{search}"
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}

export default CommandPalette
