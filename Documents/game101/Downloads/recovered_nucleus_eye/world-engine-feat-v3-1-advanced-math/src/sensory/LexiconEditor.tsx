import React, { useState, useMemo, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Separator } from '@/components/ui/separator'
import {
  Download,
  Upload,
  Plus,
  Trash2,
  Copy,
  Eye,
  Volume2,
  Hand,
  Flower2,
  Coffee,
  Brain,
  RotateCcw,
  Save
} from 'lucide-react'
import { SensoryLexicon, SensoryChannel, createBaseMoment } from './types'
import { DEFAULT_LEXICON } from './SensoryTokenStream'
import { useUI } from '../state/ui'

// Channel icons and colors
const CHANNEL_CONFIG = {
  sight: { icon: Eye, color: '#3b82f6', label: 'Sight' },
  sound: { icon: Volume2, color: '#10b981', label: 'Sound' },
  touch: { icon: Hand, color: '#f59e0b', label: 'Touch' },
  scent: { icon: Flower2, color: '#8b5cf6', label: 'Scent' },
  taste: { icon: Coffee, color: '#ef4444', label: 'Taste' },
  inner: { icon: Brain, color: '#6366f1', label: 'Inner' }
} as const

interface Props {
  lexicon: SensoryLexicon
  onLexiconChange: (lexicon: SensoryLexicon) => void
  onExport?: (lexicon: SensoryLexicon, filename: string) => void
  onImport?: (lexicon: SensoryLexicon) => void
}

export function LexiconEditor({
  lexicon,
  onLexiconChange,
  onExport,
  onImport
}: Props) {
  const { dyslexiaMode } = useUI()
  const [newWord, setNewWord] = useState('')
  const [selectedWord, setSelectedWord] = useState<string | null>(null)
  const [exportName, setExportName] = useState('custom-lexicon')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Get sorted word list
  const words = useMemo(() => {
    return Object.keys(lexicon).sort()
  }, [lexicon])

  // Add new word to lexicon
  const addWord = () => {
    if (!newWord.trim() || lexicon[newWord.toLowerCase()]) return

    const word = newWord.toLowerCase().trim()
    const newLexicon = {
      ...lexicon,
      [word]: { inner: 0.5 } // Default to inner channel
    }

    onLexiconChange(newLexicon)
    setSelectedWord(word)
    setNewWord('')
  }

  // Remove word from lexicon
  const removeWord = (word: string) => {
    const newLexicon = { ...lexicon }
    delete newLexicon[word]
    onLexiconChange(newLexicon)

    if (selectedWord === word) {
      setSelectedWord(null)
    }
  }

  // Update channel strength for a word
  const updateChannelStrength = (word: string, channel: SensoryChannel, strength: number) => {
    const newLexicon = {
      ...lexicon,
      [word]: {
        ...lexicon[word],
        [channel]: strength > 0 ? strength : undefined
      }
    }

    // Clean up empty channels
    Object.keys(newLexicon[word]).forEach(key => {
      if (newLexicon[word][key as SensoryChannel] === undefined) {
        delete newLexicon[word][key as SensoryChannel]
      }
    })

    onLexiconChange(newLexicon)
  }

  // Export lexicon as JSON
  const handleExport = () => {
    const filename = exportName.endsWith('.json') ? exportName : `${exportName}.json`

    if (onExport) {
      onExport(lexicon, filename)
    } else {
      // Default browser download
      const blob = new Blob([JSON.stringify(lexicon, null, 2)], {
        type: 'application/json'
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  // Import lexicon from file
  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const imported = JSON.parse(e.target?.result as string) as SensoryLexicon
        if (onImport) {
          onImport(imported)
        } else {
          onLexiconChange(imported)
        }
      } catch (error) {
        console.error('Failed to import lexicon:', error)
      }
    }
    reader.readAsText(file)
  }

  // Reset to default lexicon
  const resetToDefault = () => {
    onLexiconChange({ ...DEFAULT_LEXICON })
    setSelectedWord(null)
  }

  // Get channel strength for selected word
  const getChannelStrength = (channel: SensoryChannel): number => {
    if (!selectedWord) return 0
    return lexicon[selectedWord]?.[channel] ?? 0
  }

  return (
    <Card className="bg-gray-900 text-white border-gray-800 h-full">
      <CardHeader className="py-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Brain className="w-4 h-4" />
          Sensory Lexicon Editor
          <Badge variant="outline" className="ml-auto text-xs">
            {words.length} words
          </Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4 text-xs max-h-96 overflow-y-auto">

        {/* Add New Word */}
        <div className="space-y-2">
          <Label className="text-xs font-medium">Add New Word</Label>
          <div className="flex gap-2">
            <Input
              value={newWord}
              onChange={(e) => setNewWord(e.target.value)}
              placeholder="Enter word..."
              className="flex-1 bg-gray-800 border-gray-600 text-xs h-7"
              onKeyDown={(e) => e.key === 'Enter' && addWord()}
            />
            <Button
              size="sm"
              onClick={addWord}
              disabled={!newWord.trim() || !!lexicon[newWord.toLowerCase()]}
              className="h-7 px-2"
            >
              <Plus className="w-3 h-3" />
            </Button>
          </div>
        </div>

        <Separator className="bg-gray-700" />

        {/* Word List */}
        <div className="space-y-2">
          <Label className="text-xs font-medium">Words</Label>
          <div className="max-h-32 overflow-y-auto space-y-1">
            {words.map((word) => {
              const isSelected = selectedWord === word
              const channels = Object.keys(lexicon[word]) as SensoryChannel[]

              return (
                <div
                  key={word}
                  className={`
                    p-2 rounded cursor-pointer border transition-colors
                    ${isSelected
                      ? 'bg-blue-900/50 border-blue-600'
                      : 'bg-gray-800/50 border-gray-700 hover:border-gray-600'
                    }
                  `}
                  onClick={() => setSelectedWord(isSelected ? null : word)}
                >
                  <div className="flex items-center justify-between">
                    <span className={`font-medium ${dyslexiaMode ? 'tracking-wide' : ''}`}>
                      {word}
                    </span>
                    <div className="flex items-center gap-1">
                      {channels.map((channel) => {
                        const config = CHANNEL_CONFIG[channel]
                        const Icon = config.icon
                        const strength = lexicon[word][channel] ?? 0

                        return (
                          <div
                            key={channel}
                            className="w-4 h-4 rounded-full flex items-center justify-center"
                            style={{
                              backgroundColor: `${config.color}${Math.round(strength * 255).toString(16).padStart(2, '0')}`,
                              border: `1px solid ${config.color}`
                            }}
                          >
                            <Icon className="w-2 h-2" style={{ color: config.color }} />
                          </div>
                        )
                      })}
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          removeWord(word)
                        }}
                        className="h-4 w-4 p-0 text-red-400 hover:text-red-300"
                      >
                        <Trash2 className="w-2 h-2" />
                      </Button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <Separator className="bg-gray-700" />

        {/* Channel Editor */}
        {selectedWord && (
          <div className="space-y-3">
            <Label className="text-xs font-medium">
              Editing: <span className="text-blue-400">{selectedWord}</span>
            </Label>

            <div className="space-y-3">
              {(Object.keys(CHANNEL_CONFIG) as SensoryChannel[]).map((channel) => {
                const config = CHANNEL_CONFIG[channel]
                const Icon = config.icon
                const strength = getChannelStrength(channel)

                return (
                  <div key={channel} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Icon className="w-3 h-3" style={{ color: config.color }} />
                        <span className="text-xs">{config.label}</span>
                      </div>
                      <Badge
                        variant="outline"
                        className="text-xs"
                        style={{ borderColor: config.color, color: config.color }}
                      >
                        {(strength * 100).toFixed(0)}%
                      </Badge>
                    </div>

                    <Slider
                      value={[strength]}
                      min={0}
                      max={1}
                      step={0.05}
                      onValueChange={([value]) => updateChannelStrength(selectedWord, channel, value)}
                      className="w-full"
                    />
                  </div>
                )
              })}
            </div>
          </div>
        )}

        <Separator className="bg-gray-700" />

        {/* Export/Import/Reset */}
        <div className="space-y-3">
          <Label className="text-xs font-medium">Lexicon Management</Label>

          {/* Export */}
          <div className="flex gap-2">
            <Input
              value={exportName}
              onChange={(e) => setExportName(e.target.value)}
              placeholder="filename"
              className="flex-1 bg-gray-800 border-gray-600 text-xs h-7"
            />
            <Button size="sm" onClick={handleExport} className="h-7 px-2">
              <Download className="w-3 h-3 mr-1" />
              Export
            </Button>
          </div>

          {/* Import & Reset */}
          <div className="flex gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
            />
            <Button
              size="sm"
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 h-7"
            >
              <Upload className="w-3 h-3 mr-1" />
              Import
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={resetToDefault}
              className="flex-1 h-7"
            >
              <RotateCcw className="w-3 h-3 mr-1" />
              Reset
            </Button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="text-xs text-gray-400 space-y-1">
          <div>Total words: {words.length}</div>
          <div>
            Channels: {(Object.keys(CHANNEL_CONFIG) as SensoryChannel[]).map(channel => {
              const count = words.filter(word => lexicon[word][channel]).length
              const config = CHANNEL_CONFIG[channel]
              return (
                <Badge
                  key={channel}
                  variant="outline"
                  className="text-xs mx-1"
                  style={{ borderColor: config.color, color: config.color }}
                >
                  {config.label}: {count}
                </Badge>
              )
            })}
          </div>
        </div>

      </CardContent>
    </Card>
  )
}

// Hook for managing lexicon state
export function useLexiconEditor(initialLexicon: SensoryLexicon = DEFAULT_LEXICON) {
  const [lexicon, setLexicon] = useState<SensoryLexicon>(initialLexicon)

  const exportLexicon = (customLexicon: SensoryLexicon, filename: string) => {
    const blob = new Blob([JSON.stringify(customLexicon, null, 2)], {
      type: 'application/json'
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const importLexicon = (file: File): Promise<SensoryLexicon> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const imported = JSON.parse(e.target?.result as string) as SensoryLexicon
          setLexicon(imported)
          resolve(imported)
        } catch (error) {
          reject(error)
        }
      }
      reader.onerror = reject
      reader.readAsText(file)
    })
  }

  return {
    lexicon,
    setLexicon,
    exportLexicon,
    importLexicon
  }
}

export default LexiconEditor
