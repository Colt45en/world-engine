import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import {
  Brain,
  Volume2,
  Eye,
  Hand,
  Flower2,
  Coffee,
  Play,
  Pause,
  RotateCcw,
  Settings,
  Zap,
  Info
} from 'lucide-react'
import { useUI } from '../state/ui'
import { SceneMoment, SensoryChannel, createBaseMoment } from './types'
import { LexiconEditor, useLexiconEditor } from './LexiconEditor'

// Channel configuration with accessibility considerations
const CHANNEL_CONFIG = {
  sight: {
    icon: Eye,
    color: '#3b82f6',
    label: 'Sight',
    accessibilityNote: 'Visual effects may be reduced in reduced motion mode'
  },
  sound: {
    icon: Volume2,
    color: '#10b981',
    label: 'Sound',
    accessibilityNote: 'Audio cues respect system volume settings'
  },
  touch: {
    icon: Hand,
    color: '#f59e0b',
    label: 'Touch',
    accessibilityNote: 'Haptic feedback when available'
  },
  scent: {
    icon: Flower2,
    color: '#8b5cf6',
    label: 'Scent',
    accessibilityNote: 'Atmospheric and particle effects'
  },
  taste: {
    icon: Coffee,
    color: '#ef4444',
    label: 'Taste',
    accessibilityNote: 'Color and texture modulation'
  },
  inner: {
    icon: Brain,
    color: '#6366f1',
    label: 'Inner',
    accessibilityNote: 'Mood and focus enhancement'
  }
} as const

interface Props {
  moment: SceneMoment
  onMomentChange: (moment: SceneMoment) => void
  isActive: boolean
  onActiveChange: (active: boolean) => void
}

export default function SensoryTool({
  moment,
  onMomentChange,
  isActive,
  onActiveChange
}: Props) {
  const { dyslexiaMode, reducedMotion, uiScale } = useUI()
  const [showLexiconEditor, setShowLexiconEditor] = useState(false)
  const [testText, setTestText] = useState('sunrise storm memory warmth candle whisper')
  const [tokenSpeed, setTokenSpeed] = useState(6)
  const [proximityRadius, setProximityRadius] = useState(4)
  const [respectAccessibility, setRespectAccessibility] = useState(true)

  const { lexicon, setLexicon } = useLexiconEditor()

  // Update channel strength
  const updateChannelStrength = (channel: SensoryChannel, strength: number) => {
    const clone = structuredClone(moment)
    const detail = clone.details.find(d => d.channel === channel)
    if (detail) {
      detail.strength = Math.max(0, Math.min(1, strength))
      onMomentChange(clone)
    }
  }

  // Reset to base moment
  const resetMoment = () => {
    onMomentChange(createBaseMoment())
  }

  // Get channel strength
  const getChannelStrength = (channel: SensoryChannel): number => {
    return moment.details.find(d => d.channel === channel)?.strength ?? 0
  }

  // Apply accessibility constraints
  const getMaxStrength = (channel: SensoryChannel): number => {
    if (!respectAccessibility || !reducedMotion) return 1.0

    // Reduced intensity caps for motion-sensitive users
    switch (channel) {
      case 'sight': return 0.8  // Reduce visual intensity
      case 'sound': return 0.9  // Moderate sound reduction
      case 'touch': return 0.7  // Significant haptic reduction
      case 'scent': return 0.8  // Moderate particle reduction
      case 'taste': return 0.8  // Moderate color shift reduction
      case 'inner': return 1.0  // Inner effects are usually safe
      default: return 0.8
    }
  }

  return (
    <Card className="bg-gray-900 text-white border-gray-800">
      <CardHeader className="py-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Brain className="w-4 h-4" />
          Sensory System
          <div className="flex items-center gap-2 ml-auto">
            <Switch
              checked={isActive}
              onCheckedChange={onActiveChange}
              className="scale-75"
            />
            <Badge variant={isActive ? 'default' : 'outline'} className="text-xs">
              {isActive ? 'Active' : 'Paused'}
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4 text-xs">

        {/* Accessibility Status */}
        {(dyslexiaMode || reducedMotion) && (
          <div className="bg-blue-900/20 p-2 rounded border border-blue-800">
            <div className="flex items-start gap-2">
              <Info className="w-3 h-3 mt-0.5 text-blue-400" />
              <div className="text-xs">
                <div className="font-medium text-blue-300">Accessibility Mode Active</div>
                <ul className="mt-1 space-y-0.5 text-xs">
                  {dyslexiaMode && <li>• Enhanced text readability</li>}
                  {reducedMotion && <li>• Reduced effect intensity</li>}
                  {respectAccessibility && <li>• Gentle sensory modulation</li>}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Live Channel Strengths */}
        <div className="space-y-3">
          <Label className="text-xs font-medium flex items-center gap-2">
            <Zap className="w-3 h-3" />
            Channel Strengths
          </Label>

          {(Object.keys(CHANNEL_CONFIG) as SensoryChannel[]).map((channel) => {
            const config = CHANNEL_CONFIG[channel]
            const Icon = config.icon
            const strength = getChannelStrength(channel)
            const maxStrength = getMaxStrength(channel)
            const isReduced = maxStrength < 1.0

            return (
              <div key={channel} className="space-y-1">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className="w-3 h-3" style={{ color: config.color }} />
                    <span className={dyslexiaMode ? 'tracking-wide' : ''}>{config.label}</span>
                    {isReduced && (
                      <Badge variant="outline" className="text-xs px-1 py-0">
                        Capped
                      </Badge>
                    )}
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
                  max={maxStrength}
                  step={0.05}
                  onValueChange={([value]) => updateChannelStrength(channel, value)}
                  className="w-full"
                  disabled={!isActive}
                />

                {reducedMotion && respectAccessibility && (
                  <div className="text-xs text-gray-500">
                    {config.accessibilityNote}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Test Text Input */}
        <div className="space-y-2">
          <Label className="text-xs font-medium">Token Stream Test</Label>
          <Textarea
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            placeholder="Enter words to test sensory mapping..."
            className="bg-gray-800 border-gray-600 text-xs resize-none"
            rows={2}
            disabled={!isActive}
          />
          <div className="flex items-center gap-2">
            <Label className="text-xs">Speed:</Label>
            <Slider
              value={[tokenSpeed]}
              min={1}
              max={20}
              step={1}
              onValueChange={([value]) => setTokenSpeed(value)}
              className="flex-1"
              disabled={!isActive}
            />
            <Badge variant="outline" className="text-xs">
              {tokenSpeed} tps
            </Badge>
          </div>
        </div>

        {/* Proximity Settings */}
        <div className="space-y-2">
          <Label className="text-xs font-medium">Proximity Volume</Label>
          <div className="flex items-center gap-2">
            <Label className="text-xs">Radius:</Label>
            <Slider
              value={[proximityRadius]}
              min={1}
              max={20}
              step={0.5}
              onValueChange={([value]) => setProximityRadius(value)}
              className="flex-1"
              disabled={!isActive}
            />
            <Badge variant="outline" className="text-xs">
              {proximityRadius}m
            </Badge>
          </div>
        </div>

        {/* Accessibility Options */}
        <div className="space-y-2 pt-2 border-t border-gray-700">
          <Label className="text-xs font-medium">Accessibility Options</Label>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Label className="text-xs">Respect Motion Settings</Label>
              <Info className="w-3 h-3 text-gray-400" title="Automatically reduce intensity for motion-sensitive users" />
            </div>
            <Switch
              checked={respectAccessibility}
              onCheckedChange={setRespectAccessibility}
              className="scale-75"
            />
          </div>

          {respectAccessibility && reducedMotion && (
            <div className="text-xs text-gray-400 bg-gray-800/50 p-2 rounded">
              Sensory effects are automatically reduced by 20-30% to respect your motion preferences.
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2 pt-2 border-t border-gray-700">
          <Button
            size="sm"
            variant="outline"
            onClick={resetMoment}
            className="flex-1 text-xs h-7"
          >
            <RotateCcw className="w-3 h-3 mr-1" />
            Reset
          </Button>

          <Button
            size="sm"
            variant="outline"
            onClick={() => setShowLexiconEditor(!showLexiconEditor)}
            className="flex-1 text-xs h-7"
          >
            <Settings className="w-3 h-3 mr-1" />
            Lexicon
          </Button>
        </div>

        {/* Lexicon Editor */}
        {showLexiconEditor && (
          <div className="pt-2 border-t border-gray-700">
            <LexiconEditor
              lexicon={lexicon}
              onLexiconChange={setLexicon}
            />
          </div>
        )}

        {/* Live Status */}
        <div className="text-xs text-gray-400 space-y-1 pt-2 border-t border-gray-700">
          <div>System: {isActive ? '🟢 Active' : '⚫ Paused'}</div>
          <div>Motion Mode: {reducedMotion ? '🔵 Reduced' : '🟡 Full'}</div>
          <div>UI Scale: {(uiScale * 100).toFixed(0)}%</div>
          <div>Channels: {moment.details.filter(d => d.strength > 0.1).length}/6 active</div>
        </div>

      </CardContent>
    </Card>
  )
}
