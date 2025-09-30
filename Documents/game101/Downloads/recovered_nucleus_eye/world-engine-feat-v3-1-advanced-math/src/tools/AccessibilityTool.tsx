import React from 'react'
import { useUI } from '../state/ui'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Eye, Minimize2, Type, RotateCcw, Info } from 'lucide-react'

export default function AccessibilityTool() {
  const {
    dyslexiaMode,
    reducedMotion,
    uiScale,
    toggleReaderMode,
    toggleReducedMotion,
    setUiScale,
    bumpUiScale
  } = useUI()

  const scalePercentage = Math.round(uiScale * 100)

  return (
    <Card className="bg-gray-900 text-white border-gray-800">
      <CardHeader className="py-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Eye className="w-4 h-4" />
          Accessibility
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-xs">

        {/* Reader Mode Section */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Type className="w-3 h-3 text-blue-400" />
            <span className="font-medium text-blue-400">Reader Mode</span>
          </div>
          <Button
            size="sm"
            onClick={toggleReaderMode}
            variant={dyslexiaMode ? 'default' : 'outline'}
            className="w-full justify-start text-xs"
          >
            {dyslexiaMode ? '✓ Enabled' : 'Enable'} Dyslexia-Friendly Text
          </Button>
          {dyslexiaMode && (
            <div className="text-xs text-gray-400 bg-blue-900/20 p-2 rounded border border-blue-800">
              <div className="flex items-start gap-2">
                <Info className="w-3 h-3 mt-0.5 text-blue-400" />
                <div>
                  <div className="font-medium text-blue-300">Active Features:</div>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    <li>• Atkinson Hyperlegible font</li>
                    <li>• Enhanced letter spacing</li>
                    <li>• Stronger focus indicators</li>
                    <li>• Optimized line height</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Motion Section */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Minimize2 className="w-3 h-3 text-purple-400" />
            <span className="font-medium text-purple-400">Motion Settings</span>
          </div>
          <Button
            size="sm"
            onClick={toggleReducedMotion}
            variant={reducedMotion ? 'default' : 'outline'}
            className="w-full justify-start text-xs"
          >
            {reducedMotion ? '✓ Enabled' : 'Enable'} Reduce Motion
          </Button>
          {reducedMotion && (
            <div className="text-xs text-gray-400 bg-purple-900/20 p-2 rounded border border-purple-800">
              <div className="flex items-start gap-2">
                <Info className="w-3 h-3 mt-0.5 text-purple-400" />
                <div>
                  <div className="font-medium text-purple-300">Active Features:</div>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    <li>• Shortened UI animations</li>
                    <li>• Reduced camera damping</li>
                    <li>• Slower scene animations</li>
                    <li>• Minimal visual effects</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* UI Scale Section */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Type className="w-3 h-3 text-green-400" />
            <span className="font-medium text-green-400">UI Font Scale</span>
            <Badge variant="outline" className="text-xs ml-auto">
              {scalePercentage}%
            </Badge>
          </div>

          <div className="space-y-2">
            <Slider
              value={[uiScale]}
              min={0.8}
              max={1.6}
              step={0.05}
              onValueChange={([v]) => setUiScale(v)}
              className="w-full"
            />

            <div className="flex gap-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => bumpUiScale(-0.05)}
                className="flex-1 text-xs"
                disabled={uiScale <= 0.8}
              >
                –
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setUiScale(1)}
                className="flex-1 text-xs"
              >
                <RotateCcw className="w-3 h-3" />
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => bumpUiScale(+0.05)}
                className="flex-1 text-xs"
                disabled={uiScale >= 1.6}
              >
                +
              </Button>
            </div>

            <div className="text-xs text-gray-500 text-center">
              Range: 80% - 160%
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="space-y-2 pt-2 border-t border-gray-700">
          <div className="font-medium text-gray-300 text-xs">Quick Actions</div>
          <div className="grid grid-cols-2 gap-1 text-xs">
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                if (!dyslexiaMode) toggleReaderMode()
                if (!reducedMotion) toggleReducedMotion()
                setUiScale(1.2)
              }}
              className="text-xs"
            >
              Max Comfort
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                if (dyslexiaMode) toggleReaderMode()
                if (reducedMotion) toggleReducedMotion()
                setUiScale(1)
              }}
              className="text-xs"
            >
              Reset All
            </Button>
          </div>
        </div>

        {/* Keyboard Shortcuts Info */}
        <div className="space-y-1 pt-2 border-t border-gray-700">
          <div className="font-medium text-gray-300 text-xs">Keyboard Shortcuts</div>
          <div className="space-y-1 text-xs text-gray-400">
            <div className="flex justify-between">
              <span>Toggle Reader Mode</span>
              <Badge variant="outline" className="text-xs">R</Badge>
            </div>
            <div className="flex justify-between">
              <span>Toggle Reduce Motion</span>
              <Badge variant="outline" className="text-xs">M</Badge>
            </div>
            <div className="flex justify-between">
              <span>Font Size +/-</span>
              <div className="flex gap-1">
                <Badge variant="outline" className="text-xs">+</Badge>
                <Badge variant="outline" className="text-xs">-</Badge>
              </div>
            </div>
            <div className="flex justify-between">
              <span>Command Palette</span>
              <Badge variant="outline" className="text-xs">Ctrl+K</Badge>
            </div>
          </div>
        </div>

      </CardContent>
    </Card>
  )
}
