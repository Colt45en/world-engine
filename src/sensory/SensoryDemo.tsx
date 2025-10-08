import React, { useState, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { Box, Sphere, Text } from '@react-three/drei'
import { useUI } from '../state/ui'
import {
  SceneMoment,
  SensoryChannel,
  createBaseMoment,
  ChannelUtils
} from '../sensory/types'
import { SensoryTokenStream } from '../sensory/SensoryTokenStream'
import { ProximityVolume, MultiProximityVolume } from '../sensory/ProximityVolume'
import { SensoryFX, SensoryFXDebug } from './SensoryFX'

// Demo text with rich sensory vocabulary
const DEMO_TEXTS = {
  forest: "sunrise through canopy whisper rustle warm earth scent memory peace",
  storm: "thunder lightning rain wind cold sharp ozone electric dark power",
  ocean: "waves splash salt breeze foam deep blue rhythm endless flow",
  memory: "time drift soft warm candle glow inner peace silence dream",
  city: "bright neon rush concrete metal glass echo busy sharp focus"
}

interface SensoryDemoProps {
  selectedText?: keyof typeof DEMO_TEXTS
  showDebug?: boolean
  enablePostFX?: boolean
}

export function SensoryDemo({
  selectedText = 'forest',
  showDebug = false,
  enablePostFX = true
}: SensoryDemoProps) {
  const { reducedMotion } = useUI()

  // Scene objects
  const centerBox = useRef<THREE.Mesh>(null!)
  const ambientSphere = useRef<THREE.Mesh>(null!)

  // Sensory state
  const [baseMoment] = useState(createBaseMoment())
  const [currentMoment, setCurrentMoment] = useState<SceneMoment>(baseMoment)
  const [proximityGains, setProximityGains] = useState<Record<string, number>>({})

  // Animation state
  const timeRef = useRef(0)

  // Get channel strength helper
  const getChannelStrength = (channel: SensoryChannel): number => {
    return ChannelUtils.getStrength(currentMoment, channel)
  }

  // Process proximity gains to modulate moment
  const handleProximityGains = (gains: Record<string, number>) => {
    setProximityGains(gains)

    // Apply proximity modulation to the current moment
    const clone = structuredClone(currentMoment)

    // Intimate zone affects inner and scent channels
    const intimateGain = gains.intimate ?? 0
    if (intimateGain > 0.1) {
      const innerDetail = clone.details.find(d => d.channel === 'inner')
      const scentDetail = clone.details.find(d => d.channel === 'scent')

      if (innerDetail) {
        innerDetail.strength = Math.min(1, innerDetail.strength * (0.8 + intimateGain * 0.4))
      }
      if (scentDetail) {
        scentDetail.strength = Math.min(1, scentDetail.strength * (0.9 + intimateGain * 0.3))
      }
    }

    // Personal zone affects sound and touch
    const personalGain = gains.personal ?? 0
    if (personalGain > 0.1) {
      const soundDetail = clone.details.find(d => d.channel === 'sound')
      const touchDetail = clone.details.find(d => d.channel === 'touch')

      if (soundDetail) {
        soundDetail.strength = Math.min(1, soundDetail.strength * (0.7 + personalGain * 0.5))
      }
      if (touchDetail) {
        touchDetail.strength = Math.min(1, touchDetail.strength * (0.8 + personalGain * 0.4))
      }
    }

    setCurrentMoment(clone)
  }

  // Animate scene based on sensory channels
  useFrame((state) => {
    timeRef.current = state.clock.getElapsedTime()
    const motionMultiplier = reducedMotion ? 0.3 : 1.0

    // Animate center box based on channels
    if (centerBox.current) {
      const sightStrength = getChannelStrength('sight')
      const soundStrength = getChannelStrength('sound')
      const innerStrength = getChannelStrength('inner')

      // Scale based on sight + inner
      const scale = 0.8 + (sightStrength + innerStrength) * 0.4
      centerBox.current.scale.setScalar(scale)

      // Color modulation based on channels
      const material = centerBox.current.material as THREE.MeshStandardMaterial
      material.color.setRGB(
        0.2 + sightStrength * 0.6,
        0.3 + soundStrength * 0.5,
        0.4 + innerStrength * 0.4
      )

      // Subtle rotation based on sound
      centerBox.current.rotation.y += soundStrength * 0.01 * motionMultiplier

      // Position bobbing based on inner strength
      const bob = Math.sin(timeRef.current * 2) * innerStrength * 0.1 * motionMultiplier
      centerBox.current.position.y = 0.5 + bob
    }

    // Animate ambient sphere
    if (ambientSphere.current) {
      const touchStrength = getChannelStrength('touch')
      const scentStrength = getChannelStrength('scent')
      const tasteStrength = getChannelStrength('taste')

      // Scale based on atmospheric channels
      const atmosphereIntensity = (touchStrength + scentStrength + tasteStrength) / 3
      const scale = 2 + atmosphereIntensity * 1.5
      ambientSphere.current.scale.setScalar(scale)

      // Opacity modulation
      const material = ambientSphere.current.material as THREE.MeshStandardMaterial
      material.opacity = 0.1 + atmosphereIntensity * 0.3
      material.color.setRGB(
        0.5 + tasteStrength * 0.3,
        0.4 + scentStrength * 0.4,
        0.6 + touchStrength * 0.3
      )

      // Slow rotation
      ambientSphere.current.rotation.x += 0.005 * motionMultiplier
      ambientSphere.current.rotation.z += 0.003 * motionMultiplier
    }
  })

  return (
    <>
      {/* Token Stream Processing */}
      <SensoryTokenStream
        text={DEMO_TEXTS[selectedText]}
        base={baseMoment}
        onMoment={setCurrentMoment}
        tps={reducedMotion ? 3 : 6} // Slower for reduced motion
        respectAccessibility={true}
      />

      {/* Multi-zone Proximity System */}
      <MultiProximityVolume
        center={centerBox.current ?? new THREE.Vector3(0, 0, 0)}
        presets={['intimate', 'personal', 'environmental']}
        onGains={handleProximityGains}
        debug={showDebug}
        respectAccessibility={true}
      />

      {/* Center Interactive Object */}
      <Box
        ref={centerBox}
        position={[0, 0.5, 0]}
        args={[1, 1, 1]}
      >
        <meshStandardMaterial
          color="#2a2f3a"
          roughness={0.4}
          metalness={0.1}
        />
      </Box>

      {/* Ambient Atmosphere Sphere */}
      <Sphere
        ref={ambientSphere}
        position={[0, 0, 0]}
        args={[3, 32, 32]}
      >
        <meshStandardMaterial
          color="#4a5568"
          transparent
          opacity={0.2}
          wireframe
        />
      </Sphere>

      {/* Floating Text Labels */}
      <Text
        position={[0, 2, 0]}
        fontSize={0.2}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {selectedText.toUpperCase()} SCENE
      </Text>

      {/* Channel Strength Indicators */}
      {(['sight', 'sound', 'touch', 'scent', 'taste', 'inner'] as SensoryChannel[]).map((channel, index) => {
        const strength = getChannelStrength(channel)
        const angle = (index / 6) * Math.PI * 2
        const radius = 4
        const x = Math.cos(angle) * radius
        const z = Math.sin(angle) * radius

        return (
          <Text
            key={channel}
            position={[x, 1 + strength, z]}
            fontSize={0.15}
            color={strength > 0.5 ? '#66ccff' : '#888888'}
            anchorX="center"
            anchorY="middle"
          >
            {channel.toUpperCase()}\n{(strength * 100).toFixed(0)}%
          </Text>
        )
      })}

      {/* Proximity Indicators */}
      {showDebug && (
        <>
          {Object.entries(proximityGains).map(([zone, gain]) => (
            <Text
              key={zone}
              position={[-4, 2 - Object.keys(proximityGains).indexOf(zone) * 0.4, 0]}
              fontSize={0.12}
              color={gain > 0.1 ? '#ffcc66' : '#666666'}
              anchorX="left"
              anchorY="middle"
            >
              {zone}: {(gain * 100).toFixed(0)}%
            </Text>
          ))}
        </>
      )}

      {/* Ground Plane */}
      <mesh position={[0, -1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial
          color="#1a1a1a"
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Debug FX Visualization */}
      {showDebug && (
        <SensoryFXDebug
          moment={currentMoment}
          position={[5, 1, 0]}
        />
      )}

      {/* Sensory PostFX Pipeline */}
      {enablePostFX && (
        <SensoryFX
          moment={currentMoment}
          intensity={1.0}
          respectAccessibility={true}
        />
      )}
    </>
  )
}

// Scene selector component
interface SensorySceneSelectorProps {
  selectedScene: keyof typeof DEMO_TEXTS
  onSceneChange: (scene: keyof typeof DEMO_TEXTS) => void
  showDebug: boolean
  onDebugChange: (debug: boolean) => void
  enablePostFX: boolean
  onPostFXChange: (enabled: boolean) => void
}

export function SensorySceneSelector({
  selectedScene,
  onSceneChange,
  showDebug,
  onDebugChange,
  enablePostFX,
  onPostFXChange
}: SensorySceneSelectorProps) {
  return (
    <div className="space-y-3">
      {/* Scene Selection */}
      <div className="grid grid-cols-2 gap-2">
        {(Object.keys(DEMO_TEXTS) as Array<keyof typeof DEMO_TEXTS>).map((scene) => (
          <button
            key={scene}
            onClick={() => onSceneChange(scene)}
            className={`
              px-3 py-2 rounded text-xs font-medium transition-colors
              ${selectedScene === scene
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            {scene.charAt(0).toUpperCase() + scene.slice(1)}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span>Show Debug Info</span>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showDebug}
              onChange={(e) => onDebugChange(e.target.checked)}
              className="rounded"
            />
            <span>{showDebug ? 'On' : 'Off'}</span>
          </label>
        </div>

        <div className="flex items-center justify-between text-xs">
          <span>PostFX Effects</span>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={enablePostFX}
              onChange={(e) => onPostFXChange(e.target.checked)}
              className="rounded"
            />
            <span>{enablePostFX ? 'On' : 'Off'}</span>
          </label>
        </div>
      </div>

      {/* Scene Description */}
      <div className="text-xs text-gray-400 bg-gray-800/50 p-2 rounded">
        <strong>"{selectedScene}" Scene:</strong>
        <br />
        {DEMO_TEXTS[selectedScene]}
      </div>
    </div>
  )
}

// Complete demo integration component
export function CompleteSensoryDemo() {
  const [selectedScene, setSelectedScene] = useState<keyof typeof DEMO_TEXTS>('forest')
  const [showDebug, setShowDebug] = useState(false)
  const [enablePostFX, setEnablePostFX] = useState(true)
  const [currentMoment, setCurrentMoment] = useState(createBaseMoment())
  const [isActive, setIsActive] = useState(true)

  return (
    <div className="h-full flex">
      {/* 3D Scene */}
      <div className="flex-1">
        {isActive && (
          <SensoryDemo
            selectedText={selectedScene}
            showDebug={showDebug}
            enablePostFX={enablePostFX}
          />
        )}
      </div>

      {/* Control Panel */}
      <div className="w-80 bg-gray-900 p-4 overflow-y-auto space-y-4">
        {/* Scene Controls */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-white">Sensory Scene Demo</h3>

          <SensorySceneSelector
            selectedScene={selectedScene}
            onSceneChange={setSelectedScene}
            showDebug={showDebug}
            onDebugChange={setShowDebug}
            enablePostFX={enablePostFX}
            onPostFXChange={setEnablePostFX}
          />
        </div>

        {/* System Status */}
        <div className="text-xs text-gray-400 bg-gray-800/50 p-3 rounded space-y-2">
          <div className="font-medium text-gray-300">System Status</div>
          <div>Demo: {isActive ? '🟢 Running' : '⚫ Paused'}</div>
          <div>Scene: {selectedScene}</div>
          <div>Debug: {showDebug ? '🔍 Visible' : '👁️ Hidden'}</div>
          <div>PostFX: {enablePostFX ? '✨ Enabled' : '🚫 Disabled'}</div>
        </div>

        {/* Instructions */}
        <div className="text-xs text-gray-400 bg-blue-900/20 p-3 rounded border border-blue-800">
          <div className="font-medium text-blue-300 mb-2">How it Works</div>
          <ul className="space-y-1">
            <li>• Words trigger sensory channels in real-time</li>
            <li>• Camera proximity modulates channel strengths</li>
            <li>• Scene objects respond to channel intensities</li>
            <li>• Accessibility settings reduce effect intensity</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
