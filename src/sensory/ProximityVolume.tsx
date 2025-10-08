import React, { useRef, useMemo } from 'react'
import * as THREE from 'three'
import { useThree, useFrame } from '@react-three/fiber'
import { useUI } from '../state/ui'
import { SensoryChannel } from './types'

interface Props {
  center: THREE.Object3D | THREE.Vector3
  radius: number
  falloff?: number // 0..1 softness of the falloff
  onGain: (gain: number) => void // callback with 0..1 proximity gain
  channels?: SensoryChannel[] // which channels this affects
  debug?: boolean
  respectAccessibility?: boolean
  maxGain?: number // cap gain for accessibility
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

export function ProximityVolume({
  center,
  radius,
  falloff = 0.6,
  onGain,
  channels,
  debug = false,
  respectAccessibility = true,
  maxGain = 1.0
}: Props) {
  const { camera } = useThree()
  const { reducedMotion } = useUI()
  const vCenter = useMemo(() => new THREE.Vector3(), [])
  const sphere = useRef<THREE.Mesh>(null!)

  // Accessibility adjustments
  const effectiveMaxGain = respectAccessibility && reducedMotion ?
    Math.min(maxGain, 0.7) : // Cap intensity for motion-sensitive users
    maxGain

  const updateRate = respectAccessibility && reducedMotion ? 30 : 60 // Lower update rate
  const frameCounter = useRef(0)

  useFrame(() => {
    // Throttle updates for accessibility
    frameCounter.current++
    if (respectAccessibility && reducedMotion && frameCounter.current % 2 !== 0) {
      return // Skip every other frame
    }

    // Get world position of center
    if (center instanceof THREE.Object3D) {
      center.getWorldPosition(vCenter)
    } else {
      vCenter.copy(center)
    }

    // Calculate distance from camera
    const distance = camera.position.distanceTo(vCenter)

    // Calculate gain using smoothstep (closer = higher gain)
    const innerRadius = radius * (1 - falloff)
    const rawGain = smoothstep(radius, innerRadius, Math.max(0.0001, distance))

    // Apply accessibility limits
    const finalGain = Math.min(effectiveMaxGain, rawGain)

    // Call the gain callback
    onGain(finalGain)

    // Update debug visualization
    if (sphere.current) {
      sphere.current.visible = debug
      if (debug) {
        // Color-code the sphere based on proximity
        const material = sphere.current.material as THREE.MeshBasicMaterial
        const intensity = finalGain
        material.color.setRGB(
          intensity,
          1 - intensity,
          0.5 + intensity * 0.5
        )
        material.opacity = 0.3 + intensity * 0.4
      }
    }
  })

  return debug ? (
    <mesh ref={sphere} position={vCenter.toArray()}>
      <sphereGeometry args={[radius, 32, 16]} />
      <meshBasicMaterial
        wireframe
        transparent
        color="#66ccff"
        opacity={0.3}
      />
    </mesh>
  ) : null
}

// Hook for managing multiple proximity volumes
export function useProximityVolumes() {
  const volumes = useRef<Map<string, {
    gain: number
    lastUpdate: number
    config: Omit<Props, 'onGain'>
  }>>(new Map())

  const addVolume = (id: string, config: Omit<Props, 'onGain'>) => {
    volumes.current.set(id, {
      gain: 0,
      lastUpdate: 0,
      config
    })
  }

  const removeVolume = (id: string) => {
    volumes.current.delete(id)
  }

  const updateVolume = (id: string, gain: number) => {
    const volume = volumes.current.get(id)
    if (volume) {
      volume.gain = gain
      volume.lastUpdate = Date.now()
    }
  }

  const getVolume = (id: string) => {
    return volumes.current.get(id)
  }

  const getAllGains = (): Record<string, number> => {
    const gains: Record<string, number> = {}
    volumes.current.forEach((volume, id) => {
      gains[id] = volume.gain
    })
    return gains
  }

  // Get combined gain for specific channels
  const getChannelGain = (channel: SensoryChannel): number => {
    let totalGain = 0
    let count = 0

    volumes.current.forEach((volume) => {
      if (!volume.config.channels || volume.config.channels.includes(channel)) {
        totalGain += volume.gain
        count++
      }
    })

    return count > 0 ? totalGain / count : 0
  }

  return {
    addVolume,
    removeVolume,
    updateVolume,
    getVolume,
    getAllGains,
    getChannelGain
  }
}

// Predefined proximity configurations
export const ProximityPresets = {
  // Inner contemplation zone
  intimate: {
    radius: 2,
    falloff: 0.8,
    channels: ['inner', 'scent'] as SensoryChannel[],
    maxGain: 0.9
  },

  // Social interaction zone
  personal: {
    radius: 4,
    falloff: 0.6,
    channels: ['sound', 'touch'] as SensoryChannel[],
    maxGain: 0.8
  },

  // Environmental awareness zone
  environmental: {
    radius: 8,
    falloff: 0.4,
    channels: ['sight', 'sound'] as SensoryChannel[],
    maxGain: 0.7
  },

  // Distant atmospheric zone
  atmospheric: {
    radius: 16,
    falloff: 0.2,
    channels: ['sight'] as SensoryChannel[],
    maxGain: 0.5
  }
} as const

// Multi-zone proximity component
interface MultiProximityProps {
  center: THREE.Object3D | THREE.Vector3
  presets?: Array<keyof typeof ProximityPresets>
  onGains?: (gains: Record<string, number>) => void
  debug?: boolean
  respectAccessibility?: boolean
}

export function MultiProximityVolume({
  center,
  presets = ['intimate', 'personal', 'environmental'],
  onGains,
  debug = false,
  respectAccessibility = true
}: MultiProximityProps) {
  const gainsRef = useRef<Record<string, number>>({})

  const handleGainUpdate = (preset: string, gain: number) => {
    gainsRef.current[preset] = gain
    if (onGains) {
      onGains({ ...gainsRef.current })
    }
  }

  return (
    <>
      {presets.map((preset) => {
        const config = ProximityPresets[preset]
        return (
          <ProximityVolume
            key={preset}
            center={center}
            radius={config.radius}
            falloff={config.falloff}
            onGain={(gain) => handleGainUpdate(preset, gain)}
            channels={config.channels}
            maxGain={config.maxGain}
            debug={debug}
            respectAccessibility={respectAccessibility}
          />
        )
      })}
    </>
  )
}
