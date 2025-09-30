import React, { useState, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import {
  EffectComposer,
  Bloom,
  Vignette,
  Noise,
  ChromaticAberration,
  DepthOfField,
  SSAO,
  ToneMapping
} from '@react-three/postprocessing'
import { Vector2 } from 'three'
import { SceneMoment, SensoryChannel, ChannelUtils } from './types'
import { useUI } from '../state/ui'

interface Props {
  moment: SceneMoment
  intensity?: number // 0..1 global FX intensity multiplier
  respectAccessibility?: boolean
  disabled?: boolean
}

interface FXValues {
  bloom: number
  vignette: number
  noise: number
  chroma: number
  dof: number
  ssao: number
  tonemap: number
}

// Smooth damping function
function damp(current: number, target: number, lambda = 0.12): number {
  return current + (target - current) * lambda
}

// Utility functions
function avg(values: number[]): number {
  return values.length ? values.reduce((x, y) => x + y, 0) / values.length : 0
}

function map01(x: number): number {
  return Math.max(0, Math.min(1, x))
}

// Get average strength for a channel
function getChannelStrength(moment: SceneMoment, channel: SensoryChannel): number {
  return ChannelUtils.getStrength(moment, channel)
}

export function SensoryFX({
  moment,
  intensity = 1.0,
  respectAccessibility = true,
  disabled = false
}: Props) {
  const { reducedMotion } = useUI()

  // Get individual channel strengths
  const sSight = getChannelStrength(moment, 'sight')
  const sSound = getChannelStrength(moment, 'sound')
  const sTouch = getChannelStrength(moment, 'touch')
  const sScent = getChannelStrength(moment, 'scent')
  const sTaste = getChannelStrength(moment, 'taste')
  const sInner = getChannelStrength(moment, 'inner')

  // Accessibility adjustments
  const effectiveIntensity = respectAccessibility && reducedMotion ?
    intensity * 0.4 : // Reduce FX intensity for motion-sensitive users
    intensity

  const dampingRate = respectAccessibility && reducedMotion ?
    0.05 : // Slower transitions for reduced motion
    0.12

  // Smoothed FX values
  const [fxValues, setFxValues] = useState<FXValues>({
    bloom: 0,
    vignette: 0,
    noise: 0,
    chroma: 0,
    dof: 0,
    ssao: 0,
    tonemap: 0
  })

  // Update FX values each frame
  useFrame(() => {
    if (disabled) return

    // Map channels to FX targets with accessibility considerations
    const baseIntensity = effectiveIntensity

    const targets = {
      // SIGHT → Bloom (light sensitivity, brightness)
      bloom: map01((sSight * 0.9 + sInner * 0.3) * baseIntensity),

      // INNER → Vignette (focus, consciousness compression)
      vignette: map01(sInner * baseIntensity),

      // SOUND → Noise (auditory grain, air texture)
      noise: map01((sSound * 0.8 + sTouch * 0.2) * baseIntensity),

      // TOUCH → Chromatic Aberration (physical "feel", pressure distortion)
      chroma: map01((sTouch * 0.9 + sSound * 0.2) * baseIntensity),

      // SCENT + TASTE + INNER → Depth of Field (memory haze, olfactory focus)
      dof: map01((sTaste * 0.6 + sScent * 0.4 + sInner * 0.2) * baseIntensity),

      // TOUCH + SIGHT → SSAO (spatial awareness, surface definition)
      ssao: map01((sTouch * 0.7 + sSight * 0.3) * baseIntensity * 0.6),

      // ALL CHANNELS → Tone Mapping (overall sensory saturation)
      tonemap: map01((sSight + sSound + sTouch + sScent + sTaste + sInner) / 6 * baseIntensity)
    }

    // Smooth transition to targets
    setFxValues(prev => ({
      bloom: damp(prev.bloom, targets.bloom, dampingRate),
      vignette: damp(prev.vignette, targets.vignette, dampingRate),
      noise: damp(prev.noise, targets.noise, dampingRate),
      chroma: damp(prev.chroma, targets.chroma, dampingRate),
      dof: damp(prev.dof, targets.dof, dampingRate),
      ssao: damp(prev.ssao, targets.ssao, dampingRate),
      tonemap: damp(prev.tonemap, targets.tonemap, dampingRate)
    }))
  })

  // ChromaticAberration offset vector
  const chromaOffset = useMemo(() => {
    const offset = 0.001 + fxValues.chroma * (respectAccessibility && reducedMotion ? 0.001 : 0.002)
    return new Vector2(offset, offset)
  }, [fxValues.chroma, respectAccessibility, reducedMotion])

  // DepthOfField parameters
  const dofParams = useMemo(() => {
    const intensity = fxValues.dof
    return {
      focusDistance: 0.02 + intensity * (respectAccessibility && reducedMotion ? 0.1 : 0.2),
      bokehScale: 0.3 + intensity * (respectAccessibility && reducedMotion ? 1.0 : 2.0),
      focalLength: 0.015 + intensity * 0.02
    }
  }, [fxValues.dof, respectAccessibility, reducedMotion])

  // Bloom parameters
  const bloomParams = useMemo(() => {
    const intensity = fxValues.bloom
    return {
      intensity: 0.2 + intensity * (respectAccessibility && reducedMotion ? 0.6 : 1.2),
      luminanceThreshold: 0.2,
      luminanceSmoothing: 0.1
    }
  }, [fxValues.bloom, respectAccessibility, reducedMotion])

  // Vignette parameters
  const vignetteParams = useMemo(() => {
    const intensity = fxValues.vignette
    return {
      offset: 0.2 + intensity * (respectAccessibility && reducedMotion ? 0.3 : 0.5),
      darkness: 0.4 + intensity * (respectAccessibility && reducedMotion ? 0.4 : 0.6)
    }
  }, [fxValues.vignette, respectAccessibility, reducedMotion])

  // Noise parameters
  const noiseParams = useMemo(() => {
    const intensity = fxValues.noise
    return {
      opacity: 0.02 + intensity * (respectAccessibility && reducedMotion ? 0.04 : 0.08)
    }
  }, [fxValues.noise, respectAccessibility, reducedMotion])

  // SSAO parameters
  const ssaoParams = useMemo(() => {
    const intensity = fxValues.ssao
    return {
      samples: respectAccessibility && reducedMotion ? 16 : 32, // Lower quality for performance
      radius: 0.1 + intensity * 0.2,
      intensity: intensity * (respectAccessibility && reducedMotion ? 0.5 : 1.0),
      bias: 0.005
    }
  }, [fxValues.ssao, respectAccessibility, reducedMotion])

  if (disabled) {
    return null
  }

  return (
    <EffectComposer
      disableNormalPass={respectAccessibility && reducedMotion} // Skip normal pass for performance
      multisampling={respectAccessibility && reducedMotion ? 0 : 4}
    >
      {/* Bloom - Light sensitivity (sight + inner) */}
      <Bloom
        intensity={bloomParams.intensity}
        luminanceThreshold={bloomParams.luminanceThreshold}
        luminanceSmoothing={bloomParams.luminanceSmoothing}
        mipmapBlur={!reducedMotion} // Disable expensive blur for reduced motion
      />

      {/* Vignette - Focus compression (inner) */}
      <Vignette
        eskil={true}
        offset={vignetteParams.offset}
        darkness={vignetteParams.darkness}
      />

      {/* Noise - Auditory grain (sound + touch) */}
      <Noise
        opacity={noiseParams.opacity}
        premultiply
      />

      {/* Chromatic Aberration - Physical distortion (touch + sound) */}
      <ChromaticAberration
        offset={chromaOffset}
        radialModulation={!reducedMotion} // Disable modulation for reduced motion
      />

      {/* Depth of Field - Memory/focus haze (scent + taste + inner) */}
      {(!respectAccessibility || !reducedMotion) && ( // Skip expensive DOF for reduced motion
        <DepthOfField
          focusDistance={dofParams.focusDistance}
          focalLength={dofParams.focalLength}
          bokehScale={dofParams.bokehScale}
          height={respectAccessibility && reducedMotion ? 240 : 480} // Lower resolution for performance
        />
      )}

      {/* SSAO - Spatial awareness (touch + sight) */}
      {(!respectAccessibility || !reducedMotion) && ( // Skip expensive SSAO for reduced motion
        <SSAO
          samples={ssaoParams.samples}
          radius={ssaoParams.radius}
          intensity={ssaoParams.intensity}
          bias={ssaoParams.bias}
          fade={0.01}
          color="black"
        />
      )}

      {/* Tone Mapping - Overall sensory saturation */}
      <ToneMapping
        adaptive={true}
        resolution={256}
        middleGrey={0.6}
        maxLuminance={16}
        averageLuminance={1}
        adaptationRate={1 + fxValues.tonemap * 2}
      />
    </EffectComposer>
  )
}

// Hook for easier FX management
export function useSensoryFX(
  moment: SceneMoment,
  options?: {
    intensity?: number
    respectAccessibility?: boolean
    disabled?: boolean
  }
) {
  const [fxIntensity, setFxIntensity] = useState(options?.intensity ?? 1.0)
  const [isDisabled, setIsDisabled] = useState(options?.disabled ?? false)

  const toggleFX = () => setIsDisabled(prev => !prev)
  const setIntensity = (intensity: number) => setFxIntensity(Math.max(0, Math.min(1, intensity)))

  return {
    fxIntensity,
    isDisabled,
    setIntensity,
    toggleFX,
    SensoryFXComponent: (props: Partial<Props>) => (
      <SensoryFX
        moment={moment}
        intensity={fxIntensity}
        disabled={isDisabled}
        respectAccessibility={options?.respectAccessibility ?? true}
        {...props}
      />
    )
  }
}

// Debug component to visualize FX mappings
interface SensoryFXDebugProps {
  moment: SceneMoment
  position?: [number, number, number]
}

export function SensoryFXDebug({ moment, position = [0, 0, 0] }: SensoryFXDebugProps) {
  const channels: SensoryChannel[] = ['sight', 'sound', 'touch', 'scent', 'taste', 'inner']

  return (
    <group position={position}>
      {channels.map((channel, index) => {
        const strength = getChannelStrength(moment, channel)
        const y = index * 0.3

        return (
          <mesh key={channel} position={[0, y, 0]}>
            <boxGeometry args={[strength * 2, 0.1, 0.1]} />
            <meshBasicMaterial color={`hsl(${index * 60}, 70%, ${50 + strength * 30}%)`} />
          </mesh>
        )
      })}
    </group>
  )
}

export default SensoryFX
