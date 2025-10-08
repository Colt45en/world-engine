import React, { useRef, useEffect, useMemo } from 'react'
import { SceneMoment, SensoryDetail, SensoryLexicon, SensoryChannel } from './types'
import { useUI } from '../state/ui'

interface Props {
  text: string
  base: SceneMoment
  onMoment: (m: SceneMoment) => void
  tps?: number // tokens per second
  lexicon?: SensoryLexicon // custom lexicon override
  decay?: number // decay rate when no token match
  respectAccessibility?: boolean // respect reduced motion settings
}

// Default lexicon mapping words to channel intensities
const DEFAULT_LEXICON: SensoryLexicon = {
  // === VISUAL WORDS ===
  sunrise: { sight: 0.9, inner: 0.6 },
  dawn: { sight: 0.8, inner: 0.5 },
  sunset: { sight: 0.8, inner: 0.7 },
  twilight: { sight: 0.7, inner: 0.8 },
  shadow: { sight: 0.6, inner: 0.6 },
  light: { sight: 0.8, inner: 0.4 },
  dark: { sight: 0.3, inner: 0.7 },
  bright: { sight: 0.9, inner: 0.3 },
  dim: { sight: 0.4, inner: 0.6 },
  glow: { sight: 0.7, inner: 0.5 },
  shimmer: { sight: 0.6, inner: 0.4 },
  sparkle: { sight: 0.8, inner: 0.3 },
  flash: { sight: 1.0, inner: 0.2 },
  lightning: { sight: 1.0, sound: 0.8, inner: 0.7 },

  // === WEATHER & ATMOSPHERE ===
  storm: { sight: 0.8, sound: 0.9, touch: 0.7, taste: 0.6, inner: 0.8 },
  rain: { sight: 0.6, sound: 0.8, touch: 0.7, scent: 0.6 },
  thunder: { sight: 0.7, sound: 1.0, inner: 0.6 },
  wind: { sound: 0.7, touch: 0.8, scent: 0.5 },
  breeze: { touch: 0.6, sound: 0.4, scent: 0.4 },
  fog: { sight: 0.3, touch: 0.6, scent: 0.4 },
  mist: { sight: 0.4, touch: 0.5, scent: 0.3 },
  snow: { sight: 0.7, touch: 0.5, sound: 0.2 },
  ice: { sight: 0.6, touch: 0.8, sound: 0.3 },

  // === SOUND WORDS ===
  whisper: { sound: 0.6, inner: 0.6 },
  shout: { sound: 0.9, inner: 0.4 },
  rustle: { sound: 0.7, touch: 0.4 },
  crack: { sound: 0.8, touch: 0.3 },
  splash: { sound: 0.7, touch: 0.6, sight: 0.3 },
  echo: { sound: 0.8, inner: 0.7 },
  silence: { sound: 0.1, inner: 0.8 },
  music: { sound: 0.8, inner: 0.6 },
  melody: { sound: 0.7, inner: 0.7 },
  rhythm: { sound: 0.8, inner: 0.5 },

  // === TOUCH/TEXTURE ===
  rough: { touch: 0.8, sight: 0.3 },
  smooth: { touch: 0.6, sight: 0.4 },
  soft: { touch: 0.7, inner: 0.4 },
  hard: { touch: 0.8, sound: 0.3 },
  sharp: { touch: 0.9, sight: 0.4, inner: 0.3 },
  warm: { touch: 0.7, inner: 0.5 },
  cool: { touch: 0.6, inner: 0.4 },
  cold: { touch: 0.8, inner: 0.3 },
  hot: { touch: 0.9, sight: 0.4, inner: 0.2 },
  humidity: { touch: 0.8, scent: 0.5, taste: 0.4 },
  dry: { touch: 0.6, scent: 0.2 },
  wet: { touch: 0.8, sight: 0.3 },

  // === SCENT WORDS ===
  fragrant: { scent: 0.8, inner: 0.5 },
  floral: { scent: 0.7, inner: 0.6 },
  earthy: { scent: 0.6, touch: 0.4 },
  fresh: { scent: 0.7, inner: 0.4 },
  stale: { scent: 0.5, inner: 0.3 },
  sweet: { scent: 0.6, taste: 0.7 },
  bitter: { scent: 0.4, taste: 0.8 },
  smoky: { scent: 0.8, sight: 0.4, taste: 0.3 },
  ozone: { scent: 0.7, taste: 0.8 },
  candle: { scent: 0.8, sight: 0.4, inner: 0.5 },

  // === TASTE WORDS ===
  salty: { taste: 0.8, touch: 0.3 },
  sour: { taste: 0.9, touch: 0.4 },
  spicy: { taste: 0.8, touch: 0.7, scent: 0.5 },
  bland: { taste: 0.2, inner: 0.3 },
  rich: { taste: 0.7, scent: 0.6, inner: 0.4 },

  // === INNER/EMOTIONAL ===
  memory: { inner: 0.9, taste: 0.5, scent: 0.4 },
  dream: { inner: 0.8, sight: 0.4 },
  focus: { inner: 0.7, sight: 0.5 },
  calm: { inner: 0.6, sound: 0.3 },
  peace: { inner: 0.8, sound: 0.2 },
  tension: { inner: 0.7, touch: 0.6 },
  anxiety: { inner: 0.8, touch: 0.5, sound: 0.4 },
  joy: { inner: 0.8, sight: 0.6 },
  wonder: { inner: 0.9, sight: 0.5 },
  time: { inner: 0.8 },
  space: { inner: 0.6, sight: 0.4 },
  depth: { inner: 0.7, sight: 0.5 },

  // === NATURE ===
  forest: { scent: 0.7, sound: 0.6, sight: 0.5, inner: 0.6 },
  ocean: { sound: 0.8, scent: 0.6, taste: 0.7, sight: 0.6 },
  mountain: { sight: 0.7, touch: 0.5, inner: 0.7 },
  fire: { sight: 0.9, sound: 0.4, scent: 0.6, touch: 0.7 },
  water: { sound: 0.6, touch: 0.5, sight: 0.4 },
  earth: { scent: 0.6, touch: 0.7, sight: 0.3 },
  sky: { sight: 0.8, inner: 0.6 },

  // === MATERIALS ===
  metal: { touch: 0.7, sound: 0.6, sight: 0.5 },
  wood: { touch: 0.6, scent: 0.5, sound: 0.4 },
  stone: { touch: 0.8, sight: 0.5, sound: 0.5 },
  glass: { sight: 0.8, sound: 0.7, touch: 0.5 },
  fabric: { touch: 0.7, scent: 0.3 },
  paper: { touch: 0.5, sound: 0.4, scent: 0.3 },

  // === MOVEMENT ===
  flow: { sight: 0.6, sound: 0.5, inner: 0.5 },
  drift: { sight: 0.4, inner: 0.6 },
  rush: { sight: 0.7, sound: 0.6, touch: 0.5 },
  still: { inner: 0.7, sound: 0.1 },
  pulse: { inner: 0.6, touch: 0.5, sight: 0.4 },
}

export function SensoryTokenStream({
  text,
  base,
  onMoment,
  tps = 6,
  lexicon = DEFAULT_LEXICON,
  decay = 0.98,
  respectAccessibility = true
}: Props) {
  const { reducedMotion } = useUI()

  // Tokenize text into words
  const tokens = useMemo(() => {
    return text
      .split(/(\b|\W)/)
      .map(s => s.trim().toLowerCase())
      .filter(Boolean)
  }, [text])

  const idxRef = useRef(0)
  const lastRef = useRef(0)

  // Adjust speed based on accessibility preferences
  const effectiveTps = respectAccessibility && reducedMotion ? tps * 0.5 : tps
  const effectiveDecay = respectAccessibility && reducedMotion ? 0.99 : decay

  useEffect(() => {
    let raf = 0

    const step = (timestamp: number) => {
      raf = requestAnimationFrame(step)

      const intervalMs = 1000 / effectiveTps
      if (timestamp - lastRef.current < intervalMs) return

      lastRef.current = timestamp

      // Get current token
      const currentToken = tokens[idxRef.current] ?? ""
      idxRef.current = (idxRef.current + 1) % Math.max(tokens.length, 1)

      // Create moment clone
      const clone = structuredClone(base) as SceneMoment
      clone.timestamp = timestamp
      clone.source = `token:${currentToken}`

      // Apply lexicon boosts
      const boosts = lexicon[currentToken]

      if (boosts) {
        for (const detail of clone.details) {
          const boost = boosts[detail.channel]
          if (boost != null) {
            // Gentle blend toward boost (cap at 1.0)
            const currentStrength = detail.strength ?? 0.5
            const targetStrength = respectAccessibility && reducedMotion ?
              Math.min(0.8, boost) : // Cap intensity for reduced motion
              boost

            detail.strength = Math.min(1, currentStrength * 0.6 + targetStrength * 0.7)

            // Update description if empty
            if (!detail.description) {
              detail.description = currentToken
            }
          }
        }
      } else {
        // No token match - apply decay
        for (const detail of clone.details) {
          detail.strength = Math.max(0, (detail.strength ?? 0.5) * effectiveDecay)
        }
      }

      onMoment(clone)
    }

    raf = requestAnimationFrame(step)

    return () => cancelAnimationFrame(raf)
  }, [tokens, effectiveTps, effectiveDecay, base, onMoment, lexicon, reducedMotion, respectAccessibility])

  return null // This is a data-only component
}

// Hook for easier integration
export function useSensoryTokenStream(
  text: string,
  baseMoment: SceneMoment,
  options?: {
    tps?: number
    lexicon?: SensoryLexicon
    decay?: number
    respectAccessibility?: boolean
  }
): [SceneMoment, (moment: SceneMoment) => void] {
  const [currentMoment, setCurrentMoment] = React.useState<SceneMoment>(baseMoment)

  return [currentMoment, setCurrentMoment]
}

// Export the default lexicon for customization
export { DEFAULT_LEXICON }
