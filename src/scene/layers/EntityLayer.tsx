import React, { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { useEditor } from '../../state/editor'
import { useUI } from '../../state/ui'
import * as THREE from 'three'

interface CubeMeshProps {
  entity: {
    id: string
    type: string
    position: [number, number, number]
    rotation: [number, number, number]
    scale: [number, number, number]
    color: string
    offset: number
  }
  isSelected: boolean
  onClick: () => void
}

function CubeMesh({ entity, isSelected, onClick }: CubeMeshProps) {
  const ref = useRef<THREE.Mesh>(null!)
  const { animationSpeed } = useEditor()
  const { reducedMotion } = useUI()

  // Respect reduced motion preferences
  const motionMultiplier = reducedMotion ? 0.3 : 1.0
  const bobAmplitude = reducedMotion ? 0.03 : 0.1
  const rotationSpeed = reducedMotion ? 0.001 : 0.003

  useFrame((state) => {
    if (!ref.current) return

    const t = state.clock.getElapsedTime()
    const effectiveSpeed = animationSpeed * motionMultiplier

    // Gentle bobbing animation
    const bob = Math.sin(t * 0.6 * motionMultiplier + entity.offset) * bobAmplitude * effectiveSpeed
    ref.current.position.set(
      entity.position[0],
      entity.position[1] + bob,
      entity.position[2]
    )

    // Subtle rotation
    ref.current.rotation.x += rotationSpeed * effectiveSpeed
    ref.current.rotation.y += (rotationSpeed * 1.5) * effectiveSpeed

    // Selection highlight pulse (reduced for motion sensitivity)
    if (isSelected) {
      const pulseIntensity = reducedMotion ? 0.05 : 0.1
      const pulse = Math.sin(t * (reducedMotion ? 2 : 4)) * pulseIntensity + 1
      ref.current.scale.setScalar(entity.scale[0] * pulse)
    } else {
      ref.current.scale.set(...entity.scale)
    }
  })

  return (
    <mesh
      ref={ref}
      position={entity.position}
      rotation={entity.rotation}
      onClick={onClick}
      onPointerOver={(e) => {
        e.stopPropagation()
        document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => {
        document.body.style.cursor = 'auto'
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        color={isSelected ? '#66ccff' : entity.color}
        transparent
        opacity={reducedMotion ? 0.9 : 0.8}
        roughness={0.4}
        metalness={0.1}
      />
      {isSelected && (
        <meshBasicMaterial
          color="#66ccff"
          wireframe
          transparent
          opacity={reducedMotion ? 0.3 : 0.5}
        />
      )}
    </mesh>
  )
}

interface SphereMeshProps {
  entity: {
    id: string
    type: string
    position: [number, number, number]
    rotation: [number, number, number]
    scale: [number, number, number]
    color: string
    offset: number
  }
  isSelected: boolean
  onClick: () => void
}

function SphereMesh({ entity, isSelected, onClick }: SphereMeshProps) {
  const ref = useRef<THREE.Mesh>(null!)
  const { animationSpeed } = useEditor()
  const { reducedMotion } = useUI()

  const motionMultiplier = reducedMotion ? 0.2 : 1.0
  const floatAmplitude = reducedMotion ? 0.02 : 0.08

  useFrame((state) => {
    if (!ref.current) return

    const t = state.clock.getElapsedTime()
    const effectiveSpeed = animationSpeed * motionMultiplier

    // Gentle floating animation
    const float = Math.sin(t * 0.8 * motionMultiplier + entity.offset) * floatAmplitude * effectiveSpeed
    ref.current.position.set(
      entity.position[0],
      entity.position[1] + float,
      entity.position[2]
    )

    // Slow rotation
    ref.current.rotation.y += 0.002 * effectiveSpeed

    // Selection highlight
    if (isSelected) {
      const pulseIntensity = reducedMotion ? 0.03 : 0.08
      const pulse = Math.sin(t * (reducedMotion ? 1.5 : 3)) * pulseIntensity + 1
      ref.current.scale.setScalar(entity.scale[0] * pulse)
    } else {
      ref.current.scale.set(...entity.scale)
    }
  })

  return (
    <mesh
      ref={ref}
      position={entity.position}
      rotation={entity.rotation}
      onClick={onClick}
      onPointerOver={(e) => {
        e.stopPropagation()
        document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => {
        document.body.style.cursor = 'auto'
      }}
    >
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshStandardMaterial
        color={isSelected ? '#66ccff' : entity.color}
        transparent
        opacity={reducedMotion ? 0.9 : 0.8}
        roughness={0.3}
        metalness={0.2}
      />
      {isSelected && (
        <meshBasicMaterial
          color="#66ccff"
          wireframe
          transparent
          opacity={reducedMotion ? 0.3 : 0.5}
        />
      )}
    </mesh>
  )
}

export default function EntityLayer() {
  const { entities, selectedEntityId, selectEntity } = useEditor()

  return (
    <>
      {entities.map((entity) => {
        const isSelected = entity.id === selectedEntityId
        const handleClick = (e: any) => {
          e.stopPropagation()
          selectEntity(entity.id)
        }

        if (entity.type === 'cube') {
          return (
            <CubeMesh
              key={entity.id}
              entity={entity}
              isSelected={isSelected}
              onClick={handleClick}
            />
          )
        }

        if (entity.type === 'sphere') {
          return (
            <SphereMesh
              key={entity.id}
              entity={entity}
              isSelected={isSelected}
              onClick={handleClick}
            />
          )
        }

        return null
      })}
    </>
  )
}
