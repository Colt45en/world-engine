// @ts-nocheck
import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { Line, Sparkles, Float, Text } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'

function GlassShell({ radius = 5, detail = 3, onVertices = () => {} }) {
  const meshRef = useRef()
  const geometry = useMemo(() => new THREE.IcosahedronGeometry(radius, detail), [radius, detail])
  const edges = useMemo(() => new THREE.EdgesGeometry(geometry, 12), [geometry])

  useEffect(() => {
    const unique = new Map()
    const positions = geometry.attributes.position
    for (let i = 0; i < positions.count; i++) {
      const vertex = new THREE.Vector3().fromBufferAttribute(positions, i)
      if (vertex.y >= 0) {
        const key = `${vertex.x.toFixed(3)}|${vertex.y.toFixed(3)}|${vertex.z.toFixed(3)}`
        if (!unique.has(key)) {
          unique.set(key, vertex.toArray())
        }
      }
    }
    onVertices(Array.from(unique.values()))
  }, [geometry, onVertices])

  useEffect(() => () => geometry.dispose(), [geometry])
  useEffect(() => () => edges.dispose(), [edges])

  useFrame((state) => {
    if (meshRef.current) {
      const t = state.clock.getElapsedTime()
      meshRef.current.material.opacity = 0.3 + 0.08 * Math.sin(t * 1.5)
      meshRef.current.material.thickness = 0.35 + 0.1 * Math.sin(t * 0.6)
    }
  })

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
        <meshPhysicalMaterial
          color="#63d9ff"
          transmission={0.95}
          roughness={0.12}
          thickness={0.35}
          reflectivity={0.68}
          iridescence={0.42}
          iridescenceIOR={1.4}
          attenuationColor="#6ad7ff"
          attenuationDistance={16}
          transparent
          opacity={0.28}
          envMapIntensity={0.75}
          side={THREE.DoubleSide}
        />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color="#8cdfff" transparent opacity={0.4} />
      </lineSegments>
    </group>
  )
}

function FlowerPlatform({ radius = 0.9, rings = 4 }) {
  const circlePoints = useMemo(() => {
    const segments = 96
    return new Array(segments + 1).fill(0).map((_, i) => {
      const angle = (i / segments) * Math.PI * 2
      return [Math.cos(angle) * radius, 0, Math.sin(angle) * radius]
    })
  }, [radius])

  const centers = useMemo(() => {
    const results = []
    for (let q = -rings; q <= rings; q++) {
      for (let r = -rings; r <= rings; r++) {
        const s = -q - r
        if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) <= rings) {
          const x = radius * 1.5 * q
          const z = radius * Math.sqrt(3) * (r + q / 2)
          results.push([x, z])
        }
      }
    }
    return results
  }, [radius, rings])

  return (
    <group position={[0, -1.15, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[radius * (rings + 1.8), 72]} />
        <meshStandardMaterial
          color="#061627"
          metalness={0.4}
          roughness={0.8}
          transparent
          opacity={0.85}
        />
      </mesh>

      {centers.map(([x, z], idx) => (
        <Line
          key={`${x}-${z}-${idx}`}
          points={circlePoints.map(([px, py, pz]) => [px + x, 0.02 * Math.sin(idx * 0.12), pz + z])}
          color="#a7f5ff"
          lineWidth={1.1}
          transparent
          opacity={0.35 + 0.25 * Math.exp(-idx / centers.length)}
        />
      ))}

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
        <ringGeometry args={[radius * 0.65, radius * 0.85, 64]} />
        <meshBasicMaterial color="#77f8ff" transparent opacity={0.5} />
      </mesh>

      <Sparkles count={140} size={3} scale={[6, 1, 6]} position={[0, 0.4, 0]} speed={0.4} color="#7bdeff" />
    </group>
  )
}

function EnergyStream({ anchor, index }) {
  const lineRef = useRef()
  useFrame((state) => {
    if (!lineRef.current) return
    const t = state.clock.getElapsedTime()
    const pulse = 0.35 + 0.22 * Math.sin(t * 2.4 + index * 0.9)
    const material = lineRef.current.material
    if (material) {
      material.opacity = pulse
    }
  })

  return (
    <Line
      ref={lineRef}
      points={[anchor, [0, -1.15, 0]]}
      color="#7fe6ff"
      lineWidth={1.3}
      transparent
      opacity={0.45}
    />
  )
}

function EnergyStreams({ anchors }) {
  const limitedAnchors = useMemo(() => anchors.slice(0, 120), [anchors])
  return (
    <group>
      {limitedAnchors.map((point, idx) => (
        <EnergyStream key={`${point[0]}-${point[1]}-${point[2]}-${idx}`} anchor={point} index={idx} />
      ))}
    </group>
  )
}

export default function GeoHexFlowerScene() {
  const [anchors, setAnchors] = useState([])
  const handleVertices = useCallback((verts) => {
    setAnchors(verts)
  }, [])

  return (
    <>
      <color attach="background" args={['#050910']} />
      <fog attach="fog" args={['#050910', 18, 42]} />

      <hemisphereLight intensity={0.4} groundColor="#0f172a" color="#6bd9ff" />
      <directionalLight
        position={[6, 12, 6]}
        intensity={1.1}
        color="#9fe0ff"
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <pointLight position={[-8, 4, -6]} intensity={0.6} color="#4fa0ff" />
      <pointLight position={[8, 3, -2]} intensity={0.4} color="#ff7ff9" />

      <GlassShell radius={5} detail={3} onVertices={handleVertices} />
      <EnergyStreams anchors={anchors} />
      <FlowerPlatform radius={0.9} rings={4} />

      <Float speed={0.6} rotationIntensity={0.25} floatIntensity={0.8}>
        <Text
          position={[0, 2.6, 0]}
          fontSize={0.4}
          color="#b6f7ff"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.015}
          outlineColor="#0c1e33"
        >
          GeoHex Flower
        </Text>
      </Float>

      <EffectComposer multisampling={0}>
        <Bloom intensity={0.7} luminanceThreshold={0.18} luminanceSmoothing={0.35} mipmapBlur />
      </EffectComposer>
    </>
  )
}
