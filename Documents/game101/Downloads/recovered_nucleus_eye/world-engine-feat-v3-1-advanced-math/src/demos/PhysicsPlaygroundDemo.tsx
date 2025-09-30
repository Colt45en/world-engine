import React, { useState, useRef, useEffect } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Line, Text, Grid } from '@react-three/drei'
import { useUI } from '../state/ui'

// Physics body similar to your C++ Body struct
interface PhysicsBody {
  id: string
  position: THREE.Vector3
  velocity: THREE.Vector3
  mass: number
  radius: number
  color: string
  pinned: boolean
}

// Spring connection between two bodies
interface Spring {
  id: string
  bodyA: string
  bodyB: string
  restLength: number
  stiffness: number
  damping: number
  color: string
}

// Physics parameters (like your C++ sliders)
interface PhysicsParams {
  gravity: THREE.Vector3
  globalDamping: number
  linearDrag: number
  restitution: number
  floorY: number
  timeStep: number
}

// Interactive slider component (inspired by your UISlider)
interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
}

function Slider({ label, value, min, max, step = 0.01, onChange }: SliderProps) {
  return (
    <div className="slider-container">
      <label className="slider-label">{label}: {value.toFixed(3)}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="slider"
      />
      <div className="slider-range">
        <span>{min.toFixed(1)}</span>
        <span>{max.toFixed(1)}</span>
      </div>
    </div>
  )
}

// Physics simulation hook (inspired by your main.cpp simulation)
function usePhysicsSimulation(
  bodies: PhysicsBody[],
  springs: Spring[],
  params: PhysicsParams,
  paused: boolean,
  stepOne: boolean
) {
  const frameCount = useRef(0)
  const lastStepRef = useRef(false)

  useFrame(() => {
    frameCount.current++

    // Fixed timestep simulation (like your dtFixed = 1.0f/120.0f)
    const shouldStep = !paused || (stepOne && !lastStepRef.current)
    lastStepRef.current = stepOne

    if (!shouldStep) return

    const dt = params.timeStep

    // Apply forces to each body
    bodies.forEach(body => {
      if (body.pinned) return

      // Gravity force
      const gravityForce = params.gravity.clone().multiplyScalar(body.mass)

      // Spring forces
      let springForce = new THREE.Vector3()
      springs.forEach(spring => {
        if (spring.bodyA === body.id) {
          const other = bodies.find(b => b.id === spring.bodyB)
          if (other) {
            const displacement = other.position.clone().sub(body.position)
            const distance = displacement.length()
            const direction = displacement.normalize()
            const extension = distance - spring.restLength

            // Hooke's law: F = -k * x
            const force = direction.multiplyScalar(spring.stiffness * extension)
            springForce.add(force)

            // Spring damping based on relative velocity
            const relativeVelocity = other.velocity.clone().sub(body.velocity)
            const dampingForce = direction.multiplyScalar(
              spring.damping * relativeVelocity.dot(direction)
            )
            springForce.add(dampingForce)
          }
        }

        if (spring.bodyB === body.id) {
          const other = bodies.find(b => b.id === spring.bodyA)
          if (other) {
            const displacement = body.position.clone().sub(other.position)
            const distance = displacement.length()
            const direction = displacement.normalize()
            const extension = distance - spring.restLength

            const force = direction.multiplyScalar(spring.stiffness * extension)
            springForce.add(force)

            const relativeVelocity = body.velocity.clone().sub(other.velocity)
            const dampingForce = direction.multiplyScalar(
              spring.damping * relativeVelocity.dot(direction)
            )
            springForce.add(dampingForce)
          }
        }
      })

      // Total force
      const totalForce = gravityForce.add(springForce)

      // Integrate velocity (semi-implicit Euler)
      body.velocity.add(totalForce.multiplyScalar(dt / body.mass))

      // Apply global damping (like your multiplicative damping)
      body.velocity.multiplyScalar(params.globalDamping)

      // Apply linear drag (subtractive)
      const dragScale = Math.max(0, 1 - params.linearDrag * dt)
      body.velocity.multiplyScalar(dragScale)

      // Integrate position
      body.position.add(body.velocity.clone().multiplyScalar(dt))

      // Floor collision (like your collideFloor function)
      const penetration = (body.radius + params.floorY) - body.position.y
      if (penetration > 0 && body.velocity.y < 0) {
        body.position.y += penetration
        body.velocity.y = -body.velocity.y * params.restitution
        // Friction
        body.velocity.x *= 0.99
        body.velocity.z *= 0.99
      }
    })
  })
}

// Rendered physics body
function PhysicsBodyMesh({ body }: { body: PhysicsBody }) {
  const meshRef = useRef<THREE.Mesh>(null!)

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.position.copy(body.position)
    }
  })

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[body.radius, 16, 12]} />
        <meshStandardMaterial
          color={body.color}
          opacity={body.pinned ? 0.8 : 1}
          transparent={body.pinned}
        />
      </mesh>

      {/* Velocity vector (like your DrawLine3D for velocity) */}
      <Line
        points={[
          body.position,
          body.position.clone().add(body.velocity.clone().multiplyScalar(0.25))
        ]}
        color={body.pinned ? "#666" : "#ff9800"}
        lineWidth={2}
      />

      {/* Pinned indicator */}
      {body.pinned && (
        <Text
          position={[body.position.x, body.position.y + body.radius + 0.2, body.position.z]}
          fontSize={0.1}
          color="#e91e63"
          anchorX="center"
          anchorY="middle"
        >
          📌
        </Text>
      )}
    </group>
  )
}

// Rendered spring connection
function SpringMesh({ spring, bodies }: { spring: Spring; bodies: PhysicsBody[] }) {
  const bodyA = bodies.find(b => b.id === spring.bodyA)
  const bodyB = bodies.find(b => b.id === spring.bodyB)

  if (!bodyA || !bodyB) return null

  const points = [bodyA.position, bodyB.position]
  const currentLength = bodyA.position.distanceTo(bodyB.position)
  const strain = Math.abs(currentLength - spring.restLength) / spring.restLength

  // Color based on strain (blue = relaxed, red = strained)
  const strainColor = new THREE.Color().lerpColors(
    new THREE.Color('#64b5f6'),
    new THREE.Color('#f44336'),
    Math.min(strain * 2, 1)
  )

  return (
    <Line
      points={points}
      color={strainColor}
      lineWidth={Math.max(1, strain * 5)}
    />
  )
}

// Main physics playground scene
function PhysicsScene({
  bodies,
  springs,
  params,
  paused,
  stepOne,
  onBodyClick
}: {
  bodies: PhysicsBody[]
  springs: Spring[]
  params: PhysicsParams
  paused: boolean
  stepOne: boolean
  onBodyClick: (bodyId: string) => void
}) {
  usePhysicsSimulation(bodies, springs, params, paused, stepOne)

  const { raycaster, mouse, camera } = useThree()
  const [hoveredBody, setHoveredBody] = useState<string | null>(null)

  // Handle mouse interactions
  const handleClick = (event: any) => {
    // Simple ray casting to find clicked body
    raycaster.setFromCamera(mouse, camera)

    for (const body of bodies) {
      const sphere = new THREE.Sphere(body.position, body.radius)
      const intersects = raycaster.ray.intersectSphere(sphere, new THREE.Vector3())

      if (intersects) {
        onBodyClick(body.id)
        break
      }
    }
  }

  return (
    <group onClick={handleClick}>
      {/* Floor plane */}
      <mesh position={[0, params.floorY, 0]} rotation={[-Math.PI/2, 0, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#1a2332" transparent opacity={0.8} />
      </mesh>

      {/* Grid */}
      <Grid
        args={[20, 20]}
        position={[0, params.floorY + 0.001, 0]}
        cellSize={0.5}
        cellColor="#334155"
        sectionSize={2}
        sectionColor="#475569"
      />

      {/* Physics bodies */}
      {bodies.map(body => (
        <PhysicsBodyMesh key={body.id} body={body} />
      ))}

      {/* Springs */}
      {springs.map(spring => (
        <SpringMesh key={spring.id} spring={spring} bodies={bodies} />
      ))}
    </group>
  )
}

// Main physics playground component
export function PhysicsPlaygroundDemo() {
  const { reducedMotion } = useUI()

  // Simulation state
  const [paused, setPaused] = useState(false)
  const [stepOne, setStepOne] = useState(false)

  // Physics parameters (like your C++ slider values)
  const [params, setParams] = useState<PhysicsParams>({
    gravity: new THREE.Vector3(0, -9.81, 0),
    globalDamping: 0.98,
    linearDrag: 0.05,
    restitution: 0.6,
    floorY: 0,
    timeStep: 1/120
  })

  // Demo bodies (inspired by your two-body spring example)
  const [bodies, setBodies] = useState<PhysicsBody[]>([
    {
      id: 'A',
      position: new THREE.Vector3(-1, 2, 0),
      velocity: new THREE.Vector3(0.5, 0, 0),
      mass: 1.0,
      radius: 0.18,
      color: '#e3f2fd',
      pinned: false
    },
    {
      id: 'B',
      position: new THREE.Vector3(1, 2.2, 0),
      velocity: new THREE.Vector3(-0.5, 0, 0),
      mass: 1.0,
      radius: 0.18,
      color: '#e8f5e8',
      pinned: false
    },
    {
      id: 'C',
      position: new THREE.Vector3(0, 3.5, 0),
      velocity: new THREE.Vector3(0, 0, 0),
      mass: 0.5,
      radius: 0.12,
      color: '#fff3e0',
      pinned: false
    }
  ])

  // Springs connecting the bodies
  const [springs, setSprings] = useState<Spring[]>([
    {
      id: 'AB',
      bodyA: 'A',
      bodyB: 'B',
      restLength: 1.8,
      stiffness: 8.0,
      damping: 0.5,
      color: '#64b5f6'
    },
    {
      id: 'AC',
      bodyA: 'A',
      bodyB: 'C',
      restLength: 1.5,
      stiffness: 6.0,
      damping: 0.3,
      color: '#81c784'
    },
    {
      id: 'BC',
      bodyA: 'B',
      bodyB: 'C',
      restLength: 1.5,
      stiffness: 6.0,
      damping: 0.3,
      color: '#ffb74d'
    }
  ])

  // Keyboard controls (like your main.cpp input handling)
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      switch (e.key) {
        case ' ':
          setPaused(p => !p)
          break
        case 'n':
        case 'N':
          if (paused) setStepOne(true)
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [paused])

  const handleBodyClick = (bodyId: string) => {
    setBodies(prev => prev.map(body =>
      body.id === bodyId
        ? { ...body, pinned: !body.pinned }
        : body
    ))
  }

  const resetDemo = () => {
    setBodies(prev => prev.map((body, i) => ({
      ...body,
      position: new THREE.Vector3(
        (i - 1) * 1.2,
        2 + i * 0.5,
        0
      ),
      velocity: new THREE.Vector3(
        (Math.random() - 0.5) * 2,
        (Math.random() - 0.5) * 2,
        (Math.random() - 0.5) * 2
      ),
      pinned: false
    })))
  }

  return (
    <div className="physics-playground">
      {/* Control Panel */}
      <div className="controls-panel">
        <div className="control-section">
          <h3>Physics Playground</h3>
          <div className="button-row">
            <button
              onClick={() => setPaused(p => !p)}
              className={`btn ${paused ? 'primary' : 'danger'}`}
            >
              {paused ? '▶️ Resume' : '⏸️ Pause'} (Space)
            </button>
            <button
              onClick={() => setStepOne(true)}
              disabled={!paused}
              className="btn secondary"
            >
              👣 Step (N)
            </button>
            <button onClick={resetDemo} className="btn secondary">
              🔄 Reset
            </button>
          </div>
          <div className="status">
            {paused ? '⏸️ PAUSED - Step with N' : '▶️ RUNNING - Space to pause'}
          </div>
        </div>

        <div className="control-section">
          <h4>Global Physics</h4>
          <Slider
            label="Gravity Y"
            value={params.gravity.y}
            min={-20}
            max={5}
            step={0.1}
            onChange={(value) => setParams(p => ({ ...p, gravity: new THREE.Vector3(0, value, 0) }))}
          />
          <Slider
            label="Global Damping"
            value={params.globalDamping}
            min={0.9}
            max={0.999}
            step={0.001}
            onChange={(value) => setParams(p => ({ ...p, globalDamping: value }))}
          />
          <Slider
            label="Linear Drag"
            value={params.linearDrag}
            min={0}
            max={0.5}
            step={0.01}
            onChange={(value) => setParams(p => ({ ...p, linearDrag: value }))}
          />
          <Slider
            label="Restitution"
            value={params.restitution}
            min={0}
            max={0.95}
            step={0.01}
            onChange={(value) => setParams(p => ({ ...p, restitution: value }))}
          />
        </div>

        <div className="control-section">
          <h4>Spring AB</h4>
          <Slider
            label="Stiffness"
            value={springs[0]?.stiffness || 8}
            min={0.5}
            max={40}
            step={0.1}
            onChange={(value) => setSprings(prev =>
              prev.map((s, i) => i === 0 ? { ...s, stiffness: value } : s)
            )}
          />
          <Slider
            label="Rest Length"
            value={springs[0]?.restLength || 1.8}
            min={0.5}
            max={4}
            step={0.1}
            onChange={(value) => setSprings(prev =>
              prev.map((s, i) => i === 0 ? { ...s, restLength: value } : s)
            )}
          />
        </div>

        <div className="control-section">
          <h4>Bodies</h4>
          {bodies.map(body => (
            <div key={body.id} className="body-info">
              <span className="body-label" style={{ color: body.color }}>
                {body.id}: {body.pinned ? '📌' : '🔵'}
              </span>
              <span className="body-coords">
                p({body.position.x.toFixed(2)}, {body.position.y.toFixed(2)}, {body.position.z.toFixed(2)})
              </span>
            </div>
          ))}
          <small>💡 Click bodies to pin/unpin them</small>
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="canvas-container">
        <Canvas
          camera={{ position: [4, 3, 6], fov: 60 }}
          dpr={reducedMotion ? 1 : [1, 2]}
        >
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[10, 10, 5]}
            intensity={0.8}
            castShadow
          />
          <pointLight position={[-5, 5, 5]} intensity={0.3} />

          {/* Controls */}
          <OrbitControls
            enableDamping
            dampingFactor={0.05}
            maxPolarAngle={Math.PI * 0.8}
            minDistance={2}
            maxDistance={20}
          />

          {/* Physics Scene */}
          <PhysicsScene
            bodies={bodies}
            springs={springs}
            params={params}
            paused={paused}
            stepOne={stepOne}
            onBodyClick={handleBodyClick}
          />
        </Canvas>
      </div>

      <style jsx>{`
        .physics-playground {
          display: flex;
          height: 100vh;
          background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
          color: white;
          font-family: 'SF Mono', monospace;
        }

        .controls-panel {
          width: 320px;
          background: rgba(15, 23, 42, 0.95);
          border-right: 1px solid rgba(255, 255, 255, 0.1);
          padding: 16px;
          overflow-y: auto;
        }

        .control-section {
          margin-bottom: 24px;
        }

        .control-section h3 {
          margin: 0 0 16px 0;
          font-size: 18px;
          color: #64b5f6;
        }

        .control-section h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          color: #81c784;
        }

        .button-row {
          display: flex;
          gap: 8px;
          margin-bottom: 12px;
        }

        .btn {
          padding: 8px 12px;
          border: none;
          border-radius: 4px;
          font-size: 11px;
          cursor: pointer;
          font-family: inherit;
          transition: all 0.2s;
        }

        .btn.primary {
          background: #64b5f6;
          color: white;
        }

        .btn.secondary {
          background: rgba(255, 255, 255, 0.1);
          color: white;
        }

        .btn.danger {
          background: #f44336;
          color: white;
        }

        .btn:hover:not(:disabled) {
          transform: translateY(-1px);
          filter: brightness(1.1);
        }

        .btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .status {
          font-size: 11px;
          color: rgba(255, 255, 255, 0.7);
          font-style: italic;
        }

        .slider-container {
          margin-bottom: 16px;
        }

        .slider-label {
          display: block;
          font-size: 11px;
          color: rgba(255, 255, 255, 0.9);
          margin-bottom: 4px;
        }

        .slider {
          width: 100%;
          margin: 4px 0;
          height: 4px;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 2px;
          outline: none;
          cursor: pointer;
        }

        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          background: #64b5f6;
          border-radius: 50%;
          cursor: pointer;
        }

        .slider-range {
          display: flex;
          justify-content: space-between;
          font-size: 9px;
          color: rgba(255, 255, 255, 0.5);
        }

        .body-info {
          display: flex;
          justify-content: space-between;
          font-size: 10px;
          margin-bottom: 4px;
          padding: 4px 8px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 2px;
        }

        .body-label {
          font-weight: bold;
        }

        .body-coords {
          font-family: 'Courier New', monospace;
          color: rgba(255, 255, 255, 0.7);
        }

        .control-section small {
          font-size: 9px;
          color: rgba(255, 255, 255, 0.6);
          font-style: italic;
          display: block;
          margin-top: 8px;
        }

        .canvas-container {
          flex: 1;
          background: radial-gradient(ellipse at center, #1e293b 0%, #0f172a 100%);
        }
      `}</style>
    </div>
  )
}

export default PhysicsPlaygroundDemo
