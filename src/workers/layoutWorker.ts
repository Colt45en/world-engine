// Web Worker for graph physics simulation
// Implements Barnes-Hut octree for O(N log N) repulsion and mass-spring layout

interface Vector3 {
    x: number
    y: number
    z: number
}

interface WorkerMessage {
    type: 'init' | 'tick' | 'pin' | 'setParams' | 'reset' | 'terminate'
    positions?: Float32Array
    velocities?: Float32Array
    edges?: Uint32Array
    pinnedNodes?: number[]
    params?: Partial<LayoutParams>
    dt?: number
    nodeId?: number
    pinned?: boolean
}

interface WorkerResponse {
    type: 'ready' | 'positions' | 'state' | 'error'
    positions?: Float32Array
    velocities?: Float32Array
    forces?: Float32Array
    iteration?: number
    energy?: number
    converged?: boolean
    error?: string
}

interface LayoutParams {
    edgeStiffness: number
    edgeLength: number
    repulsion: number
    damping: number
    theta: number
    bounds: number
    gravity: number
    gravityCenter: Vector3
    timeStep: number
    maxVelocity: number
    minEnergyThreshold: number
    stabilizationSteps: number
}

// Octree for efficient N-body repulsion calculation
class OctreeNode {
    center: Vector3
    size: number
    mass: number = 0
    centerOfMass: Vector3 = { x: 0, y: 0, z: 0 }
    children: OctreeNode[] | null = null
    particles: number[] = []

    constructor(center: Vector3, size: number) {
        this.center = center
        this.size = size
    }

    isLeaf(): boolean {
        return this.children === null
    }

    contains(x: number, y: number, z: number): boolean {
        const half = this.size / 2
        return (
            x >= this.center.x - half && x <= this.center.x + half &&
            y >= this.center.y - half && y <= this.center.y + half &&
            z >= this.center.z - half && z <= this.center.z + half
        )
    }

    subdivide(): void {
        const quarter = this.size / 4
        const { x, y, z } = this.center

        this.children = [
            new OctreeNode({ x: x - quarter, y: y - quarter, z: z - quarter }, this.size / 2),
            new OctreeNode({ x: x + quarter, y: y - quarter, z: z - quarter }, this.size / 2),
            new OctreeNode({ x: x - quarter, y: y + quarter, z: z - quarter }, this.size / 2),
            new OctreeNode({ x: x + quarter, y: y + quarter, z: z - quarter }, this.size / 2),
            new OctreeNode({ x: x - quarter, y: y - quarter, z: z + quarter }, this.size / 2),
            new OctreeNode({ x: x + quarter, y: y - quarter, z: z + quarter }, this.size / 2),
            new OctreeNode({ x: x - quarter, y: y + quarter, z: z + quarter }, this.size / 2),
            new OctreeNode({ x: x + quarter, y: y + quarter, z: z + quarter }, this.size / 2)
        ]
    }

    insert(particleIndex: number, x: number, y: number, z: number): void {
        if (!this.contains(x, y, z)) return

        this.mass += 1
        this.centerOfMass.x = (this.centerOfMass.x * (this.mass - 1) + x) / this.mass
        this.centerOfMass.y = (this.centerOfMass.y * (this.mass - 1) + y) / this.mass
        this.centerOfMass.z = (this.centerOfMass.z * (this.mass - 1) + z) / this.mass

        if (this.isLeaf()) {
            this.particles.push(particleIndex)

            // Subdivide if we have too many particles
            if (this.particles.length > 1) {
                this.subdivide()

                // Redistribute particles to children
                for (const pIndex of this.particles) {
                    const px = positions[pIndex * 3]
                    const py = positions[pIndex * 3 + 1]
                    const pz = positions[pIndex * 3 + 2]

                    for (const child of this.children!) {
                        child.insert(pIndex, px, py, pz)
                    }
                }

                this.particles = []
            }
        } else {
            // Insert into appropriate child
            for (const child of this.children!) {
                child.insert(particleIndex, x, y, z)
            }
        }
    }

    calculateForce(particleIndex: number, x: number, y: number, z: number, theta: number, repulsion: number): Vector3 {
        const force: Vector3 = { x: 0, y: 0, z: 0 }

        if (this.mass === 0) return force

        const dx = this.centerOfMass.x - x
        const dy = this.centerOfMass.y - y
        const dz = this.centerOfMass.z - z
        const distanceSquared = dx * dx + dy * dy + dz * dz

        if (distanceSquared < 0.0001) return force // avoid self-interaction and division by zero

        const distance = Math.sqrt(distanceSquared)

        // Barnes-Hut criterion: if node is far enough, treat as single body
        if (this.isLeaf() || (this.size / distance) < theta) {
            const forceStrength = repulsion * this.mass / (distanceSquared + 0.01) // epsilon to avoid singularities
            force.x = dx * forceStrength / distance
            force.y = dy * forceStrength / distance
            force.z = dz * forceStrength / distance
        } else {
            // Recurse to children for more accurate calculation
            if (this.children) {
                for (const child of this.children) {
                    const childForce = child.calculateForce(particleIndex, x, y, z, theta, repulsion)
                    force.x += childForce.x
                    force.y += childForce.y
                    force.z += childForce.z
                }
            }
        }

        return force
    }
}

// Worker state
let positions: Float32Array = new Float32Array()
let velocities: Float32Array = new Float32Array()
let forces: Float32Array = new Float32Array()
let edges: Uint32Array = new Uint32Array()
let pinnedNodes: Set<number> = new Set()
let nodeCount = 0
let iteration = 0

let params: LayoutParams = {
    edgeStiffness: 2.0,
    edgeLength: 1.1,
    repulsion: 0.8,
    damping: 0.9,
    theta: 0.5,
    bounds: 0,
    gravity: 0,
    gravityCenter: { x: 0, y: 0, z: 0 },
    timeStep: 1 / 60,
    maxVelocity: 10.0,
    minEnergyThreshold: 0.01,
    stabilizationSteps: 1000
}

// Utility functions
function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
}

function calculateBounds(): { center: Vector3, size: number } {
    if (nodeCount === 0) return { center: { x: 0, y: 0, z: 0 }, size: 10 }

    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity
    let minZ = Infinity, maxZ = -Infinity

    for (let i = 0; i < nodeCount; i++) {
        const x = positions[i * 3]
        const y = positions[i * 3 + 1]
        const z = positions[i * 3 + 2]

        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minY = Math.min(minY, y)
        maxY = Math.max(maxY, y)
        minZ = Math.min(minZ, z)
        maxZ = Math.max(maxZ, z)
    }

    const center = {
        x: (minX + maxX) / 2,
        y: (minY + maxY) / 2,
        z: (minZ + maxZ) / 2
    }

    const sizeX = maxX - minX
    const sizeY = maxY - minY
    const sizeZ = maxZ - minZ
    const size = Math.max(sizeX, sizeY, sizeZ) * 1.2 // padding

    return { center, size: Math.max(size, 2) }
}

function buildOctree(): OctreeNode {
    const { center, size } = calculateBounds()
    const octree = new OctreeNode(center, size)

    for (let i = 0; i < nodeCount; i++) {
        const x = positions[i * 3]
        const y = positions[i * 3 + 1]
        const z = positions[i * 3 + 2]
        octree.insert(i, x, y, z)
    }

    return octree
}

function calculateRepulsionForces(octree: OctreeNode): void {
    for (let i = 0; i < nodeCount; i++) {
        const x = positions[i * 3]
        const y = positions[i * 3 + 1]
        const z = positions[i * 3 + 2]

        const repulsionForce = octree.calculateForce(i, x, y, z, params.theta, params.repulsion)

        forces[i * 3] += repulsionForce.x
        forces[i * 3 + 1] += repulsionForce.y
        forces[i * 3 + 2] += repulsionForce.z
    }
}

function calculateSpringForces(): void {
    for (let i = 0; i < edges.length; i += 2) {
        const sourceIndex = edges[i]
        const targetIndex = edges[i + 1]

        if (sourceIndex >= nodeCount || targetIndex >= nodeCount) continue

        const sx = positions[sourceIndex * 3]
        const sy = positions[sourceIndex * 3 + 1]
        const sz = positions[sourceIndex * 3 + 2]

        const tx = positions[targetIndex * 3]
        const ty = positions[targetIndex * 3 + 1]
        const tz = positions[targetIndex * 3 + 2]

        const dx = tx - sx
        const dy = ty - sy
        const dz = tz - sz

        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)

        if (distance < 0.0001) continue // avoid division by zero

        const displacement = distance - params.edgeLength
        const force = params.edgeStiffness * displacement

        const fx = (dx / distance) * force
        const fy = (dy / distance) * force
        const fz = (dz / distance) * force

        // Apply force to both nodes (Newton's 3rd law)
        forces[sourceIndex * 3] += fx
        forces[sourceIndex * 3 + 1] += fy
        forces[sourceIndex * 3 + 2] += fz

        forces[targetIndex * 3] -= fx
        forces[targetIndex * 3 + 1] -= fy
        forces[targetIndex * 3 + 2] -= fz
    }
}

function calculateGravityForces(): void {
    if (params.gravity === 0) return

    for (let i = 0; i < nodeCount; i++) {
        const x = positions[i * 3]
        const y = positions[i * 3 + 1]
        const z = positions[i * 3 + 2]

        const dx = params.gravityCenter.x - x
        const dy = params.gravityCenter.y - y
        const dz = params.gravityCenter.z - z

        forces[i * 3] += dx * params.gravity
        forces[i * 3 + 1] += dy * params.gravity
        forces[i * 3 + 2] += dz * params.gravity
    }
}

function integrate(dt: number): number {
    let totalEnergy = 0

    for (let i = 0; i < nodeCount; i++) {
        if (pinnedNodes.has(i)) {
            // Reset forces and velocities for pinned nodes
            velocities[i * 3] = 0
            velocities[i * 3 + 1] = 0
            velocities[i * 3 + 2] = 0
            forces[i * 3] = 0
            forces[i * 3 + 1] = 0
            forces[i * 3 + 2] = 0
            continue
        }

        // Semi-implicit Euler integration
        const vx = velocities[i * 3] + forces[i * 3] * dt
        const vy = velocities[i * 3 + 1] + forces[i * 3 + 1] * dt
        const vz = velocities[i * 3 + 2] + forces[i * 3 + 2] * dt

        // Apply damping
        velocities[i * 3] = vx * params.damping
        velocities[i * 3 + 1] = vy * params.damping
        velocities[i * 3 + 2] = vz * params.damping

        // Clamp velocity
        const speed = Math.sqrt(vx * vx + vy * vy + vz * vz)
        if (speed > params.maxVelocity) {
            const scale = params.maxVelocity / speed
            velocities[i * 3] *= scale
            velocities[i * 3 + 1] *= scale
            velocities[i * 3 + 2] *= scale
        }

        // Update position
        positions[i * 3] += velocities[i * 3] * dt
        positions[i * 3 + 1] += velocities[i * 3 + 1] * dt
        positions[i * 3 + 2] += velocities[i * 3 + 2] * dt

        // Apply bounds constraint
        if (params.bounds > 0) {
            const x = positions[i * 3]
            const y = positions[i * 3 + 1]
            const z = positions[i * 3 + 2]
            const dist = Math.sqrt(x * x + y * y + z * z)

            if (dist > params.bounds) {
                const scale = params.bounds / dist
                positions[i * 3] *= scale
                positions[i * 3 + 1] *= scale
                positions[i * 3 + 2] *= scale

                // Bounce velocity
                velocities[i * 3] *= -0.5
                velocities[i * 3 + 1] *= -0.5
                velocities[i * 3 + 2] *= -0.5
            }
        }

        // Accumulate kinetic energy
        const vSq = velocities[i * 3] ** 2 + velocities[i * 3 + 1] ** 2 + velocities[i * 3 + 2] ** 2
        totalEnergy += 0.5 * vSq
    }

    return totalEnergy
}

function simulationStep(): void {
    // Clear forces
    forces.fill(0)

    // Build octree for this frame
    const octree = buildOctree()

    // Calculate forces
    calculateRepulsionForces(octree)
    calculateSpringForces()
    calculateGravityForces()

    // Integrate
    const energy = integrate(params.timeStep)

    iteration++

    // Check convergence
    const converged = energy < params.minEnergyThreshold || iteration > params.stabilizationSteps

    // Send positions back to main thread (transferable)
    const positionsCopy = positions.slice()
    const velocitiesCopy = velocities.slice()
    const forcesCopy = forces.slice()

    const response: WorkerResponse = {
        type: 'state',
        positions: positionsCopy,
        velocities: velocitiesCopy,
        forces: forcesCopy,
        iteration,
        energy,
        converged
    }

    self.postMessage(response, [positionsCopy.buffer, velocitiesCopy.buffer, forcesCopy.buffer])

    // Recreate views after transfer
    positions = new Float32Array(positions.length)
    positions.set(new Float32Array(positionsCopy.buffer))

    velocities = new Float32Array(velocities.length)
    velocities.set(new Float32Array(velocitiesCopy.buffer))

    forces = new Float32Array(forces.length)
    forces.set(new Float32Array(forcesCopy.buffer))
}

// Message handler
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
    const message = event.data

    try {
        switch (message.type) {
            case 'init':
                if (!message.positions || !message.edges) {
                    throw new Error('Missing positions or edges in init message')
                }

                positions = message.positions.slice()
                velocities = message.velocities?.slice() || new Float32Array(positions.length)
                forces = new Float32Array(positions.length)
                edges = message.edges.slice()

                nodeCount = positions.length / 3
                iteration = 0

                if (message.pinnedNodes) {
                    pinnedNodes = new Set(message.pinnedNodes)
                }

                if (message.params) {
                    Object.assign(params, message.params)
                }

                self.postMessage({ type: 'ready' } as WorkerResponse)
                break

            case 'tick':
                simulationStep()
                break

            case 'pin':
                if (typeof message.nodeId === 'number') {
                    if (message.pinned) {
                        pinnedNodes.add(message.nodeId)
                    } else {
                        pinnedNodes.delete(message.nodeId)
                    }
                }
                break

            case 'setParams':
                if (message.params) {
                    Object.assign(params, message.params)
                }
                break

            case 'reset':
                iteration = 0
                velocities.fill(0)
                forces.fill(0)
                break

            case 'terminate':
                self.close()
                break

            default:
                throw new Error(`Unknown message type: ${(message as any).type}`)
        }
    } catch (error) {
        const response: WorkerResponse = {
            type: 'error',
            error: error instanceof Error ? error.message : String(error)
        }
        self.postMessage(response)
    }
}

// Handle uncaught errors
self.onerror = (error) => {
    const response: WorkerResponse = {
        type: 'error',
        error: error.message || 'Unknown worker error'
    }
    self.postMessage(response)
}

export { } // Make this a module
