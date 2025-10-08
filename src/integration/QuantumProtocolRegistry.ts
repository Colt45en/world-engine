/**
 * Quantum Protocol Registry - TypeScript integration for Unity-inspired quantum systems
 * ====================================================================================
 *
 * Integrates the Quantum Protocol Engine with VS Code extension and multi-engine suite:
 * - Quantum agent management and real-time path tracking
 * - Environmental event visualization and interaction
 * - Recursive infrastructure analysis with swarm mind coordination
 * - Memory ghost replay and amplitude resolution
 * - Unity-style event handling in TypeScript
 */

import { EngineMessage, EngineResponse, sendMessage } from './MultiEngineRegistry';

// Quantum Protocol Types (matching Python/C++ definitions)
export interface Vector3 {
    x: number;
    y: number;
    z: number;
}

export enum MathFunctionType {
    WAVE = 0,
    RIPPLE = 1,
    MULTIWAVE = 2,
    SPHERE = 3,
    TORUS = 4
}

export enum QuantumEventType {
    AGENT_COLLAPSE = "agent_collapse",
    FUNCTION_CHANGE = "function_change",
    ENVIRONMENTAL_EVENT = "environmental_event",
    MEMORY_ECHO = "memory_echo",
    SWARM_CONVERGENCE = "swarm_convergence"
}

export enum EnvironmentalEventType {
    STORM = "storm",
    FLUX_SURGE = "flux_surge",
    MEMORY_ECHO = "memory_echo",
    QUANTUM_TUNNEL = "quantum_tunnel",
    REALITY_DISTORTION = "reality_distortion"
}

export interface QuantumStep {
    position: Vector3;
    agent_id: string;
    step_number: number;
    timestamp: number;
    energy_level: number;
    coherence: number;
    entanglement_strength: number;
    is_collapsed: boolean;
    features: Record<string, number>;
}

export interface QuantumAgent {
    id: string;
    position: Vector3;
    velocity: Vector3;
    energy_level: number;
    coherence: number;
    step_count: number;
    max_steps: number;
    is_active: boolean;
    is_collapsed: boolean;
    path_history: QuantumStep[];
    quantum_state: Record<string, number>;
    last_update: number;
}

export interface EnvironmentalEvent {
    type: EnvironmentalEventType;
    origin: Vector3;
    radius: number;
    duration: number;
    time_elapsed: number;
    intensity: number;
    effects: Record<string, number>;
}

export interface GlyphData {
    id: string;
    position: Vector3;
    energy_level: number;
    amplitude: number;
    metadata: Record<string, string>;
    features: Record<string, number>;
    memory_awakened: boolean;
    mutated: boolean;
    creation_time: number;
    last_update: number;
}

export interface RecursiveInfrastructureNode {
    topic: string;
    visible_infrastructure: string;
    unseen_infrastructure: string;
    solid_state: string;
    liquid_state: string;
    gas_state: string;
    derived_topic: string;
    timestamp: number;
    iteration_depth: number;
}

export interface SwarmAnalysis {
    total_nodes: number;
    unique_topics: string[];
    latest_timestamp: number;
    topic_frequency: Record<string, number>;
    convergence_metric: number;
    dominant_theme: string;
}

export interface AmplitudeResult {
    winner_id: string;
    max_amplitude: number;
    all_scores: Record<string, number>;
    resolution_time: number;
}

// Unity-style Protocol Events
export interface QuantumProtocolEvent {
    type: QuantumEventType;
    data: Record<string, any>;
    timestamp: number;
}

// Quantum Protocol Registry Class
export class QuantumProtocolRegistry {
    private eventCallbacks: Map<QuantumEventType, Array<(data: any) => void>> = new Map();
    private agentCallbacks: Map<string, Array<(agent: QuantumAgent) => void>> = new Map();
    private isRunning: boolean = false;
    private updateFrequency: number = 30; // Hz
    private updateInterval?: NodeJS.Timeout;

    constructor() {
        this.initializeEventHandlers();
    }

    // ============================================================================
    // Unity-style Event System
    // ============================================================================

    /**
     * Unity-style: Handle agent collapse event
     */
    async onAgentCollapse(agentId: string): Promise<void> {
        console.log(`[QuantumProtocol] COLLAPSE EVENT for agent: ${agentId}`);

        // Spawn memory ghost (placeholder for Unity integration)
        await this.spawnMemoryGhost(agentId);

        // Show score glyph (placeholder for UI integration)
        await this.showScoreGlyph(agentId);

        // Trigger visual burst (placeholder for graphics integration)
        await this.triggerVisualBurst(agentId);

        // Archive in quantum lore
        await this.archiveInQuantumLore(agentId);

        // Dispatch protocol event
        this.dispatchEvent(QuantumEventType.AGENT_COLLAPSE, { agent_id: agentId });
    }

    /**
     * Unity-style: Handle global collapse event
     */
    async onCollapseAll(): Promise<void> {
        console.log("[QuantumProtocol] GLOBAL COLLAPSE triggered.");

        // Replay all memory ghosts
        await this.replayAllGhosts();

        // Fade world visuals
        await this.fadeWorldVisuals();

        // Play echo field audio
        await this.playEchoFieldAudio();

        // Dispatch swarm convergence event
        this.dispatchEvent(QuantumEventType.SWARM_CONVERGENCE, {});
    }

    /**
     * Unity-style: Handle math function change
     */
    async onFunctionChanged(newFunction: MathFunctionType): Promise<void> {
        console.log(`[QuantumProtocol] Function shift to: ${MathFunctionType[newFunction]}`);

        // Set global shader parameters (placeholder)
        await this.setGlobalShaderParams(newFunction);

        // Play function tone audio
        await this.playFunctionTone(newFunction);

        // Update UI function display
        await this.updateFunctionDisplay(newFunction);

        // Sync trail palette
        await this.syncTrailPalette(newFunction);

        // Dispatch function change event
        this.dispatchEvent(QuantumEventType.FUNCTION_CHANGE, { function_type: newFunction });
    }

    /**
     * Unity-style: Handle agent completion
     */
    async onAgentComplete(agentId: string): Promise<void> {
        console.log(`[QuantumProtocol] Agent ${agentId} completed its journey.`);
        await this.onAgentCollapse(agentId);
    }

    // ============================================================================
    // Quantum Agent Management
    // ============================================================================

    /**
     * Register a new quantum agent
     */
    async registerAgent(agentId: string, initialPosition: Vector3): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "register_agent",
                params: {
                    agent_id: agentId,
                    position: initialPosition
                }
            });

            return response.success;
        } catch (error) {
            console.error(`Failed to register agent ${agentId}:`, error);
            return false;
        }
    }

    /**
     * Remove a quantum agent
     */
    async removeAgent(agentId: string): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "remove_agent",
                params: { agent_id: agentId }
            });

            return response.success;
        } catch (error) {
            console.error(`Failed to remove agent ${agentId}:`, error);
            return false;
        }
    }

    /**
     * Register a quantum step for an agent
     */
    async registerStep(position: Vector3, agentId: string, stepNumber: number, energyLevel: number = 1.0): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "register_step",
                params: {
                    position,
                    agent_id: agentId,
                    step_number: stepNumber,
                    energy_level: energyLevel
                }
            });

            return response.success;
        } catch (error) {
            console.error(`Failed to register step for agent ${agentId}:`, error);
            return false;
        }
    }

    /**
     * Get quantum agent by ID
     */
    async getAgent(agentId: string): Promise<QuantumAgent | null> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "get_agent",
                params: { agent_id: agentId }
            });

            return response.success ? response.data : null;
        } catch (error) {
            console.error(`Failed to get agent ${agentId}:`, error);
            return null;
        }
    }

    /**
     * Get agent path history
     */
    async getAgentPath(agentId: string): Promise<QuantumStep[]> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "get_agent_path",
                params: { agent_id: agentId }
            });

            return response.success ? response.data : [];
        } catch (error) {
            console.error(`Failed to get agent path ${agentId}:`, error);
            return [];
        }
    }

    /**
     * Get all quantum steps
     */
    async getAllSteps(): Promise<QuantumStep[]> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "get_all_steps",
                params: {}
            });

            return response.success ? response.data : [];
        } catch (error) {
            console.error("Failed to get all steps:", error);
            return [];
        }
    }

    // ============================================================================
    // Environmental Event System
    // ============================================================================

    /**
     * Spawn environmental event
     */
    async spawnEnvironmentalEvent(eventType: EnvironmentalEventType, origin: Vector3,
        radius: number = 3.0, duration: number = 10.0): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "spawn_environmental_event",
                params: {
                    event_type: eventType,
                    origin,
                    radius,
                    duration
                }
            });

            return response.success;
        } catch (error) {
            console.error("Failed to spawn environmental event:", error);
            return false;
        }
    }

    /**
     * Get active environmental events
     */
    async getActiveEvents(): Promise<EnvironmentalEvent[]> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "get_active_events",
                params: {}
            });

            return response.success ? response.data : [];
        } catch (error) {
            console.error("Failed to get active events:", error);
            return [];
        }
    }

    // ============================================================================
    // Glyph Amplitude Resolution
    // ============================================================================

    /**
     * Resolve quantum amplitudes and determine collapse winner
     */
    async resolveAndCollapse(): Promise<AmplitudeResult | null> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "resolve_and_collapse",
                params: {}
            });

            return response.success ? response.data : null;
        } catch (error) {
            console.error("Failed to resolve amplitudes:", error);
            return null;
        }
    }

    // ============================================================================
    // Recursive Infrastructure & Swarm Mind
    // ============================================================================

    /**
     * Activate nexus with recursive analysis
     */
    async activateNexus(seedTopic: string = "Quantum Origin"): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "activate_nexus",
                params: { seed_topic: seedTopic }
            });

            if (response.success) {
                console.log(`[QuantumProtocol] Nexus activated with seed: ${seedTopic}`);
            }

            return response.success;
        } catch (error) {
            console.error("Failed to activate nexus:", error);
            return false;
        }
    }

    /**
     * Get recursive infrastructure memory
     */
    async getInfrastructureMemory(): Promise<{ forward_memory: RecursiveInfrastructureNode[], reverse_memory: RecursiveInfrastructureNode[] } | null> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "get_infrastructure_memory",
                params: {}
            });

            return response.success ? response.data : null;
        } catch (error) {
            console.error("Failed to get infrastructure memory:", error);
            return null;
        }
    }

    /**
     * Get swarm mind analysis
     */
    async getSwarmAnalysis(): Promise<SwarmAnalysis | null> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "get_swarm_analysis",
                params: {}
            });

            return response.success ? response.data : null;
        } catch (error) {
            console.error("Failed to get swarm analysis:", error);
            return null;
        }
    }

    /**
     * Perform recursive topic analysis
     */
    async analyzeTopicRecursively(topic: string, iterations: number = 5): Promise<RecursiveInfrastructureNode[]> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "recursive_analysis",
                params: {
                    starting_topic: topic,
                    iterations
                }
            });

            return response.success ? response.data : [];
        } catch (error) {
            console.error("Failed to analyze topic recursively:", error);
            return [];
        }
    }

    // ============================================================================
    // Control & Configuration
    // ============================================================================

    /**
     * Set math function type
     */
    async setFunctionType(functionType: MathFunctionType): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "set_function_type",
                params: { function_type: functionType }
            });

            if (response.success) {
                await this.onFunctionChanged(functionType);
            }

            return response.success;
        } catch (error) {
            console.error("Failed to set function type:", error);
            return false;
        }
    }

    /**
     * Start quantum protocol daemon
     */
    async start(): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "start",
                params: {}
            });

            if (response.success) {
                this.isRunning = true;
                this.startUpdateLoop();
                console.log("[QuantumProtocolRegistry] Started");
            }

            return response.success;
        } catch (error) {
            console.error("Failed to start quantum protocol:", error);
            return false;
        }
    }

    /**
     * Stop quantum protocol daemon
     */
    async stop(): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "stop",
                params: {}
            });

            if (response.success) {
                this.isRunning = false;
                if (this.updateInterval) {
                    clearInterval(this.updateInterval);
                    this.updateInterval = undefined;
                }
                console.log("[QuantumProtocolRegistry] Stopped");
            }

            return response.success;
        } catch (error) {
            console.error("Failed to stop quantum protocol:", error);
            return false;
        }
    }

    /**
     * Configure quantum protocol settings
     */
    async configure(settings: {
        updateFrequency?: number;
        autoCollapseEnabled?: boolean;
        collapseThreshold?: number;
    }): Promise<boolean> {
        try {
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "configure",
                params: settings
            });

            if (response.success && settings.updateFrequency) {
                this.updateFrequency = settings.updateFrequency;
            }

            return response.success;
        } catch (error) {
            console.error("Failed to configure quantum protocol:", error);
            return false;
        }
    }

    // ============================================================================
    // Event System & Callbacks
    // ============================================================================

    /**
     * Register event callback
     */
    onEvent(eventType: QuantumEventType, callback: (data: any) => void): void {
        if (!this.eventCallbacks.has(eventType)) {
            this.eventCallbacks.set(eventType, []);
        }
        this.eventCallbacks.get(eventType)!.push(callback);
    }

    /**
     * Register agent-specific callback
     */
    onAgentEvent(agentId: string, callback: (agent: QuantumAgent) => void): void {
        if (!this.agentCallbacks.has(agentId)) {
            this.agentCallbacks.set(agentId, []);
        }
        this.agentCallbacks.get(agentId)!.push(callback);
    }

    /**
     * Remove event callback
     */
    removeEventCallback(eventType: QuantumEventType, callback: (data: any) => void): void {
        const callbacks = this.eventCallbacks.get(eventType);
        if (callbacks) {
            const index = callbacks.indexOf(callback);
            if (index !== -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    /**
     * Dispatch event to callbacks
     */
    private dispatchEvent(eventType: QuantumEventType, data: any): void {
        const callbacks = this.eventCallbacks.get(eventType);
        if (callbacks) {
            callbacks.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in quantum protocol event callback:`, error);
                }
            });
        }
    }

    // ============================================================================
    // Unity-style Placeholder Methods (for integration with rendering/audio systems)
    // ============================================================================

    private async spawnMemoryGhost(agentId: string): Promise<void> {
        console.log(`[QuantumProtocol] Spawning memory ghost for: ${agentId}`);
        // TODO: Integrate with 3D rendering system to replay agent path
    }

    private async showScoreGlyph(agentId: string): Promise<void> {
        console.log(`[QuantumProtocol] Showing score glyph for: ${agentId}`);
        // TODO: Integrate with UI system to show floating score display
    }

    private async triggerVisualBurst(agentId: string): Promise<void> {
        console.log(`[QuantumProtocol] Triggering visual burst for: ${agentId}`);
        // TODO: Integrate with particle system for collapse effects
    }

    private async archiveInQuantumLore(agentId: string): Promise<void> {
        console.log(`[QuantumProtocol] Archiving in quantum lore: ${agentId}`);
        // TODO: Store in persistent lore/history system
    }

    private async replayAllGhosts(): Promise<void> {
        console.log("[QuantumProtocol] Replaying all memory ghosts");
        // TODO: Replay all agent paths simultaneously
    }

    private async fadeWorldVisuals(): Promise<void> {
        console.log("[QuantumProtocol] Fading world visuals");
        // TODO: Apply visual fade/darken effect
    }

    private async playEchoFieldAudio(): Promise<void> {
        console.log("[QuantumProtocol] Playing echo field audio");
        // TODO: Trigger ambient collapse audio
    }

    private async setGlobalShaderParams(functionType: MathFunctionType): Promise<void> {
        console.log(`[QuantumProtocol] Setting shader params for: ${MathFunctionType[functionType]}`);
        // TODO: Set global shader uniforms for math function visualization
    }

    private async playFunctionTone(functionType: MathFunctionType): Promise<void> {
        console.log(`[QuantumProtocol] Playing function tone: ${MathFunctionType[functionType]}`);
        // TODO: Play audio tone corresponding to math function
    }

    private async updateFunctionDisplay(functionType: MathFunctionType): Promise<void> {
        console.log(`[QuantumProtocol] Updating function display: ${MathFunctionType[functionType]}`);
        // TODO: Update UI panels and controls to reflect function change
    }

    private async syncTrailPalette(functionType: MathFunctionType): Promise<void> {
        console.log(`[QuantumProtocol] Syncing trail palette: ${MathFunctionType[functionType]}`);
        // TODO: Update particle trail colors based on function
    }

    // ============================================================================
    // Internal Update System
    // ============================================================================

    private initializeEventHandlers(): void {
        // Set up default event handlers for protocol events
        this.onEvent(QuantumEventType.AGENT_COLLAPSE, (data) => {
            console.log(`[Event] Agent collapsed: ${data.agent_id}`);
        });

        this.onEvent(QuantumEventType.FUNCTION_CHANGE, (data) => {
            console.log(`[Event] Function changed: ${data.function_type}`);
        });

        this.onEvent(QuantumEventType.SWARM_CONVERGENCE, (data) => {
            console.log(`[Event] Swarm convergence detected`);
        });
    }

    private startUpdateLoop(): void {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        const intervalMs = 1000 / this.updateFrequency;
        this.updateInterval = setInterval(async () => {
            if (this.isRunning) {
                await this.update();
            }
        }, intervalMs);
    }

    private async update(): Promise<void> {
        try {
            // Check for quantum protocol events from daemon
            const response = await sendMessage({
                engine: "quantum_protocol",
                operation: "get_events",
                params: {}
            });

            if (response.success && response.data.events) {
                for (const event of response.data.events) {
                    this.dispatchEvent(event.type, event.data);
                }
            }
        } catch (error) {
            // Silently handle update errors to avoid spam
        }
    }
}

// ============================================================================
// Convenience Functions & Macros (Unity-style)
// ============================================================================

// Global quantum protocol registry instance
export const QuantumProtocol = new QuantumProtocolRegistry();

// Unity-style convenience macros
export const QUANTUM_PROTOCOL = QuantumProtocol;

export const ON_AGENT_COLLAPSE = (agentId: string) => QuantumProtocol.onAgentCollapse(agentId);
export const ON_COLLAPSE_ALL = () => QuantumProtocol.onCollapseAll();
export const ON_FUNCTION_CHANGED = (func: MathFunctionType) => QuantumProtocol.onFunctionChanged(func);
export const ON_AGENT_COMPLETE = (agentId: string) => QuantumProtocol.onAgentComplete(agentId);

export const SPAWN_EVENT = (type: EnvironmentalEventType, origin: Vector3, radius?: number, duration?: number) =>
    QuantumProtocol.spawnEnvironmentalEvent(type, origin, radius, duration);

export const REGISTER_AGENT = (id: string, pos: Vector3) =>
    QuantumProtocol.registerAgent(id, pos);

export const ACTIVATE_NEXUS = (seed: string) =>
    QuantumProtocol.activateNexus(seed);

// Utility functions
export namespace QuantumOps {
    export function createVector3(x: number = 0, y: number = 0, z: number = 0): Vector3 {
        return { x, y, z };
    }

    export function distance(a: Vector3, b: Vector3): number {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dz = a.z - b.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    export function addVector3(a: Vector3, b: Vector3): Vector3 {
        return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
    }

    export function multiplyVector3(v: Vector3, scalar: number): Vector3 {
        return { x: v.x * scalar, y: v.y * scalar, z: v.z * scalar };
    }

    export function generateAgentId(): string {
        return `agent_${Math.random().toString(36).substr(2, 9)}`;
    }

    export function createSinusoidalPath(steps: number, amplitude: number = 2.0): Vector3[] {
        const path: Vector3[] = [];
        for (let i = 0; i < steps; i++) {
            path.push({
                x: i * 0.5,
                y: Math.sin(i * 0.1) * amplitude,
                z: 0
            });
        }
        return path;
    }

    export function createRadialPath(steps: number, radius: number = 5.0): Vector3[] {
        const path: Vector3[] = [];
        for (let i = 0; i < steps; i++) {
            const angle = (i / steps) * 2 * Math.PI;
            path.push({
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius,
                z: 0
            });
        }
        return path;
    }
}

export default QuantumProtocolRegistry;
