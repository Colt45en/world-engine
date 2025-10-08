// Room Adapter - Tier 4 State Management Types and Utilities
// Provides the core interface for Tier-4 state representation

export interface Tier4State {
    /** 4D vector representing the current tier 4 coordinate */
    x: [number, number, number, number];

    /** Meta-parameter controlling system behavior */
    kappa: number;

    /** Current tier level (0-4) */
    level: number;

    /** Optional session metadata */
    meta?: {
        sessionId?: string;
        timestamp?: number;
        operator?: string;
        [key: string]: any;
    };

    /** Optional nucleus state */
    nucleus?: {
        role: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED';
        active: boolean;
        intensity?: number;
    };

    /** Optional clock synchronization data */
    clock?: {
        bar: number;
        beat: number;
        phase: number;
        bpm: number;
        timestamp: number;
    };
}

export interface RoomAdapter {
    /** Get current state */
    getState(): Tier4State;

    /** Set new state */
    setState(state: Tier4State): void;

    /** Apply an operator transformation */
    applyOperator(operator: string, meta?: any): Tier4State;

    /** Subscribe to state changes */
    onStateChange(callback: (state: Tier4State) => void): void;

    /** Unsubscribe from state changes */
    offStateChange(callback: (state: Tier4State) => void): void;

    /** Serialize state for transmission */
    serialize(): string;

    /** Deserialize state from string */
    deserialize(data: string): Tier4State;
}

export class DefaultRoomAdapter implements RoomAdapter {
    private state: Tier4State;
    private listeners: Array<(state: Tier4State) => void> = [];

    constructor(initialState?: Tier4State) {
        this.state = initialState || {
            x: [0, 0.5, 0.4, 0.6],
            kappa: 0.6,
            level: 0
        };
    }

    getState(): Tier4State {
        return { ...this.state };
    }

    setState(state: Tier4State): void {
        this.state = { ...state };
        this.notifyListeners();
    }

    applyOperator(operator: string, meta?: any): Tier4State {
        const newState = this.transformState(operator, this.state, meta);
        this.setState(newState);
        return newState;
    }

    onStateChange(callback: (state: Tier4State) => void): void {
        this.listeners.push(callback);
    }

    offStateChange(callback: (state: Tier4State) => void): void {
        this.listeners = this.listeners.filter(cb => cb !== callback);
    }

    serialize(): string {
        return JSON.stringify(this.state);
    }

    deserialize(data: string): Tier4State {
        try {
            const parsed = JSON.parse(data);
            if (this.isValidState(parsed)) {
                return parsed;
            }
        } catch (error) {
            console.warn('Failed to deserialize state:', error);
        }

        // Return default state if deserialization fails
        return {
            x: [0, 0.5, 0.4, 0.6],
            kappa: 0.6,
            level: 0
        };
    }

    private notifyListeners(): void {
        this.listeners.forEach(callback => {
            try {
                callback(this.state);
            } catch (error) {
                console.error('Error in state change listener:', error);
            }
        });
    }

    private transformState(operator: string, currentState: Tier4State, meta?: any): Tier4State {
        // Implement tier 4 operator transformations
        const newState = { ...currentState };

        // Add meta information
        newState.meta = {
            ...currentState.meta,
            operator: operator,
            timestamp: Date.now(),
            ...meta
        };

        // Apply operator-specific transformations
        switch (operator) {
            case 'RB': // Rotate Base
                newState.x = [
                    newState.x[0] * 0.9 + newState.x[1] * 0.1,
                    newState.x[1] * 0.9 + newState.x[2] * 0.1,
                    newState.x[2] * 0.9 + newState.x[3] * 0.1,
                    newState.x[3] * 0.9 + newState.x[0] * 0.1
                ];
                break;

            case 'UP': // Uplift
                newState.x = newState.x.map(x => Math.min(1, x + 0.1)) as [number, number, number, number];
                newState.kappa = Math.min(1, newState.kappa + 0.05);
                break;

            case 'CV': // Converge
                const center = newState.x.reduce((sum, x) => sum + x, 0) / 4;
                newState.x = newState.x.map(x => x * 0.8 + center * 0.2) as [number, number, number, number];
                break;

            case 'TL': // Transform Linear
                newState.x = [
                    newState.x[1],
                    newState.x[2],
                    newState.x[3],
                    newState.x[0]
                ];
                break;

            case 'MV': // Move
                const delta = (Math.random() - 0.5) * 0.1;
                newState.x = newState.x.map(x => Math.max(0, Math.min(1, x + delta))) as [number, number, number, number];
                break;

            case 'SC': // Scale
                const scale = 0.9 + Math.random() * 0.2;
                newState.x = newState.x.map(x => Math.max(0, Math.min(1, x * scale))) as [number, number, number, number];
                break;

            case 'NG': // Negate
                newState.x = newState.x.map(x => 1 - x) as [number, number, number, number];
                break;

            case 'CN': // Connect
                newState.kappa = Math.max(0, Math.min(1, newState.kappa + (Math.random() - 0.5) * 0.2));
                break;

            case 'MO': // Modulate
                newState.x = newState.x.map((x, i) =>
                    Math.max(0, Math.min(1, x + Math.sin(Date.now() / 1000 + i) * 0.05))
                ) as [number, number, number, number];
                break;

            case 'SA': // Sample
                const sampleIndex = Math.floor(Math.random() * 4);
                const sampleValue = newState.x[sampleIndex];
                newState.x = newState.x.map((x, i) =>
                    i === sampleIndex ? x : x * 0.9 + sampleValue * 0.1
                ) as [number, number, number, number];
                break;

            case 'PS': // Phase Shift
                newState.x = [
                    newState.x[3],
                    newState.x[0],
                    newState.x[1],
                    newState.x[2]
                ];
                break;

            case 'CO': // Collapse
                const collapse = newState.x.reduce((sum, x) => sum + x * x, 0) / 4;
                newState.x = [collapse, collapse, collapse, collapse];
                newState.level = Math.max(0, newState.level - 1);
                break;

            default:
                console.warn(`Unknown operator: ${operator}`);
        }

        // Ensure values stay in valid ranges
        newState.x = newState.x.map(x => Math.max(0, Math.min(1, x))) as [number, number, number, number];
        newState.kappa = Math.max(0, Math.min(1, newState.kappa));
        newState.level = Math.max(0, Math.min(4, newState.level));

        return newState;
    }

    private isValidState(obj: any): obj is Tier4State {
        return obj &&
            Array.isArray(obj.x) &&
            obj.x.length === 4 &&
            obj.x.every((x: any) => typeof x === 'number') &&
            typeof obj.kappa === 'number' &&
            typeof obj.level === 'number';
    }
}

// Utility functions for state manipulation
export const StateUtils = {
    /** Create a default Tier4State */
    createDefault(): Tier4State {
        return {
            x: [0, 0.5, 0.4, 0.6],
            kappa: 0.6,
            level: 0
        };
    },

    /** Calculate distance between two states */
    distance(state1: Tier4State, state2: Tier4State): number {
        const dx = state1.x.map((x, i) => x - state2.x[i]);
        const vectorDistance = Math.sqrt(dx.reduce((sum, d) => sum + d * d, 0));
        const kappaDistance = Math.abs(state1.kappa - state2.kappa);
        const levelDistance = Math.abs(state1.level - state2.level);

        return vectorDistance + kappaDistance + levelDistance;
    },

    /** Interpolate between two states */
    lerp(state1: Tier4State, state2: Tier4State, t: number): Tier4State {
        const clampedT = Math.max(0, Math.min(1, t));

        return {
            x: state1.x.map((x, i) => x + (state2.x[i] - x) * clampedT) as [number, number, number, number],
            kappa: state1.kappa + (state2.kappa - state1.kappa) * clampedT,
            level: Math.round(state1.level + (state2.level - state1.level) * clampedT),
            meta: {
                interpolated: true,
                t: clampedT,
                timestamp: Date.now()
            }
        };
    },

    /** Validate state integrity */
    validate(state: Tier4State): boolean {
        if (!Array.isArray(state.x) || state.x.length !== 4) return false;
        if (!state.x.every(x => typeof x === 'number' && x >= 0 && x <= 1)) return false;
        if (typeof state.kappa !== 'number' || state.kappa < 0 || state.kappa > 1) return false;
        if (typeof state.level !== 'number' || state.level < 0 || state.level > 4) return false;

        return true;
    },

    /** Clone a state deeply */
    clone(state: Tier4State): Tier4State {
        return JSON.parse(JSON.stringify(state));
    }
};
