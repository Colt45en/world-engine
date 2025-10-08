// Clock Debug Type Definitions
// Provides type definitions for the Clock Debug utilities used in EnhancedEngineRoom

export interface ClockState {
    bar: number;
    beat: number;
    phase: number;
    bpm: number;
    running: boolean;
    startTime: number;
}

export interface ClockAnomaly {
    type: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH';
    description: string;
    timestamp: number;
    value?: number;
}

export interface ClockSample {
    timestamp: number;
    state: ClockState;
    drift?: number;
    jitter?: number;
}

export interface ClockStats {
    totalSamples: number;
    averageBPM: number;
    driftVariance: number;
    jitterVariance: number;
    anomalies: ClockAnomaly[];
    uptime: number;
}

export interface ClockBugDetector {
    analyzeClock(state: ClockState, timestamp: number): ClockSample;
    detectBugs(): ClockAnomaly[];
    getStats(): ClockStats;
    exportData(): any;
    reset(): void;
}

export interface ClockDebugOverlay {
    update(state: ClockState, timestamp: number): void;
    toggle(): void;
    show(): void;
    hide(): void;
    destroy(): void;
}

// Extend the Window interface to include ClockDebug
declare global {
    interface Window {
        ClockDebug?: {
            ClockBugDetector: new () => ClockBugDetector;
            ClockDebugOverlay: new (container: HTMLElement) => ClockDebugOverlay;
        };
    }
}

// Mock implementations for when clock debugging is not available
export class MockClockBugDetector implements ClockBugDetector {
    private samples: ClockSample[] = [];
    private anomalies: ClockAnomaly[] = [];

    analyzeClock(state: ClockState, timestamp: number): ClockSample {
        const sample: ClockSample = {
            timestamp,
            state: { ...state },
            drift: 0,
            jitter: 0
        };

        this.samples.push(sample);

        // Keep only last 1000 samples
        if (this.samples.length > 1000) {
            this.samples = this.samples.slice(-1000);
        }

        return sample;
    }

    detectBugs(): ClockAnomaly[] {
        // Return copy of current anomalies and clear them
        const result = [...this.anomalies];
        this.anomalies = [];
        return result;
    }

    getStats(): ClockStats {
        return {
            totalSamples: this.samples.length,
            averageBPM: 120,
            driftVariance: 0,
            jitterVariance: 0,
            anomalies: [...this.anomalies],
            uptime: this.samples.length > 0 ? Date.now() - this.samples[0].timestamp : 0
        };
    }

    exportData(): any {
        return {
            samples: this.samples,
            anomalies: this.anomalies,
            stats: this.getStats()
        };
    }

    reset(): void {
        this.samples = [];
        this.anomalies = [];
    }
}

export class MockClockDebugOverlay implements ClockDebugOverlay {
    private container: HTMLElement;
    private overlay: HTMLElement | null = null;
    private visible = false;

    constructor(container: HTMLElement) {
        this.container = container;
        this.createOverlay();
    }

    private createOverlay(): void {
        this.overlay = document.createElement('div');
        this.overlay.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 10px;
      border-radius: 5px;
      font-family: monospace;
      font-size: 12px;
      z-index: 10000;
      display: none;
      min-width: 200px;
    `;

        this.overlay.innerHTML = `
      <div>Clock Debug Overlay</div>
      <div>Bar: 0 | Beat: 0</div>
      <div>Phase: 0.000 | BPM: 120</div>
      <div>Status: Stopped</div>
    `;

        this.container.appendChild(this.overlay);
    }

    update(state: ClockState, timestamp: number): void {
        if (!this.overlay || !this.visible) return;

        this.overlay.innerHTML = `
      <div style="font-weight: bold; margin-bottom: 5px;">Clock Debug Overlay</div>
      <div>Bar: ${state.bar} | Beat: ${state.beat}</div>
      <div>Phase: ${state.phase.toFixed(3)} | BPM: ${state.bpm}</div>
      <div>Status: ${state.running ? 'Running' : 'Stopped'}</div>
      <div style="font-size: 10px; opacity: 0.7; margin-top: 5px;">
        Updated: ${new Date(timestamp).toLocaleTimeString()}
      </div>
    `;
    }

    toggle(): void {
        if (this.visible) {
            this.hide();
        } else {
            this.show();
        }
    }

    show(): void {
        if (this.overlay) {
            this.overlay.style.display = 'block';
            this.visible = true;
        }
    }

    hide(): void {
        if (this.overlay) {
            this.overlay.style.display = 'none';
            this.visible = false;
        }
    }

    destroy(): void {
        if (this.overlay && this.container.contains(this.overlay)) {
            this.container.removeChild(this.overlay);
        }
        this.overlay = null;
        this.visible = false;
    }
}

// Utility to get clock debug utilities with fallback to mocks
export function getClockDebugUtils() {
    if (typeof window !== 'undefined' && window.ClockDebug) {
        return window.ClockDebug;
    }

    return {
        ClockBugDetector: MockClockBugDetector,
        ClockDebugOverlay: MockClockDebugOverlay
    };
}

export default {
    MockClockBugDetector,
    MockClockDebugOverlay,
    getClockDebugUtils
};
