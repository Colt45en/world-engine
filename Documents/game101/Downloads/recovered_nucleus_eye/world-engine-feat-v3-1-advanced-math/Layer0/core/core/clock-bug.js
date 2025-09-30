/**
 * CLOCK-BUG.JS - Clock/Timing Debug Utility
 * =========================================
 *
 * Debug utility to identify timing precision issues, drift, and synchronization bugs
 * in the Holy Beat Clock system. Integrates with pad interface for real-time monitoring.
 */

export class ClockBugDetector {
    constructor() {
        this.samples = [];
        this.maxSamples = 1000;
        this.startTime = null;
        this.lastBeatTime = null;
        this.beatCount = 0;
        this.driftSamples = [];
        this.anomalies = [];

        console.log('🐛 ClockBugDetector initialized');
    }

    // Monitor clock performance
    analyzeClock(clockState, actualTime) {
        if (!this.startTime) {
            this.startTime = actualTime;
        }

        const sample = {
            timestamp: actualTime,
            elapsed: actualTime - this.startTime,
            bar: clockState.bar,
            beat: clockState.beat,
            phase: clockState.phase,
            bpm: clockState.bpm,
            expectedPhase: null,
            phaseDrift: 0,
            beatInterval: 0,
            jitter: 0
        };

        // Calculate expected phase based on elapsed time
        const secPerBeat = 60 / clockState.bpm;
        const expectedTotalBeats = sample.elapsed / secPerBeat;
        sample.expectedPhase = expectedTotalBeats - Math.floor(expectedTotalBeats);

        // Calculate phase drift
        sample.phaseDrift = Math.abs(clockState.phase - sample.expectedPhase);

        // Detect beat timing issues
        if (clockState.beat !== this.lastBeat && this.lastBeatTime) {
            sample.beatInterval = actualTime - this.lastBeatTime;
            const expectedInterval = secPerBeat;
            sample.jitter = Math.abs(sample.beatInterval - expectedInterval);

            // High jitter indicates timing problems
            if (sample.jitter > expectedInterval * 0.05) { // 5% tolerance
                this.anomalies.push({
                    type: 'BEAT_JITTER',
                    severity: sample.jitter > expectedInterval * 0.1 ? 'HIGH' : 'MEDIUM',
                    value: sample.jitter,
                    expected: expectedInterval,
                    timestamp: actualTime,
                    message: `Beat jitter: ${(sample.jitter * 1000).toFixed(2)}ms (expected: ${(expectedInterval * 1000).toFixed(2)}ms)`
                });
            }
        }

        // Check for phase drift
        if (sample.phaseDrift > 0.02) { // 2% phase tolerance
            this.anomalies.push({
                type: 'PHASE_DRIFT',
                severity: sample.phaseDrift > 0.05 ? 'HIGH' : 'MEDIUM',
                value: sample.phaseDrift,
                timestamp: actualTime,
                message: `Phase drift: ${(sample.phaseDrift * 100).toFixed(2)}% (φ=${clockState.phase.toFixed(4)}, expected=${sample.expectedPhase.toFixed(4)})`
            });
        }

        // Store sample
        this.samples.push(sample);
        if (this.samples.length > this.maxSamples) {
            this.samples.shift();
        }

        if (clockState.beat !== this.lastBeat) {
            this.lastBeat = clockState.beat;
            this.lastBeatTime = actualTime;
            this.beatCount++;
        }

        return sample;
    }

    // Detect common clock bugs
    detectBugs() {
        if (this.samples.length < 10) return [];

        const bugs = [];
        const recentSamples = this.samples.slice(-50);

        // 1. Consistent drift pattern
        const drifts = recentSamples.map(s => s.phaseDrift);
        const avgDrift = drifts.reduce((a, b) => a + b, 0) / drifts.length;
        if (avgDrift > 0.01) {
            bugs.push({
                type: 'SYSTEMATIC_DRIFT',
                severity: 'HIGH',
                description: `Consistent phase drift detected: ${(avgDrift * 100).toFixed(2)}% average`,
                fix: 'Check clock.tick() calculation precision or audio context timing'
            });
        }

        // 2. Beat quantization errors
        const beatJitters = recentSamples
            .filter(s => s.jitter > 0)
            .map(s => s.jitter);

        if (beatJitters.length > 5) {
            const maxJitter = Math.max(...beatJitters);
            const avgJitter = beatJitters.reduce((a, b) => a + b, 0) / beatJitters.length;

            if (avgJitter > 0.01) { // 10ms average jitter
                bugs.push({
                    type: 'BEAT_QUANTIZATION_ERROR',
                    severity: maxJitter > 0.05 ? 'HIGH' : 'MEDIUM',
                    description: `Beat timing inconsistent: ${(avgJitter * 1000).toFixed(2)}ms avg jitter, ${(maxJitter * 1000).toFixed(2)}ms max`,
                    fix: 'Check requestAnimationFrame vs setInterval usage, or audio buffer scheduling'
                });
            }
        }

        // 3. BPM calculation errors
        if (this.beatCount > 10 && this.samples.length > 50) {
            const totalTime = this.samples[this.samples.length - 1].elapsed;
            const measuredBPM = (this.beatCount / totalTime) * 60;
            const expectedBPM = recentSamples[recentSamples.length - 1].bpm;
            const bpmError = Math.abs(measuredBPM - expectedBPM) / expectedBPM;

            if (bpmError > 0.02) { // 2% BPM error
                bugs.push({
                    type: 'BPM_CALCULATION_ERROR',
                    severity: bpmError > 0.05 ? 'HIGH' : 'MEDIUM',
                    description: `BPM mismatch: measured ${measuredBPM.toFixed(2)}, expected ${expectedBPM.toFixed(2)} (${(bpmError * 100).toFixed(2)}% error)`,
                    fix: 'Verify secPerBeat = 60/BPM calculation and clock start/stop logic'
                });
            }
        }

        return bugs;
    }

    // Generate debug report
    generateReport() {
        const bugs = this.detectBugs();
        const recentAnomalies = this.anomalies.slice(-20);

        return {
            timestamp: Date.now(),
            samples: this.samples.length,
            beats: this.beatCount,
            bugs: bugs,
            anomalies: recentAnomalies,
            stats: this.getStats()
        };
    }

    getStats() {
        if (this.samples.length === 0) return {};

        const recentSamples = this.samples.slice(-100);
        const drifts = recentSamples.map(s => s.phaseDrift);
        const jitters = recentSamples.filter(s => s.jitter > 0).map(s => s.jitter);

        return {
            avgPhaseDrift: drifts.reduce((a, b) => a + b, 0) / drifts.length,
            maxPhaseDrift: Math.max(...drifts),
            avgJitter: jitters.length > 0 ? jitters.reduce((a, b) => a + b, 0) / jitters.length : 0,
            maxJitter: jitters.length > 0 ? Math.max(...jitters) : 0,
            beatCount: this.beatCount,
            uptime: this.samples.length > 0 ? this.samples[this.samples.length - 1].elapsed : 0
        };
    }

    // Clear anomalies and reset
    reset() {
        this.samples = [];
        this.anomalies = [];
        this.startTime = null;
        this.lastBeatTime = null;
        this.beatCount = 0;
        console.log('🐛 ClockBugDetector reset');
    }

    // Export data for external analysis
    exportData() {
        return {
            samples: this.samples,
            anomalies: this.anomalies,
            stats: this.getStats(),
            timestamp: Date.now()
        };
    }
}

// Real-time clock monitoring overlay for pad interface
export class ClockDebugOverlay {
    constructor(container) {
        this.container = container;
        this.detector = new ClockBugDetector();
        this.element = null;
        this.isVisible = false;
        this.createOverlay();
    }

    createOverlay() {
        this.element = document.createElement('div');
        this.element.className = 'clock-debug-overlay';
        this.element.innerHTML = `
            <div class="debug-header">
                <span>🐛 Clock Debug</span>
                <button class="debug-close">×</button>
            </div>
            <div class="debug-content">
                <div class="debug-stats"></div>
                <div class="debug-anomalies"></div>
                <div class="debug-controls">
                    <button class="debug-reset">Reset</button>
                    <button class="debug-export">Export</button>
                </div>
            </div>
        `;

        // Add CSS
        const style = document.createElement('style');
        style.textContent = `
            .clock-debug-overlay {
                position: fixed;
                top: 10px;
                right: 10px;
                width: 320px;
                max-height: 500px;
                background: rgba(20, 20, 20, 0.95);
                border: 1px solid #444;
                border-radius: 8px;
                color: #fff;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                z-index: 10000;
                overflow: hidden;
                backdrop-filter: blur(8px);
            }
            .debug-header {
                background: #333;
                padding: 8px 12px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #555;
            }
            .debug-close {
                background: transparent;
                border: none;
                color: #fff;
                cursor: pointer;
                font-size: 16px;
            }
            .debug-content {
                padding: 12px;
                max-height: 400px;
                overflow-y: auto;
            }
            .debug-stats {
                margin-bottom: 12px;
            }
            .debug-anomalies {
                margin-bottom: 12px;
                max-height: 200px;
                overflow-y: auto;
                font-size: 10px;
            }
            .anomaly-high { color: #ff4444; }
            .anomaly-medium { color: #ffaa00; }
            .anomaly-low { color: #44ff44; }
            .debug-controls {
                display: flex;
                gap: 8px;
            }
            .debug-controls button {
                padding: 4px 8px;
                background: #444;
                border: 1px solid #666;
                color: #fff;
                border-radius: 4px;
                cursor: pointer;
                font-size: 10px;
            }
        `;
        document.head.appendChild(style);

        // Event handlers
        this.element.querySelector('.debug-close').addEventListener('click', () => this.hide());
        this.element.querySelector('.debug-reset').addEventListener('click', () => {
            this.detector.reset();
            this.updateDisplay();
        });
        this.element.querySelector('.debug-export').addEventListener('click', () => {
            const data = this.detector.exportData();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `clock-debug-${Date.now()}.json`;
            a.click();
        });

        this.container.appendChild(this.element);
    }

    show() {
        this.isVisible = true;
        this.element.style.display = 'block';
    }

    hide() {
        this.isVisible = false;
        this.element.style.display = 'none';
    }

    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    update(clockState, actualTime) {
        this.detector.analyzeClock(clockState, actualTime);

        // Update display every 10 samples to reduce overhead
        if (this.detector.samples.length % 10 === 0) {
            this.updateDisplay();
        }
    }

    updateDisplay() {
        if (!this.isVisible) return;

        const stats = this.detector.getStats();
        const bugs = this.detector.detectBugs();
        const recentAnomalies = this.detector.anomalies.slice(-10);

        // Update stats
        const statsEl = this.element.querySelector('.debug-stats');
        statsEl.innerHTML = `
            <div><strong>Clock Stats:</strong></div>
            <div>Beats: ${stats.beatCount}</div>
            <div>Uptime: ${stats.uptime?.toFixed(2)}s</div>
            <div>Avg Drift: ${(stats.avgPhaseDrift * 100)?.toFixed(3)}%</div>
            <div>Max Drift: ${(stats.maxPhaseDrift * 100)?.toFixed(3)}%</div>
            <div>Avg Jitter: ${(stats.avgJitter * 1000)?.toFixed(2)}ms</div>
        `;

        // Update anomalies
        const anomaliesEl = this.element.querySelector('.debug-anomalies');
        anomaliesEl.innerHTML = `
            <div><strong>Recent Issues:</strong></div>
            ${bugs.map(bug => `
                <div class="anomaly-${bug.severity.toLowerCase()}">
                    ${bug.type}: ${bug.description}
                </div>
            `).join('')}
            ${recentAnomalies.map(anomaly => `
                <div class="anomaly-${anomaly.severity.toLowerCase()}">
                    ${anomaly.message}
                </div>
            `).join('')}
        `;
    }
}

// Global debug instance for console access
if (typeof window !== 'undefined') {
    window.ClockDebug = { ClockBugDetector, ClockDebugOverlay };
    console.log('🐛 Clock debugging utilities loaded. Use window.ClockDebug');
}
