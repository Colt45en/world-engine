/**
 * NEXUS Forge Performance Monitor
 * • Real-time system performance tracking
 * • Memory usage monitoring
 * • Frame rate optimization
 * • Resource load tracking
 * • Bottleneck detection and reporting
 */

class NexusForgePerformanceMonitor {
    constructor() {
        this.metrics = {
            fps: 60,
            frameTime: 16.67,
            memoryUsage: 0,
            cpuUsage: 0,
            drawCalls: 0,
            triangles: 0,
            textures: 0,
            shaders: 0,
            buffers: 0,
            networkLatency: 0,
            loadTime: 0
        };

        this.history = {
            fps: [],
            frameTime: [],
            memory: [],
            cpu: []
        };

        this.thresholds = {
            fpsWarning: 30,
            fpsCritical: 15,
            memoryWarning: 0.7, // 70%
            memoryCritical: 0.9, // 90%
            frameTimeWarning: 33.33, // 30 FPS
            frameTimeCritical: 66.67 // 15 FPS
        };

        this.warnings = [];
        this.isMonitoring = false;
        this.lastUpdate = performance.now();
        this.updateInterval = 500; // Update every 500ms

        console.log('📊 Performance Monitor initialized');
    }

    startMonitoring() {
        if (this.isMonitoring) return;

        this.isMonitoring = true;
        this.lastUpdate = performance.now();
        this.monitoringLoop();

        console.log('🔍 Performance monitoring started');
    }

    stopMonitoring() {
        this.isMonitoring = false;
        console.log('⏸️ Performance monitoring stopped');
    }

    monitoringLoop() {
        if (!this.isMonitoring) return;

        const now = performance.now();
        const deltaTime = now - this.lastUpdate;

        if (deltaTime >= this.updateInterval) {
            this.updateMetrics();
            this.updateHistory();
            this.checkThresholds();
            this.lastUpdate = now;
        }

        requestAnimationFrame(() => this.monitoringLoop());
    }

    updateMetrics() {
        // Frame rate calculation
        this.metrics.frameTime = performance.now() - this.lastFrameTime || 16.67;
        this.metrics.fps = Math.round(1000 / Math.max(this.metrics.frameTime, 1));
        this.lastFrameTime = performance.now();

        // Memory usage (if available)
        if (performance.memory) {
            const memory = performance.memory;
            this.metrics.memoryUsage = memory.usedJSHeapSize / memory.jsHeapSizeLimit;
        }

        // Estimate CPU usage based on frame time consistency
        this.estimateCPUUsage();

        // WebGL metrics (if available)
        this.updateWebGLMetrics();
    }

    estimateCPUUsage() {
        const frameTimeDeviation = Math.abs(this.metrics.frameTime - 16.67);
        const normalizedDeviation = Math.min(frameTimeDeviation / 16.67, 1);
        this.metrics.cpuUsage = normalizedDeviation * 0.5 + 0.1; // Base 10% + deviation
    }

    updateWebGLMetrics() {
        // Try to get WebGL context info
        const canvas = document.querySelector('canvas');
        if (canvas) {
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            if (gl) {
                this.metrics.drawCalls = this.estimateDrawCalls();
                this.metrics.triangles = this.estimateTriangles();
                this.metrics.textures = this.estimateTextures();
            }
        }
    }

    estimateDrawCalls() {
        // Rough estimate based on active chunks and objects
        if (typeof nexusForge !== 'undefined' && nexusForge) {
            const chunks = nexusForge.world ? nexusForge.world.chunks.size : 0;
            const objects = nexusForge.getTotalObjects ? nexusForge.getTotalObjects() : 0;
            return chunks * 2 + objects; // Rough estimate
        }
        return 0;
    }

    estimateTriangles() {
        // Rough estimate based on terrain and objects
        if (typeof nexusForge !== 'undefined' && nexusForge) {
            const chunks = nexusForge.world ? nexusForge.world.chunks.size : 0;
            const objects = nexusForge.getTotalObjects ? nexusForge.getTotalObjects() : 0;
            return chunks * 2048 + objects * 12; // Rough estimate
        }
        return 0;
    }

    estimateTextures() {
        // Estimate based on biome types and objects
        if (typeof nexusForge !== 'undefined' && nexusForge) {
            const chunks = nexusForge.world ? nexusForge.world.chunks.size : 0;
            return Math.min(chunks * 5, 64); // Max 64 textures
        }
        return 8; // Default textures
    }

    updateHistory() {
        const maxHistoryLength = 60; // Keep 60 samples (30 seconds at 0.5s intervals)

        // Add current metrics to history
        this.history.fps.push(this.metrics.fps);
        this.history.frameTime.push(this.metrics.frameTime);
        this.history.memory.push(this.metrics.memoryUsage);
        this.history.cpu.push(this.metrics.cpuUsage);

        // Trim history to max length
        Object.keys(this.history).forEach(key => {
            if (this.history[key].length > maxHistoryLength) {
                this.history[key].shift();
            }
        });
    }

    checkThresholds() {
        this.warnings = [];

        // FPS warnings
        if (this.metrics.fps <= this.thresholds.fpsCritical) {
            this.warnings.push({
                type: 'critical',
                category: 'performance',
                message: `Critical FPS drop: ${this.metrics.fps} FPS`,
                suggestion: 'Reduce render distance or object complexity'
            });
        } else if (this.metrics.fps <= this.thresholds.fpsWarning) {
            this.warnings.push({
                type: 'warning',
                category: 'performance',
                message: `Low FPS detected: ${this.metrics.fps} FPS`,
                suggestion: 'Consider reducing graphics quality'
            });
        }

        // Memory warnings
        if (this.metrics.memoryUsage >= this.thresholds.memoryCritical) {
            this.warnings.push({
                type: 'critical',
                category: 'memory',
                message: `Critical memory usage: ${(this.metrics.memoryUsage * 100).toFixed(1)}%`,
                suggestion: 'Clear unused chunks or restart application'
            });
        } else if (this.metrics.memoryUsage >= this.thresholds.memoryWarning) {
            this.warnings.push({
                type: 'warning',
                category: 'memory',
                message: `High memory usage: ${(this.metrics.memoryUsage * 100).toFixed(1)}%`,
                suggestion: 'Consider reducing active chunks'
            });
        }

        // Frame time warnings
        if (this.metrics.frameTime >= this.thresholds.frameTimeCritical) {
            this.warnings.push({
                type: 'critical',
                category: 'performance',
                message: `Frame time spike: ${this.metrics.frameTime.toFixed(2)}ms`,
                suggestion: 'Check for performance bottlenecks'
            });
        }
    }

    getAverageMetric(metric, samples = 10) {
        const history = this.history[metric];
        if (!history || history.length === 0) return 0;

        const recentSamples = history.slice(-samples);
        return recentSamples.reduce((sum, val) => sum + val, 0) / recentSamples.length;
    }

    getPerformanceReport() {
        const avgFPS = this.getAverageMetric('fps');
        const avgFrameTime = this.getAverageMetric('frameTime');
        const avgMemory = this.getAverageMetric('memory');
        const avgCPU = this.getAverageMetric('cpu');

        return {
            current: { ...this.metrics },
            averages: {
                fps: avgFPS,
                frameTime: avgFrameTime,
                memory: avgMemory,
                cpu: avgCPU
            },
            warnings: [...this.warnings],
            grade: this.getPerformanceGrade(),
            recommendations: this.getRecommendations()
        };
    }

    getPerformanceGrade() {
        let score = 100;

        // FPS scoring
        if (this.metrics.fps >= 60) score += 0;
        else if (this.metrics.fps >= 30) score -= 10;
        else if (this.metrics.fps >= 15) score -= 30;
        else score -= 50;

        // Memory scoring
        if (this.metrics.memoryUsage <= 0.5) score += 0;
        else if (this.metrics.memoryUsage <= 0.7) score -= 10;
        else if (this.metrics.memoryUsage <= 0.9) score -= 20;
        else score -= 40;

        // CPU scoring
        if (this.metrics.cpuUsage <= 0.3) score += 0;
        else if (this.metrics.cpuUsage <= 0.6) score -= 5;
        else if (this.metrics.cpuUsage <= 0.8) score -= 15;
        else score -= 25;

        score = Math.max(0, Math.min(100, score));

        if (score >= 90) return 'A+';
        if (score >= 80) return 'A';
        if (score >= 70) return 'B';
        if (score >= 60) return 'C';
        if (score >= 50) return 'D';
        return 'F';
    }

    getRecommendations() {
        const recommendations = [];

        if (this.metrics.fps < 30) {
            recommendations.push('Reduce render distance');
            recommendations.push('Lower chunk resolution');
            recommendations.push('Disable complex effects');
        }

        if (this.metrics.memoryUsage > 0.7) {
            recommendations.push('Clear inactive chunks');
            recommendations.push('Reduce texture quality');
            recommendations.push('Limit active objects');
        }

        if (this.metrics.drawCalls > 1000) {
            recommendations.push('Batch similar objects');
            recommendations.push('Use instanced rendering');
            recommendations.push('Reduce geometry complexity');
        }

        if (recommendations.length === 0) {
            recommendations.push('Performance is optimal');
        }

        return recommendations;
    }

    // Real-time performance adjustment
    autoOptimize(nexusForge) {
        if (!nexusForge) return;

        const report = this.getPerformanceReport();

        // Auto-adjust render distance based on FPS
        if (this.metrics.fps < 20 && nexusForge.world.renderDistance > 1) {
            nexusForge.world.renderDistance = Math.max(1, nexusForge.world.renderDistance - 1);
            console.log(`🔧 Auto-reduced render distance to ${nexusForge.world.renderDistance}`);
        } else if (this.metrics.fps > 55 && nexusForge.world.renderDistance < 5) {
            nexusForge.world.renderDistance = Math.min(5, nexusForge.world.renderDistance + 1);
            console.log(`🔧 Auto-increased render distance to ${nexusForge.world.renderDistance}`);
        }

        // Auto-adjust chunk size based on memory
        if (this.metrics.memoryUsage > 0.8 && nexusForge.world.chunkSize > 32) {
            nexusForge.world.chunkSize = Math.max(32, nexusForge.world.chunkSize - 16);
            console.log(`🔧 Auto-reduced chunk size to ${nexusForge.world.chunkSize}`);
        }

        return report;
    }

    // Create visual performance overlay
    createPerformanceOverlay() {
        let overlay = document.getElementById('performance-overlay');

        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'performance-overlay';
            overlay.style.cssText = `
                position: fixed;
                top: 70px;
                right: 10px;
                background: rgba(11, 14, 20, 0.9);
                border: 1px solid #1e2b46;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                color: #e6f0ff;
                z-index: 1000;
                min-width: 200px;
            `;
            document.body.appendChild(overlay);
        }

        return overlay;
    }

    updatePerformanceOverlay() {
        const overlay = this.createPerformanceOverlay();
        const report = this.getPerformanceReport();

        const fpsColor = this.metrics.fps >= 30 ? '#28a745' : this.metrics.fps >= 15 ? '#ffc107' : '#dc3545';
        const memoryColor = this.metrics.memoryUsage <= 0.7 ? '#28a745' : this.metrics.memoryUsage <= 0.9 ? '#ffc107' : '#dc3545';

        overlay.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 8px; color: #54f0b8;">
                📊 Performance Monitor
            </div>
            <div style="color: ${fpsColor};">FPS: ${this.metrics.fps}</div>
            <div>Frame: ${this.metrics.frameTime.toFixed(2)}ms</div>
            <div style="color: ${memoryColor};">Memory: ${(this.metrics.memoryUsage * 100).toFixed(1)}%</div>
            <div>CPU: ${(this.metrics.cpuUsage * 100).toFixed(1)}%</div>
            <div>Draws: ${this.metrics.drawCalls}</div>
            <div>Triangles: ${this.metrics.triangles.toLocaleString()}</div>
            <div style="margin-top: 8px; font-weight: bold;">
                Grade: <span style="color: ${report.grade.startsWith('A') ? '#28a745' : report.grade === 'B' ? '#54f0b8' : report.grade === 'C' ? '#ffc107' : '#dc3545'}">${report.grade}</span>
            </div>
            ${this.warnings.length > 0 ? `
                <div style="margin-top: 8px; color: #dc3545; font-size: 10px;">
                    ⚠️ ${this.warnings.length} warning(s)
                </div>
            ` : ''}
        `;
    }

    // Toggle performance overlay
    toggleOverlay() {
        const overlay = document.getElementById('performance-overlay');
        if (overlay) {
            overlay.style.display = overlay.style.display === 'none' ? 'block' : 'none';
        } else {
            this.createPerformanceOverlay();
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgePerformanceMonitor;
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    window.NexusForgePerformanceMonitor = NexusForgePerformanceMonitor;
    console.log('📊 Performance Monitor available globally');
}
