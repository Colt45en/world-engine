/**
 * VortexLab Animation Engine for NEXUS FORGE PRIMORDIAL
 * ====================================================
 *
 * Advanced oscillating limb animation system for AI pattern flow
 * visualization and system health metrics. Features:
 * • Oscillating limb animations with quantum effects
 * • Real-time responsiveness to AI intelligence patterns
 * • System health visualization through organic movement
 * • Multi-limb coordination for complex data representation
 * • Integration with React-based components
 * • Dynamic vortex effects for pattern emergence visualization
 */

class VortexLabAnimationEngine {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.canvas = null;
        this.ctx = null;

        // Animation configuration
        this.config = {
            limbCount: options.limbCount || 6,
            centralRadius: options.centralRadius || 40,
            limbLength: options.limbLength || 120,
            segmentCount: options.segmentCount || 8,
            baseFrequency: options.baseFrequency || 0.02,
            amplitude: options.amplitude || 0.8,
            quantumIntensity: options.quantumIntensity || 1.0,
            ...options
        };

        // Animation state
        this.animationFrame = 0;
        this.isAnimating = false;
        this.lastFrameTime = 0;
        this.deltaTime = 0;

        // Limb system
        this.limbs = [];
        this.vortexCore = {
            x: 0, y: 0,
            energy: 1.0,
            resonance: 0.0,
            phase: 0.0
        };

        // AI Intelligence integration
        this.aiMetrics = {
            codeQuality: 0.85,
            userSatisfaction: 0.72,
            developmentVelocity: 0.91,
            technicalDebt: 0.43,
            painLevel: 0.2,
            burstActivity: 0.1
        };

        // Visual effects
        this.effects = {
            particleSystem: [],
            trailSystem: [],
            quantumField: [],
            energyRings: []
        };

        // Color palettes for different states
        this.palettes = {
            healthy: ['#00ff7f', '#28f49b', '#7cfccb', '#26de81'],
            warning: ['#ffa502', '#ff9f43', '#ffb142', '#ff8c00'],
            critical: ['#ff4757', '#ff3838', '#ff6b9d', '#ff4757'],
            processing: ['#7b68ee', '#9370db', '#ba55d3', '#8a2be2'],
            quantum: ['#00ffff', '#40e0d0', '#48cae4', '#0077be']
        };

        this.initialize();
        console.log('🌀 VortexLab Animation Engine initialized');
    }

    initialize() {
        this.createCanvas();
        this.initializeLimbs();
        this.initializeEffects();
        this.setupEventHandlers();
        this.startAnimation();
    }

    createCanvas() {
        // Create main animation canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = 400;
        this.canvas.height = 300;
        this.canvas.style.cssText = `
            width: 100%;
            height: 100%;
            background: transparent;
            border-radius: 12px;
        `;

        this.ctx = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);

        // Set center point
        this.vortexCore.x = this.canvas.width / 2;
        this.vortexCore.y = this.canvas.height / 2;

        console.log('🎨 VortexLab canvas created:', {
            width: this.canvas.width,
            height: this.canvas.height
        });
    }

    initializeLimbs() {
        this.limbs = [];

        for (let i = 0; i < this.config.limbCount; i++) {
            const limb = {
                id: i,
                angle: (Math.PI * 2 * i) / this.config.limbCount,
                baseAngle: (Math.PI * 2 * i) / this.config.limbCount,
                segments: [],
                energy: 1.0,
                frequency: this.config.baseFrequency + (Math.random() - 0.5) * 0.01,
                phase: Math.random() * Math.PI * 2,
                amplitude: this.config.amplitude + (Math.random() - 0.5) * 0.2,
                health: 1.0,
                activity: 0.5
            };

            // Initialize limb segments
            for (let j = 0; j < this.config.segmentCount; j++) {
                const segment = {
                    index: j,
                    length: this.config.limbLength / this.config.segmentCount,
                    angle: 0,
                    x: 0, y: 0,
                    energy: 1.0,
                    oscillation: 0,
                    trail: []
                };
                limb.segments.push(segment);
            }

            this.limbs.push(limb);
        }

        console.log(`🦾 Initialized ${this.config.limbCount} limbs with ${this.config.segmentCount} segments each`);
    }

    initializeEffects() {
        // Initialize particle system
        this.effects.particleSystem = [];

        // Initialize quantum field points
        for (let i = 0; i < 20; i++) {
            this.effects.quantumField.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                phase: Math.random() * Math.PI * 2,
                frequency: 0.01 + Math.random() * 0.02,
                amplitude: 2 + Math.random() * 3,
                energy: Math.random()
            });
        }

        // Initialize energy rings
        for (let i = 0; i < 3; i++) {
            this.effects.energyRings.push({
                radius: 30 + i * 20,
                baseRadius: 30 + i * 20,
                phase: i * Math.PI / 3,
                frequency: 0.015 + i * 0.005,
                amplitude: 5 + i * 2,
                energy: 1.0,
                opacity: 0.6 - i * 0.15
            });
        }

        console.log('✨ VortexLab effects initialized');
    }

    setupEventHandlers() {
        // Canvas interaction for manual energy injection
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            this.injectEnergy(x, y, 0.5);
        });

        // Resize handler
        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });

        console.log('👆 VortexLab event handlers ready');
    }

    startAnimation() {
        if (this.isAnimating) return;

        this.isAnimating = true;
        this.lastFrameTime = performance.now();
        this.animate();

        console.log('▶️ VortexLab animation started');
    }

    stopAnimation() {
        this.isAnimating = false;
        console.log('⏸️ VortexLab animation stopped');
    }

    animate(currentTime = performance.now()) {
        if (!this.isAnimating) return;

        this.deltaTime = currentTime - this.lastFrameTime;
        this.lastFrameTime = currentTime;
        this.animationFrame++;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Update and render all components
        this.updateVortexCore();
        this.updateLimbs();
        this.updateEffects();

        this.renderQuantumField();
        this.renderEnergyRings();
        this.renderLimbs();
        this.renderVortexCore();
        this.renderParticles();
        this.renderMetrics();

        requestAnimationFrame(() => this.animate());
    }

    updateVortexCore() {
        // Core energy based on AI metrics
        const avgMetrics = (this.aiMetrics.codeQuality +
            this.aiMetrics.userSatisfaction +
            this.aiMetrics.developmentVelocity +
            (1 - this.aiMetrics.technicalDebt)) / 4;

        this.vortexCore.energy = 0.5 + avgMetrics * 0.5;
        this.vortexCore.resonance = this.aiMetrics.burstActivity;
        this.vortexCore.phase += 0.02 + this.vortexCore.resonance * 0.05;

        // Pulsation based on pain level
        const painIntensity = this.aiMetrics.painLevel;
        this.vortexCore.pulsation = Math.sin(this.vortexCore.phase * 2) * painIntensity;
    }

    updateLimbs() {
        this.limbs.forEach((limb, limbIndex) => {
            // Update limb energy based on AI metrics
            limb.energy = this.vortexCore.energy * (0.8 + Math.random() * 0.4);
            limb.health = this.getLimbHealth(limbIndex);
            limb.activity = this.getLimbActivity(limbIndex);

            // Update limb phase and frequency
            limb.phase += limb.frequency * (1 + this.vortexCore.resonance);
            limb.angle = limb.baseAngle +
                Math.sin(limb.phase) * limb.amplitude * limb.activity;

            // Update segments
            let cumulativeAngle = limb.angle;
            let currentX = this.vortexCore.x;
            let currentY = this.vortexCore.y;

            limb.segments.forEach((segment, segmentIndex) => {
                // Segment oscillation based on position and limb energy
                const segmentPhase = limb.phase + segmentIndex * 0.5;
                const segmentOscillation = Math.sin(segmentPhase) * 0.3 * limb.energy;

                segment.angle = cumulativeAngle + segmentOscillation;
                segment.oscillation = segmentOscillation;
                segment.energy = limb.energy * (1 - segmentIndex * 0.1);

                // Calculate position
                const segmentEndX = currentX + Math.cos(segment.angle) * segment.length;
                const segmentEndY = currentY + Math.sin(segment.angle) * segment.length;

                segment.x = segmentEndX;
                segment.y = segmentEndY;

                // Update trail
                segment.trail.push({ x: segmentEndX, y: segmentEndY, time: this.animationFrame });
                if (segment.trail.length > 20) {
                    segment.trail.shift();
                }

                // Prepare for next segment
                currentX = segmentEndX;
                currentY = segmentEndY;
                cumulativeAngle = segment.angle;
            });
        });
    }

    updateEffects() {
        // Update quantum field
        this.effects.quantumField.forEach(point => {
            point.phase += point.frequency;
            point.energy = 0.3 + Math.sin(point.phase) * 0.7;
        });

        // Update energy rings
        this.effects.energyRings.forEach(ring => {
            ring.phase += ring.frequency;
            ring.radius = ring.baseRadius + Math.sin(ring.phase) * ring.amplitude;
            ring.energy = this.vortexCore.energy;
        });

        // Update particles
        this.effects.particleSystem.forEach((particle, index) => {
            particle.life -= 0.02;
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.opacity = particle.life;

            if (particle.life <= 0) {
                this.effects.particleSystem.splice(index, 1);
            }
        });

        // Spawn new particles based on activity
        if (Math.random() < this.vortexCore.resonance) {
            this.spawnParticle();
        }
    }

    renderQuantumField() {
        this.ctx.globalAlpha = 0.15;
        this.ctx.fillStyle = '#40e0d0';

        this.effects.quantumField.forEach(point => {
            const size = point.energy * 3;
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, size, 0, Math.PI * 2);
            this.ctx.fill();
        });

        this.ctx.globalAlpha = 1.0;
    }

    renderEnergyRings() {
        this.effects.energyRings.forEach(ring => {
            this.ctx.globalAlpha = ring.opacity * ring.energy;
            this.ctx.strokeStyle = '#00ff7f';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);

            this.ctx.beginPath();
            this.ctx.arc(this.vortexCore.x, this.vortexCore.y, ring.radius, 0, Math.PI * 2);
            this.ctx.stroke();
        });

        this.ctx.setLineDash([]);
        this.ctx.globalAlpha = 1.0;
    }

    renderLimbs() {
        this.limbs.forEach(limb => {
            const healthColor = this.getHealthColor(limb.health);

            // Render limb segments
            let currentX = this.vortexCore.x;
            let currentY = this.vortexCore.y;

            limb.segments.forEach((segment, index) => {
                // Segment color based on energy and health
                const segmentAlpha = segment.energy * limb.health;
                this.ctx.globalAlpha = segmentAlpha;

                // Gradient from center to tip
                const gradient = this.ctx.createLinearGradient(
                    currentX, currentY, segment.x, segment.y
                );
                gradient.addColorStop(0, healthColor);
                gradient.addColorStop(1, healthColor + '40');

                this.ctx.strokeStyle = gradient;
                this.ctx.lineWidth = 4 - (index * 0.4);

                // Draw segment
                this.ctx.beginPath();
                this.ctx.moveTo(currentX, currentY);
                this.ctx.lineTo(segment.x, segment.y);
                this.ctx.stroke();

                // Draw segment joint
                this.ctx.fillStyle = healthColor;
                this.ctx.beginPath();
                this.ctx.arc(segment.x, segment.y, 3, 0, Math.PI * 2);
                this.ctx.fill();

                // Render trail
                this.renderSegmentTrail(segment, healthColor);

                currentX = segment.x;
                currentY = segment.y;
            });
        });

        this.ctx.globalAlpha = 1.0;
    }

    renderSegmentTrail(segment, color) {
        if (segment.trail.length < 2) return;

        this.ctx.strokeStyle = color + '30';
        this.ctx.lineWidth = 1;

        this.ctx.beginPath();
        this.ctx.moveTo(segment.trail[0].x, segment.trail[0].y);

        segment.trail.forEach((point, index) => {
            const alpha = index / segment.trail.length;
            this.ctx.globalAlpha = alpha * 0.3;
            this.ctx.lineTo(point.x, point.y);
        });

        this.ctx.stroke();
    }

    renderVortexCore() {
        const coreSize = this.config.centralRadius +
            this.vortexCore.pulsation * 10;

        // Core energy field
        const gradient = this.ctx.createRadialGradient(
            this.vortexCore.x, this.vortexCore.y, 0,
            this.vortexCore.x, this.vortexCore.y, coreSize
        );

        const coreColor = this.getCoreColor();
        gradient.addColorStop(0, coreColor);
        gradient.addColorStop(0.7, coreColor + '60');
        gradient.addColorStop(1, 'transparent');

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(this.vortexCore.x, this.vortexCore.y, coreSize, 0, Math.PI * 2);
        this.ctx.fill();

        // Core ring
        this.ctx.strokeStyle = coreColor;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.arc(this.vortexCore.x, this.vortexCore.y,
            this.config.centralRadius, 0, Math.PI * 2);
        this.ctx.stroke();

        // Core center
        this.ctx.fillStyle = coreColor;
        this.ctx.beginPath();
        this.ctx.arc(this.vortexCore.x, this.vortexCore.y, 6, 0, Math.PI * 2);
        this.ctx.fill();
    }

    renderParticles() {
        this.effects.particleSystem.forEach(particle => {
            this.ctx.globalAlpha = particle.opacity;
            this.ctx.fillStyle = particle.color;

            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
        });

        this.ctx.globalAlpha = 1.0;
    }

    renderMetrics() {
        // Render AI metrics as small indicators
        this.ctx.font = '10px monospace';
        this.ctx.fillStyle = '#c9f7db80';

        const metrics = [
            `Q:${(this.aiMetrics.codeQuality * 100).toFixed(0)}%`,
            `S:${(this.aiMetrics.userSatisfaction * 100).toFixed(0)}%`,
            `V:${(this.aiMetrics.developmentVelocity * 100).toFixed(0)}%`,
            `D:${(this.aiMetrics.technicalDebt * 100).toFixed(0)}%`
        ];

        metrics.forEach((metric, index) => {
            this.ctx.fillText(metric, 10, 20 + index * 15);
        });
    }

    // Helper methods
    getLimbHealth(limbIndex) {
        // Map AI metrics to limb health
        const metricMap = [
            this.aiMetrics.codeQuality,
            this.aiMetrics.userSatisfaction,
            this.aiMetrics.developmentVelocity,
            1 - this.aiMetrics.technicalDebt,
            1 - this.aiMetrics.painLevel,
            this.aiMetrics.burstActivity
        ];

        return metricMap[limbIndex % metricMap.length] || 0.5;
    }

    getLimbActivity(limbIndex) {
        // Different limbs respond to different aspects
        const activityMap = [
            this.aiMetrics.burstActivity,
            this.aiMetrics.painLevel,
            this.aiMetrics.developmentVelocity,
            this.aiMetrics.technicalDebt,
            1 - this.aiMetrics.codeQuality,
            this.aiMetrics.userSatisfaction
        ];

        return 0.3 + activityMap[limbIndex % activityMap.length] * 0.7;
    }

    getHealthColor(health) {
        if (health > 0.8) return '#00ff7f';
        if (health > 0.6) return '#28f49b';
        if (health > 0.4) return '#ffa502';
        if (health > 0.2) return '#ff6b42';
        return '#ff4757';
    }

    getCoreColor() {
        const avgHealth = this.limbs.reduce((sum, limb) => sum + limb.health, 0) / this.limbs.length;
        return this.getHealthColor(avgHealth);
    }

    spawnParticle() {
        // Spawn particle from limb tips
        const randomLimb = this.limbs[Math.floor(Math.random() * this.limbs.length)];
        const tipSegment = randomLimb.segments[randomLimb.segments.length - 1];

        this.effects.particleSystem.push({
            x: tipSegment.x,
            y: tipSegment.y,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            size: 2 + Math.random() * 3,
            color: this.getHealthColor(randomLimb.health),
            life: 1.0,
            opacity: 1.0
        });
    }

    injectEnergy(x, y, intensity) {
        // Manual energy injection for interaction
        const distance = Math.sqrt(
            (x - this.vortexCore.x) ** 2 +
            (y - this.vortexCore.y) ** 2
        );

        if (distance < this.config.centralRadius * 2) {
            this.vortexCore.resonance += intensity;
            this.vortexCore.resonance = Math.min(this.vortexCore.resonance, 2.0);

            // Spawn interaction particles
            for (let i = 0; i < 5; i++) {
                this.effects.particleSystem.push({
                    x: x + (Math.random() - 0.5) * 20,
                    y: y + (Math.random() - 0.5) * 20,
                    vx: (Math.random() - 0.5) * 4,
                    vy: (Math.random() - 0.5) * 4,
                    size: 3,
                    color: '#40e0d0',
                    life: 1.0,
                    opacity: 1.0
                });
            }

            console.log(`⚡ Energy injected at (${x.toFixed(0)}, ${y.toFixed(0)}) intensity: ${intensity}`);
        }

        // Decay resonance over time
        setTimeout(() => {
            this.vortexCore.resonance *= 0.9;
        }, 1000);
    }

    resizeCanvas() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;

        // Update core position
        this.vortexCore.x = this.canvas.width / 2;
        this.vortexCore.y = this.canvas.height / 2;
    }

    // Public API methods
    updateAIMetrics(newMetrics) {
        this.aiMetrics = { ...this.aiMetrics, ...newMetrics };
        console.log('🧠 VortexLab AI metrics updated:', this.aiMetrics);
    }

    setQuantumIntensity(intensity) {
        this.config.quantumIntensity = Math.max(0, Math.min(2, intensity));
        console.log(`🌀 Quantum intensity set to: ${this.config.quantumIntensity}`);
    }

    triggerBurst(intensity = 1.0) {
        this.vortexCore.resonance += intensity;
        this.aiMetrics.burstActivity = Math.min(1.0, this.aiMetrics.burstActivity + intensity * 0.5);

        // Spawn burst particles
        for (let i = 0; i < intensity * 10; i++) {
            const angle = Math.random() * Math.PI * 2;
            const distance = Math.random() * 50;

            this.effects.particleSystem.push({
                x: this.vortexCore.x + Math.cos(angle) * distance,
                y: this.vortexCore.y + Math.sin(angle) * distance,
                vx: Math.cos(angle) * 3,
                vy: Math.sin(angle) * 3,
                size: 4,
                color: '#ff6b9d',
                life: 1.0,
                opacity: 1.0
            });
        }

        console.log(`💥 Burst triggered with intensity: ${intensity}`);
    }

    getAnimationStats() {
        return {
            frame: this.animationFrame,
            isAnimating: this.isAnimating,
            limbCount: this.limbs.length,
            particleCount: this.effects.particleSystem.length,
            coreEnergy: this.vortexCore.energy.toFixed(3),
            resonance: this.vortexCore.resonance.toFixed(3)
        };
    }

    // Cleanup
    destroy() {
        this.stopAnimation();

        // Remove event listeners
        this.canvas?.removeEventListener('click', this.injectEnergy);
        window.removeEventListener('resize', this.resizeCanvas);

        // Clear all data
        this.limbs = [];
        this.effects = { particleSystem: [], trailSystem: [], quantumField: [], energyRings: [] };

        console.log('🗑️ VortexLab Animation Engine destroyed');
    }
}

// Integration utilities
window.VortexLabAnimationEngine = VortexLabAnimationEngine;
