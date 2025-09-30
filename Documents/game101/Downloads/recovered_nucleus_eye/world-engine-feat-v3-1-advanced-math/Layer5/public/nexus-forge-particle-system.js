/**
 * NEXUS Forge Particle System
 * • Advanced particle effects for environmental storytelling
 * • Audio-reactive particle behaviors
 * • Heart-synchronized particle pulses
 * • Memory echo visualizations
 * • Storm and weather particle effects
 */

class NexusForgeParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.emitters = new Map();
        this.maxParticles = 1000;
        this.globalForces = {
            gravity: { x: 0, y: 0.1 },
            wind: { x: 0, y: 0 },
            turbulence: 0.1
        };

        this.audioReactivity = {
            bassResponse: 0,
            midResponse: 0,
            trebleResponse: 0,
            beatPulse: 0
        };

        this.heartSync = {
            resonance: 0.5,
            pulsePhase: 0,
            emotionalColor: [84, 240, 184] // Default to accent color
        };

        this.presets = this.createParticlePresets();

        console.log('✨ Particle System initialized');
    }

    createParticlePresets() {
        return {
            memory_echo: {
                count: 15,
                life: 3000,
                speed: 0.5,
                size: 2,
                color: [157, 78, 221, 0.8], // Temporal purple
                behavior: 'drift',
                trail: true,
                glow: true
            },
            heart_pulse: {
                count: 30,
                life: 2000,
                speed: 1.0,
                size: 3,
                color: [255, 105, 180, 0.9], // Heart pink
                behavior: 'radial',
                trail: false,
                glow: true
            },
            storm_chaos: {
                count: 50,
                life: 5000,
                speed: 2.0,
                size: 1.5,
                color: [76, 201, 240, 0.7], // Mechanical blue
                behavior: 'chaotic',
                trail: true,
                glow: false
            },
            terrain_genesis: {
                count: 25,
                life: 4000,
                speed: 0.3,
                size: 2.5,
                color: [247, 127, 0, 0.8], // Worldshift orange
                behavior: 'emergence',
                trail: true,
                glow: true
            },
            quantum_flux: {
                count: 40,
                life: 1500,
                speed: 3.0,
                size: 1,
                color: [6, 255, 165, 0.9], // Emotional green
                behavior: 'quantum',
                trail: false,
                glow: true
            }
        };
    }

    createParticle(x, y, preset = 'memory_echo', customOptions = {}) {
        const config = { ...this.presets[preset], ...customOptions };

        const particle = {
            id: Date.now() + Math.random(),
            x: x,
            y: y,
            vx: (Math.random() - 0.5) * config.speed,
            vy: (Math.random() - 0.5) * config.speed,
            life: config.life,
            maxLife: config.life,
            size: config.size,
            color: [...config.color],
            originalColor: [...config.color],
            behavior: config.behavior,
            trail: config.trail ? [] : null,
            glow: config.glow,
            age: 0,
            rotation: Math.random() * Math.PI * 2,
            rotationSpeed: (Math.random() - 0.5) * 0.1,
            pulsePhase: Math.random() * Math.PI * 2,
            created: Date.now()
        };

        return particle;
    }

    createEmitter(name, x, y, preset, options = {}) {
        const emitter = {
            x: x,
            y: y,
            preset: preset,
            rate: options.rate || 10, // particles per second
            duration: options.duration || -1, // -1 = infinite
            lastEmit: 0,
            active: true,
            created: Date.now(),
            options: options
        };

        this.emitters.set(name, emitter);
        console.log(`✨ Emitter "${name}" created at [${x}, ${y}]`);
        return emitter;
    }

    updateParticles(deltaTime) {
        // Update existing particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];

            this.updateParticle(particle, deltaTime);

            // Remove dead particles
            if (particle.life <= 0) {
                this.particles.splice(i, 1);
            }
        }

        // Update emitters
        this.updateEmitters(deltaTime);

        // Apply global forces
        this.applyGlobalForces();

        // Apply audio reactivity
        this.applyAudioEffects();

        // Apply heart synchronization
        this.applyHeartEffects();
    }

    updateParticle(particle, deltaTime) {
        particle.age += deltaTime;
        particle.life -= deltaTime;

        // Update position
        particle.x += particle.vx * deltaTime * 0.1;
        particle.y += particle.vy * deltaTime * 0.1;

        // Update rotation
        particle.rotation += particle.rotationSpeed * deltaTime * 0.01;

        // Apply behavior-specific updates
        this.applyBehavior(particle, deltaTime);

        // Update trail
        if (particle.trail) {
            particle.trail.push({ x: particle.x, y: particle.y, time: Date.now() });
            // Keep trail length reasonable
            if (particle.trail.length > 20) {
                particle.trail.shift();
            }
        }

        // Update alpha based on life
        const lifeRatio = particle.life / particle.maxLife;
        if (particle.color.length > 3) {
            particle.color[3] = particle.originalColor[3] * lifeRatio;
        }
    }

    applyBehavior(particle, deltaTime) {
        switch (particle.behavior) {
            case 'drift':
                // Gentle drifting motion
                particle.vx += (Math.random() - 0.5) * 0.01;
                particle.vy += (Math.random() - 0.5) * 0.01;
                particle.vx *= 0.99; // Slight damping
                particle.vy *= 0.99;
                break;

            case 'radial': {
                // Expand outward from origin
                const angle = Math.atan2(particle.vy, particle.vx);
                particle.vx = Math.cos(angle) * 2;
                particle.vy = Math.sin(angle) * 2;
                break;
            }

            case 'chaotic':
                // Chaotic storm-like movement
                particle.vx += (Math.random() - 0.5) * 0.1;
                particle.vy += (Math.random() - 0.5) * 0.1;
                particle.vx = Math.max(-3, Math.min(3, particle.vx));
                particle.vy = Math.max(-3, Math.min(3, particle.vy));
                break;

            case 'emergence':
                // Rising from the ground
                particle.vy -= 0.02;
                particle.vx *= 0.98;
                break;

            case 'quantum':
                // Quantum tunneling effect
                if (Math.random() < 0.01) {
                    particle.x += (Math.random() - 0.5) * 50;
                    particle.y += (Math.random() - 0.5) * 50;
                }
                particle.vx += Math.sin(particle.age * 0.01) * 0.05;
                particle.vy += Math.cos(particle.age * 0.01) * 0.05;
                break;
        }
    }

    updateEmitters(deltaTime) {
        const now = Date.now();

        for (const [, emitter] of this.emitters) {
            if (!emitter.active) continue;

            // Check if emitter should stop
            if (emitter.duration > 0 && (now - emitter.created) > emitter.duration) {
                emitter.active = false;
                continue;
            }

            // Emit particles based on rate
            const timeSinceLastEmit = now - emitter.lastEmit;
            const emitInterval = 1000 / emitter.rate;

            if (timeSinceLastEmit >= emitInterval && this.particles.length < this.maxParticles) {
                const particle = this.createParticle(emitter.x, emitter.y, emitter.preset, emitter.options);
                this.particles.push(particle);
                emitter.lastEmit = now;
            }
        }
    }

    applyGlobalForces() {
        this.particles.forEach(particle => {
            particle.vx += this.globalForces.gravity.x;
            particle.vy += this.globalForces.gravity.y;
            particle.vx += this.globalForces.wind.x;
            particle.vy += this.globalForces.wind.y;

            // Add turbulence
            if (this.globalForces.turbulence > 0) {
                particle.vx += (Math.random() - 0.5) * this.globalForces.turbulence;
                particle.vy += (Math.random() - 0.5) * this.globalForces.turbulence;
            }
        });
    }

    applyAudioEffects() {
        if (this.audioReactivity.beatPulse > 0.8) {
            // Create beat pulse particles
            this.createBeatPulse();
        }

        // Modify particle colors based on audio
        this.particles.forEach(particle => {
            if (particle.behavior === 'quantum') {
                const bassInfluence = this.audioReactivity.bassResponse * 50;
                particle.color[0] = Math.min(255, particle.originalColor[0] + bassInfluence);
                particle.size = particle.originalColor[0] + this.audioReactivity.midResponse * 2;
            }
        });
    }

    applyHeartEffects() {
        const heartPhase = Math.sin(this.heartSync.pulsePhase);

        // Apply heart synchronization to particles
        this.particles.forEach(particle => {
            if (particle.behavior === 'radial') {
                // Heart pulse affects radial particles
                const heartInfluence = heartPhase * this.heartSync.resonance;
                particle.vx *= (1 + heartInfluence * 0.2);
                particle.vy *= (1 + heartInfluence * 0.2);

                // Adjust color based on emotional state
                const [r, g, b] = this.heartSync.emotionalColor;
                particle.color[0] = Math.min(255, particle.originalColor[0] + r * heartInfluence * 0.3);
                particle.color[1] = Math.min(255, particle.originalColor[1] + g * heartInfluence * 0.3);
                particle.color[2] = Math.min(255, particle.originalColor[2] + b * heartInfluence * 0.3);
            }
        });
    }

    createBeatPulse() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;

        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            const x = centerX + Math.cos(angle) * 50;
            const y = centerY + Math.sin(angle) * 50;

            const particle = this.createParticle(x, y, 'heart_pulse', {
                vx: Math.cos(angle) * 2,
                vy: Math.sin(angle) * 2
            });

            this.particles.push(particle);
        }
    }

    render() {
        // Clear with slight trail effect
        this.ctx.fillStyle = 'rgba(11, 14, 20, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Render particles
        this.particles.forEach(particle => {
            this.renderParticle(particle);
        });
    }

    renderParticle(particle) {
        const [r, g, b, a] = particle.color;

        // Render trail if exists
        if (particle.trail && particle.trail.length > 1) {
            this.renderTrail(particle);
        }

        // Apply glow effect
        if (particle.glow) {
            this.ctx.shadowColor = `rgba(${r}, ${g}, ${b}, ${a})`;
            this.ctx.shadowBlur = particle.size * 3;
        } else {
            this.ctx.shadowBlur = 0;
        }

        // Render particle
        this.ctx.save();
        this.ctx.translate(particle.x, particle.y);
        this.ctx.rotate(particle.rotation);

        this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a})`;
        this.ctx.beginPath();

        // Different shapes based on behavior
        switch (particle.behavior) {
            case 'memory_echo':
                this.renderMemoryParticle(particle);
                break;
            case 'quantum':
                this.renderQuantumParticle(particle);
                break;
            default:
                this.ctx.arc(0, 0, particle.size, 0, Math.PI * 2);
                break;
        }

        this.ctx.fill();
        this.ctx.restore();
    }

    renderTrail(particle) {
        if (particle.trail.length < 2) return;

        this.ctx.strokeStyle = `rgba(${particle.color[0]}, ${particle.color[1]}, ${particle.color[2]}, 0.3)`;
        this.ctx.lineWidth = particle.size * 0.5;
        this.ctx.beginPath();

        const trail = particle.trail;
        this.ctx.moveTo(trail[0].x, trail[0].y);

        for (let i = 1; i < trail.length; i++) {
            this.ctx.lineTo(trail[i].x, trail[i].y);
        }

        this.ctx.stroke();
    }

    renderMemoryParticle(particle) {
        // Render as a diamond shape for memory echoes
        const size = particle.size;
        this.ctx.beginPath();
        this.ctx.moveTo(0, -size);
        this.ctx.lineTo(size, 0);
        this.ctx.lineTo(0, size);
        this.ctx.lineTo(-size, 0);
        this.ctx.closePath();
    }

    renderQuantumParticle(particle) {
        // Render as a flickering star for quantum particles
        const size = particle.size;
        const flicker = Math.sin(particle.age * 0.1) * 0.3 + 0.7;

        this.ctx.globalAlpha *= flicker;

        // Draw star shape
        this.ctx.beginPath();
        for (let i = 0; i < 5; i++) {
            const angle = (i * 4 * Math.PI) / 5;
            const x = Math.cos(angle) * size;
            const y = Math.sin(angle) * size;
            if (i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        }
        this.ctx.closePath();
    }

    // Public API methods
    spawnMemoryEcho(x, y) {
        this.createEmitter(`memory_${Date.now()}`, x, y, 'memory_echo', {
            rate: 5,
            duration: 3000
        });
    }

    spawnHeartPulse(x, y, intensity = 1.0) {
        for (let i = 0; i < 12 * intensity; i++) {
            const angle = (i / (12 * intensity)) * Math.PI * 2;
            const particle = this.createParticle(
                x + Math.cos(angle) * 10,
                y + Math.sin(angle) * 10,
                'heart_pulse',
                {
                    vx: Math.cos(angle) * intensity,
                    vy: Math.sin(angle) * intensity
                }
            );
            this.particles.push(particle);
        }
    }

    spawnStorm(x, y, intensity = 1.0) {
        this.createEmitter(`storm_${Date.now()}`, x, y, 'storm_chaos', {
            rate: 20 * intensity,
            duration: 5000
        });

        // Add wind effects
        this.globalForces.wind.x = (Math.random() - 0.5) * intensity;
        this.globalForces.wind.y = (Math.random() - 0.5) * intensity * 0.5;
        this.globalForces.turbulence = 0.1 * intensity;

        // Reset wind after storm duration
        setTimeout(() => {
            this.globalForces.wind.x *= 0.5;
            this.globalForces.wind.y *= 0.5;
            this.globalForces.turbulence *= 0.5;
        }, 5000);
    }

    spawnTerrainGenesis(x, y) {
        this.createEmitter(`terrain_${Date.now()}`, x, y, 'terrain_genesis', {
            rate: 8,
            duration: 4000
        });
    }

    spawnQuantumFlux(x, y) {
        this.createEmitter(`quantum_${Date.now()}`, x, y, 'quantum_flux', {
            rate: 15,
            duration: 2000
        });
    }

    // Integration methods
    setAudioReactivity(bassResponse, midResponse, trebleResponse, beatPulse) {
        this.audioReactivity.bassResponse = bassResponse;
        this.audioReactivity.midResponse = midResponse;
        this.audioReactivity.trebleResponse = trebleResponse;
        this.audioReactivity.beatPulse = beatPulse;
    }

    setHeartSync(resonance, pulsePhase, emotionalColor) {
        this.heartSync.resonance = resonance;
        this.heartSync.pulsePhase = pulsePhase;
        if (emotionalColor) {
            this.heartSync.emotionalColor = emotionalColor;
        }
    }

    clearAllParticles() {
        this.particles = [];
        this.emitters.clear();
    }

    getParticleCount() {
        return {
            active: this.particles.length,
            emitters: this.emitters.size,
            max: this.maxParticles
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeParticleSystem;
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    window.NexusForgeParticleSystem = NexusForgeParticleSystem;
    console.log('✨ Particle System available globally');
}
