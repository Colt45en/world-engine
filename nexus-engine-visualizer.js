/**
 * RNES 6.0 + Nexus Lab Unified Visualizer
 * Combines ResourceEngineCore, CodeOptimizerAI, and Audio Visualization
 */

import { ResourceEngineCore } from './ResourceEngineCore.js';
import { CodeOptimizerAI, ManagerAI } from './CodeOptimizerAI.js';

// ====================================
// Camera Class - 360° Orbit System
// ====================================
class Camera {
    constructor(target = { x: 500, y: 500, z: 0 }) {
        // Orbit parameters
        this.target = target;           // Look-at point
        this.distance = 800;            // Distance from target
        this.azimuth = 0;               // Horizontal angle (radians)
        this.elevation = Math.PI / 4;   // Vertical angle (radians, 45°)
        
        // Pan offset
        this.panOffset = { x: 0, y: 0 };
        
        // Computed position
        this.position = { x: 0, y: 0, z: 0 };
        this.updatePosition();
        
        // Limits
        this.minDistance = 100;
        this.maxDistance = 3000;
        this.minElevation = 0.1;        // Don't go below ground
        this.maxElevation = Math.PI / 2 - 0.1;  // Don't flip over
    }

    updatePosition() {
        // Convert spherical to cartesian
        const x = this.distance * Math.cos(this.elevation) * Math.sin(this.azimuth);
        const y = this.distance * Math.sin(this.elevation);
        const z = this.distance * Math.cos(this.elevation) * Math.cos(this.azimuth);
        
        this.position.x = this.target.x + x + this.panOffset.x;
        this.position.y = this.target.y + y + this.panOffset.y;
        this.position.z = z;
    }

    orbit(deltaAzimuth, deltaElevation) {
        this.azimuth += deltaAzimuth;
        this.elevation += deltaElevation;
        
        // Clamp elevation
        this.elevation = Math.max(this.minElevation, Math.min(this.maxElevation, this.elevation));
        
        this.updatePosition();
    }

    zoom(delta) {
        this.distance += delta;
        this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance));
        this.updatePosition();
    }

    pan(dx, dy) {
        // Pan perpendicular to view direction
        const panSpeed = this.distance * 0.001;
        this.panOffset.x += dx * panSpeed;
        this.panOffset.y -= dy * panSpeed;
        this.updatePosition();
    }

    reset() {
        this.target = { x: 500, y: 500, z: 0 };
        this.distance = 800;
        this.azimuth = 0;
        this.elevation = Math.PI / 4;
        this.panOffset = { x: 0, y: 0 };
        this.updatePosition();
    }

    update() {
        // Smooth any animated transitions here if needed
    }
}

// ====================================
// Audio Visualizer
// ====================================
class AudioVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.bufferLength = 0;
        this.active = false;
        this.mode = 'waveform'; // 'waveform', 'circle', 'particles'
    }

    async initialize() {
        try {
            // Try to get microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyser);
            
            this.bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(this.bufferLength);
            this.active = true;
            
            console.log('[AudioVisualizer] Initialized successfully');
        } catch (err) {
            console.warn('[AudioVisualizer] Could not initialize:', err.message);
            this.active = false;
        }
    }

    draw(camera) {
        if (!this.active || !this.analyser) return;
        
        this.analyser.getByteFrequencyData(this.dataArray);
        
        switch (this.mode) {
            case 'waveform':
                this.drawWaveform(camera);
                break;
            case 'circle':
                this.drawCircle(camera);
                break;
            case 'particles':
                this.drawParticles(camera);
                break;
        }
    }

    drawWaveform(camera) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerX = width / 2 - camera.position.x + 500;
        const centerY = height / 2 - camera.position.y + 500;
        
        this.ctx.strokeStyle = '#10e0e0';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        const sliceWidth = width / this.bufferLength;
        let x = 0;
        
        for (let i = 0; i < this.bufferLength; i++) {
            const v = this.dataArray[i] / 255.0;
            const y = centerY + (v * 100 - 50);
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        this.ctx.stroke();
    }

    drawCircle(camera) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = 150;
        
        for (let i = 0; i < this.bufferLength; i++) {
            const angle = (i / this.bufferLength) * Math.PI * 2;
            const amp = (this.dataArray[i] / 255) * 100;
            
            const x = centerX + Math.cos(angle) * (radius + amp);
            const y = centerY + Math.sin(angle) * (radius + amp);
            
            const hue = (i / this.bufferLength) * 360;
            this.ctx.fillStyle = `hsla(${hue}, 100%, 60%, 0.8)`;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    drawParticles(camera) {
        for (let i = 0; i < 30; i++) {
            const dataIndex = Math.floor((i / 30) * this.bufferLength);
            const amp = this.dataArray[dataIndex] / 255;
            
            const x = Math.random() * this.canvas.width;
            const y = (Math.random() * this.canvas.height) * amp;
            const size = amp * 8;
            
            const hue = (i / 30) * 360;
            this.ctx.fillStyle = `hsla(${hue}, 100%, 60%, 0.6)`;
            this.ctx.beginPath();
            this.ctx.arc(x, y, size, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    cycleMode() {
        const modes = ['waveform', 'circle', 'particles'];
        const currentIndex = modes.indexOf(this.mode);
        this.mode = modes[(currentIndex + 1) % modes.length];
        console.log(`[AudioVisualizer] Mode: ${this.mode}`);
    }
}

// ====================================
// Nexus AI Color Line Producer
// ====================================
class NexusColorAI {
    constructor() {
        this.colorPalettes = [];
        this.activeLines = [];
        this.shaderPresets = new Map();
        this.generation = 0;
        
        // Initialize with epic preset palettes
        this.initializePresets();
    }

    initializePresets() {
        // Cyberpunk Neon
        this.addPalette('cyberpunk', [
            { r: 16, g: 224, b: 224 },   // Cyan
            { r: 255, g: 16, b: 240 },   // Magenta
            { r: 255, g: 200, b: 0 },    // Gold
            { r: 0, g: 255, b: 100 }     // Green
        ]);

        // Cosmic Void
        this.addPalette('cosmic', [
            { r: 75, g: 0, b: 130 },     // Deep Purple
            { r: 255, g: 0, b: 255 },    // Magenta
            { r: 0, g: 191, b: 255 },    // Deep Sky Blue
            { r: 138, g: 43, b: 226 }    // Blue Violet
        ]);

        // Fire & Ice
        this.addPalette('fireice', [
            { r: 255, g: 69, b: 0 },     // Red Orange
            { r: 255, g: 140, b: 0 },    // Dark Orange
            { r: 0, g: 255, b: 255 },    // Cyan
            { r: 64, g: 224, b: 208 }    // Turquoise
        ]);

        // Electric Dream
        this.addPalette('electric', [
            { r: 0, g: 255, b: 255 },    // Cyan
            { r: 255, g: 255, b: 0 },    // Yellow
            { r: 255, g: 0, b: 255 },    // Magenta
            { r: 0, g: 255, b: 0 }       // Green
        ]);
    }

    addPalette(name, colors) {
        this.colorPalettes.push({ name, colors });
    }

    // Generate epic color line with AI-driven interpolation
    generateColorLine(pointCount = 100, paletteIndex = 0) {
        const palette = this.colorPalettes[paletteIndex % this.colorPalettes.length];
        const line = {
            id: `line_${this.generation++}`,
            palette: palette.name,
            points: [],
            shaderCode: ''
        };

        for (let i = 0; i < pointCount; i++) {
            const t = i / (pointCount - 1);
            const color = this.interpolateColor(palette.colors, t);
            line.points.push({
                position: t,
                color: color,
                rgba: `rgba(${color.r}, ${color.g}, ${color.b}, 1.0)`,
                hex: this.rgbToHex(color)
            });
        }

        // Generate shader code
        line.shaderCode = this.generateShaderCode(line);
        this.activeLines.push(line);
        
        return line;
    }

    // AI-driven smooth color interpolation (cubic bezier)
    interpolateColor(colors, t) {
        const segmentCount = colors.length - 1;
        const segment = Math.floor(t * segmentCount);
        const localT = (t * segmentCount) - segment;
        
        const c1 = colors[Math.min(segment, colors.length - 1)];
        const c2 = colors[Math.min(segment + 1, colors.length - 1)];
        
        // Smooth cubic easing
        const easedT = localT * localT * (3 - 2 * localT);
        
        return {
            r: Math.round(c1.r + (c2.r - c1.r) * easedT),
            g: Math.round(c1.g + (c2.g - c1.g) * easedT),
            b: Math.round(c1.b + (c2.b - c1.b) * easedT)
        };
    }

    // Generate GLSL shader code for the color line
    generateShaderCode(line) {
        let code = `// Nexus AI Generated Shader - ${line.palette}\n`;
        code += `// Generation ID: ${line.id}\n\n`;
        code += `vec3 getNexusColor(float t) {\n`;
        code += `    vec3 colors[${line.palette === 'cyberpunk' ? 4 : line.points.length}];\n`;
        
        // Sample key colors for shader
        const samples = 8;
        for (let i = 0; i < samples; i++) {
            const idx = Math.floor((i / samples) * line.points.length);
            const c = line.points[idx].color;
            code += `    colors[${i}] = vec3(${(c.r / 255).toFixed(3)}, ${(c.g / 255).toFixed(3)}, ${(c.b / 255).toFixed(3)});\n`;
        }
        
        code += `    int idx = int(t * ${samples - 1}.0);\n`;
        code += `    float localT = fract(t * ${samples - 1}.0);\n`;
        code += `    return mix(colors[idx], colors[idx + 1], localT);\n`;
        code += `}\n`;
        
        return code;
    }

    rgbToHex(color) {
        return '#' + [color.r, color.g, color.b]
            .map(x => x.toString(16).padStart(2, '0'))
            .join('');
    }

    // Generate gradient CSS string
    generateGradient(lineId) {
        const line = this.activeLines.find(l => l.id === lineId);
        if (!line) return '';
        
        const stops = line.points
            .filter((_, i) => i % 10 === 0) // Sample every 10th point
            .map(p => `${p.rgba} ${(p.position * 100).toFixed(1)}%`)
            .join(', ');
        
        return `linear-gradient(90deg, ${stops})`;
    }

    // Export all lines as JSON
    exportLines() {
        return JSON.stringify(this.activeLines, null, 2);
    }

    // Get random palette
    getRandomPalette() {
        return Math.floor(Math.random() * this.colorPalettes.length);
    }
}

// ====================================
// Circular Particle Accumulation System
// ====================================
class CircularParticleAccumulator {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.centerX = width / 2;
        this.centerY = height / 2;
        
        // Particle buffer (persists between frames)
        this.particles = [];
        this.maxParticles = 5000;
        
        // Spawn settings
        this.spawnRate = 20; // particles per frame
        this.time = 0;
        
        // Accumulation canvas (particles paint onto this)
        this.accumulationCanvas = document.createElement('canvas');
        this.accumulationCanvas.width = width;
        this.accumulationCanvas.height = height;
        this.accumulationCtx = this.accumulationCanvas.getContext('2d');
        
        // Start with transparent black
        this.accumulationCtx.fillStyle = 'rgba(0, 0, 0, 0)';
        this.accumulationCtx.fillRect(0, 0, width, height);
        
        this.enabled = true;
    }

    update(colorAI, timeScale = 1.0) {
        if (!this.enabled || colorAI.activeLines.length === 0) return;
        
        const dt = 0.016 * timeScale; // Apply time scaling
        this.time += dt;
        
        const latest = colorAI.activeLines[colorAI.activeLines.length - 1];
        
        // Spawn new particles in circular pattern
        for (let i = 0; i < this.spawnRate; i++) {
            if (this.particles.length >= this.maxParticles) {
                this.particles.shift(); // Remove oldest
            }
            
            // Circular emission pattern
            const angle = Math.random() * Math.PI * 2;
            const speed = 0.5 + Math.random() * 2;
            const lifetime = 2 + Math.random() * 3; // 2-5 seconds
            
            // Pick color from gradient
            const colorIndex = Math.floor(Math.random() * latest.points.length);
            const color = latest.points[colorIndex].color;
            
            // Spawn from center, move outward in spiral
            this.particles.push({
                x: this.centerX,
                y: this.centerY,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                angle: angle,
                angularVel: (Math.random() - 0.5) * 0.02, // Slight rotation
                color: color,
                alpha: 1.0,
                size: 2 + Math.random() * 4,
                life: lifetime,
                maxLife: lifetime,
                trail: []
            });
        }
        
        // Update existing particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            
            p.life -= dt;
            if (p.life <= 0) {
                this.particles.splice(i, 1);
                continue;
            }
            
            // Move in spiral
            p.angle += p.angularVel;
            
            // Add slight curve to motion
            const curve = 0.02;
            p.vx += Math.cos(p.angle + Math.PI / 2) * curve;
            p.vy += Math.sin(p.angle + Math.PI / 2) * curve;
            
            // Apply velocity
            p.x += p.vx;
            p.y += p.vy;
            
            // Fade out near end of life
            p.alpha = Math.min(1, p.life / 1);
            
            // Apply slight drag
            p.vx *= 0.99;
            p.vy *= 0.99;
            
            // Save trail position for accumulation
            if (p.trail.length > 10) p.trail.shift();
            p.trail.push({ x: p.x, y: p.y, alpha: p.alpha });
        }
    }

    draw(ctx) {
        if (!this.enabled) return;
        
        // First, paint new particles onto accumulation canvas (persistent)
        this.accumulationCtx.globalCompositeOperation = 'lighter'; // Additive blending
        
        for (const p of this.particles) {
            // Draw particle with glow
            const gradient = this.accumulationCtx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 2);
            gradient.addColorStop(0, `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${p.alpha * 0.3})`);
            gradient.addColorStop(0.5, `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${p.alpha * 0.15})`);
            gradient.addColorStop(1, `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, 0)`);
            
            this.accumulationCtx.fillStyle = gradient;
            this.accumulationCtx.beginPath();
            this.accumulationCtx.arc(p.x, p.y, p.size * 2, 0, Math.PI * 2);
            this.accumulationCtx.fill();
        }
        
        // Very slow fade (creates accumulation effect)
        this.accumulationCtx.globalCompositeOperation = 'destination-out';
        this.accumulationCtx.fillStyle = 'rgba(0, 0, 0, 0.005)'; // Fade 0.5% per frame
        this.accumulationCtx.fillRect(0, 0, this.width, this.height);
        this.accumulationCtx.globalCompositeOperation = 'source-over';
        
        // Draw accumulated buffer to main canvas
        ctx.save();
        ctx.globalAlpha = 1.0;
        ctx.globalCompositeOperation = 'lighter'; // Additive blend with scene
        ctx.drawImage(this.accumulationCanvas, 0, 0);
        ctx.restore();
    }

    clear() {
        this.particles = [];
        this.accumulationCtx.fillStyle = 'rgba(0, 0, 0, 1)';
        this.accumulationCtx.fillRect(0, 0, this.width, this.height);
    }

    toggle() {
        this.enabled = !this.enabled;
    }
}

// ====================================
// Renderer
// ====================================
class Renderer {
    constructor(canvas, engine, camera) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.engine = engine;
        this.camera = camera;
    }

    draw() {
        // Clear canvas
        this.ctx.fillStyle = '#0a0f17';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Center view on camera
        this.ctx.save();
        this.ctx.translate(
            -this.camera.position.x + this.canvas.width / 2,
            -this.camera.position.y + this.canvas.height / 2
        );

        // Draw Chunks
        for (const chunk of this.engine.chunks) {
            this.ctx.strokeStyle = chunk.loaded ? '#55aaff' : '#1a2a52';
            this.ctx.lineWidth = chunk.loaded ? 2 : 1;
            this.ctx.strokeRect(
                chunk.position.x,
                chunk.position.y,
                this.engine.chunkSize,
                this.engine.chunkSize
            );
            
            // Draw chunk ID if loaded
            if (chunk.loaded) {
                this.ctx.fillStyle = 'rgba(85, 170, 255, 0.3)';
                this.ctx.font = '10px monospace';
                this.ctx.fillText(
                    chunk.id,
                    chunk.position.x + 5,
                    chunk.position.y + 15
                );
            }
        }

        // Draw Resources
        for (const resource of this.engine.loadedResources.values()) {
            if (resource.visible) {
                switch (resource.currentLOD) {
                    case 'high':
                        this.ctx.fillStyle = '#00ff00';
                        this.ctx.beginPath();
                        this.ctx.arc(resource.position.x, resource.position.y, 5, 0, Math.PI * 2);
                        this.ctx.fill();
                        break;
                    case 'medium':
                        this.ctx.fillStyle = '#ffff00';
                        this.ctx.fillRect(resource.position.x - 3, resource.position.y - 3, 6, 6);
                        break;
                    case 'low':
                        this.ctx.fillStyle = '#ff8800';
                        this.ctx.fillRect(resource.position.x - 2, resource.position.y - 2, 4, 4);
                        break;
                }
            } else {
                this.ctx.fillStyle = '#333';
                this.ctx.fillRect(resource.position.x - 1, resource.position.y - 1, 2, 2);
            }
        }

        // Draw Camera and Load Radii
        const camPos = this.camera.position;
        
        // Unload distance (red)
        this.ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(camPos.x, camPos.y, this.engine.unloadDistance, 0, 2 * Math.PI);
        this.ctx.stroke();
        
        // Load distance (white)
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        this.ctx.beginPath();
        this.ctx.arc(camPos.x, camPos.y, this.engine.loadDistance, 0, 2 * Math.PI);
        this.ctx.stroke();
        
        // Camera itself
        this.ctx.fillStyle = '#00ccff';
        this.ctx.shadowColor = '#00ccff';
        this.ctx.shadowBlur = 20;
        this.ctx.beginPath();
        this.ctx.arc(camPos.x, camPos.y, 12, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
        
        // Camera direction indicator (show orbit azimuth)
        const dirX = Math.sin(this.camera.azimuth) * 40;
        const dirY = -Math.cos(this.camera.azimuth) * 40;
        this.ctx.strokeStyle = '#00ccff';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(camPos.x, camPos.y);
        this.ctx.lineTo(camPos.x + dirX, camPos.y + dirY);
        this.ctx.stroke();

        this.ctx.restore();
    }

    // Draw Nexus AI Color Lines with Particle Accumulation
    drawColorLines(colorAI, particleSystem) {
        if (colorAI.activeLines.length === 0) return;
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        
        // Draw accumulated particles from particle system
        if (particleSystem) {
            particleSystem.update(colorAI, timeScale); // Pass timeScale
            particleSystem.draw(this.ctx);
        }
        
        // Draw latest color line as epic gradient stream (lighter overlay)
        const latest = colorAI.activeLines[colorAI.activeLines.length - 1];
        const time = Date.now() * 0.001;
        
        // Animated spiral pattern (lighter, for guidance)
        this.ctx.save();
        this.ctx.globalAlpha = 0.2;
        
        const arms = 5;
        
        for (let arm = 0; arm < arms; arm++) {
            this.ctx.beginPath();
            const armAngle = (arm / arms) * Math.PI * 2 + time * 0.2 * timeScale; // Apply time scale
            
            for (let i = 0; i < latest.points.length; i++) {
                const t = i / (latest.points.length - 1);
                const angle = armAngle + t * Math.PI * 4;
                const radius = 50 + t * 250;
                
                const x = centerX + Math.cos(angle) * radius;
                const y = centerY + Math.sin(angle) * radius;
                
                const point = latest.points[i];
                this.ctx.strokeStyle = point.rgba;
                this.ctx.lineWidth = 2;
                
                if (i === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            this.ctx.stroke();
        }
        
        this.ctx.restore();
    }

    // Draw Mouse-Tracking Spotlight
    drawSpotlight(x, y) {
        this.ctx.save();
        
        // Create radial gradient spotlight
        const size = 300;
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, size);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
        gradient.addColorStop(0.3, 'rgba(255, 255, 255, 0.1)');
        gradient.addColorStop(0.7, 'rgba(255, 255, 255, 0)');
        
        this.ctx.globalCompositeOperation = 'lighter';
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x - size, y - size, size * 2, size * 2);
        
        // Glow ring
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.shadowColor = 'rgba(255, 255, 255, 0.8)';
        this.ctx.shadowBlur = 15;
        this.ctx.beginPath();
        this.ctx.arc(x, y, size * 0.7, 0, Math.PI * 2);
        this.ctx.stroke();
        
        this.ctx.restore();
    }
}

// ====================================
// Console Logger
// ====================================
class ConsoleLogger {
    constructor(consoleElement) {
        this.element = consoleElement;
        this.maxLines = 50;
    }

    log(message, type = 'info') {
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        this.element.appendChild(line);
        
        // Auto-scroll
        this.element.scrollTop = this.element.scrollHeight;
        
        // Limit lines
        while (this.element.children.length > this.maxLines) {
            this.element.removeChild(this.element.firstChild);
        }
    }
}

// ====================================
// Main Application
// ====================================
const canvas = document.getElementById('worldCanvas');
const vizCanvas = document.getElementById('visualizerCanvas');
const consoleElement = document.getElementById('console');

const camera = new Camera({ x: 500, y: 500, z: 0 });
const engine = new ResourceEngineCore();
const renderer = new Renderer(canvas, engine, camera);
const audioViz = new AudioVisualizer(vizCanvas);
const logger = new ConsoleLogger(consoleElement);

// Initialize Code Optimizer
const manager = new ManagerAI();
const optimizer = new CodeOptimizerAI(manager);
manager.add_optimizer(optimizer);

// Initialize Nexus Color AI
const colorAI = new NexusColorAI();
let currentPaletteIndex = 0;
logger.log('[NexusAI] Color line producer initialized with 4 epic palettes', 'success');

// Initialize Circular Particle Accumulator
const particleSystem = new CircularParticleAccumulator(canvas.width, canvas.height);
logger.log('[Particles] Circular accumulation system ready', 'success');

// Initialize Spotlight Effect
let spotlightEnabled = false;
let spotlightX = canvas.width / 2;
let spotlightY = canvas.height / 2;

// Initialize Time Control
let timeScale = 1.0; // 1.0 = normal speed

// Initialize Engine
logger.log('[RNES] Initializing Resource Engine Core...', 'info');
engine.initialize(camera);

// Register resource types
engine.registerResourceType('tree', {
    model: 'tree.obj',
    textures: [],
    lodModels: { high: 'tree_high.obj', medium: 'tree_med.obj', low: 'tree_low.obj' }
});
engine.registerResourceType('rock', {
    model: 'rock.obj',
    textures: [],
    lodModels: { high: 'rock_high.obj', medium: 'rock_med.obj', low: 'rock_low.obj' }
});
engine.registerResourceType('building', {
    model: 'building.obj',
    textures: [],
    lodModels: { high: 'building_high.obj', medium: 'building_med.obj', low: 'building_low.obj' }
});

logger.log('[RNES] Populating world with resources...', 'info');

// Populate world with random resources
const resourceCounts = { tree: 0, rock: 0, building: 0 };
for (let i = 0; i < 3000; i++) {
    const types = ['tree', 'rock', 'building'];
    const weights = [0.6, 0.3, 0.1]; // More trees, fewer buildings
    const rand = Math.random();
    let type = 'tree';
    
    if (rand < weights[2]) type = 'building';
    else if (rand < weights[1] + weights[2]) type = 'rock';
    
    const pos = {
        x: Math.random() * engine.worldBounds.x,
        y: Math.random() * engine.worldBounds.y,
        z: 0
    };
    
    engine.placeResource(type, pos);
    resourceCounts[type]++;
}

logger.log(`[RNES] World populated: ${resourceCounts.tree} trees, ${resourceCounts.rock} rocks, ${resourceCounts.building} buildings`, 'success');

// Add sample optimization tasks
manager.register_task('renderLoop', 'function render() { /* complex rendering */ }', 10);
manager.register_task('physics', 'function updatePhysics() { /* physics calculations */ }', 8);
manager.register_task('ai', 'function updateAI() { /* AI logic */ }', 5);

// ====================================
// Input Handling - 360° Orbit Controls
// ====================================
const keys = {};
let isDragging = false;
let isPanning = false;
let lastMouseX = 0;
let lastMouseY = 0;

window.addEventListener('keydown', (e) => {
    keys[e.key] = true;
    
    if (e.key === 'r' || e.key === 'R') {
        camera.reset();
        logger.log('[Camera] Reset to center', 'info');
    }
});
window.addEventListener('keyup', (e) => (keys[e.key] = false));

// Mouse orbit controls
canvas.addEventListener('mousedown', (e) => {
    if (e.shiftKey) {
        isPanning = true;
    } else {
        isDragging = true;
    }
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    canvas.style.cursor = e.shiftKey ? 'move' : 'grabbing';
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging && !isPanning) return;
    
    const deltaX = e.clientX - lastMouseX;
    const deltaY = e.clientY - lastMouseY;
    
    if (isDragging) {
        // Orbit camera
        camera.orbit(deltaX * 0.005, -deltaY * 0.005);
    } else if (isPanning) {
        // Pan camera
        camera.pan(deltaX, deltaY);
    }
    
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
});

window.addEventListener('mouseup', () => {
    isDragging = false;
    isPanning = false;
    canvas.style.cursor = 'default';
});

// Mouse wheel zoom
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    camera.zoom(e.deltaY * 0.5);
}, { passive: false });

// ====================================
// UI Controls
// ====================================
document.getElementById('btnResetCamera').addEventListener('click', () => {
    camera.reset();
    logger.log('[Camera] Reset to center', 'info');
});

document.getElementById('btnToggleParticles').addEventListener('click', () => {
    particleSystem.toggle();
    const status = particleSystem.enabled ? 'ON' : 'OFF';
    document.getElementById('btnToggleParticles').textContent = `✨ Particles ${status}`;
    logger.log(`[Particles] Circular accumulation ${status}`, 'info');
});

document.getElementById('btnClearParticles').addEventListener('click', () => {
    particleSystem.clear();
    logger.log('[Particles] Accumulation buffer cleared', 'info');
});

document.getElementById('btnToggleVisualizer').addEventListener('click', async () => {
    if (!audioViz.active) {
        await audioViz.initialize();
        logger.log('[AudioViz] Activating visualizer...', 'info');
    } else {
        audioViz.cycleMode();
        logger.log(`[AudioViz] Mode changed to: ${audioViz.mode}`, 'info');
    }
});

document.getElementById('btnOptimizeAll').addEventListener('click', () => {
    logger.log('[Optimizer] Starting optimization pass...', 'info');
    manager.optimize_all();
    updateOptimizerStats();
    logger.log('[Optimizer] Optimization complete!', 'success');
});

document.getElementById('btnQueueTask').addEventListener('click', () => {
    const taskId = `task_${Date.now()}`;
    const code = `function example_${Date.now()}() { /* placeholder */ }`;
    manager.register_task(taskId, code, Math.floor(Math.random() * 10));
    optimizer.queue_upgrade(taskId);
    updateTaskList();
    logger.log(`[Optimizer] Task ${taskId} queued`, 'info');
});

document.getElementById('btnProcessQueue').addEventListener('click', () => {
    logger.log('[Optimizer] Processing upgrade queue...', 'info');
    optimizer.process_upgrade_queue();
    updateOptimizerStats();
    logger.log('[Optimizer] Queue processed!', 'success');
});

// Nexus Color AI Controls
document.getElementById('btnGenerateColors').addEventListener('click', () => {
    const line = colorAI.generateColorLine(100, currentPaletteIndex);
    logger.log(`[NexusAI] Generated epic color line: ${line.palette} (${line.points.length} points)`, 'success');
    
    // Update preview
    const gradient = colorAI.generateGradient(line.id);
    document.getElementById('colorPreview').style.background = gradient;
    
    // Update stats
    document.getElementById('statColorLines').textContent = colorAI.activeLines.length;
    document.getElementById('statPalette').textContent = line.palette;
    document.getElementById('statParticles').textContent = particleSystem.particles.length;
});

document.getElementById('btnCyclePalette').addEventListener('click', () => {
    currentPaletteIndex = (currentPaletteIndex + 1) % colorAI.colorPalettes.length;
    const palette = colorAI.colorPalettes[currentPaletteIndex];
    document.getElementById('statPalette').textContent = palette.name;
    logger.log(`[NexusAI] Switched to ${palette.name} palette`, 'info');
});

document.getElementById('btnExportShader').addEventListener('click', () => {
    if (colorAI.activeLines.length === 0) {
        logger.log('[NexusAI] No color lines to export! Generate one first.', 'warning');
        return;
    }
    
    const latest = colorAI.activeLines[colorAI.activeLines.length - 1];
    
    // Copy shader code to clipboard
    navigator.clipboard.writeText(latest.shaderCode).then(() => {
        logger.log(`[NexusAI] Shader code copied to clipboard! (${latest.id})`, 'success');
    }).catch(() => {
        // Fallback: show in console
        console.log(latest.shaderCode);
        logger.log('[NexusAI] Shader code logged to browser console', 'info');
    });
    
    // Also export full JSON
    const json = colorAI.exportLines();
    console.log('Nexus Color Lines Export:', json);
});

// ====================================
// Timeline Toolbar Controls
// ====================================

// Spotlight Effect
document.getElementById('btnSpotlight').addEventListener('click', () => {
    spotlightEnabled = !spotlightEnabled;
    const btn = document.getElementById('btnSpotlight');
    btn.classList.toggle('active');
    btn.textContent = spotlightEnabled ? '💡 Spotlight ON' : '💡 Spotlight';
    logger.log(`[Spotlight] ${spotlightEnabled ? 'Enabled' : 'Disabled'}`, 'info');
});

// Track mouse for spotlight
canvas.addEventListener('mousemove', (e) => {
    if (spotlightEnabled) {
        const rect = canvas.getBoundingClientRect();
        spotlightX = e.clientX - rect.left;
        spotlightY = e.clientY - rect.top;
    }
});

// Time Scale Controls
document.getElementById('btnSlowMo').addEventListener('click', () => {
    timeScale = 0.5;
    updateTimeButtons();
    logger.log('[Time] Slow motion (0.5x)', 'info');
});

document.getElementById('btnNormalTime').addEventListener('click', () => {
    timeScale = 1.0;
    updateTimeButtons();
    logger.log('[Time] Normal speed (1x)', 'info');
});

document.getElementById('btnFastFwd').addEventListener('click', () => {
    timeScale = 2.0;
    updateTimeButtons();
    logger.log('[Time] Fast forward (2x)', 'info');
});

function updateTimeButtons() {
    ['btnSlowMo', 'btnNormalTime', 'btnFastFwd'].forEach(id => {
        document.getElementById(id).classList.remove('active');
    });
    if (timeScale === 0.5) document.getElementById('btnSlowMo').classList.add('active');
    else if (timeScale === 1.0) document.getElementById('btnNormalTime').classList.add('active');
    else if (timeScale === 2.0) document.getElementById('btnFastFwd').classList.add('active');
}
updateTimeButtons(); // Set initial state

// Geometry Graph Generator
document.getElementById('btnGeometry').addEventListener('click', () => {
    // TODO: Create geometric-graph-tool.html
    logger.log('[Geometry] Graph generator tool - dashboard integration coming soon!', 'warning');
    // openDashboard('Geometric Graph Generator', './geometric-graph-tool.html');
});

// Line Drawing Mode
document.getElementById('btnLineMode').addEventListener('click', () => {
    // TODO: Create line-drawing-tool.html
    logger.log('[Lines] Line drawing tool - dashboard integration coming soon!', 'warning');
    // openDashboard('Line Drawing Tool', './line-drawing-tool.html');
});

// ====================================
// Dashboard Panel System
// ====================================
const dashboardPanel = document.getElementById('dashboardPanel');
const dashboardIframe = document.getElementById('dashboardIframe');
const dashboardTitle = document.getElementById('dashboardTitle');

function openDashboard(title, url) {
    dashboardTitle.textContent = title;
    dashboardIframe.src = url;
    dashboardPanel.classList.add('open');
    logger.log(`[Dashboard] Opening: ${title} from ${url}`, 'info');
    console.log('Dashboard opening:', title, url);
    
    // Debug: Check if iframe loads
    dashboardIframe.addEventListener('load', function() {
        console.log('Iframe loaded successfully:', url);
    }, { once: true });
    
    dashboardIframe.addEventListener('error', function(e) {
        console.error('Iframe load error:', e);
        logger.log(`[Dashboard] Failed to load: ${url}`, 'error');
    }, { once: true });
}

function closeDashboard() {
    dashboardPanel.classList.remove('open');
    setTimeout(() => {
        dashboardIframe.src = 'about:blank';
    }, 300); // Wait for animation
    logger.log('[Dashboard] Closed', 'info');
}

document.getElementById('btnCloseDashboard').addEventListener('click', closeDashboard);

// Close dashboard with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && dashboardPanel.classList.contains('open')) {
        closeDashboard();
    }
});

// Math Labs
const labConfigs = [
    { title: 'Lab 1: Node/Electron Cubes', file: './math-lab-1.html' },
    { title: 'Lab 2: Spotlight Hue/Thermal', file: './math-lab-2.html' },
    { title: 'Lab 3 & 4: Color Fill System', file: './math-lab-3.html' },
    { title: 'Lab 3 & 4: Color Fill System', file: './math-lab-3.html' },
    { title: 'Lab 5: Time Flow Control', file: './math-lab-5.html' },
    { title: 'Lab 6: Orb Graph System', file: './math-lab-6.html' },
    { title: 'Lab 7: Grid Wire System', file: './math-lab-7.html' }
];

for (let i = 1; i <= 7; i++) {
    const btnId = `btnLab${i}`;
    const btn = document.getElementById(btnId);
    
    if (!btn) {
        console.error(`Button not found: ${btnId}`);
        continue;
    }
    
    btn.addEventListener('click', () => {
        console.log(`Lab ${i} button clicked!`);
        currentLabIndex = i - 1; // Set current lab index
        const config = labConfigs[i - 1];
        console.log('Config:', config);
        openDashboard(config.title, config.file);
    });
    
    console.log(`Lab ${i} button registered:`, btn);
}

// Lens Compositor button
const btnCompositor = document.getElementById('btnCompositor');
if (btnCompositor) {
    btnCompositor.addEventListener('click', () => {
        console.log('Opening Lens Compositor...');
        openDashboard('🔬 Lens Tunnel Compositor', './lens-compositor.html');
    });
    console.log('Compositor button registered');
}

let isRecording = false;
let mediaRecorder = null;

document.getElementById('btnRecord').addEventListener('click', async () => {
    if (!isRecording) {
        // Start recording
        try {
            const stream = canvas.captureStream(30);
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
            const chunks = [];
            
            mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `rnes-capture-${Date.now()}.webm`;
                a.click();
                logger.log('[Recorder] Video saved!', 'success');
            };
            
            mediaRecorder.start();
            isRecording = true;
            document.getElementById('btnRecord').textContent = '■ Stop';
            document.getElementById('btnRecord').classList.add('active');
            logger.log('[Recorder] Recording started', 'info');
        } catch (err) {
            logger.log(`[Recorder] Error: ${err.message}`, 'error');
        }
    } else {
        // Stop recording
        mediaRecorder.stop();
        isRecording = false;
        document.getElementById('btnRecord').textContent = '● Record';
        document.getElementById('btnRecord').classList.remove('active');
        logger.log('[Recorder] Recording stopped', 'info');
    }
});

// ====================================
// UI Update Functions
// ====================================
function updateStats() {
    const stats = engine.getStats();
    
    // HUD
    document.getElementById('hudCamX').textContent = Math.floor(camera.position.x);
    document.getElementById('hudCamY').textContent = Math.floor(camera.position.y);
    document.getElementById('hudChunks').textContent = engine.chunks.filter(c => c.loaded).length;
    document.getElementById('hudResources').textContent = stats.resourcesVisible;
    document.getElementById('hudTotal').textContent = engine.resources.length;
    
    // Stats Panel
    document.getElementById('statChunksLoaded').textContent = engine.chunks.filter(c => c.loaded).length;
    document.getElementById('statResourcesLoaded').textContent = stats.resourcesLoaded;
    document.getElementById('statResourcesVisible').textContent = stats.resourcesVisible;
    document.getElementById('statLODHigh').textContent = stats.lodHigh;
    document.getElementById('statLODMedium').textContent = stats.lodMedium;
    document.getElementById('statLODLow').textContent = stats.lodLow;
    
    // Particle stats
    if (particleSystem) {
        document.getElementById('statParticles').textContent = particleSystem.particles.length;
    }
}

function updateOptimizerStats() {
    const report = optimizer.report_optimizations();
    document.getElementById('statOptimizations').textContent = report.stats.optimizations;
    document.getElementById('statUpgrades').textContent = report.stats.upgrades;
    
    // Calculate average improvement
    let totalImprovement = 0;
    for (const entry of report.history) {
        totalImprovement += entry.improvement;
    }
    const avgImprovement = report.history.length > 0 ? totalImprovement / report.history.length : 0;
    document.getElementById('statImprovement').textContent = avgImprovement.toFixed(1) + '%';
}

function updateTaskList() {
    const taskList = document.getElementById('taskList');
    taskList.innerHTML = '';
    
    for (const [taskId, task] of manager.tasks.entries()) {
        const item = document.createElement('div');
        item.className = 'task-item';
        item.innerHTML = `
            <div class="task-name">${taskId}</div>
            <div class="task-status">Priority: ${task.priority} | Status: ${task.status}</div>
        `;
        taskList.appendChild(item);
    }
}

// ====================================
// Main Game Loop
// ====================================
let lastTime = performance.now();
let frameCount = 0;
let fpsTime = 0;

function gameLoop(currentTime) {
    const deltaTime = currentTime - lastTime;
    lastTime = currentTime;
    
    // FPS counter
    frameCount++;
    fpsTime += deltaTime;
    if (fpsTime >= 1000) {
        document.getElementById('hudFPS').textContent = frameCount;
        frameCount = 0;
        fpsTime = 0;
    }
    
    // Camera now controlled by mouse orbit (no keyboard movement)
    if (keys['ArrowRight'] || keys['d']) camera.accelerate(moveSpeed, 0);
    
    camera.update();
    
    // Update and Render
    engine.update();
    renderer.draw();
    
    // Draw Nexus AI Color Lines with Particle Accumulation (epic!)
    renderer.drawColorLines(colorAI, particleSystem);
    
    // Draw spotlight effect
    if (spotlightEnabled) {
        renderer.drawSpotlight(spotlightX, spotlightY);
    }
    
    // Draw audio visualizer
    if (audioViz.active) {
        vizCanvas.getContext('2d').clearRect(0, 0, vizCanvas.width, vizCanvas.height);
        audioViz.draw(camera);
    }
    
    updateStats();
    
    requestAnimationFrame(gameLoop);
}

// ====================================
// Initialize and Start
// ====================================
logger.log('[System] All systems initialized', 'success');
logger.log('[Controls] Drag: Orbit camera 360°', 'info');
logger.log('[Controls] Shift+Drag: Pan camera', 'info');
logger.log('[Controls] Wheel: Zoom in/out', 'info');
logger.log('[Controls] R: Reset camera', 'info');
logger.log('[NexusAI] Click "Generate Epic Colors" to create shader-ready color lines!', 'info');
logger.log('[System] Ready!', 'success');

updateTaskList();
requestAnimationFrame(gameLoop);
