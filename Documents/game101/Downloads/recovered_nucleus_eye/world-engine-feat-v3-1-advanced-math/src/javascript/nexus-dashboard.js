/**
 * Nexus Dashboard Controller - Layer 4 Interface Management
 * Unified dashboard for all system layers with real-time monitoring
 * Based on upgrade specifications from Untitled-11.html
 */

class NexusDashboard {
    constructor() {
        // Core dashboard state
        this.isInitialized = false;
        this.activeTab = 'dashboard';
        this.audioEngine = null;
        this.canvasRenderer = null;

        // Layout management
        this.layout = {
            col1: '300px',
            col3: '380px',
            row1: '58vh',
            dockH: '44px'
        };

        this.layoutKey = 'nexus-dashboard-layout-v2';

        // System monitoring
        this.systemMetrics = {
            performance: 0,
            memoryUsage: 0,
            cpuUsage: 0,
            layerStatus: {
                layer0: 'active',
                layer1: 'active',
                layer2: 'active',
                layer3: 'warning',
                layer4: 'active'
            }
        };

        // Feature tracking
        this.audioFeatures = {
            loudness: 0,
            spectralCentroid: 0,
            spectralFlux: 0,
            pitch: 0
        };

        // Canvas and rendering
        this.canvas = null;
        this.ctx = null;
        this.animationId = null;

        // Event handlers map
        this.eventHandlers = new Map();

        // System log
        this.logBuffer = [];
        this.maxLogLines = 50;

        this.init();
    }

    /**
     * Initialize dashboard components
     */
    init() {
        this.log('Initializing Nexus Dashboard...');

        // Setup DOM elements
        this.setupCanvas();
        this.setupEventListeners();
        this.setupLayoutManager();
        this.setupAudioVisualEngine();

        // Restore saved layout
        this.restoreLayout();

        // Start monitoring loops
        this.startPerformanceMonitoring();
        this.startCanvasAnimation();

        this.isInitialized = true;
        this.log('Nexus Dashboard fully operational');
    }

    /**
     * Setup main canvas for visualization
     */
    setupCanvas() {
        this.canvas = document.getElementById('main-canvas');
        if (!this.canvas) {
            this.log('ERROR: Main canvas not found');
            return;
        }

        this.ctx = this.canvas.getContext('2d');

        // Handle canvas resize
        const resizeCanvas = () => {
            const container = this.canvas.parentElement;
            const rect = container.getBoundingClientRect();

            this.canvas.width = rect.width;
            this.canvas.height = rect.height;

            // Setup canvas properties
            this.ctx.lineCap = 'round';
            this.ctx.lineJoin = 'round';
        };

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        this.log('Canvas initialized');
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Top bar controls
        this.bindEvent('#menuBtn', 'click', this.toggleSystemMenu);
        this.bindEvent('#saveLayoutBtn', 'click', this.saveLayout);
        this.bindEvent('#killSwitchBtn', 'click', this.triggerKillSwitch);
        this.bindEvent('#commandInput', 'keydown', this.handleCommand);

        // Audio controls
        this.bindEvent('#startAudioBtn', 'click', this.startAudioCapture);
        this.bindEvent('#stopAudioBtn', 'click', this.stopAudioCapture);
        this.bindEvent('#exportDataBtn', 'click', this.exportSystemData);
        this.bindEvent('#resetSystemBtn', 'click', this.resetSystem);

        // Shape selector
        document.querySelectorAll('.shape-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Remove active from all
                document.querySelectorAll('.shape-btn').forEach(b => b.classList.remove('active'));
                // Add active to clicked
                e.target.classList.add('active');

                const shape = e.target.dataset.shape;
                this.setActiveShape(shape);
            });
        });

        // Tab switching
        document.querySelectorAll('.tab[data-tab]').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Dock controls
        this.bindEvent('#expandDock', 'click', this.expandDock);
        this.bindEvent('#minimizeDock', 'click', this.minimizeDock);

        // System action buttons
        document.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.handleSystemAction(e.target.dataset.action);
            });
        });

        this.log('Event listeners bound');
    }

    /**
     * Setup layout management with drag handles
     */
    setupLayoutManager() {
        this.bindVerticalGutter('gV1', 'left');
        this.bindVerticalGutter('gV2', 'right');
        this.bindHorizontalGutter('gH1');

        this.log('Layout manager initialized');
    }

    /**
     * Initialize audio-visual engine integration
     */
    async setupAudioVisualEngine() {
        try {
            // Import the audio-visual engine (assuming it's available)
            if (window.AudioVisualShapeEngine) {
                this.audioEngine = new window.AudioVisualShapeEngine({
                    sampleRate: 44100,
                    bufferSize: 1024,
                    bpm: 120
                });

                // Set up shape generation callback
                this.audioEngine.onShapeGenerated = (shape, features, time) => {
                    this.audioFeatures = features;
                    this.updateFeatureDisplay(features);
                    this.renderShape(shape, time);
                };

                this.log('Audio-Visual Engine integrated');
            } else {
                this.log('Audio-Visual Engine not available - running in demo mode');
                this.startDemoMode();
            }
        } catch (error) {
            this.log(`Audio engine setup failed: ${error.message}`);
            this.startDemoMode();
        }
    }

    /**
     * Start demo mode with simulated data
     */
    startDemoMode() {
        setInterval(() => {
            const demoFeatures = {
                loudness: Math.random() * 0.5,
                spectralCentroid: Math.random(),
                spectralFlux: Math.random() * 0.3,
                pitch: 200 + Math.random() * 400
            };

            this.audioFeatures = demoFeatures;
            this.updateFeatureDisplay(demoFeatures);

            // Generate demo shape
            this.renderDemoShape();

        }, 100);

        this.log('Demo mode active');
    }

    /**
     * Bind event with error handling
     */
    bindEvent(selector, event, handler) {
        const element = document.querySelector(selector);
        if (element) {
            const boundHandler = handler.bind(this);
            element.addEventListener(event, boundHandler);
            this.eventHandlers.set(`${selector}:${event}`, boundHandler);
        } else {
            this.log(`WARNING: Element ${selector} not found`);
        }
    }

    /**
     * Layout management - vertical gutters
     */
    bindVerticalGutter(handleId, side) {
        const handle = document.getElementById(handleId);
        if (!handle) return;

        let isDown = false;
        let startX = 0;
        let startValue = 0;

        handle.addEventListener('mousedown', (e) => {
            isDown = true;
            startX = e.clientX;
            startValue = this.getLayoutValue(side === 'left' ? 'col1' : 'col3');
            e.preventDefault();
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDown) return;

            const dx = e.clientX - startX;
            const newValue = side === 'left'
                ? this.clamp(startValue + dx, 160, 520)
                : this.clamp(startValue - dx, 240, 640);

            this.setLayoutValue(side === 'left' ? 'col1' : 'col3', `${newValue}px`);
        });

        window.addEventListener('mouseup', () => {
            if (isDown) {
                isDown = false;
                this.saveLayout();
            }
        });
    }

    /**
     * Layout management - horizontal gutter
     */
    bindHorizontalGutter(handleId) {
        const handle = document.getElementById(handleId);
        if (!handle) return;

        let isDown = false;
        let startY = 0;
        let startValue = 0;

        handle.addEventListener('mousedown', (e) => {
            isDown = true;
            startY = e.clientY;
            startValue = this.getLayoutValue('row1', true);
            e.preventDefault();
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDown) return;

            const dy = e.clientY - startY;
            const newValue = this.clamp(startValue + dy, 180, window.innerHeight - 220);
            this.setLayoutValue('row1', `${newValue}px`);
        });

        window.addEventListener('mouseup', () => {
            if (isDown) {
                isDown = false;
                this.saveLayout();
            }
        });
    }

    /**
     * Utility functions for layout management
     */
    getLayoutValue(property, isPixel = false) {
        const value = getComputedStyle(document.documentElement)
            .getPropertyValue(`--${property}`).trim();

        if (isPixel || value.includes('px')) {
            return parseFloat(value);
        }

        // Convert vh to pixels
        if (value.includes('vh')) {
            return (parseFloat(value) / 100) * window.innerHeight;
        }

        return parseFloat(value);
    }

    setLayoutValue(property, value) {
        document.documentElement.style.setProperty(`--${property}`, value);
        this.layout[property] = value;
    }

    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    /**
     * Save and restore layout
     */
    saveLayout() {
        try {
            localStorage.setItem(this.layoutKey, JSON.stringify(this.layout));
            this.log('Layout saved');
        } catch (error) {
            this.log(`Layout save failed: ${error.message}`);
        }
    }

    restoreLayout() {
        try {
            const saved = localStorage.getItem(this.layoutKey);
            if (saved) {
                const layout = JSON.parse(saved);
                Object.entries(layout).forEach(([key, value]) => {
                    this.setLayoutValue(key, value);
                });
                this.log('Layout restored');
            }
        } catch (error) {
            this.log(`Layout restore failed: ${error.message}`);
        }
    }

    /**
     * Feature display updates
     */
    updateFeatureDisplay(features) {
        const updates = [
            { id: 'loudnessValue', value: features.loudness.toFixed(2) },
            { id: 'centroidValue', value: features.spectralCentroid.toFixed(2) },
            { id: 'fluxValue', value: features.spectralFlux.toFixed(2) },
            { id: 'pitchValue', value: `${features.pitch.toFixed(1)} Hz` }
        ];

        updates.forEach(({ id, value }) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    /**
     * Shape rendering on canvas
     */
    renderShape(shapePoints, time) {
        if (!this.ctx || !shapePoints || shapePoints.length === 0) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Setup rendering context
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const scale = Math.min(this.canvas.width, this.canvas.height) / 4;

        // Begin path
        this.ctx.beginPath();
        this.ctx.strokeStyle = `hsl(${(time * 60) % 360}, 70%, 60%)`;
        this.ctx.lineWidth = 2;
        this.ctx.shadowColor = this.ctx.strokeStyle;
        this.ctx.shadowBlur = 10;

        // Draw shape
        shapePoints.forEach((point, index) => {
            const x = centerX + point.x * scale;
            const y = centerY + point.y * scale;

            if (index === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        });

        this.ctx.closePath();
        this.ctx.stroke();

        // Reset shadow
        this.ctx.shadowBlur = 0;
    }

    /**
     * Demo shape rendering
     */
    renderDemoShape() {
        if (!this.ctx) return;

        const time = Date.now() / 1000;
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const radius = 50 + this.audioFeatures.loudness * 100;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw animated circle
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        this.ctx.strokeStyle = `hsl(${(time * 60) % 360}, 70%, 60%)`;
        this.ctx.lineWidth = 3;
        this.ctx.shadowColor = this.ctx.strokeStyle;
        this.ctx.shadowBlur = 20;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
    }

    /**
     * Canvas animation loop
     */
    startCanvasAnimation() {
        const animate = () => {
            if (this.isInitialized) {
                // Update performance meter
                this.updatePerformanceMeter();
            }
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }

    /**
     * Performance monitoring
     */
    startPerformanceMonitoring() {
        setInterval(() => {
            // Simulate performance metrics
            this.systemMetrics.performance = 40 + Math.random() * 50;
            this.systemMetrics.memoryUsage = 30 + Math.random() * 40;
            this.systemMetrics.cpuUsage = 20 + Math.random() * 60;
        }, 2000);
    }

    updatePerformanceMeter() {
        const meter = document.getElementById('performanceMeter');
        if (meter) {
            const value = Math.min(100, this.systemMetrics.performance);
            meter.style.width = `${value}%`;
            document.documentElement.style.setProperty('--meter-value', `${value}%`);
        }
    }

    /**
     * Event handlers
     */
    toggleSystemMenu() {
        this.log('System menu toggled');
        // Implement system menu
    }

    triggerKillSwitch() {
        this.log('EMERGENCY: Kill switch activated');

        // Stop all audio processing
        if (this.audioEngine) {
            this.audioEngine.stop();
        }

        // Cancel animation
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        // Clear canvas
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }

        // Show emergency status
        document.body.style.filter = 'saturate(0.2)';
        this.log('System halted by kill switch');
    }

    async startAudioCapture() {
        try {
            if (this.audioEngine) {
                const success = await this.audioEngine.initAudio();
                if (success) {
                    this.audioEngine.start();
                    this.log('Audio capture started');
                } else {
                    this.log('Audio initialization failed');
                }
            } else {
                this.log('Audio engine not available');
            }
        } catch (error) {
            this.log(`Audio start failed: ${error.message}`);
        }
    }

    stopAudioCapture() {
        if (this.audioEngine) {
            this.audioEngine.stop();
            this.log('Audio capture stopped');
        }
    }

    setActiveShape(shape) {
        if (this.audioEngine) {
            this.audioEngine.setShape(shape);
            this.log(`Active shape changed to: ${shape}`);
        }
    }

    handleCommand(event) {
        if (event.key === 'Enter') {
            const command = event.target.value.trim();
            if (command) {
                this.executeCommand(command);
                event.target.value = '';
            }
        }
    }

    executeCommand(command) {
        this.log(`Command: ${command}`);

        // Basic command processing
        const cmd = command.toLowerCase();

        if (cmd.includes('start audio')) {
            this.startAudioCapture();
        } else if (cmd.includes('stop audio')) {
            this.stopAudioCapture();
        } else if (cmd.includes('save')) {
            this.saveLayout();
        } else if (cmd.includes('reset')) {
            this.resetSystem();
        } else if (cmd.includes('shape')) {
            const shapes = ['heart', 'rose', 'spiral', 'lissajous', 'circle', 'dragon'];
            const shape = shapes.find(s => cmd.includes(s));
            if (shape) {
                this.setActiveShape(shape);
            }
        } else {
            this.log(`Unknown command: ${command}`);
        }
    }

    switchTab(tab) {
        // Update active tab
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelector(`[data-tab="${tab}"]`).classList.add('active');

        this.activeTab = tab;
        this.log(`Switched to tab: ${tab}`);
    }

    expandDock() {
        const currentHeight = this.getLayoutValue('dockH');
        const newHeight = this.clamp(currentHeight + 12, 32, 120);
        this.setLayoutValue('dockH', `${newHeight}px`);
        this.saveLayout();
    }

    minimizeDock() {
        const currentHeight = this.getLayoutValue('dockH');
        const newHeight = this.clamp(currentHeight - 12, 32, 120);
        this.setLayoutValue('dockH', `${newHeight}px`);
        this.saveLayout();
    }

    handleSystemAction(action) {
        this.log(`System action: ${action}`);

        switch (action) {
            case 'layer0':
                this.log('Layer 0 Overseer status: SEALED AND ACTIVE');
                break;
            case 'bridge':
                this.log('Studio Bridge transport active');
                break;
            case 'metabase':
                this.log('Meta-Base Thought Engine processing...');
                break;
            case 'morphology':
                this.log('Morphology system analyzing...');
                break;
            case 'audiovisual':
                this.startAudioCapture();
                break;
            default:
                this.log(`Action ${action} not implemented`);
        }
    }

    exportSystemData() {
        const data = {
            layout: this.layout,
            systemMetrics: this.systemMetrics,
            audioFeatures: this.audioFeatures,
            logs: this.logBuffer,
            timestamp: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(data, null, 2)],
            { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `nexus-dashboard-export-${Date.now()}.json`;
        a.click();

        URL.revokeObjectURL(url);
        this.log('System data exported');
    }

    resetSystem() {
        if (confirm('Reset system to defaults? This will clear all settings.')) {
            // Reset layout
            this.layout = {
                col1: '300px',
                col3: '380px',
                row1: '58vh',
                dockH: '44px'
            };

            Object.entries(this.layout).forEach(([key, value]) => {
                this.setLayoutValue(key, value);
            });

            // Clear logs
            this.logBuffer = [];
            this.updateLogDisplay();

            // Remove saved data
            localStorage.removeItem(this.layoutKey);

            this.log('System reset to defaults');
        }
    }

    /**
     * Logging system
     */
    log(message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = `${timestamp} • ${message}`;

        this.logBuffer.push(logEntry);
        if (this.logBuffer.length > this.maxLogLines) {
            this.logBuffer.shift();
        }

        this.updateLogDisplay();
    }

    updateLogDisplay() {
        const logElement = document.getElementById('systemLog');
        if (logElement) {
            logElement.innerHTML = this.logBuffer.join('<br>');
            logElement.scrollTop = logElement.scrollHeight;
        }
    }

    /**
     * Cleanup on destroy
     */
    destroy() {
        // Cancel animation
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        // Stop audio engine
        if (this.audioEngine) {
            this.audioEngine.stop();
        }

        // Clear event handlers
        this.eventHandlers.clear();

        this.log('Nexus Dashboard destroyed');
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.nexusDashboard = new NexusDashboard();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.nexusDashboard) {
        window.nexusDashboard.destroy();
    }
});
