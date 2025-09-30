/**
 * NEXUS FORGE UNIFIED ENGINES
 * ============================
 *
 * Secondary engine systems that complement the main unified system:
 * • Rendering Engine (Canvas/WebGL graphics)
 * • Audio Engine (Real-time audio processing)
 * • Asset Management System
 * • Network Engine (Multiplayer/WebSocket integration)
 * • UI Engine (Interface management)
 * • Animation Engine (Smooth interpolation and transitions)
 * • Physics Engine (Basic collision and movement)
 * • Event System (Game event coordination)
 *
 * These engines work in harmony with the main NexusForgeUnified system.
 */

// =============================================================================
// SECTION A: RENDERING ENGINE
// =============================================================================

class UnifiedRenderEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.webglCtx = null;
        this.useWebGL = false;

        // Rendering state
        this.camera = {
            x: 0, y: 50, z: 0,
            pitch: -15, yaw: 0,
            fov: 75,
            near: 0.1,
            far: 1000
        };

        this.viewport = {
            width: canvas.width,
            height: canvas.height,
            scale: window.devicePixelRatio || 1
        };

        // Rendering options
        this.options = {
            showWireframe: false,
            enableLighting: true,
            enableShadows: false,
            enableFog: true,
            fogDistance: 200,
            ambientLight: 0.3,
            sunAngle: Math.PI / 4
        };

        // Performance tracking
        this.stats = {
            triangles: 0,
            drawCalls: 0,
            frameTime: 0
        };

        this.initializeWebGL();
        console.log('🎨 Unified Render Engine initialized');
    }

    initializeWebGL() {
        try {
            this.webglCtx = this.canvas.getContext('webgl2') || this.canvas.getContext('webgl');
            if (this.webglCtx) {
                this.useWebGL = true;
                this.setupWebGLShaders();
                console.log('✅ WebGL enabled for 3D rendering');
            }
        } catch (error) {
            console.warn('WebGL not available, using 2D canvas fallback');
        }
    }

    setupWebGLShaders() {
        // Vertex shader for terrain rendering
        const vertexShaderSource = `
            attribute vec3 position;
            attribute vec3 normal;
            attribute vec2 texCoord;

            uniform mat4 modelMatrix;
            uniform mat4 viewMatrix;
            uniform mat4 projMatrix;
            uniform float time;

            varying vec3 worldPosition;
            varying vec3 worldNormal;
            varying vec2 vTexCoord;
            varying float vTime;

            void main() {
                worldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                worldNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
                vTexCoord = texCoord;
                vTime = time;

                gl_Position = projMatrix * viewMatrix * vec4(worldPosition, 1.0);
            }
        `;

        // Fragment shader with beat-reactive effects
        const fragmentShaderSource = `
            precision mediump float;

            varying vec3 worldPosition;
            varying vec3 worldNormal;
            varying vec2 vTexCoord;
            varying float vTime;

            uniform vec3 cameraPosition;
            uniform vec3 sunDirection;
            uniform float beatPhase;
            uniform vec4 biomeColor;
            uniform float fogDistance;

            void main() {
                vec3 normal = normalize(worldNormal);
                vec3 viewDir = normalize(cameraPosition - worldPosition);

                // Basic lighting
                float NdotL = max(0.0, dot(normal, sunDirection));
                float lighting = 0.3 + 0.7 * NdotL;

                // Beat-reactive color modulation
                float beatMod = sin(beatPhase * 6.28318) * 0.2 + 0.8;
                vec3 color = biomeColor.rgb * lighting * beatMod;

                // Distance fog
                float distance = length(worldPosition - cameraPosition);
                float fogFactor = exp(-distance / fogDistance);
                fogFactor = clamp(fogFactor, 0.0, 1.0);

                vec3 fogColor = vec3(0.7, 0.8, 0.9);
                color = mix(fogColor, color, fogFactor);

                gl_FragColor = vec4(color, 1.0);
            }
        `;

        this.shaderProgram = this.createShaderProgram(vertexShaderSource, fragmentShaderSource);
    }

    createShaderProgram(vertexSource, fragmentSource) {
        const gl = this.webglCtx;

        const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentSource);

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Shader program failed to link:', gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    }

    compileShader(type, source) {
        const gl = this.webglCtx;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    render(worldData, beatState) {
        const renderStart = performance.now();

        if (this.useWebGL && this.shaderProgram) {
            this.renderWebGL(worldData, beatState);
        } else {
            this.render2D(worldData, beatState);
        }

        this.stats.frameTime = performance.now() - renderStart;
    }

    render2D(worldData, beatState) {
        const ctx = this.ctx;
        const canvas = this.canvas;

        // Clear canvas
        ctx.fillStyle = '#001122';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Render chunks as 2D heightmap
        if (worldData.chunks && worldData.chunks.size > 0) {
            this.render2DHeightmap(worldData.chunks, beatState);
        }

        // Render UI overlay
        this.renderUI(beatState);
    }

    render2DHeightmap(chunks, beatState) {
        const ctx = this.ctx;
        const scale = 4;
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;

        for (const [, chunk] of chunks) {
            const startX = centerX + (chunk.x * chunk.heightMap.length * scale) - (this.camera.x * scale);
            const startY = centerY + (chunk.z * chunk.heightMap[0].length * scale) - (this.camera.z * scale);

            for (let x = 0; x < chunk.heightMap.length; x++) {
                for (let z = 0; z < chunk.heightMap[0].length; z++) {
                    const height = chunk.heightMap[x][z];
                    const biome = chunk.biomeMap[x][z];

                    // Beat-reactive color modulation
                    const beatMod = Math.sin(beatState.beatPhase * Math.PI * 2) * 0.3 + 0.7;
                    const heightNorm = Math.max(0, Math.min(255, height * 2 + 128)) * beatMod;

                    // Biome colors
                    const biomeColors = [
                        `rgb(${heightNorm * 0.7}, ${heightNorm * 0.6}, ${heightNorm * 0.4})`, // Mountains
                        `rgb(${heightNorm * 0.8}, ${heightNorm * 0.7}, ${heightNorm * 0.3})`, // Desert
                        `rgb(${heightNorm * 0.3}, ${heightNorm * 0.8}, ${heightNorm * 0.3})`, // Forest
                        `rgb(${heightNorm * 0.5}, ${heightNorm * 0.9}, ${heightNorm * 0.4})`, // Plains
                        `rgb(${heightNorm * 0.2}, ${heightNorm * 0.4}, ${heightNorm * 0.8})`  // Ocean
                    ];

                    ctx.fillStyle = biomeColors[biome] || biomeColors[0];
                    ctx.fillRect(startX + x * scale, startY + z * scale, scale, scale);
                }
            }
        }
    }

    renderWebGL(worldData, beatState) {
        const gl = this.webglCtx;

        // Clear and setup
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.useProgram(this.shaderProgram);

        // Set uniforms
        const beatPhaseLocation = gl.getUniformLocation(this.shaderProgram, 'beatPhase');
        gl.uniform1f(beatPhaseLocation, beatState.beatPhase);

        // Render chunks as 3D meshes
        this.render3DChunks(worldData.chunks, beatState);
    }

    render3DChunks(chunks, beatState) {
        // Simplified 3D chunk rendering
        // In production, this would generate proper triangle meshes
        this.stats.triangles = 0;
        this.stats.drawCalls = 0;

        for (const [, chunk] of chunks) {
            this.stats.drawCalls++;
            // Generate and render mesh for chunk
            // This is a placeholder for actual 3D mesh generation
            this.stats.triangles += chunk.heightMap.length * chunk.heightMap[0].length * 2;
        }
    }

    renderUI(beatState) {
        const ctx = this.ctx;

        // Beat indicator
        ctx.fillStyle = `rgba(84, 240, 184, ${beatState.beatPhase})`;
        ctx.fillRect(10, 10, 20, 20);

        // Performance stats
        ctx.fillStyle = '#e6f0ff';
        ctx.font = '12px monospace';
        ctx.fillText(`Frame: ${this.stats.frameTime.toFixed(1)}ms`, 40, 25);

        if (this.useWebGL) {
            ctx.fillText(`Triangles: ${this.stats.triangles}`, 40, 40);
            ctx.fillText(`Draw Calls: ${this.stats.drawCalls}`, 40, 55);
        }
    }

    setCamera(x, y, z, pitch, yaw) {
        this.camera.x = x;
        this.camera.y = y;
        this.camera.z = z;
        this.camera.pitch = pitch;
        this.camera.yaw = yaw;
    }

    resize(width, height) {
        this.canvas.width = width * this.viewport.scale;
        this.canvas.height = height * this.viewport.scale;
        this.canvas.style.width = width + 'px';
        this.canvas.style.height = height + 'px';

        this.viewport.width = width;
        this.viewport.height = height;

        if (this.useWebGL) {
            this.webglCtx.viewport(0, 0, this.canvas.width, this.canvas.height);
        }
    }
}

// =============================================================================
// SECTION B: AUDIO ENGINE
// =============================================================================

class UnifiedAudioEngine {
    constructor() {
        this.audioContext = null;
        this.masterGain = null;
        this.analyzer = null;
        this.bufferSize = 2048;
        this.sampleRate = 44100;

        // Audio reactive data
        this.audioData = {
            frequencyData: null,
            timeData: null,
            volume: 0,
            bass: 0,
            mid: 0,
            treble: 0,
            beatDetected: false
        };

        // Beat detection
        this.beatDetection = {
            threshold: 0.3,
            minTimeBetweenBeats: 200,
            lastBeatTime: 0,
            energyHistory: [],
            historySize: 42
        };

        // Synthesis oscillators
        this.oscillators = new Map();

        this.initializeAudioContext();
        console.log('🔊 Unified Audio Engine initialized');
    }

    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Create master gain
            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = 0.3;
            this.masterGain.connect(this.audioContext.destination);

            // Create analyzer
            this.analyzer = this.audioContext.createAnalyser();
            this.analyzer.fftSize = this.bufferSize;
            this.analyzer.connect(this.masterGain);

            // Initialize audio data arrays
            this.audioData.frequencyData = new Uint8Array(this.analyzer.frequencyBinCount);
            this.audioData.timeData = new Uint8Array(this.analyzer.fftSize);

            console.log('✅ Audio context ready');
        } catch (error) {
            console.warn('Audio initialization failed:', error);
        }
    }

    analyzeAudio() {
        if (!this.analyzer) return;

        // Get frequency and time domain data
        this.analyzer.getByteFrequencyData(this.audioData.frequencyData);
        this.analyzer.getByteTimeDomainData(this.audioData.timeData);

        // Calculate volume and frequency bands
        this.calculateAudioFeatures();

        // Detect beats
        this.detectBeats();
    }

    calculateAudioFeatures() {
        const freqData = this.audioData.frequencyData;
        const binCount = freqData.length;

        // Overall volume (RMS)
        let sum = 0;
        for (let i = 0; i < binCount; i++) {
            sum += freqData[i] * freqData[i];
        }
        this.audioData.volume = Math.sqrt(sum / binCount) / 255;

        // Frequency bands
        const bassEnd = Math.floor(binCount * 0.1);
        const midEnd = Math.floor(binCount * 0.5);

        this.audioData.bass = this.getAverageFrequency(freqData, 0, bassEnd);
        this.audioData.mid = this.getAverageFrequency(freqData, bassEnd, midEnd);
        this.audioData.treble = this.getAverageFrequency(freqData, midEnd, binCount);
    }

    getAverageFrequency(freqData, startBin, endBin) {
        let sum = 0;
        const count = endBin - startBin;

        for (let i = startBin; i < endBin; i++) {
            sum += freqData[i];
        }

        return (sum / count) / 255;
    }

    detectBeats() {
        const currentEnergy = this.audioData.volume;
        const currentTime = performance.now();

        // Add to energy history
        this.beatDetection.energyHistory.push(currentEnergy);
        if (this.beatDetection.energyHistory.length > this.beatDetection.historySize) {
            this.beatDetection.energyHistory.shift();
        }

        // Calculate average energy
        const avgEnergy = this.beatDetection.energyHistory.reduce((sum, e) => sum + e, 0) / this.beatDetection.energyHistory.length;

        // Beat detection logic
        const timeSinceLastBeat = currentTime - this.beatDetection.lastBeatTime;
        const energyRatio = avgEnergy > 0 ? currentEnergy / avgEnergy : 0;

        this.audioData.beatDetected = energyRatio > (1 + this.beatDetection.threshold) &&
            timeSinceLastBeat > this.beatDetection.minTimeBetweenBeats;

        if (this.audioData.beatDetected) {
            this.beatDetection.lastBeatTime = currentTime;
        }
    }

    createOscillator(frequency, type = 'sine') {
        if (!this.audioContext) return null;

        const oscillator = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();

        oscillator.frequency.value = frequency;
        oscillator.type = type;
        gain.gain.value = 0.1;

        oscillator.connect(gain);
        gain.connect(this.analyzer);

        return { oscillator, gain };
    }

    playTone(frequency, duration, type = 'sine') {
        const { oscillator } = this.createOscillator(frequency, type);

        if (oscillator) {
            oscillator.start();
            oscillator.stop(this.audioContext.currentTime + duration);
        }
    }

    synthesizeHarmonics(baseFreq, harmonics, duration) {
        const oscillators = [];

        for (let i = 1; i <= harmonics; i++) {
            const { oscillator, gain } = this.createOscillator(baseFreq * i, 'sine');
            if (oscillator) {
                gain.gain.value = 0.1 / i; // 1/n amplitude for harmonics
                oscillator.start();
                oscillator.stop(this.audioContext.currentTime + duration);
                oscillators.push({ oscillator, gain });
            }
        }

        return oscillators;
    }

    connectMicrophone() {
        return navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                const microphone = this.audioContext.createMediaStreamSource(stream);
                microphone.connect(this.analyzer);
                console.log('🎤 Microphone connected for audio analysis');
                return microphone;
            })
            .catch(error => {
                console.warn('Microphone access denied:', error);
                return null;
            });
    }

    getAudioData() {
        return { ...this.audioData };
    }

    setMasterVolume(volume) {
        if (this.masterGain) {
            this.masterGain.gain.value = Math.max(0, Math.min(1, volume));
        }
    }
}

// =============================================================================
// SECTION C: ASSET MANAGEMENT SYSTEM
// =============================================================================

class UnifiedAssetManager {
    constructor() {
        this.assets = new Map();
        this.loadingQueue = new Set();
        this.cache = new Map();
        this.maxCacheSize = 100 * 1024 * 1024; // 100MB
        this.currentCacheSize = 0;

        // Asset types
        this.supportedTypes = {
            image: ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
            audio: ['.mp3', '.wav', '.ogg', '.m4a'],
            model: ['.obj', '.gltf', '.glb'],
            texture: ['.png', '.jpg', '.dds'],
            shader: ['.glsl', '.vert', '.frag'],
            json: ['.json'],
            text: ['.txt', '.md']
        };

        console.log('📦 Unified Asset Manager initialized');
    }

    async loadAsset(url, type = null) {
        // Check if already loaded
        if (this.assets.has(url)) {
            return this.assets.get(url);
        }

        // Check if currently loading
        if (this.loadingQueue.has(url)) {
            return new Promise((resolve) => {
                const checkLoaded = () => {
                    if (this.assets.has(url)) {
                        resolve(this.assets.get(url));
                    } else {
                        setTimeout(checkLoaded, 10);
                    }
                };
                checkLoaded();
            });
        }

        this.loadingQueue.add(url);

        try {
            const assetType = type || this.detectAssetType(url);
            let asset;

            switch (assetType) {
                case 'image':
                    asset = await this.loadImage(url);
                    break;
                case 'audio':
                    asset = await this.loadAudio(url);
                    break;
                case 'json':
                    asset = await this.loadJSON(url);
                    break;
                case 'text':
                    asset = await this.loadText(url);
                    break;
                default:
                    asset = await this.loadBinary(url);
            }

            asset.url = url;
            asset.type = assetType;
            asset.loadTime = Date.now();
            asset.size = this.calculateAssetSize(asset);

            this.assets.set(url, asset);
            this.updateCache(url, asset);

            return asset;
        } catch (error) {
            console.error(`Failed to load asset: ${url}`, error);
            return null;
        } finally {
            this.loadingQueue.delete(url);
        }
    }

    detectAssetType(url) {
        const extension = url.toLowerCase().substring(url.lastIndexOf('.'));

        for (const [type, extensions] of Object.entries(this.supportedTypes)) {
            if (extensions.includes(extension)) {
                return type;
            }
        }

        return 'binary';
    }

    loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve({ data: img, width: img.width, height: img.height });
            img.onerror = reject;
            img.src = url;
        });
    }

    loadAudio(url) {
        return fetch(url)
            .then(response => response.arrayBuffer())
            .then(buffer => ({ data: buffer, duration: 0 })); // Duration would be calculated after decoding
    }

    loadJSON(url) {
        return fetch(url)
            .then(response => response.json())
            .then(data => ({ data }));
    }

    loadText(url) {
        return fetch(url)
            .then(response => response.text())
            .then(data => ({ data }));
    }

    loadBinary(url) {
        return fetch(url)
            .then(response => response.arrayBuffer())
            .then(data => ({ data }));
    }

    calculateAssetSize(asset) {
        if (asset.data instanceof ArrayBuffer) {
            return asset.data.byteLength;
        } else if (asset.data instanceof HTMLImageElement) {
            return asset.width * asset.height * 4; // Approximate RGBA size
        } else if (typeof asset.data === 'string') {
            return asset.data.length * 2; // Approximate Unicode string size
        } else if (typeof asset.data === 'object') {
            return JSON.stringify(asset.data).length * 2;
        }
        return 0;
    }

    updateCache(url, asset) {
        this.cache.set(url, asset);
        this.currentCacheSize += asset.size;

        // Implement LRU cache eviction if over limit
        if (this.currentCacheSize > this.maxCacheSize) {
            this.evictLRU();
        }
    }

    evictLRU() {
        // Simple LRU eviction - remove oldest assets until under limit
        const sorted = Array.from(this.cache.entries())
            .sort((a, b) => a[1].loadTime - b[1].loadTime);

        while (this.currentCacheSize > this.maxCacheSize && sorted.length > 0) {
            const [url, asset] = sorted.shift();
            this.cache.delete(url);
            this.assets.delete(url);
            this.currentCacheSize -= asset.size;
        }
    }

    generateProceduralTexture(width, height, pattern = 'noise') {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            const x = (i / 4) % width;
            const y = Math.floor((i / 4) / width);

            let value;
            switch (pattern) {
                case 'noise':
                    value = Math.random() * 255;
                    break;
                case 'checkerboard':
                    value = ((Math.floor(x / 8) + Math.floor(y / 8)) % 2) * 255;
                    break;
                case 'gradient':
                    value = (x / width) * 255;
                    break;
                default:
                    value = 128;
            }

            data[i] = value;     // R
            data[i + 1] = value; // G
            data[i + 2] = value; // B
            data[i + 3] = 255;   // A
        }

        ctx.putImageData(imageData, 0, 0);

        const asset = {
            data: canvas,
            width,
            height,
            type: 'image',
            procedural: true,
            pattern,
            loadTime: Date.now(),
            size: width * height * 4
        };

        const assetKey = `procedural_${pattern}_${width}x${height}`;
        this.assets.set(assetKey, asset);

        return asset;
    }

    preloadAssets(urls) {
        const promises = urls.map(url => this.loadAsset(url));
        return Promise.all(promises);
    }

    getAsset(url) {
        return this.assets.get(url) || null;
    }

    unloadAsset(url) {
        const asset = this.assets.get(url);
        if (asset) {
            this.assets.delete(url);
            this.cache.delete(url);
            this.currentCacheSize -= asset.size;
            return true;
        }
        return false;
    }

    getLoadingProgress() {
        return {
            loading: this.loadingQueue.size,
            loaded: this.assets.size,
            cacheSize: this.currentCacheSize,
            maxCacheSize: this.maxCacheSize
        };
    }
}

// =============================================================================
// SECTION D: UI ENGINE
// =============================================================================

class UnifiedUIEngine {
    constructor(container) {
        this.container = container;
        this.elements = new Map();
        this.panels = new Map();
        this.activePanel = null;

        // UI state
        this.theme = {
            bg: '#0b0e14',
            panel: '#0f1523',
            text: '#e6f0ff',
            accent: '#54f0b8',
            secondary: '#9ab0d6',
            warning: '#ffc107',
            error: '#dc3545'
        };

        this.animations = new Map();

        this.initializeUI();
        console.log('🖥️ Unified UI Engine initialized');
    }

    initializeUI() {
        // Create main UI structure
        this.container.innerHTML = `
            <div id="nexus-ui-root" class="nexus-ui">
                <div id="nexus-toolbar" class="nexus-toolbar"></div>
                <div id="nexus-panels" class="nexus-panels"></div>
                <div id="nexus-status" class="nexus-status"></div>
                <div id="nexus-notifications" class="nexus-notifications"></div>
            </div>
        `;

        this.applyStyles();
        this.createDefaultPanels();
    }

    applyStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .nexus-ui {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                font-family: 'Segoe UI', system-ui, sans-serif;
                color: ${this.theme.text};
                z-index: 1000;
            }

            .nexus-ui * {
                box-sizing: border-box;
            }

            .nexus-toolbar {
                position: absolute;
                top: 10px;
                left: 10px;
                right: 10px;
                height: 50px;
                background: ${this.theme.panel};
                border: 1px solid ${this.theme.accent}33;
                border-radius: 8px;
                display: flex;
                align-items: center;
                padding: 0 16px;
                pointer-events: auto;
            }

            .nexus-panels {
                position: absolute;
                top: 70px;
                bottom: 60px;
                left: 10px;
                right: 10px;
                pointer-events: none;
            }

            .nexus-panel {
                position: absolute;
                background: ${this.theme.panel};
                border: 1px solid ${this.theme.accent}33;
                border-radius: 8px;
                backdrop-filter: blur(10px);
                pointer-events: auto;
                opacity: 0;
                transform: scale(0.9);
                transition: all 0.3s ease;
            }

            .nexus-panel.active {
                opacity: 1;
                transform: scale(1);
            }

            .nexus-status {
                position: absolute;
                bottom: 10px;
                left: 10px;
                right: 10px;
                height: 40px;
                background: ${this.theme.panel};
                border: 1px solid ${this.theme.accent}33;
                border-radius: 8px;
                display: flex;
                align-items: center;
                padding: 0 16px;
                font-family: monospace;
                font-size: 12px;
                pointer-events: auto;
            }

            .nexus-notifications {
                position: absolute;
                top: 70px;
                right: 10px;
                width: 300px;
                pointer-events: none;
            }

            .nexus-button {
                background: ${this.theme.accent}22;
                border: 1px solid ${this.theme.accent};
                color: ${this.theme.text};
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s ease;
                margin: 0 4px;
            }

            .nexus-button:hover {
                background: ${this.theme.accent};
                color: ${this.theme.bg};
            }

            .nexus-slider {
                -webkit-appearance: none;
                appearance: none;
                background: ${this.theme.accent}33;
                border-radius: 4px;
                height: 6px;
                margin: 0 8px;
                outline: none;
            }

            .nexus-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                background: ${this.theme.accent};
                border-radius: 50%;
                height: 16px;
                width: 16px;
                cursor: pointer;
            }
        `;
        document.head.appendChild(style);
    }

    createDefaultPanels() {
        // Performance panel
        this.createPanel('performance', {
            title: 'Performance',
            position: { top: 0, left: 0, width: 300, height: 200 },
            content: `
                <div style="padding: 16px;">
                    <div id="fps-counter">FPS: 60</div>
                    <div id="frame-time">Frame Time: 16ms</div>
                    <div id="memory-usage">Memory: 0MB</div>
                    <canvas id="perf-graph" width="260" height="100" style="margin-top: 10px; border: 1px solid ${this.theme.accent}33;"></canvas>
                </div>
            `
        });

        // Beat panel
        this.createPanel('beat', {
            title: 'Beat Engine',
            position: { top: 0, right: 0, width: 250, height: 180 },
            content: `
                <div style="padding: 16px;">
                    <div>BPM: <span id="beat-bpm">120</span></div>
                    <div>Beat: <span id="beat-current">0</span></div>
                    <div>Phase: <span id="beat-phase">0.0</span></div>
                    <div style="margin-top: 10px;">
                        <div id="beat-visualizer" style="width: 200px; height: 20px; background: ${this.theme.accent}33; border-radius: 4px; position: relative;">
                            <div id="beat-indicator" style="width: 4px; height: 20px; background: ${this.theme.accent}; border-radius: 2px; position: absolute; transition: left 0.1s;"></div>
                        </div>
                    </div>
                </div>
            `
        });

        // Controls panel
        this.createPanel('controls', {
            title: 'Controls',
            position: { bottom: 0, left: 0, width: 300, height: 150 },
            content: `
                <div style="padding: 16px;">
                    <label>Render Distance: <input type="range" class="nexus-slider" id="render-distance" min="1" max="10" value="3"></label><br><br>
                    <label>Audio Reactivity: <input type="range" class="nexus-slider" id="audio-react" min="0" max="1" step="0.1" value="0.5"></label><br><br>
                    <button class="nexus-button" id="reset-camera">Reset Camera</button>
                    <button class="nexus-button" id="toggle-wireframe">Wireframe</button>
                </div>
            `
        });
    }

    createPanel(id, options) {
        const panel = document.createElement('div');
        panel.id = `nexus-panel-${id}`;
        panel.className = 'nexus-panel';

        // Set position
        const pos = options.position;
        if (pos.top !== undefined) panel.style.top = pos.top + 'px';
        if (pos.bottom !== undefined) panel.style.bottom = pos.bottom + 'px';
        if (pos.left !== undefined) panel.style.left = pos.left + 'px';
        if (pos.right !== undefined) panel.style.right = pos.right + 'px';
        if (pos.width !== undefined) panel.style.width = pos.width + 'px';
        if (pos.height !== undefined) panel.style.height = pos.height + 'px';

        // Add title bar
        const titleBar = document.createElement('div');
        titleBar.style.cssText = `
            background: ${this.theme.accent}22;
            border-bottom: 1px solid ${this.theme.accent}33;
            padding: 8px 16px;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
        `;
        titleBar.textContent = options.title;

        // Add content
        const content = document.createElement('div');
        content.innerHTML = options.content;

        panel.appendChild(titleBar);
        panel.appendChild(content);

        document.getElementById('nexus-panels').appendChild(panel);
        this.panels.set(id, { element: panel, options });

        return panel;
    }

    showPanel(id) {
        const panel = this.panels.get(id);
        if (panel) {
            panel.element.classList.add('active');
            this.activePanel = id;
        }
    }

    hidePanel(id) {
        const panel = this.panels.get(id);
        if (panel) {
            panel.element.classList.remove('active');
            if (this.activePanel === id) {
                this.activePanel = null;
            }
        }
    }

    updatePerformancePanel(stats) {
        document.getElementById('fps-counter').textContent = `FPS: ${Math.round(stats.fps || 0)}`;
        document.getElementById('frame-time').textContent = `Frame Time: ${(stats.frameTime || 0).toFixed(1)}ms`;
        document.getElementById('memory-usage').textContent = `Memory: ${Math.round((stats.memoryUsage || 0) * 100)}%`;
    }

    updateBeatPanel(beatState) {
        document.getElementById('beat-bpm').textContent = beatState.bpm || 120;
        document.getElementById('beat-current').textContent = beatState.beat || 0;
        document.getElementById('beat-phase').textContent = (beatState.beatPhase || 0).toFixed(2);

        const indicator = document.getElementById('beat-indicator');
        if (indicator) {
            indicator.style.left = ((beatState.beatPhase || 0) * 196) + 'px';
        }
    }

    updateStatusBar(text) {
        const statusElement = document.getElementById('nexus-status');
        if (statusElement) {
            statusElement.textContent = text;
        }
    }

    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            background: ${this.theme.panel};
            border: 1px solid ${type === 'error' ? this.theme.error : this.theme.accent}66;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 8px;
            pointer-events: auto;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;

        notification.textContent = message;
        document.getElementById('nexus-notifications').appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Auto remove
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, duration);
    }

    addEventListener(elementId, event, callback) {
        const element = document.getElementById(elementId);
        if (element) {
            element.addEventListener(event, callback);
        }
    }

    setTheme(newTheme) {
        this.theme = { ...this.theme, ...newTheme };
        this.applyStyles(); // Reapply with new theme
    }
}

// =============================================================================
// SECTION E: ANIMATION ENGINE
// =============================================================================

class UnifiedAnimationEngine {
    constructor() {
        this.animations = new Map();
        this.runningAnimations = new Set();
        this.animationId = 0;
        this.running = false;

        // Easing functions
        this.easingFunctions = {
            linear: t => t,
            easeIn: t => t * t,
            easeOut: t => 1 - (1 - t) * (1 - t),
            easeInOut: t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2,
            bounce: t => {
                if (t < 1 / 2.75) return 7.5625 * t * t;
                if (t < 2 / 2.75) {
                    t -= 1.5 / 2.75;
                    return 7.5625 * t * t + 0.75;
                }
                if (t < 2.5 / 2.75) {
                    t -= 2.25 / 2.75;
                    return 7.5625 * t * t + 0.9375;
                }
                t -= 2.625 / 2.75;
                return 7.5625 * t * t + 0.984375;
            }
        };

        console.log('🎬 Unified Animation Engine initialized');
    }

    animate(target, properties, options = {}) {
        const animationId = ++this.animationId;
        const startTime = performance.now();
        const duration = options.duration || 1000;
        const easing = this.easingFunctions[options.easing] || this.easingFunctions.linear;
        const onComplete = options.onComplete;
        const onUpdate = options.onUpdate;

        // Store initial values
        const startValues = {};
        for (const prop in properties) {
            startValues[prop] = target[prop];
        }

        const animation = {
            id: animationId,
            target,
            properties,
            startValues,
            duration,
            easing,
            startTime,
            onComplete,
            onUpdate
        };

        this.animations.set(animationId, animation);
        this.runningAnimations.add(animationId);

        if (!this.running) {
            this.start();
        }

        return animationId;
    }

    start() {
        this.running = true;
        this.update();
    }

    stop() {
        this.running = false;
    }

    update() {
        if (!this.running) return;

        const currentTime = performance.now();
        const completedAnimations = this.updateRunningAnimations(currentTime);
        this.cleanupCompletedAnimations(completedAnimations);
        this.scheduleNextUpdate();
    }

    updateRunningAnimations(currentTime) {
        const completedAnimations = [];

        for (const animationId of this.runningAnimations) {
            const animation = this.animations.get(animationId);
            if (!animation) continue;

            const elapsed = currentTime - animation.startTime;
            const progress = Math.min(elapsed / animation.duration, 1);
            const easedProgress = animation.easing(progress);

            this.updateAnimationProperties(animation, easedProgress);

            if (animation.onUpdate) {
                animation.onUpdate(easedProgress, animation.target);
            }

            if (progress >= 1) {
                completedAnimations.push(animationId);
                if (animation.onComplete) {
                    animation.onComplete(animation.target);
                }
            }
        }

        return completedAnimations;
    }

    updateAnimationProperties(animation, easedProgress) {
        for (const prop in animation.properties) {
            const startValue = animation.startValues[prop];
            const endValue = animation.properties[prop];
            const currentValue = startValue + (endValue - startValue) * easedProgress;
            animation.target[prop] = currentValue;
        }
    }

    cleanupCompletedAnimations(completedAnimations) {
        for (const animationId of completedAnimations) {
            this.runningAnimations.delete(animationId);
            this.animations.delete(animationId);
        }
    }

    scheduleNextUpdate() {
        if (this.runningAnimations.size > 0) {
            requestAnimationFrame(() => this.update());
        } else {
            this.running = false;
        }
    }

    cancelAnimation(animationId) {
        this.runningAnimations.delete(animationId);
        this.animations.delete(animationId);
    }

    cancelAllAnimations() {
        this.runningAnimations.clear();
        this.animations.clear();
        this.running = false;
    }

    // Specialized animation methods
    fadeIn(element, duration = 500) {
        return this.animate(element.style, { opacity: 1 }, {
            duration,
            easing: 'easeOut'
        });
    }

    fadeOut(element, duration = 500) {
        return this.animate(element.style, { opacity: 0 }, {
            duration,
            easing: 'easeIn'
        });
    }

    slideIn(element, direction = 'left', duration = 500) {
        const startPos = direction === 'left' ? '-100%' : '100%';
        element.style.transform = `translateX(${startPos})`;

        return this.animate(element.style, { transform: 'translateX(0%)' }, {
            duration,
            easing: 'easeOut'
        });
    }

    pulse(target, property, amplitude = 0.1, frequency = 2) {
        const startValue = target[property];
        const startTime = performance.now();

        return this.animate(target, {}, {
            duration: 1000, // Run for 1 second
            onUpdate: (progress) => {
                const time = (performance.now() - startTime) / 1000;
                const pulse = Math.sin(time * frequency * Math.PI * 2) * amplitude;
                target[property] = startValue * (1 + pulse);
            }
        });
    }
}

// =============================================================================
// GLOBAL EXPORT
// =============================================================================

window.NexusForgeEngines = {
    UnifiedRenderEngine,
    UnifiedAudioEngine,
    UnifiedAssetManager,
    UnifiedUIEngine,
    UnifiedAnimationEngine
};

console.log('🔧 NEXUS Forge Unified Engines loaded successfully');
