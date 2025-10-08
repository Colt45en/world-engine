/**
 * NEXUS Forge Networking Engine
 * • Real-time multiplayer synchronization
 * • Audio-reactive collaborative environments
 * • P2P and server-based networking options
 * • Synchronized world state and particle effects
 * • Voice chat with spatial audio integration
 */

class NexusForgeNetworkEngine {
    constructor() {
        this.isHost = false;
        this.peerId = this.generatePeerId();
        this.peers = new Map();
        this.worldSyncData = new Map();
        this.audioSyncData = new Map();

        this.networkMode = 'p2p'; // 'p2p', 'server', 'hybrid'
        this.serverUrl = 'wss://nexus-forge-server.com';
        this.webSocket = null;
        this.peerConnection = null;
        this.dataChannel = null;

        this.syncSettings = {
            worldSync: true,
            audioSync: true,
            particleSync: true,
            cameraSync: false,
            voiceChat: true,
            maxPeers: 8,
            syncRate: 60, // Hz
            compressionLevel: 'medium'
        };

        this.audioContext = null;
        this.spatialAudio = {
            enabled: true,
            maxDistance: 100,
            rolloffFactor: 1.0,
            dopplerFactor: 1.0,
            sources: new Map()
        };

        this.networkStats = {
            latency: 0,
            bandwidth: 0,
            packetLoss: 0,
            connectedPeers: 0,
            totalDataSent: 0,
            totalDataReceived: 0
        };

        this.eventHandlers = new Map();
        this.messageQueue = [];
        this.syncBuffer = new Map();

        console.log('🌐 Network Engine initialized');
    }

    async initialize(mode = 'p2p') {
        this.networkMode = mode;

        try {
            // Initialize audio context for spatial audio
            if (this.syncSettings.voiceChat) {
                await this.initializeSpatialAudio();
            }

            switch (mode) {
                case 'p2p':
                    await this.initializeP2P();
                    break;
                case 'server':
                    await this.initializeServerMode();
                    break;
                case 'hybrid':
                    await this.initializeHybridMode();
                    break;
                default:
                    throw new Error(`Unknown network mode: ${mode}`);
            }

            this.startSyncLoop();
            console.log(`✅ Network Engine initialized in ${mode} mode`);
            return true;

        } catch (error) {
            console.error('❌ Network initialization failed:', error);
            return false;
        }
    }

    async initializeP2P() {
        // Initialize WebRTC for peer-to-peer networking
        const configuration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };

        this.peerConnection = new RTCPeerConnection(configuration);

        // Setup data channel for game data
        this.dataChannel = this.peerConnection.createDataChannel('nexus-forge', {
            ordered: false, // Allow out-of-order delivery for better performance
            maxPacketLifeTime: 100 // 100ms max retry time
        });

        this.setupDataChannelHandlers();
        this.setupPeerConnectionHandlers();

        console.log('🔗 P2P networking initialized');
    }

    async initializeServerMode() {
        try {
            this.webSocket = new WebSocket(this.serverUrl);
            this.setupWebSocketHandlers();

            // Wait for connection
            await new Promise((resolve, reject) => {
                this.webSocket.onopen = resolve;
                this.webSocket.onerror = reject;
                setTimeout(reject, 5000); // 5 second timeout
            });

            console.log('🖥️ Server mode initialized');
        } catch (error) {
            console.error('❌ Server connection failed:', error);
            throw error;
        }
    }

    async initializeHybridMode() {
        // Initialize both P2P and server connections
        await Promise.all([
            this.initializeP2P(),
            this.initializeServerMode()
        ]);

        console.log('🔄 Hybrid mode initialized');
    }

    async initializeSpatialAudio() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create spatial audio listener
            if (this.audioContext.listener) {
                this.audioContext.listener.positionX.value = 0;
                this.audioContext.listener.positionY.value = 0;
                this.audioContext.listener.positionZ.value = 0;
                this.audioContext.listener.forwardX.value = 0;
                this.audioContext.listener.forwardY.value = 0;
                this.audioContext.listener.forwardZ.value = -1;
                this.audioContext.listener.upX.value = 0;
                this.audioContext.listener.upY.value = 1;
                this.audioContext.listener.upZ.value = 0;
            }

            console.log('🔊 Spatial audio initialized');
        } catch (error) {
            console.error('❌ Spatial audio initialization failed:', error);
            this.syncSettings.voiceChat = false;
        }
    }

    setupDataChannelHandlers() {
        this.dataChannel.onopen = () => {
            console.log('📡 Data channel opened');
            this.triggerEvent('peer-connected', { peerId: this.peerId });
        };

        this.dataChannel.onmessage = (event) => {
            this.handleDataChannelMessage(JSON.parse(event.data));
        };

        this.dataChannel.onerror = (error) => {
            console.error('❌ Data channel error:', error);
        };

        this.dataChannel.onclose = () => {
            console.log('📡 Data channel closed');
            this.triggerEvent('peer-disconnected', { peerId: this.peerId });
        };
    }

    setupPeerConnectionHandlers() {
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                this.sendSignaling({
                    type: 'ice-candidate',
                    candidate: event.candidate
                });
            }
        };

        this.peerConnection.oniceconnectionstatechange = () => {
            console.log('🧊 ICE connection state:', this.peerConnection.iceConnectionState);
            if (this.peerConnection.iceConnectionState === 'connected') {
                this.updateNetworkStats();
            }
        };

        this.peerConnection.ondatachannel = (event) => {
            const channel = event.channel;
            channel.onmessage = (event) => {
                this.handleDataChannelMessage(JSON.parse(event.data));
            };
        };
    }

    setupWebSocketHandlers() {
        this.webSocket.onopen = () => {
            console.log('🌐 WebSocket connected to server');
            this.sendServerMessage({
                type: 'join-room',
                peerId: this.peerId,
                capabilities: this.getCapabilities()
            });
        };

        this.webSocket.onmessage = (event) => {
            this.handleServerMessage(JSON.parse(event.data));
        };

        this.webSocket.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
        };

        this.webSocket.onclose = () => {
            console.log('🌐 WebSocket connection closed');
            this.attemptReconnection();
        };
    }

    handleDataChannelMessage(message) {
        switch (message.type) {
            case 'world-sync':
                this.handleWorldSync(message.data);
                break;
            case 'audio-sync':
                this.handleAudioSync(message.data);
                break;
            case 'particle-sync':
                this.handleParticleSync(message.data);
                break;
            case 'voice-data':
                this.handleVoiceData(message.data);
                break;
            case 'camera-update':
                this.handleCameraUpdate(message.data);
                break;
            case 'user-action':
                this.handleUserAction(message.data);
                break;
            default:
                console.warn('🤷 Unknown message type:', message.type);
        }

        this.updateNetworkStats();
    }

    handleServerMessage(message) {
        switch (message.type) {
            case 'room-joined':
                this.handleRoomJoined(message.data);
                break;
            case 'peer-list':
                this.handlePeerList(message.data);
                break;
            case 'world-state':
                this.handleWorldState(message.data);
                break;
            case 'server-broadcast':
                this.handleServerBroadcast(message.data);
                break;
            default:
                console.warn('🤷 Unknown server message:', message.type);
        }
    }

    // World Synchronization
    syncWorldState(worldData) {
        if (!this.syncSettings.worldSync) return;

        const syncData = {
            timestamp: Date.now(),
            chunks: this.compressWorldChunks(worldData.chunks),
            activeChunks: Array.from(worldData.activeChunks),
            weatherState: worldData.weather,
            timeOfDay: worldData.timeOfDay
        };

        this.broadcastMessage({
            type: 'world-sync',
            data: syncData
        });
    }

    handleWorldSync(data) {
        // Apply world state from remote peer
        const decompressedChunks = this.decompressWorldChunks(data.chunks);

        this.triggerEvent('world-sync-received', {
            chunks: decompressedChunks,
            activeChunks: new Set(data.activeChunks),
            weather: data.weatherState,
            timeOfDay: data.timeOfDay,
            timestamp: data.timestamp
        });
    }

    // Audio Synchronization
    syncAudioState(audioData) {
        if (!this.syncSettings.audioSync) return;

        const syncData = {
            timestamp: Date.now(),
            frequencyBands: audioData.frequencyBands,
            beatDetected: audioData.beat.detected,
            bpm: audioData.beat.bpm,
            emotionalState: audioData.heart.emotionalState,
            energy: audioData.energy.overall
        };

        this.broadcastMessage({
            type: 'audio-sync',
            data: syncData
        });
    }

    handleAudioSync(data) {
        // Synchronize audio reactive effects across peers
        this.triggerEvent('audio-sync-received', data);
    }

    // Particle Synchronization
    syncParticleEffects(particleData) {
        if (!this.syncSettings.particleSync) return;

        const syncData = {
            timestamp: Date.now(),
            effects: particleData.effects.map(effect => ({
                type: effect.type,
                position: effect.position,
                intensity: effect.intensity,
                duration: effect.duration,
                id: effect.id
            }))
        };

        this.broadcastMessage({
            type: 'particle-sync',
            data: syncData
        });
    }

    handleParticleSync(data) {
        // Apply particle effects from remote peers
        this.triggerEvent('particle-sync-received', data);
    }

    // Voice Chat with Spatial Audio
    async startVoiceChat() {
        if (!this.syncSettings.voiceChat || !this.audioContext) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 48000
                }
            });

            // Create audio processing nodes
            const source = this.audioContext.createMediaStreamSource(stream);
            const processor = this.audioContext.createScriptProcessor(1024, 1, 1);
            const compressor = this.audioContext.createDynamicsCompressor();

            // Setup audio processing pipeline
            source.connect(compressor);
            compressor.connect(processor);

            processor.onaudioprocess = (event) => {
                const audioData = event.inputBuffer.getChannelData(0);
                this.sendVoiceData(audioData);
            };

            console.log('🎤 Voice chat started');
            return true;

        } catch (error) {
            console.error('❌ Voice chat failed to start:', error);
            return false;
        }
    }

    sendVoiceData(audioData) {
        // Compress and send voice data
        const compressedData = this.compressAudioData(audioData);

        this.broadcastMessage({
            type: 'voice-data',
            data: {
                peerId: this.peerId,
                audioData: compressedData,
                timestamp: Date.now()
            }
        });
    }

    handleVoiceData(data) {
        if (data.peerId === this.peerId) return; // Don't play back own voice

        // Create spatial audio source for the peer
        this.createSpatialVoiceSource(data.peerId, data.audioData);
    }

    createSpatialVoiceSource(peerId, audioData) {
        let source = this.spatialAudio.sources.get(peerId);

        if (!source) {
            // Create new spatial audio source
            const panner = this.audioContext.createPanner();
            panner.panningModel = 'HRTF';
            panner.distanceModel = 'exponential';
            panner.maxDistance = this.spatialAudio.maxDistance;
            panner.rolloffFactor = this.spatialAudio.rolloffFactor;

            source = {
                panner: panner,
                gainNode: this.audioContext.createGain()
            };

            source.panner.connect(source.gainNode);
            source.gainNode.connect(this.audioContext.destination);

            this.spatialAudio.sources.set(peerId, source);
        }

        // Update position based on peer location
        const peerData = this.peers.get(peerId);
        if (peerData && peerData.position) {
            source.panner.positionX.value = peerData.position.x;
            source.panner.positionY.value = peerData.position.y;
            source.panner.positionZ.value = peerData.position.z;
        }

        // Play decompressed audio data
        const decompressedAudio = this.decompressAudioData(audioData);
        this.playAudioBuffer(source, decompressedAudio);
    }

    // Camera and Player Synchronization
    syncCameraUpdate(cameraData) {
        if (!this.syncSettings.cameraSync) return;

        this.broadcastMessage({
            type: 'camera-update',
            data: {
                peerId: this.peerId,
                position: cameraData.position,
                rotation: cameraData.rotation,
                timestamp: Date.now()
            }
        });
    }

    handleCameraUpdate(data) {
        // Update peer camera/player position
        this.peers.set(data.peerId, {
            position: data.position,
            rotation: data.rotation,
            lastUpdate: data.timestamp
        });

        this.triggerEvent('peer-moved', data);
    }

    // User Action Synchronization
    broadcastUserAction(action) {
        this.broadcastMessage({
            type: 'user-action',
            data: {
                peerId: this.peerId,
                action: action,
                timestamp: Date.now()
            }
        });
    }

    handleUserAction(data) {
        // Handle actions from other players
        this.triggerEvent('peer-action', data);
    }

    // Networking Utilities
    broadcastMessage(message) {
        if (this.dataChannel && this.dataChannel.readyState === 'open') {
            this.dataChannel.send(JSON.stringify(message));
        }

        if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
            this.sendServerMessage({
                type: 'broadcast',
                message: message
            });
        }
    }

    sendServerMessage(message) {
        if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
            this.webSocket.send(JSON.stringify(message));
        }
    }

    sendSignaling(message) {
        // Send WebRTC signaling through server
        this.sendServerMessage({
            type: 'signaling',
            data: message
        });
    }

    // Compression and Optimization
    compressWorldChunks(chunks) {
        // Implement world data compression
        const compressed = {};
        for (const [key, chunk] of chunks) {
            compressed[key] = {
                heightmap: this.compressFloatArray(chunk.heightmap),
                biomes: this.compressStringArray(chunk.biomes),
                // Only sync essential data to reduce bandwidth
                lastModified: chunk.lastModified || Date.now()
            };
        }
        return compressed;
    }

    decompressWorldChunks(compressedChunks) {
        const chunks = new Map();
        for (const [key, compressed] of Object.entries(compressedChunks)) {
            chunks.set(key, {
                heightmap: this.decompressFloatArray(compressed.heightmap),
                biomes: this.decompressStringArray(compressed.biomes),
                lastModified: compressed.lastModified
            });
        }
        return chunks;
    }

    compressAudioData(audioData) {
        // Simple audio compression for voice chat
        const compressed = new Int16Array(audioData.length / 2);
        for (let i = 0; i < compressed.length; i++) {
            compressed[i] = Math.round(audioData[i * 2] * 32767);
        }
        return Array.from(compressed);
    }

    decompressAudioData(compressedData) {
        const decompressed = new Float32Array(compressedData.length * 2);
        for (let i = 0; i < compressedData.length; i++) {
            const value = compressedData[i] / 32767;
            decompressed[i * 2] = value;
            decompressed[i * 2 + 1] = value; // Duplicate for stereo
        }
        return decompressed;
    }

    compressFloatArray(arr) {
        // Convert to 16-bit integers for compression
        return arr.map(val => Math.round(val * 100) / 100);
    }

    decompressFloatArray(arr) {
        return new Float32Array(arr);
    }

    compressStringArray(arr) {
        // Use dictionary compression for repeated strings
        const dict = [...new Set(arr)];
        const indices = arr.map(str => dict.indexOf(str));
        return { dict, indices };
    }

    decompressStringArray(compressed) {
        return compressed.indices.map(index => compressed.dict[index]);
    }

    // Network Statistics and Monitoring
    updateNetworkStats() {
        this.networkStats.connectedPeers = this.peers.size;

        if (this.peerConnection) {
            this.peerConnection.getStats().then(stats => {
                stats.forEach(report => {
                    if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                        this.networkStats.latency = report.currentRoundTripTime * 1000;
                    }
                    if (report.type === 'data-channel') {
                        this.networkStats.bandwidth = report.bytesReceived + report.bytesSent;
                    }
                });
            });
        }
    }

    getNetworkStats() {
        return { ...this.networkStats };
    }

    // Event System
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    off(event, handler) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    triggerEvent(event, data) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    // Synchronization Loop
    startSyncLoop() {
        const syncInterval = 1000 / this.syncSettings.syncRate;

        setInterval(() => {
            if (this.isConnected()) {
                this.processSyncBuffer();
                this.sendHeartbeat();
            }
        }, syncInterval);
    }

    processSyncBuffer() {
        // Process queued synchronization data
        for (const [peerId, data] of this.syncBuffer) {
            this.processBufferedData(peerId, data);
        }
        this.syncBuffer.clear();
    }

    sendHeartbeat() {
        this.broadcastMessage({
            type: 'heartbeat',
            data: {
                peerId: this.peerId,
                timestamp: Date.now()
            }
        });
    }

    // Utility Functions
    generatePeerId() {
        return 'nexus-' + Math.random().toString(36).substr(2, 9);
    }

    getCapabilities() {
        return {
            webgl: !!document.createElement('canvas').getContext('webgl'),
            webrtc: !!(window.RTCPeerConnection),
            webaudio: !!(window.AudioContext || window.webkitAudioContext),
            mediaCaptureAndStreams: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            gamepad: !!navigator.getGamepads
        };
    }

    isConnected() {
        return (this.dataChannel && this.dataChannel.readyState === 'open') ||
            (this.webSocket && this.webSocket.readyState === WebSocket.OPEN);
    }

    attemptReconnection() {
        setTimeout(() => {
            if (this.networkMode === 'server' || this.networkMode === 'hybrid') {
                console.log('🔄 Attempting server reconnection...');
                this.initializeServerMode().catch(console.error);
            }
        }, 3000);
    }

    // Cleanup
    disconnect() {
        this.isHost = false;

        if (this.dataChannel) {
            this.dataChannel.close();
        }

        if (this.peerConnection) {
            this.peerConnection.close();
        }

        if (this.webSocket) {
            this.webSocket.close();
        }

        if (this.audioContext) {
            this.audioContext.close();
        }

        this.peers.clear();
        this.spatialAudio.sources.clear();

        console.log('🌐 Network disconnected');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeNetworkEngine;
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    window.NexusForgeNetworkEngine = NexusForgeNetworkEngine;
    console.log('🌐 Network Engine available globally');
}
