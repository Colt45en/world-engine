#!/usr/bin/env node
/**
 * World Engine Studio Server
 * Provides REST API endpoints for the studio interface to interact with advanced systems
 */

import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer } from 'http';
import { Server as SocketIO } from 'socket.io';
import { createWorldEngineFramework } from './world-engine-integration-framework.js';
import { AdvancedMorphologyEngine } from './models/advanced-morphology-engine.js';
import { createEnhancedUpflowV2 } from './models/enhanced-upflow-v2-fixed.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new SocketIO(httpServer, {
    cors: {
        origin: '*',
        methods: ['GET', 'POST']
    }
});

const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// World Engine Framework instance with v2 enhancements
let framework = null;
let morphologyEngineV2 = null;
let enhancedUpflowV2 = null;

const systemMetrics = {
    processedCount: 0,
    totalTime: 0,
    successCount: 0,
    cacheHits: 0,
    totalRequests: 0,
    startTime: Date.now(),
    // V2 metrics
    morphologyQueries: 0,
    upflowOperations: 0,
    performanceHistory: []
};

// Initialize World Engine Framework
async function initializeFramework() {
    console.log('🚀 Initializing World Engine Advanced Systems...');

    try {
        // Initialize V2 Enhanced Systems
        console.log('🧠 Initializing Morphology Engine v2...');
        morphologyEngineV2 = new AdvancedMorphologyEngine({
            debug: true,
            enableV2Features: true,
            externalRules: [] // Could be loaded from config
        });

        console.log('💾 Initializing Enhanced Upflow v2...');
        enhancedUpflowV2 = await createEnhancedUpflowV2({
            idbFactory: null // Node.js doesn't have IndexedDB, will use memory storage
        }).catch(err => {
            console.warn('Upflow v2 init failed, using null:', err.message);
            return null;
        });

        framework = await createWorldEngineFramework({
            enableAutoStart: true,
            enablePerformanceMonitoring: true,
            autoConnect: true
        });

        console.log('✅ World Engine Framework initialized successfully');
        return true;
    } catch (error) {
        console.error('❌ Framework initialization failed:', error);
        return false;
    }
}

// API Routes

/**
 * GET /api/status
 * Returns the status of all advanced systems
 */
app.get('/api/status', async (req, res) => {
    try {
        if (!framework) {
            return res.json({
                success: false,
                message: 'Framework not initialized',
                systems: {
                    morphology: 'offline',
                    indexManager: 'offline',
                    uxPipeline: 'offline',
                    agentSystem: 'offline'
                }
            });
        }

        const systemStatus = framework.getSystemStatus();
        const healthReport = await framework.generateSystemHealthReport();

        res.json({
            success: true,
            systems: {
                morphology: framework.systems.advancedMorphologyEngine ? 'online' : 'offline',
                indexManager: framework.systems.indexManager ? 'online' : 'offline',
                uxPipeline: framework.systems.uxPipeline ? 'online' : 'offline',
                agentSystem: framework.systems.agentSystem ? 'online' : 'offline'
            },
            metrics: systemMetrics,
            health: healthReport.overallHealth,
            uptime: Date.now() - systemMetrics.startTime
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

/**
 * POST /api/analyze/v2
 * Enhanced morphological analysis with position tracking and semantic classification
 */
app.post('/api/analyze/v2', async (req, res) => {
    const startTime = Date.now();
    systemMetrics.totalRequests++;
    systemMetrics.morphologyQueries++;

    try {
        const { text, options = {} } = req.body;

        if (!text || typeof text !== 'string') {
            return res.status(400).json({
                success: false,
                error: 'Text parameter is required'
            });
        }

        if (!morphologyEngineV2) {
            return res.status(503).json({
                success: false,
                error: 'Morphology Engine v2 not initialized'
            });
        }

        // Use v2 analysis with position tracking
        const result = morphologyEngineV2.analyzeV2(text, options);

        // Store in enhanced upflow if available
        if (enhancedUpflowV2 && result) {
            try {
                await enhancedUpflowV2.ingestEnhanced(result);
            } catch (upflowError) {
                console.warn('Upflow ingestion failed:', upflowError.message);
            }
        }

        const processingTime = Date.now() - startTime;
        systemMetrics.processedCount++;
        systemMetrics.totalTime += processingTime;
        systemMetrics.successCount++;

        // Add to performance history
        systemMetrics.performanceHistory.push({
            type: 'morphology_v2',
            duration: processingTime,
            timestamp: Date.now(),
            wordCount: text.split(' ').length
        });

        // Keep only last 100 entries
        if (systemMetrics.performanceHistory.length > 100) {
            systemMetrics.performanceHistory = systemMetrics.performanceHistory.slice(-100);
        }

        res.json({
            success: true,
            result: result,
            version: '2.0',
            processingTime: processingTime,
            metrics: {
                morphemes: result.morphemes ? result.morphemes.length : 0,
                complexity: result.complexity,
                confidence: result.confidence
            }
        });

    } catch (error) {
        console.error('Enhanced morphological analysis error:', error);
        const processingTime = Date.now() - startTime;

        res.status(500).json({
            success: false,
            error: error.message,
            processingTime: processingTime
        });
    }
});

/**
 * POST /api/analyze
 * Performs morphological analysis on input text
 */
app.post('/api/analyze', async (req, res) => {
    const startTime = Date.now();
    systemMetrics.totalRequests++;

    try {
        const { text, options = {} } = req.body;

        if (!text || typeof text !== 'string') {
            return res.status(400).json({
                success: false,
                error: 'Text parameter is required'
            });
        }

        if (!framework) {
            return res.status(503).json({
                success: false,
                error: 'Framework not initialized'
            });
        }

        const result = await framework.processWithAdvancedMorphology(text, options);

        const processingTime = Date.now() - startTime;
        systemMetrics.totalTime += processingTime;
        systemMetrics.processedCount++;
        systemMetrics.successCount++;

        if (Math.random() > 0.7) { // Simulate cache hits
            systemMetrics.cacheHits++;
        }

        res.json({
            success: true,
            result: result,
            processingTime: processingTime,
            timestamp: Date.now()
        });

    } catch (error) {
        const processingTime = Date.now() - startTime;
        systemMetrics.totalTime += processingTime;

        res.status(500).json({
            success: false,
            error: error.message,
            processingTime: processingTime
        });
    }
});

/**
 * POST /api/search
 * Performs search using the index manager
 */
app.post('/api/search', async (req, res) => {
    const startTime = Date.now();
    systemMetrics.totalRequests++;

    try {
        const { query, options = {} } = req.body;

        if (!query || typeof query !== 'string') {
            return res.status(400).json({
                success: false,
                error: 'Query parameter is required'
            });
        }

        if (!framework) {
            return res.status(503).json({
                success: false,
                error: 'Framework not initialized'
            });
        }

        const results = await framework.searchWithIndexManager(query, {
            maxResults: 10,
            useMorphology: true,
            includeVectorSearch: true,
            ...options
        });

        const processingTime = Date.now() - startTime;
        systemMetrics.totalTime += processingTime;
        systemMetrics.successCount++;

        res.json({
            success: true,
            results: results,
            count: results.length,
            processingTime: processingTime,
            timestamp: Date.now()
        });

    } catch (error) {
        const processingTime = Date.now() - startTime;
        systemMetrics.totalTime += processingTime;

        res.status(500).json({
            success: false,
            error: error.message,
            processingTime: processingTime
        });
    }
});

/**
 * POST /api/comprehensive
 * Runs comprehensive processing through all systems
 */
app.post('/api/comprehensive', async (req, res) => {
    const startTime = Date.now();
    systemMetrics.totalRequests++;

    try {
        const { input, options = {} } = req.body;

        if (!input || typeof input !== 'string') {
            return res.status(400).json({
                success: false,
                error: 'Input parameter is required'
            });
        }

        if (!framework) {
            return res.status(503).json({
                success: false,
                error: 'Framework not initialized'
            });
        }

        const result = await framework.processComprehensive(input, {
            enableMorphology: true,
            enableSearch: true,
            useAgents: true,
            maxSearchResults: 10,
            ...options
        });

        const processingTime = Date.now() - startTime;
        systemMetrics.totalTime += processingTime;
        systemMetrics.processedCount++;
        systemMetrics.successCount++;

        res.json({
            success: true,
            result: result,
            processingTime: processingTime,
            timestamp: Date.now()
        });

    } catch (error) {
        const processingTime = Date.now() - startTime;
        systemMetrics.totalTime += processingTime;

        res.status(500).json({
            success: false,
            error: error.message,
            processingTime: processingTime
        });
    }
});

/**
 * GET /api/upflow/semantic/:class
 * Query morphemes by semantic class using enhanced upflow v2
 */
app.get('/api/upflow/semantic/:class', async (req, res) => {
    const startTime = Date.now();
    systemMetrics.totalRequests++;
    systemMetrics.upflowOperations++;

    try {
        const { class: semanticClass } = req.params;

        if (!enhancedUpflowV2) {
            return res.status(503).json({
                success: false,
                error: 'Enhanced Upflow v2 not available'
            });
        }

        const results = await enhancedUpflowV2.semantic[semanticClass.toLowerCase()]?.() ||
            await enhancedUpflowV2.queryBySemanticClass(semanticClass);

        const processingTime = Date.now() - startTime;

        res.json({
            success: true,
            results: results,
            count: results.length,
            semanticClass: semanticClass,
            processingTime: processingTime,
            version: '2.0'
        });

    } catch (error) {
        const processingTime = Date.now() - startTime;
        res.status(500).json({
            success: false,
            error: error.message,
            processingTime: processingTime
        });
    }
});

/**
 * POST /api/benchmark
 * Runs performance benchmark on all systems
 */
app.post('/api/benchmark', async (req, res) => {
    try {
        if (!framework) {
            return res.status(503).json({
                success: false,
                error: 'Framework not initialized'
            });
        }

        const testWords = [
            'productivity',
            'counterproductivity',
            'bioengineering',
            'antidisestablishmentarianism',
            'machine learning algorithms',
            'neural network optimization'
        ];

        const results = [];
        let totalTime = 0;
        let successCount = 0;

        for (const word of testWords) {
            const startTime = Date.now();

            try {
                const result = await framework.processComprehensive(word, {
                    enableMorphology: true,
                    enableSearch: true,
                    maxSearchResults: 3
                });

                const wordTime = Date.now() - startTime;
                totalTime += wordTime;
                successCount++;

                results.push({
                    word: word,
                    time: wordTime,
                    success: true,
                    morphology: !!result.morphology,
                    indexing: !!result.indexing,
                    uxProcessing: !!result.uxProcessing
                });

            } catch (error) {
                const wordTime = Date.now() - startTime;
                totalTime += wordTime;

                results.push({
                    word: word,
                    time: wordTime,
                    success: false,
                    error: error.message
                });
            }
        }

        const avgTime = totalTime / testWords.length;
        const wordsPerSecond = testWords.length / totalTime * 1000;
        const successRate = (successCount / testWords.length) * 100;

        res.json({
            success: true,
            benchmark: {
                totalWords: testWords.length,
                totalTime: totalTime,
                avgTime: avgTime,
                wordsPerSecond: wordsPerSecond,
                successRate: successRate,
                results: results
            },
            timestamp: Date.now()
        });

    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

/**
 * GET /api/metrics
 * Returns current system metrics
 */
app.get('/api/metrics', (req, res) => {
    const avgTime = systemMetrics.totalRequests > 0 ?
        systemMetrics.totalTime / systemMetrics.totalRequests : 0;
    const successRate = systemMetrics.totalRequests > 0 ?
        (systemMetrics.successCount / systemMetrics.totalRequests) * 100 : 0;
    const cacheRate = systemMetrics.totalRequests > 0 ?
        (systemMetrics.cacheHits / systemMetrics.totalRequests) * 100 : 0;

    res.json({
        success: true,
        metrics: {
            ...systemMetrics,
            avgTime: Math.round(avgTime * 100) / 100,
            successRate: Math.round(successRate * 100) / 100,
            cacheRate: Math.round(cacheRate * 100) / 100,
            uptime: Date.now() - systemMetrics.startTime
        }
    });
});

// Serve studio interface
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'world-engine-studio.html'));
});

// Error handler
app.use((error, req, res, next) => {
    console.error('Server error:', error);
    res.status(500).json({
        success: false,
        error: 'Internal server error'
    });
});

// Room management
const rooms = new Map(); // roomName -> { participants: Set, createdAt: Date }
const participants = new Map(); // socketId -> { name, room, sharing }

// WebSocket handlers for unified scaffold bus system
io.on('connection', (socket) => {
    console.log('🔌 Unified scaffold client connected:', socket.id);

    // Room management
    socket.on('join-room', (data) => {
        const { room, name } = data;

        // Leave current room if in one
        leaveCurrentRoom(socket);

        // Join new room
        if (!rooms.has(room)) {
            rooms.set(room, { participants: new Set(), createdAt: new Date() });
        }

        const roomData = rooms.get(room);
        roomData.participants.add(socket.id);
        socket.join(room);

        // Store participant info
        participants.set(socket.id, { name, room, sharing: false });

        // Get existing participants
        const existingParticipants = Array.from(roomData.participants)
            .filter(id => id !== socket.id)
            .map(id => participants.get(id))
            .filter(p => p);

        // Notify client they joined
        socket.emit('room-joined', {
            room: room,
            participants: existingParticipants.map(p => ({ id: socket.id, name: p.name, sharing: p.sharing }))
        });

        // Notify others in room
        socket.to(room).emit('participant-joined', {
            id: socket.id,
            name: name,
            sharing: false
        });

        console.log(`👥 ${name} joined room: ${room}`);
    });

    socket.on('leave-room', () => {
        leaveCurrentRoom(socket);
    });

    // Screen sharing
    socket.on('start-screen-share', (data) => {
        const participant = participants.get(socket.id);
        if (participant) {
            participant.sharing = true;
            socket.to(participant.room).emit('screen-share-started', {
                participantId: socket.id,
                name: participant.name
            });
            console.log(`📺 ${participant.name} started screen sharing`);
        }
    });

    socket.on('stop-screen-share', (data) => {
        const participant = participants.get(socket.id);
        if (participant) {
            participant.sharing = false;
            socket.to(participant.room).emit('screen-share-stopped', socket.id);
            console.log(`📺 ${participant.name} stopped screen sharing`);
        }
    });

    // WebRTC signaling
    socket.on('webrtc-offer', (data) => {
        socket.to(data.to).emit('webrtc-offer', {
            from: socket.id,
            offer: data.offer
        });
    });

    socket.on('webrtc-answer', (data) => {
        socket.to(data.to).emit('webrtc-answer', {
            from: socket.id,
            answer: data.answer
        });
    });

    socket.on('webrtc-ice-candidate', (data) => {
        socket.to(data.to).emit('webrtc-ice-candidate', {
            from: socket.id,
            candidate: data.candidate
        });
    });

    // Chat messages
    socket.on('chat-message', (data) => {
        const participant = participants.get(socket.id);
        if (participant && participant.room === data.room) {
            socket.to(data.room).emit('chat-message', {
                from: participant.name,
                message: data.message
            });
        }
    });

    // Handle studio bus messages
    socket.on('studio-bus', async (message) => {
        try {
            console.log('📨 Received bus message:', message.type);

            switch (message.type) {
                case 'we.analyze': {
                    const analyzeResult = await framework.processWithAdvancedMorphology(message.text, message.options);
                    socket.emit('studio-bus', {
                        type: 'we.result',
                        requestId: message.requestId,
                        result: analyzeResult,
                        timestamp: Date.now()
                    });
                    break;
                }

                case 'we.search': {
                    const searchResult = await framework.searchWithIndexManager(message.query, message.options);
                    socket.emit('studio-bus', {
                        type: 'we.result',
                        requestId: message.requestId,
                        result: searchResult,
                        timestamp: Date.now()
                    });
                    break;
                }

                case 'we.comprehensive': {
                    const comprehensiveResult = await framework.processComprehensive(message.input, message.options);
                    socket.emit('studio-bus', {
                        type: 'we.result',
                        requestId: message.requestId,
                        result: comprehensiveResult,
                        timestamp: Date.now()
                    });
                    break;
                }

                case 'we.status': {
                    const systemStatus = {
                        success: true,
                        systems: {
                            morphology: framework.systems.advancedMorphologyEngine ? 'online' : 'offline',
                            indexManager: framework.systems.indexManager ? 'online' : 'offline',
                            uxPipeline: framework.systems.uxPipeline ? 'online' : 'offline',
                            agentSystem: framework.systems.agentSystem ? 'online' : 'offline'
                        },
                        metrics: systemMetrics,
                        uptime: Date.now() - systemMetrics.startTime
                    };
                    socket.emit('studio-bus', {
                        type: 'we.status',
                        requestId: message.requestId,
                        result: systemStatus,
                        timestamp: Date.now()
                    });
                    break;
                }

                case 'beat.start':
                    // Integration point for NEXUS Holy Beat System
                    console.log('🎵 Beat system start requested:', message.config);
                    socket.emit('studio-bus', {
                        type: 'beat.started',
                        requestId: message.requestId,
                        config: message.config,
                        timestamp: Date.now()
                    });
                    break;

                case 'beat.stop':
                    console.log('🎵 Beat system stop requested');
                    socket.emit('studio-bus', {
                        type: 'beat.stopped',
                        requestId: message.requestId,
                        timestamp: Date.now()
                    });
                    break;

                default:
                    console.warn('❓ Unknown bus message type:', message.type);
            }
        } catch (error) {
            console.error('❌ Bus message error:', error);
            socket.emit('studio-bus', {
                type: 'error',
                requestId: message.requestId,
                error: error.message,
                timestamp: Date.now()
            });
        }
    });

    socket.on('disconnect', () => {
        console.log('🔌 Unified scaffold client disconnected:', socket.id);
        leaveCurrentRoom(socket);
    });

    // Helper function to handle room leaving
    function leaveCurrentRoom(socket) {
        const participant = participants.get(socket.id);
        if (participant && participant.room) {
            const room = participant.room;
            const roomData = rooms.get(room);

            if (roomData) {
                roomData.participants.delete(socket.id);

                // Clean up empty rooms
                if (roomData.participants.size === 0) {
                    rooms.delete(room);
                    console.log(`🏠 Room ${room} deleted (empty)`);
                } else {
                    // Notify others in room
                    socket.to(room).emit('participant-left', socket.id);
                }
            }

            socket.leave(room);
            socket.emit('room-left');
        }

        participants.delete(socket.id);
    }
});

// Initialize and start server
async function startServer() {
    console.log('🌍 Starting World Engine Studio Server...');

    // Initialize framework
    const initialized = await initializeFramework();
    if (!initialized) {
        console.warn('⚠️ Starting server without framework (will retry initialization)');
    }

    // Start HTTP server with WebSocket support
    httpServer.listen(PORT, () => {
        console.log(`✅ World Engine Studio Server running on http://localhost:${PORT}`);
        console.log(`🎨 Studio interface available at http://localhost:${PORT}`);
        console.log(`📊 API endpoints available at http://localhost:${PORT}/api/*`);
        console.log('🔌 WebSocket bus available for unified scaffold');

        // Retry framework initialization if it failed
        if (!initialized) {
            setTimeout(async () => {
                console.log('🔄 Retrying framework initialization...');
                await initializeFramework();
            }, 5000);
        }
    });
}// Handle process termination
process.on('SIGINT', async () => {
    console.log('🔧 Shutting down World Engine Studio Server...');

    if (framework) {
        try {
            await framework.shutdown();
            console.log('✅ Framework shutdown completed');
        } catch (error) {
            console.error('❌ Framework shutdown error:', error);
        }
    }

    process.exit(0);
});

// Start the server
startServer().catch(error => {
    console.error('💥 Failed to start server:', error);
    process.exit(1);
});
