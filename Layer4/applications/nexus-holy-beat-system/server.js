import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8000;

// Serve static files
app.use(express.static(__dirname));

// Serve Three.js from node_modules
app.use('/three', express.static(path.join(__dirname, 'node_modules/three')));

// Main routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'system-overview.html'));
});

app.get('/demo', (req, res) => {
    res.sendFile(path.join(__dirname, 'demos', 'master-demo.html'));
});

// Dashboard routes
app.get('/dashboards', (req, res) => {
    res.sendFile(path.join(__dirname, 'dashboards', 'dashboard_index.html'));
});

app.get('/dashboards/:dashboard', (req, res) => {
    const dashboard = req.params.dashboard;
    res.sendFile(path.join(__dirname, 'dashboards', dashboard + '.html'));
});

// API routes for system data
app.get('/api/status', (req, res) => {
    res.json({
        timestamp: new Date().toISOString(),
        status: 'running',
        version: '1.0.0',
        components: {
            clockBus: 'active',
            audioEngine: 'active',
            artEngine: 'active',
            worldEngine: 'active',
            training: 'ready'
        },
        parameters: {
            bpm: 120,
            harmonics: 6,
            petalCount: 8,
            terrainRoughness: 0.4
        }
    });
});

// Game Logic API routes
app.get('/api/game/status', (req, res) => {
    res.json({
        timestamp: new Date().toISOString(),
        gameEngine: 'ready',
        components: ['Transform', 'AudioSync', 'ArtSync', 'Physics'],
        entityCount: 0,
        systemSync: {
            clockBus: true,
            audioEngine: true,
            artEngine: true,
            worldEngine: true
        }
    });
});

app.post('/api/game/sync', express.json(), (req, res) => {
    // Endpoint for game logic to sync with NEXUS systems
    const { bpm, harmonics, petalCount, terrainRoughness } = req.body;

    // In a real implementation, this would update the game engine
    console.log('🎮 Game Logic Sync:', { bpm, harmonics, petalCount, terrainRoughness });

    res.json({
        success: true,
        synced: true,
        timestamp: new Date().toISOString()
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🎵 NEXUS Holy Beat Server running on http://localhost:${PORT}`);
    console.log(`📊 System Overview: http://localhost:${PORT}/`);
    console.log(`🚀 Master Demo: http://localhost:${PORT}/demo`);
    console.log(`🎛️ Dashboards: http://localhost:${PORT}/dashboards`);
    console.log(`🎮 Game Logic API: http://localhost:${PORT}/api/game/status`);
    console.log(`📡 API Status: http://localhost:${PORT}/api/status`);
});
