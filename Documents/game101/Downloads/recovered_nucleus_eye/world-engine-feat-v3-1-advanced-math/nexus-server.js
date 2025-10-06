import express from 'express';
import cors from 'cors';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const app = express();
const PORT = process.env.PORT || 3000;

// Auto-allow all requests - CORS middleware
app.use(cors({
    origin: '*', // Allow all origins
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allowedHeaders: ['*'],
    credentials: true
}));

// Parse JSON bodies
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve static files
app.use(express.static(__dirname));

// Auto-allow preflight requests
app.options('*', (req, res) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS');
    res.header('Access-Control-Allow-Headers', '*');
    res.sendStatus(200);
});

// Main launcher route
app.get('/', (req, res) => {
    res.sendFile(join(__dirname, 'nexus-world-engine-launcher.html'));
});

// NEXUS Brain Dashboard
app.get('/brain', (req, res) => {
    res.sendFile(join(__dirname, 'nexus-brain-dashboard.html'));
});

// NEXUS AI Vault
app.get('/vault', (req, res) => {
    res.sendFile(join(__dirname, 'nexus-ai-vault.html'));
});

// NEXUS Lab
app.get('/lab', (req, res) => {
    res.sendFile(join(__dirname, 'nexus-lab.html'));
});

// World Engine Demo
app.get('/engine', (req, res) => {
    res.sendFile(join(__dirname, 'world-engine-demo.html'));
});

// GameLite Applications Routes
app.get('/gamilelite/:app', (req, res) => {
    const appName = req.params.app;
    res.sendFile(join(__dirname, 'gamilelite-apps', appName));
});

// Specific GameLite app shortcuts
app.get('/react-hub', (req, res) => {
    res.sendFile(join(__dirname, 'gamilelite-apps', 'react-hub.html'));
});

app.get('/r3f-sandbox', (req, res) => {
    res.sendFile(join(__dirname, 'gamilelite-apps', 'r3f-sandbox.html'));
});

app.get('/threejs-demo', (req, res) => {
    res.sendFile(join(__dirname, 'gamilelite-apps', 'threejs-interactive-demo.html'));
});

app.get('/voxel-sculptor', (req, res) => {
    res.sendFile(join(__dirname, 'gamilelite-apps', 'advanced-voxel-sculptor.html'));
});

app.get('/terrain-visualizer', (req, res) => {
    res.sendFile(join(__dirname, 'gamilelite-apps', 'nexus-terrain-visualizer.html'));
});

app.get('/physics-engine', (req, res) => {
    res.sendFile(join(__dirname, 'gamilelite-apps', 'calculus-physics-engine.html'));
});

app.get('/sandbox-store', (req, res) => {
    res.sendFile(join(__dirname, 'gamilelite-apps', 'Sandbox-Store.html'));
});

// Multi-Language Commander
app.get('/commander', (req, res) => {
    res.sendFile(join(__dirname, 'multi-language-commander.html'));
});

app.get('/language-dashboard', (req, res) => {
    res.sendFile(join(__dirname, 'language-commanding-dashboard.html'));
});

// API Endpoints for AI integration
app.post('/api/nexus/query', (req, res) => {
    const { query, context } = req.body;
    
    // Simulate NEXUS AI response
    const aiResponse = {
        status: 'success',
        query: query,
        response: `NEXUS AI processed: "${query}"`,
        timestamp: new Date().toISOString(),
        context: context || 'world-engine',
        capabilities: [
            'Multi-language code generation',
            'Semantic knowledge processing',
            'Real-time system integration',
            'Persistent memory vault'
        ]
    };
    
    res.json(aiResponse);
});

// Python bridge endpoint
app.post('/api/python/execute', (req, res) => {
    const { code, context } = req.body;
    
    // Simulate Python execution response
    const pythonResponse = {
        status: 'success',
        code: code,
        output: `Executed Python code in ${context || 'default'} context`,
        timestamp: new Date().toISOString(),
        environment: 'Python 3.13.7'
    };
    
    res.json(pythonResponse);
});

// System status endpoint
app.get('/api/status', (req, res) => {
    const systemStatus = {
        status: 'operational',
        components: {
            'nexus-brain': 'active',
            'python-env': 'ready',
            'nodejs-env': 'ready',
            'ai-vault': 'loaded',
            'world-engine': 'initialized'
        },
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        timestamp: new Date().toISOString()
    };
    
    res.json(systemStatus);
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        service: 'nexus-world-engine',
        timestamp: new Date().toISOString()
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({
        status: 'error',
        message: err.message,
        timestamp: new Date().toISOString()
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🌍 NEXUS World Engine Server running on http://localhost:${PORT}`);
    console.log(`🧠 NEXUS Brain Dashboard: http://localhost:${PORT}/brain`);
    console.log(`🔬 NEXUS Lab: http://localhost:${PORT}/lab`);
    console.log(`💾 NEXUS Vault: http://localhost:${PORT}/vault`);
    console.log(`⚡ World Engine: http://localhost:${PORT}/engine`);
    console.log(`📊 System Status: http://localhost:${PORT}/api/status`);
    console.log(`✅ Health Check: http://localhost:${PORT}/health`);
    console.log(`\n🚀 All CORS policies disabled - Auto-allow all requests enabled!`);
});

export default app;