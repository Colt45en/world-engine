#!/bin/bash
# NEXUS Forge Complete Deployment Script
# Version: 5.0.0 - September 2025
# Complete Mathematical & AI Ecosystem Deployment

echo "🌟 NEXUS Forge Complete Ecosystem Deployment"
echo "=============================================="
echo "Version 5.0.0 - Advanced Mathematical & AI Integration"
echo "Deployment Date: $(date)"
echo ""

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NEXUS_DIR="nexus-forge-ecosystem"
WEB_PORT=8080
API_PORT=3000
AI_PORT=5000
MONITOR_PORT=9090

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}[NEXUS]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_header "Checking System Requirements"

    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js found: $NODE_VERSION"
    else
        print_error "Node.js not found. Please install Node.js 16+ to continue."
        exit 1
    fi

    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3 not found. Please install Python 3.8+ to continue."
        exit 1
    fi

    # Check Git
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version)
        print_success "Git found: $GIT_VERSION"
    else
        print_warning "Git not found. Version control features will be limited."
    fi

    # Check available ports
    check_port() {
        if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null; then
            print_warning "Port $1 is in use"
            return 1
        else
            print_success "Port $1 is available"
            return 0
        fi
    }

    check_port $WEB_PORT
    check_port $API_PORT
    check_port $AI_PORT
    check_port $MONITOR_PORT

    echo ""
}

# Create directory structure
create_structure() {
    print_header "Creating NEXUS Directory Structure"

    # Main directories
    mkdir -p $NEXUS_DIR/{public,src,config,logs,data,scripts,docs,tests}
    mkdir -p $NEXUS_DIR/src/{ai,math,webgl,education,monitoring,communication}
    mkdir -p $NEXUS_DIR/public/{assets,css,js,components}
    mkdir -p $NEXUS_DIR/config/{development,production,testing}
    mkdir -p $NEXUS_DIR/logs/{system,ai,math,webgl,error}
    mkdir -p $NEXUS_DIR/data/{training,math,visualization,education}
    mkdir -p $NEXUS_DIR/scripts/{deployment,maintenance,backup}
    mkdir -p $NEXUS_DIR/tests/{unit,integration,e2e}

    print_success "Directory structure created"
    echo ""
}

# Copy NEXUS files
deploy_files() {
    print_header "Deploying NEXUS Core Files"

    # Copy main HTML files to public directory
    if [ -f "nexus-forge-launcher.html" ]; then
        cp nexus-forge-launcher.html $NEXUS_DIR/public/index.html
        print_success "Main launcher deployed"
    else
        print_error "nexus-forge-launcher.html not found"
    fi

    # Copy component files
    COMPONENT_FILES=(
        "nexus-system-monitor.html"
        "nexus-analytics-dashboard.html"
        "nexus-config-manager.html"
        "nexus-master-control-center.html"
        "nucleus-ai-training-system.html"
        "3d-mathematical-playground.html"
        "nexus-educational-bridge.html"
        "nexus-system-integration.html"
        "nexus-system-optimizer.js"
        "upgraded code pad.css"
    )

    for file in "${COMPONENT_FILES[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" $NEXUS_DIR/public/
            print_success "Deployed $file"
        else
            print_warning "$file not found - skipping"
        fi
    done

    echo ""
}

# Generate package.json
generate_package_json() {
    print_header "Generating Package Configuration"

    cat > $NEXUS_DIR/package.json << EOL
{
  "name": "nexus-forge-ecosystem",
  "version": "5.0.0",
  "description": "Complete Mathematical & AI Ecosystem - NEXUS Forge",
  "main": "src/server.js",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js",
    "test": "jest",
    "build": "webpack --mode=production",
    "deploy": "npm run build && npm start",
    "monitor": "node src/monitoring/system-monitor.js",
    "ai-train": "python3 src/ai/training/train_model.py",
    "math-test": "node src/math/test-engine.js",
    "webgl-test": "node src/webgl/test-renderer.js"
  },
  "keywords": [
    "mathematics",
    "ai",
    "webgl",
    "education",
    "visualization",
    "neural-networks",
    "geometry",
    "physics"
  ],
  "author": "NEXUS Forge Development Team",
  "license": "MIT",
  "dependencies": {
    "express": "^4.18.2",
    "socket.io": "^4.7.2",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "compression": "^1.7.4",
    "body-parser": "^1.20.2",
    "ws": "^8.13.0",
    "mathjs": "^11.11.0",
    "three": "^0.155.0",
    "tensorflow": "^4.10.0",
    "d3": "^7.8.5",
    "chart.js": "^4.4.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.6.2",
    "webpack": "^5.88.2",
    "webpack-cli": "^5.1.4",
    "@babel/core": "^7.22.9",
    "@babel/preset-env": "^7.22.9",
    "babel-loader": "^9.1.3",
    "css-loader": "^6.8.1",
    "html-webpack-plugin": "^5.5.3"
  },
  "engines": {
    "node": ">=16.0.0",
    "npm": ">=8.0.0"
  }
}
EOL

    print_success "Package.json generated"
    echo ""
}

# Generate main server file
generate_server() {
    print_header "Generating NEXUS Server"

    cat > $NEXUS_DIR/src/server.js << 'EOL'
/**
 * NEXUS Forge Main Server
 * Complete Mathematical & AI Ecosystem
 * Version 5.0.0
 */

const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const bodyParser = require('body-parser');

// Import NEXUS modules
const MathEngine = require('./math/engine');
const AICore = require('./ai/core');
const MonitoringSystem = require('./monitoring/system');
const CommunicationHub = require('./communication/hub');

class NexusServer {
    constructor() {
        this.app = express();
        this.server = http.createServer(this.app);
        this.io = socketIo(this.server, {
            cors: {
                origin: "*",
                methods: ["GET", "POST"]
            }
        });

        this.port = process.env.PORT || 8080;
        this.mathEngine = new MathEngine();
        this.aiCore = new AICore();
        this.monitoring = new MonitoringSystem();
        this.communicationHub = new CommunicationHub(this.io);

        this.initializeServer();
        this.setupRoutes();
        this.setupSocketHandlers();
    }

    initializeServer() {
        console.log('🌟 Initializing NEXUS Forge Server...');

        // Security middleware
        this.app.use(helmet({
            contentSecurityPolicy: false // Allow inline scripts for NEXUS components
        }));

        // Compression and parsing
        this.app.use(compression());
        this.app.use(bodyParser.json({ limit: '10mb' }));
        this.app.use(bodyParser.urlencoded({ extended: true }));

        // CORS for cross-origin requests
        this.app.use(cors());

        // Static files
        this.app.use(express.static(path.join(__dirname, '../public')));

        // Request logging
        this.app.use((req, res, next) => {
            console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
            next();
        });
    }

    setupRoutes() {
        console.log('🔗 Setting up API routes...');

        // Main launcher route
        this.app.get('/', (req, res) => {
            res.sendFile(path.join(__dirname, '../public/index.html'));
        });

        // API Routes
        this.app.get('/api/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                version: '5.0.0',
                components: {
                    mathEngine: this.mathEngine.getStatus(),
                    aiCore: this.aiCore.getStatus(),
                    monitoring: this.monitoring.getStatus()
                }
            });
        });

        // Mathematical API
        this.app.post('/api/math/calculate', async (req, res) => {
            try {
                const result = await this.mathEngine.calculate(req.body);
                res.json({ success: true, result });
            } catch (error) {
                res.status(400).json({ success: false, error: error.message });
            }
        });

        // AI Training API
        this.app.post('/api/ai/train', async (req, res) => {
            try {
                const result = await this.aiCore.train(req.body);
                res.json({ success: true, result });
            } catch (error) {
                res.status(400).json({ success: false, error: error.message });
            }
        });

        // System monitoring API
        this.app.get('/api/monitor/stats', (req, res) => {
            const stats = this.monitoring.getSystemStats();
            res.json(stats);
        });

        // Configuration API
        this.app.get('/api/config', (req, res) => {
            res.json(this.getSystemConfiguration());
        });

        this.app.post('/api/config', (req, res) => {
            try {
                this.updateSystemConfiguration(req.body);
                res.json({ success: true, message: 'Configuration updated' });
            } catch (error) {
                res.status(400).json({ success: false, error: error.message });
            }
        });

        // Error handling
        this.app.use((err, req, res, next) => {
            console.error('Server error:', err);
            res.status(500).json({
                success: false,
                error: 'Internal server error',
                timestamp: new Date().toISOString()
            });
        });

        // 404 handler
        this.app.use('*', (req, res) => {
            res.status(404).json({
                success: false,
                error: 'Endpoint not found',
                path: req.originalUrl
            });
        });
    }

    setupSocketHandlers() {
        console.log('🔌 Setting up WebSocket handlers...');

        this.io.on('connection', (socket) => {
            console.log(`Client connected: ${socket.id}`);

            // Send initial system status
            socket.emit('system_status', this.monitoring.getSystemStats());

            // Handle math calculation requests
            socket.on('math_calculate', async (data, callback) => {
                try {
                    const result = await this.mathEngine.calculate(data);
                    callback({ success: true, result });
                } catch (error) {
                    callback({ success: false, error: error.message });
                }
            });

            // Handle AI training requests
            socket.on('ai_train', async (data, callback) => {
                try {
                    const result = await this.aiCore.train(data);
                    callback({ success: true, result });
                } catch (error) {
                    callback({ success: false, error: error.message });
                }
            });

            // Handle system monitoring
            socket.on('monitor_subscribe', () => {
                socket.join('monitoring');
                console.log(`Client ${socket.id} subscribed to monitoring`);
            });

            socket.on('disconnect', () => {
                console.log(`Client disconnected: ${socket.id}`);
            });
        });

        // Broadcast system updates every 5 seconds
        setInterval(() => {
            this.io.to('monitoring').emit('system_update', this.monitoring.getSystemStats());
        }, 5000);
    }

    getSystemConfiguration() {
        return {
            server: {
                port: this.port,
                environment: process.env.NODE_ENV || 'development'
            },
            mathEngine: this.mathEngine.getConfiguration(),
            aiCore: this.aiCore.getConfiguration(),
            monitoring: this.monitoring.getConfiguration()
        };
    }

    updateSystemConfiguration(config) {
        if (config.mathEngine) {
            this.mathEngine.updateConfiguration(config.mathEngine);
        }
        if (config.aiCore) {
            this.aiCore.updateConfiguration(config.aiCore);
        }
        if (config.monitoring) {
            this.monitoring.updateConfiguration(config.monitoring);
        }
    }

    start() {
        this.server.listen(this.port, () => {
            console.log(`
🌟 NEXUS Forge Server Started Successfully!
============================================
🔗 Main Interface: http://localhost:${this.port}
📊 System Monitor: http://localhost:${this.port}/nexus-system-monitor.html
📈 Analytics: http://localhost:${this.port}/nexus-analytics-dashboard.html
⚙️ Configuration: http://localhost:${this.port}/nexus-config-manager.html
🎛️ Master Control: http://localhost:${this.port}/nexus-master-control-center.html

Version: 5.0.0
Environment: ${process.env.NODE_ENV || 'development'}
============================================
            `);
        });

        // Graceful shutdown
        process.on('SIGTERM', () => {
            console.log('🔄 Gracefully shutting down NEXUS server...');
            this.server.close(() => {
                console.log('✅ NEXUS server stopped');
                process.exit(0);
            });
        });
    }
}

// Start the server
if (require.main === module) {
    const nexusServer = new NexusServer();
    nexusServer.start();
}

module.exports = NexusServer;
EOL

    print_success "Main server generated"
    echo ""
}

# Generate configuration files
generate_config() {
    print_header "Generating Configuration Files"

    # Development config
    cat > $NEXUS_DIR/config/development/server.json << EOL
{
  "server": {
    "port": 8080,
    "host": "localhost",
    "cors": {
      "enabled": true,
      "origin": "*"
    },
    "logging": {
      "level": "debug",
      "file": "logs/development.log"
    }
  },
  "database": {
    "type": "sqlite",
    "path": "data/nexus-dev.db"
  },
  "ai": {
    "modelPath": "data/models/",
    "trainingEnabled": true,
    "batchSize": 32,
    "learningRate": 0.001
  },
  "math": {
    "precision": 8,
    "cacheEnabled": true,
    "maxCacheSize": "100MB"
  },
  "monitoring": {
    "enabled": true,
    "interval": 5000,
    "metricsRetention": "7d"
  }
}
EOL

    # Production config
    cat > $NEXUS_DIR/config/production/server.json << EOL
{
  "server": {
    "port": 8080,
    "host": "0.0.0.0",
    "cors": {
      "enabled": true,
      "origin": ["https://nexus-forge.com", "https://www.nexus-forge.com"]
    },
    "logging": {
      "level": "info",
      "file": "logs/production.log"
    },
    "ssl": {
      "enabled": true,
      "cert": "ssl/cert.pem",
      "key": "ssl/key.pem"
    }
  },
  "database": {
    "type": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "nexus_forge",
    "username": "nexus_user",
    "password": "secure_password"
  },
  "ai": {
    "modelPath": "data/models/",
    "trainingEnabled": true,
    "batchSize": 64,
    "learningRate": 0.0001,
    "gpuEnabled": true
  },
  "math": {
    "precision": 12,
    "cacheEnabled": true,
    "maxCacheSize": "1GB"
  },
  "monitoring": {
    "enabled": true,
    "interval": 1000,
    "metricsRetention": "30d",
    "alerting": {
      "enabled": true,
      "thresholds": {
        "cpu": 80,
        "memory": 85,
        "disk": 90
      }
    }
  }
}
EOL

    print_success "Configuration files generated"
    echo ""
}

# Generate README
generate_documentation() {
    print_header "Generating Documentation"

    cat > $NEXUS_DIR/README.md << 'EOL'
# 🌟 NEXUS Forge - Complete Mathematical & AI Ecosystem

Version 5.0.0 - Advanced Mathematical & AI Integration

## Overview

NEXUS Forge is a comprehensive mathematical and artificial intelligence ecosystem designed for education, research, and advanced computational tasks. It combines neural network training, 3D mathematical visualization, educational curricula, and real-time system monitoring into a unified platform.

## 🚀 Features

### Core Systems
- **🤖 AI Training Core**: Neural network training with pattern recognition
- **🔢 Mathematical Engine**: Advanced mathematical processing and visualization
- **🎮 3D WebGL Playground**: Hardware-accelerated 3D mathematical demonstrations
- **🎓 Educational Bridge**: Pythagorean curriculum integration
- **📊 System Monitor**: Real-time performance monitoring
- **⚙️ Configuration Manager**: Comprehensive system settings
- **📈 Analytics Dashboard**: Advanced data visualization

### Key Features
- **Real-time Communication**: WebSocket-based system integration
- **Cross-System Synchronization**: Unified data sharing across components
- **Advanced Monitoring**: Performance metrics and health tracking
- **Configurable Architecture**: Flexible system configuration
- **Educational Integration**: Guided learning sequences
- **3D Visualization**: Interactive mathematical demonstrations

## 🛠 Installation

### Prerequisites
- Node.js 16+
- Python 3.8+
- Modern web browser with WebGL support
- Git (optional, for version control)

### Quick Start
```bash
# Clone or extract NEXUS Forge
cd nexus-forge-ecosystem

# Install dependencies
npm install

# Start development server
npm run dev

# Or start production server
npm start
```

### Alternative Installation
```bash
# Using the deployment script
chmod +x deploy-nexus.sh
./deploy-nexus.sh
```

## 🎯 Usage

### Web Interface
1. Open http://localhost:8080 in your browser
2. Use the NEXUS Forge Launcher to access all components
3. Click on any system card to launch that component

### API Access
The NEXUS system provides REST and WebSocket APIs:

```javascript
// REST API Example
fetch('/api/math/calculate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        expression: '2 + 2 * sin(pi/4)',
        precision: 8
    })
});

// WebSocket Example
const socket = io();
socket.emit('math_calculate', { expression: '2^3' }, (result) => {
    console.log('Result:', result);
});
```

## 📊 System Components

### AI Training System
- **Neural Network Training**: Real-time pattern learning
- **Data Processing**: Mathematical pattern recognition
- **Model Management**: Save and load trained models
- **Performance Tracking**: Training progress monitoring

### Mathematical Engine
- **Expression Evaluation**: Complex mathematical expressions
- **Vector Operations**: 3D mathematical computations
- **Sacred Geometry**: Advanced geometric calculations
- **Caching System**: Optimized calculation storage

### 3D WebGL Playground
- **Interactive Visualization**: Real-time 3D mathematics
- **Shader Programs**: Custom visual effects
- **Educational Demos**: Mathematical concept visualization
- **Performance Optimization**: Hardware-accelerated rendering

### Educational Bridge
- **Curriculum Integration**: Pythagorean theorem lessons
- **Progress Tracking**: Student advancement monitoring
- **Interactive Learning**: Hands-on mathematical exploration
- **Assessment Tools**: Knowledge evaluation systems

## ⚙️ Configuration

### Environment Variables
```bash
NODE_ENV=development          # Environment mode
PORT=8080                    # Server port
LOG_LEVEL=debug              # Logging level
ENABLE_AI=true               # AI training features
ENABLE_WEBGL=true            # WebGL rendering
```

### Configuration Files
- `config/development/server.json`: Development settings
- `config/production/server.json`: Production settings
- `config/ai/models.json`: AI model configurations
- `config/math/engine.json`: Mathematical engine settings

## 🔧 Development

### Project Structure
```
nexus-forge-ecosystem/
├── public/                  # Web interface files
├── src/                     # Server source code
│   ├── ai/                  # AI training modules
│   ├── math/                # Mathematical engine
│   ├── webgl/               # WebGL components
│   └── monitoring/          # System monitoring
├── config/                  # Configuration files
├── data/                    # Data storage
├── logs/                    # System logs
└── tests/                   # Test suites
```

### Available Scripts
```bash
npm start              # Start production server
npm run dev            # Start development server
npm test               # Run test suite
npm run build          # Build for production
npm run monitor        # Start system monitoring
npm run ai-train       # Run AI training
npm run math-test      # Test mathematical engine
npm run webgl-test     # Test WebGL renderer
```

## 📈 Monitoring

### Real-Time Metrics
- **System Performance**: CPU, memory, GPU usage
- **AI Training Progress**: Loss, accuracy, convergence
- **Mathematical Operations**: Calculation performance
- **WebGL Rendering**: FPS, render time, GPU load
- **Network Communication**: Latency, throughput

### Health Checks
- **Component Status**: Online/offline monitoring
- **Error Tracking**: Real-time error detection
- **Performance Alerts**: Threshold-based notifications
- **Resource Usage**: System resource monitoring

## 🔒 Security

### Security Features
- **Helmet.js**: Security headers
- **CORS Protection**: Cross-origin request filtering
- **Input Validation**: Request sanitization
- **Rate Limiting**: API request throttling
- **SSL/TLS Support**: Encrypted communication

## 📚 API Documentation

### REST Endpoints
- `GET /api/health`: System health status
- `POST /api/math/calculate`: Mathematical calculations
- `POST /api/ai/train`: AI training requests
- `GET /api/monitor/stats`: System statistics
- `GET /api/config`: System configuration
- `POST /api/config`: Update configuration

### WebSocket Events
- `system_status`: Real-time system updates
- `math_calculate`: Mathematical operations
- `ai_train`: AI training events
- `monitor_subscribe`: Subscribe to monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Documentation**: Visit the built-in help system
- **Issues**: Report bugs via the issue tracker
- **Community**: Join the NEXUS Forge community
- **Email**: support@nexus-forge.com

## 🔮 Roadmap

### Version 5.1 (Planned)
- Enhanced AI model architectures
- Advanced mathematical visualizations
- Improved educational content
- Mobile interface optimization

### Version 5.2 (Future)
- Multi-user collaboration
- Cloud deployment support
- Advanced analytics features
- Extended API capabilities

---

**NEXUS Forge** - Bridging Mathematics, AI, and Education
🌟 *Empowering the next generation of mathematical and AI innovation*
EOL

    print_success "Documentation generated"
    echo ""
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"

    cd $NEXUS_DIR

    if command -v npm &> /dev/null; then
        print_status "Installing Node.js dependencies..."
        npm install
        if [ $? -eq 0 ]; then
            print_success "Node.js dependencies installed"
        else
            print_error "Failed to install Node.js dependencies"
        fi
    fi

    # Install Python dependencies
    if command -v pip3 &> /dev/null; then
        print_status "Installing Python dependencies..."
        cat > requirements.txt << EOL
tensorflow>=2.13.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
pandas>=1.3.0
flask>=2.0.0
websocket-client>=1.0.0
EOL
        pip3 install -r requirements.txt
        if [ $? -eq 0 ]; then
            print_success "Python dependencies installed"
        else
            print_warning "Some Python dependencies may have failed to install"
        fi
    fi

    cd ..
    echo ""
}

# Start services
start_services() {
    print_header "Starting NEXUS Services"

    cd $NEXUS_DIR

    # Start the main server in background
    print_status "Starting NEXUS Forge Server..."
    nohup npm start > logs/server.log 2>&1 &
    SERVER_PID=$!

    sleep 3

    # Check if server started successfully
    if ps -p $SERVER_PID > /dev/null; then
        print_success "NEXUS Server started (PID: $SERVER_PID)"
        echo $SERVER_PID > .server.pid
    else
        print_error "Failed to start NEXUS Server"
        return 1
    fi

    # Wait a moment for server to initialize
    sleep 2

    # Test server health
    if curl -s http://localhost:$WEB_PORT/api/health > /dev/null; then
        print_success "Server health check passed"
    else
        print_warning "Server health check failed - server may still be initializing"
    fi

    cd ..
    echo ""
}

# Generate startup script
generate_startup() {
    print_header "Generating Startup Scripts"

    # Linux/Mac startup script
    cat > $NEXUS_DIR/start-nexus.sh << EOL
#!/bin/bash
# NEXUS Forge Startup Script

cd "\$(dirname "\$0")"

echo "🌟 Starting NEXUS Forge Ecosystem..."

# Check if server is already running
if [ -f .server.pid ]; then
    PID=\$(cat .server.pid)
    if ps -p \$PID > /dev/null 2>&1; then
        echo "NEXUS Server is already running (PID: \$PID)"
        echo "Visit: http://localhost:8080"
        exit 0
    else
        rm .server.pid
    fi
fi

# Start the server
npm start &
echo \$! > .server.pid

echo ""
echo "🌟 NEXUS Forge Server Started!"
echo "=================================="
echo "🔗 Main Interface: http://localhost:8080"
echo "📊 System Monitor: http://localhost:8080/nexus-system-monitor.html"
echo "📈 Analytics: http://localhost:8080/nexus-analytics-dashboard.html"
echo "⚙️ Configuration: http://localhost:8080/nexus-config-manager.html"
echo ""
echo "To stop: ./stop-nexus.sh"
echo "To restart: ./restart-nexus.sh"
EOL

    # Stop script
    cat > $NEXUS_DIR/stop-nexus.sh << EOL
#!/bin/bash
# NEXUS Forge Stop Script

cd "\$(dirname "\$0")"

if [ -f .server.pid ]; then
    PID=\$(cat .server.pid)
    if ps -p \$PID > /dev/null 2>&1; then
        echo "Stopping NEXUS Server (PID: \$PID)..."
        kill \$PID
        rm .server.pid
        echo "✅ NEXUS Server stopped"
    else
        echo "NEXUS Server is not running"
        rm .server.pid
    fi
else
    echo "NEXUS Server is not running"
fi
EOL

    # Restart script
    cat > $NEXUS_DIR/restart-nexus.sh << EOL
#!/bin/bash
# NEXUS Forge Restart Script

cd "\$(dirname "\$0")"

echo "🔄 Restarting NEXUS Forge..."
./stop-nexus.sh
sleep 2
./start-nexus.sh
EOL

    # Make scripts executable
    chmod +x $NEXUS_DIR/start-nexus.sh
    chmod +x $NEXUS_DIR/stop-nexus.sh
    chmod +x $NEXUS_DIR/restart-nexus.sh

    # Windows batch file
    cat > $NEXUS_DIR/start-nexus.bat << EOL
@echo off
cd /d "%~dp0"

echo 🌟 Starting NEXUS Forge Ecosystem...

REM Check if server is already running
if exist .server.pid (
    echo NEXUS Server may already be running
    echo Check http://localhost:8080
)

REM Start the server
start /b npm start
echo Server started in background

echo.
echo 🌟 NEXUS Forge Server Started!
echo ==================================
echo 🔗 Main Interface: http://localhost:8080
echo 📊 System Monitor: http://localhost:8080/nexus-system-monitor.html
echo 📈 Analytics: http://localhost:8080/nexus-analytics-dashboard.html
echo ⚙️ Configuration: http://localhost:8080/nexus-config-manager.html
echo.
echo To stop: stop-nexus.bat
pause
EOL

    cat > $NEXUS_DIR/stop-nexus.bat << EOL
@echo off
echo Stopping NEXUS Server...
taskkill /f /im node.exe
echo ✅ NEXUS Server stopped
pause
EOL

    print_success "Startup scripts generated"
    echo ""
}

# Main deployment function
main() {
    echo -e "${CYAN}"
    cat << "EOL"
    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
    ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝

    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
EOL
    echo -e "${NC}"

    echo "Complete Mathematical & AI Ecosystem Deployment"
    echo "==============================================="
    echo ""

    # Run deployment steps
    check_requirements
    create_structure
    deploy_files
    generate_package_json
    generate_server
    generate_config
    generate_documentation
    install_dependencies
    generate_startup
    start_services

    # Final summary
    print_header "🎉 NEXUS Forge Deployment Complete!"
    echo ""
    echo -e "${GREEN}✅ All systems deployed and running!${NC}"
    echo ""
    echo -e "${BLUE}📋 Quick Access:${NC}"
    echo -e "   🔗 Main Interface:    ${CYAN}http://localhost:8080${NC}"
    echo -e "   📊 System Monitor:    ${CYAN}http://localhost:8080/nexus-system-monitor.html${NC}"
    echo -e "   📈 Analytics:         ${CYAN}http://localhost:8080/nexus-analytics-dashboard.html${NC}"
    echo -e "   ⚙️ Configuration:     ${CYAN}http://localhost:8080/nexus-config-manager.html${NC}"
    echo -e "   🎛️ Master Control:    ${CYAN}http://localhost:8080/nexus-master-control-center.html${NC}"
    echo ""
    echo -e "${BLUE}🛠 Management:${NC}"
    echo -e "   Start:    ${CYAN}cd $NEXUS_DIR && ./start-nexus.sh${NC}"
    echo -e "   Stop:     ${CYAN}cd $NEXUS_DIR && ./stop-nexus.sh${NC}"
    echo -e "   Restart:  ${CYAN}cd $NEXUS_DIR && ./restart-nexus.sh${NC}"
    echo -e "   Logs:     ${CYAN}cd $NEXUS_DIR && tail -f logs/server.log${NC}"
    echo ""
    echo -e "${YELLOW}📚 Documentation: ${CYAN}$NEXUS_DIR/README.md${NC}"
    echo ""
    echo -e "${GREEN}🌟 NEXUS Forge is ready for mathematical and AI exploration!${NC}"
    echo ""
}

# Run main function
main "$@"
