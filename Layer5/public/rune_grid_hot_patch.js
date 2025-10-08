/**
 * Rune Grid Hot-Patch System for NEXUS FORGE PRIMORDIAL
 * ====================================================
 *
 * Advanced file watching system with wire connections for real-time
 * AI pain detection updates. Features:
 * • Live file system monitoring with hot-patch functionality
 * • Wire connection system for data flow visualization
 * • Real-time synchronization with NEXUS FORGE AI intelligence
 * • Rune-based encoding for rapid pattern recognition
 * • Grid-based visual interface for file relationships
 * • Hot-reload capabilities for AI model updates
 */

class RuneGridHotPatchSystem {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.gridCanvas = null;
        this.gridCtx = null;
        this.wireCanvas = null;
        this.wireCtx = null;

        // Grid configuration
        this.gridSize = 20; // Grid cell size in pixels
        this.gridWidth = 32; // Grid width in cells
        this.gridHeight = 24; // Grid height in cells

        // File watching system
        this.watchedFiles = new Map();
        this.fileNodes = new Map(); // File -> grid position mapping
        this.activeWires = new Map(); // Wire connections between nodes
        this.hotPatchQueue = [];

        // Rune system for encoding file states
        this.runeSymbols = {
            // File type runes
            JS: '⚡', TS: '🔮', HTML: '🌐', CSS: '🎨', JSON: '📦',
            PY: '🐍', MD: '📝', TXT: '📄', LOG: '📊', CFG: '⚙️',
            // State runes
            WATCHING: '👁️', MODIFIED: '🔥', ERROR: '❌', SYNCED: '✅',
            PROCESSING: '🌀', CONNECTED: '🔗', DISCONNECTED: '💔'
        };

        // Connection types for wire visualization
        this.wireTypes = {
            DATA_FLOW: { color: '#00ff7f', width: 2, pattern: 'solid' },
            DEPENDENCY: { color: '#ffa502', width: 1, pattern: 'dashed' },
            ERROR_LINK: { color: '#ff4757', width: 3, pattern: 'pulse' },
            SYNC_LINE: { color: '#7cfccb', width: 1, pattern: 'dotted' },
            HOT_PATCH: { color: '#ff6b9d', width: 4, pattern: 'flow' }
        };

        // Performance tracking
        this.stats = {
            filesWatched: 0,
            wiresActive: 0,
            hotPatchesApplied: 0,
            syncEvents: 0,
            lastSyncTime: null
        };

        // Initialize the system
        this.initializeGrid();
        this.initializeWireSystem();
        this.setupEventHandlers();

        console.log('🔥 Rune Grid Hot-Patch System initialized');
    }

    initializeGrid() {
        // Create dual canvas system - grid background + wire overlay
        const canvasContainer = document.createElement('div');
        canvasContainer.style.cssText = `
            position: relative;
            width: ${this.gridWidth * this.gridSize}px;
            height: ${this.gridHeight * this.gridSize}px;
            background: #0a1a14;
            border: 1px solid #123126;
            border-radius: 8px;
            overflow: hidden;
        `;

        // Background grid canvas
        this.gridCanvas = document.createElement('canvas');
        this.gridCanvas.width = this.gridWidth * this.gridSize;
        this.gridCanvas.height = this.gridHeight * this.gridSize;
        this.gridCanvas.style.cssText = 'position: absolute; top: 0; left: 0; z-index: 1;';
        this.gridCtx = this.gridCanvas.getContext('2d');

        // Wire overlay canvas
        this.wireCanvas = document.createElement('canvas');
        this.wireCanvas.width = this.gridWidth * this.gridSize;
        this.wireCanvas.height = this.gridHeight * this.gridSize;
        this.wireCanvas.style.cssText = 'position: absolute; top: 0; left: 0; z-index: 2; pointer-events: none;';
        this.wireCtx = this.wireCanvas.getContext('2d');

        canvasContainer.appendChild(this.gridCanvas);
        canvasContainer.appendChild(this.wireCanvas);

        // Insert into container
        const existingContent = this.container.querySelector('.rune-grid-content');
        if (existingContent) {
            existingContent.appendChild(canvasContainer);
        } else {
            this.container.appendChild(canvasContainer);
        }

        // Draw initial grid
        this.drawGrid();

        console.log('✅ Rune Grid initialized:', {
            dimensions: `${this.gridWidth}x${this.gridHeight}`,
            cellSize: this.gridSize,
            totalCells: this.gridWidth * this.gridHeight
        });
    }

    drawGrid() {
        const ctx = this.gridCtx;
        ctx.clearRect(0, 0, this.gridCanvas.width, this.gridCanvas.height);

        // Background
        ctx.fillStyle = '#0a1a14';
        ctx.fillRect(0, 0, this.gridCanvas.width, this.gridCanvas.height);

        // Grid lines
        ctx.strokeStyle = '#123126';
        ctx.lineWidth = 0.5;
        ctx.globalAlpha = 0.3;

        // Vertical lines
        for (let x = 0; x <= this.gridWidth; x++) {
            ctx.beginPath();
            ctx.moveTo(x * this.gridSize, 0);
            ctx.lineTo(x * this.gridSize, this.gridCanvas.height);
            ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y <= this.gridHeight; y++) {
            ctx.beginPath();
            ctx.moveTo(0, y * this.gridSize);
            ctx.lineTo(this.gridCanvas.width, y * this.gridSize);
            ctx.stroke();
        }

        ctx.globalAlpha = 1.0;

        // Draw file nodes
        this.drawFileNodes();
    }

    drawFileNodes() {
        const ctx = this.gridCtx;

        this.fileNodes.forEach((position, filePath) => {
            const { x, y, rune, state, connections } = position;
            const pixelX = x * this.gridSize;
            const pixelY = y * this.gridSize;

            // Node background based on state
            const stateColors = {
                watching: '#28f49b20',
                modified: '#ff475740',
                error: '#ff475780',
                synced: '#26de8140',
                processing: '#ffa50240'
            };

            ctx.fillStyle = stateColors[state] || stateColors.watching;
            ctx.fillRect(pixelX + 1, pixelY + 1, this.gridSize - 2, this.gridSize - 2);

            // Node border
            ctx.strokeStyle = state === 'error' ? '#ff4757' :
                state === 'modified' ? '#ffa502' : '#28f49b';
            ctx.lineWidth = state === 'error' ? 2 : 1;
            ctx.strokeRect(pixelX + 1, pixelY + 1, this.gridSize - 2, this.gridSize - 2);

            // Rune symbol
            ctx.fillStyle = '#c9f7db';
            ctx.font = `${Math.floor(this.gridSize * 0.7)}px monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(
                rune,
                pixelX + this.gridSize / 2,
                pixelY + this.gridSize / 2
            );

            // Connection indicators
            if (connections > 0) {
                ctx.fillStyle = '#00ff7f';
                ctx.fillRect(pixelX + this.gridSize - 4, pixelY, 3, 3);
            }
        });
    }

    initializeWireSystem() {
        // Wire animation system
        this.wireAnimationFrame = 0;
        this.wireFlowOffset = 0;

        // Start wire animation loop
        this.animateWires();

        console.log('🔗 Wire system initialized');
    }

    animateWires() {
        this.wireCtx.clearRect(0, 0, this.wireCanvas.width, this.wireCanvas.height);
        this.wireFlowOffset = (this.wireFlowOffset + 0.5) % 20;

        // Draw all active wires
        this.activeWires.forEach((wire, wireId) => {
            this.drawWire(wire);
        });

        // Continue animation
        requestAnimationFrame(() => this.animateWires());
    }

    drawWire(wire) {
        const ctx = this.wireCtx;
        const { from, to, type, intensity = 1, data } = wire;
        const style = this.wireTypes[type] || this.wireTypes.DATA_FLOW;

        // Calculate pixel positions
        const fromX = from.x * this.gridSize + this.gridSize / 2;
        const fromY = from.y * this.gridSize + this.gridSize / 2;
        const toX = to.x * this.gridSize + this.gridSize / 2;
        const toY = to.y * this.gridSize + this.gridSize / 2;

        // Set wire style
        ctx.strokeStyle = style.color;
        ctx.lineWidth = style.width * intensity;
        ctx.globalAlpha = 0.8;

        // Draw based on pattern type
        switch (style.pattern) {
            case 'solid':
                this.drawSolidWire(ctx, fromX, fromY, toX, toY);
                break;
            case 'dashed':
                this.drawDashedWire(ctx, fromX, fromY, toX, toY);
                break;
            case 'dotted':
                this.drawDottedWire(ctx, fromX, fromY, toX, toY);
                break;
            case 'pulse':
                this.drawPulseWire(ctx, fromX, fromY, toX, toY);
                break;
            case 'flow':
                this.drawFlowWire(ctx, fromX, fromY, toX, toY);
                break;
        }

        // Draw data indicators for hot patches
        if (type === 'HOT_PATCH' && data) {
            this.drawDataPacket(ctx, fromX, fromY, toX, toY, data);
        }

        ctx.globalAlpha = 1.0;
    }

    drawSolidWire(ctx, x1, y1, x2, y2) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }

    drawDashedWire(ctx, x1, y1, x2, y2) {
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    drawDottedWire(ctx, x1, y1, x2, y2) {
        ctx.setLineDash([2, 6]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    drawPulseWire(ctx, x1, y1, x2, y2) {
        const pulse = Math.sin(this.wireAnimationFrame * 0.1) * 0.5 + 0.5;
        ctx.globalAlpha = 0.3 + pulse * 0.7;
        this.drawSolidWire(ctx, x1, y1, x2, y2);
    }

    drawFlowWire(ctx, x1, y1, x2, y2) {
        // Draw base wire
        ctx.globalAlpha = 0.4;
        this.drawSolidWire(ctx, x1, y1, x2, y2);

        // Draw flow particles
        const distance = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        const particles = Math.floor(distance / 10);

        ctx.globalAlpha = 0.9;
        ctx.fillStyle = ctx.strokeStyle;

        for (let i = 0; i < particles; i++) {
            const progress = ((i / particles) + (this.wireFlowOffset / 20)) % 1;
            const px = x1 + (x2 - x1) * progress;
            const py = y1 + (y2 - y1) * progress;

            ctx.beginPath();
            ctx.arc(px, py, 2, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    drawDataPacket(ctx, x1, y1, x2, y2, data) {
        // Animated data packet flowing along wire
        const progress = (this.wireFlowOffset / 20) % 1;
        const px = x1 + (x2 - x1) * progress;
        const py = y1 + (y2 - y1) * progress;

        // Packet background
        ctx.fillStyle = '#ff6b9d40';
        ctx.fillRect(px - 8, py - 4, 16, 8);

        // Packet border
        ctx.strokeStyle = '#ff6b9d';
        ctx.lineWidth = 1;
        ctx.strokeRect(px - 8, py - 4, 16, 8);

        // Data indicator
        ctx.fillStyle = '#ffffff';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('📦', px, py + 2);
    }

    setupEventHandlers() {
        // Grid click handler for manual connections
        this.gridCanvas.addEventListener('click', (e) => {
            const rect = this.gridCanvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) / this.gridSize);
            const y = Math.floor((e.clientY - rect.top) / this.gridSize);

            this.handleGridClick(x, y);
        });

        // Grid hover for node information
        this.gridCanvas.addEventListener('mousemove', (e) => {
            const rect = this.gridCanvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) / this.gridSize);
            const y = Math.floor((e.clientY - rect.top) / this.gridSize);

            this.handleGridHover(x, y);
        });

        console.log('👆 Grid event handlers ready');
    }

    handleGridClick(x, y) {
        // Find node at clicked position
        const clickedNode = this.findNodeAtPosition(x, y);

        if (clickedNode) {
            console.log(`🎯 Clicked node: ${clickedNode.path}`, {
                position: { x, y },
                state: clickedNode.state,
                connections: clickedNode.connections
            });

            // Toggle node state or create connection
            this.toggleNodeState(clickedNode);
        } else {
            // Empty grid cell - could be used for new file watching
            console.log(`📍 Empty grid cell clicked: (${x}, ${y})`);
        }
    }

    handleGridHover(x, y) {
        const node = this.findNodeAtPosition(x, y);

        if (node) {
            // Show node tooltip (could enhance with DOM tooltip)
            this.gridCanvas.title = `${node.path}\nState: ${node.state}\nConnections: ${node.connections}`;
        } else {
            this.gridCanvas.title = '';
        }
    }

    findNodeAtPosition(x, y) {
        for (const [filePath, position] of this.fileNodes) {
            if (position.x === x && position.y === y) {
                return { path: filePath, ...position };
            }
        }
        return null;
    }

    toggleNodeState(node) {
        const currentState = node.state;
        const newState = currentState === 'watching' ? 'processing' : 'watching';

        this.updateNodeState(node.path, newState);
        this.drawGrid(); // Redraw to show state change

        console.log(`🔄 Node state changed: ${node.path} → ${newState}`);
    }

    // Public API methods

    watchFile(filePath, options = {}) {
        const { x, y, rune, priority = 'normal' } = options;

        // Find available grid position if not specified
        const position = this.findAvailablePosition(x, y);
        if (!position) {
            console.warn(`⚠️ No available grid positions for ${filePath}`);
            return false;
        }

        // Determine rune based on file extension
        const extension = filePath.split('.').pop().toUpperCase();
        const fileRune = rune || this.runeSymbols[extension] || this.runeSymbols.TXT;

        // Add to watched files
        this.watchedFiles.set(filePath, {
            path: filePath,
            priority,
            startTime: Date.now(),
            lastModified: null,
            errorCount: 0,
            syncCount: 0
        });

        // Add to grid
        this.fileNodes.set(filePath, {
            x: position.x,
            y: position.y,
            rune: fileRune,
            state: 'watching',
            connections: 0,
            priority
        });

        this.stats.filesWatched++;
        this.drawGrid();

        console.log(`👁️ Now watching: ${filePath} at (${position.x}, ${position.y})`);
        return true;
    }

    unwatchFile(filePath) {
        if (this.watchedFiles.has(filePath)) {
            this.watchedFiles.delete(filePath);
            this.fileNodes.delete(filePath);
            this.removeWiresForFile(filePath);

            this.stats.filesWatched--;
            this.drawGrid();

            console.log(`👋 Stopped watching: ${filePath}`);
            return true;
        }
        return false;
    }

    updateNodeState(filePath, newState, metadata = {}) {
        const node = this.fileNodes.get(filePath);
        if (!node) return false;

        const oldState = node.state;
        node.state = newState;
        node.lastUpdated = Date.now();

        // Update stats
        if (newState === 'synced') this.stats.syncEvents++;
        if (newState === 'modified') this.triggerHotPatch(filePath, metadata);

        console.log(`🔄 ${filePath}: ${oldState} → ${newState}`);
        return true;
    }

    createWire(fromFile, toFile, wireType = 'DATA_FLOW', metadata = {}) {
        const fromNode = this.fileNodes.get(fromFile);
        const toNode = this.fileNodes.get(toFile);

        if (!fromNode || !toNode) {
            console.warn('❌ Cannot create wire: missing nodes');
            return false;
        }

        const wireId = `${fromFile}→${toFile}`;
        this.activeWires.set(wireId, {
            from: { x: fromNode.x, y: fromNode.y },
            to: { x: toNode.x, y: toNode.y },
            type: wireType,
            intensity: metadata.intensity || 1,
            data: metadata.data,
            created: Date.now()
        });

        // Update connection counts
        fromNode.connections++;
        toNode.connections++;

        this.stats.wiresActive++;

        console.log(`🔗 Wire created: ${fromFile} → ${toFile} (${wireType})`);
        return wireId;
    }

    removeWire(wireId) {
        if (this.activeWires.delete(wireId)) {
            this.stats.wiresActive--;
            console.log(`💔 Wire removed: ${wireId}`);
            return true;
        }
        return false;
    }

    removeWiresForFile(filePath) {
        const wiresToRemove = [];

        for (const [wireId, wire] of this.activeWires) {
            if (wireId.includes(filePath)) {
                wiresToRemove.push(wireId);
            }
        }

        wiresToRemove.forEach(wireId => this.removeWire(wireId));
    }

    triggerHotPatch(filePath, metadata) {
        const patch = {
            id: `patch_${Date.now()}`,
            file: filePath,
            timestamp: new Date().toISOString(),
            type: metadata.changeType || 'modification',
            data: metadata.data,
            priority: this.fileNodes.get(filePath)?.priority || 'normal'
        };

        this.hotPatchQueue.push(patch);
        this.stats.hotPatchesApplied++;

        // Create temporary hot-patch wire for visualization
        const targetFiles = this.findRelatedFiles(filePath);
        targetFiles.forEach(targetFile => {
            const wireId = this.createWire(filePath, targetFile, 'HOT_PATCH', {
                intensity: metadata.severity || 1,
                data: patch
            });

            // Auto-remove after animation
            setTimeout(() => this.removeWire(wireId), 3000);
        });

        console.log(`🔥 Hot patch triggered: ${filePath}`, patch);

        // Notify connected systems
        this.notifyHotPatch(patch);
        return patch;
    }

    findAvailablePosition(preferredX, preferredY) {
        // Use preferred position if available
        if (preferredX !== undefined && preferredY !== undefined) {
            if (this.isPositionFree(preferredX, preferredY)) {
                return { x: preferredX, y: preferredY };
            }
        }

        // Find first available position
        for (let y = 0; y < this.gridHeight; y++) {
            for (let x = 0; x < this.gridWidth; x++) {
                if (this.isPositionFree(x, y)) {
                    return { x, y };
                }
            }
        }

        return null; // Grid is full
    }

    isPositionFree(x, y) {
        for (const [filePath, position] of this.fileNodes) {
            if (position.x === x && position.y === y) {
                return false;
            }
        }
        return true;
    }

    findRelatedFiles(filePath) {
        // Simplified - would use actual dependency analysis
        const related = [];
        const extension = filePath.split('.').pop();

        // Find files of related types
        for (const [path] of this.fileNodes) {
            if (path !== filePath) {
                const pathExt = path.split('.').pop();
                if (this.areRelatedTypes(extension, pathExt)) {
                    related.push(path);
                }
            }
        }

        return related.slice(0, 3); // Limit connections
    }

    areRelatedTypes(ext1, ext2) {
        const relatedSets = [
            ['js', 'ts', 'jsx', 'tsx'],
            ['html', 'css', 'js'],
            ['py', 'pyi', 'pyc'],
            ['json', 'js', 'ts']
        ];

        return relatedSets.some(set =>
            set.includes(ext1.toLowerCase()) &&
            set.includes(ext2.toLowerCase())
        );
    }

    notifyHotPatch(patch) {
        // Integration point with NEXUS FORGE
        const event = new CustomEvent('rune-grid-hot-patch', {
            detail: patch
        });

        window.dispatchEvent(event);

        // Direct integration if available
        if (window.nexusForgeEngine) {
            window.nexusForgeEngine.processHotPatch(patch);
        }
    }

    getStats() {
        return {
            ...this.stats,
            lastSyncTime: this.stats.lastSyncTime || 'Never',
            uptime: Date.now() - (this.initTime || Date.now()),
            gridUtilization: (this.fileNodes.size / (this.gridWidth * this.gridHeight) * 100).toFixed(1) + '%'
        };
    }

    exportConfiguration() {
        return {
            version: '1.0',
            timestamp: new Date().toISOString(),
            gridConfig: {
                width: this.gridWidth,
                height: this.gridHeight,
                cellSize: this.gridSize
            },
            watchedFiles: Array.from(this.watchedFiles.entries()),
            fileNodes: Array.from(this.fileNodes.entries()),
            activeWires: Array.from(this.activeWires.entries()),
            stats: this.getStats()
        };
    }

    importConfiguration(config) {
        try {
            // Clear existing state
            this.watchedFiles.clear();
            this.fileNodes.clear();
            this.activeWires.clear();

            // Restore state
            config.watchedFiles.forEach(([path, data]) =>
                this.watchedFiles.set(path, data));

            config.fileNodes.forEach(([path, node]) =>
                this.fileNodes.set(path, node));

            config.activeWires.forEach(([id, wire]) =>
                this.activeWires.set(id, wire));

            // Update stats
            this.stats = { ...this.stats, ...config.stats };

            // Redraw
            this.drawGrid();

            console.log('📥 Configuration imported successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to import configuration:', error);
            return false;
        }
    }

    // Cleanup
    destroy() {
        // Remove event listeners
        this.gridCanvas?.removeEventListener('click', this.handleGridClick);
        this.gridCanvas?.removeEventListener('mousemove', this.handleGridHover);

        // Clear all data
        this.watchedFiles.clear();
        this.fileNodes.clear();
        this.activeWires.clear();

        console.log('🗑️ Rune Grid Hot-Patch System destroyed');
    }
}

// Integration utilities
window.RuneGridHotPatchSystem = RuneGridHotPatchSystem;
