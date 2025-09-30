/**
 * Advanced Line Manager for Quantum Graphics Engine
 * ================================================
 *
 * Integrates sophisticated line drawing capabilities with:
 * • Click-to-connect mode for linking pain clusters
 * • Drag-draw mode for freeform connections
 * • Pin-draw mode for continuous path creation
 * • SVG-based overlay system with quantum styling
 * • Export/import functionality for connection persistence
 * • Integration with NEXUS FORGE AI pain detection
 */

class QuantumLineManager {
    constructor(canvasElement, svgOverlay = null) {
        // Core elements
        this.canvas = canvasElement;
        this.container = canvasElement.parentElement;

        // Create or use existing SVG overlay
        this.svg = svgOverlay || this.createSVGOverlay();

        // Line drawing states
        this.isDrawing = false;
        this.startPoint = null;
        this.currentLine = null;
        this.tempMarker = null;

        // Line storage and management
        this.lines = [];
        this.connections = new Map(); // For tracking object connections

        // Drawing modes
        this.modes = {
            CLICK_TO_CONNECT: 'click',
            DRAG_DRAW: 'drag',
            PIN_DRAW: 'pin',
            PAIN_CONNECT: 'pain' // Special mode for AI pain connections
        };
        this.currentMode = this.modes.CLICK_TO_CONNECT;

        // Quantum styling
        this.lineStyle = {
            stroke: '#00ff7f',
            strokeWidth: 2,
            opacity: 0.8,
            glowEffect: true
        };

        // Pain connection styling
        this.painStyles = {
            high: { stroke: '#ff4757', strokeWidth: 3, opacity: 0.9 },
            medium: { stroke: '#ffa502', strokeWidth: 2, opacity: 0.8 },
            low: { stroke: '#26de81', strokeWidth: 2, opacity: 0.7 }
        };

        // Initialize event listeners
        this.setupEventListeners();

        console.log('🌌 Quantum Line Manager initialized with advanced connection system');
    }

    createSVGOverlay() {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.style.position = 'absolute';
        svg.style.top = '0';
        svg.style.left = '0';
        svg.style.pointerEvents = 'none';
        svg.style.zIndex = '1000';

        // Add quantum glow filter
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'quantum-glow');
        filter.setAttribute('x', '-50%');
        filter.setAttribute('y', '-50%');
        filter.setAttribute('width', '200%');
        filter.setAttribute('height', '200%');

        const glow = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
        glow.setAttribute('stdDeviation', '3');
        glow.setAttribute('result', 'coloredBlur');

        const merge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
        const mergeNode1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
        mergeNode1.setAttribute('in', 'coloredBlur');
        const mergeNode2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
        mergeNode2.setAttribute('in', 'SourceGraphic');

        merge.appendChild(mergeNode1);
        merge.appendChild(mergeNode2);
        filter.appendChild(glow);
        filter.appendChild(merge);
        defs.appendChild(filter);
        svg.appendChild(defs);

        this.container.appendChild(svg);
        return svg;
    }

    // Core line creation method with quantum styling
    createLine(x1, y1, x2, y2, style = null) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');

        // Set line attributes
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);

        // Apply styling
        const lineStyle = style || this.lineStyle;
        line.setAttribute('stroke', lineStyle.stroke);
        line.setAttribute('stroke-width', lineStyle.strokeWidth);
        line.setAttribute('opacity', lineStyle.opacity);

        // Add quantum glow effect
        if (lineStyle.glowEffect || this.lineStyle.glowEffect) {
            line.setAttribute('filter', 'url(#quantum-glow)');
        }

        // Add animation for new lines
        line.style.strokeDasharray = '1000';
        line.style.strokeDashoffset = '1000';
        line.style.animation = 'quantumLineDraw 0.5s ease-out forwards';

        // Add CSS animation if not already added
        if (!document.getElementById('quantum-line-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'quantum-line-styles';
            styleSheet.textContent = `
                @keyframes quantumLineDraw {
                    to {
                        stroke-dashoffset: 0;
                    }
                }
                .quantum-line-pulse {
                    animation: quantumPulse 2s ease-in-out infinite;
                }
                @keyframes quantumPulse {
                    0%, 100% { opacity: 0.7; }
                    50% { opacity: 1; }
                }
            `;
            document.head.appendChild(styleSheet);
        }

        // Add interactivity
        this.addLineInteractivity(line);

        return line;
    }

    // Event setup for different drawing modes
    setupEventListeners() {
        // Use pointer events for better touch support
        this.container.addEventListener('pointerdown', this.handlePointerDown.bind(this));
        this.container.addEventListener('pointermove', this.handlePointerMove.bind(this));
        this.container.addEventListener('pointerup', this.handlePointerUp.bind(this));

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
    }

    handleKeyDown(event) {
        switch (event.key.toLowerCase()) {
            case 'c':
                if (event.ctrlKey || event.metaKey) return; // Don't interfere with copy
                this.setDrawingMode(this.modes.CLICK_TO_CONNECT);
                break;
            case 'd':
                this.setDrawingMode(this.modes.DRAG_DRAW);
                break;
            case 'p':
                this.setDrawingMode(this.modes.PIN_DRAW);
                break;
            case 'a':
                this.setDrawingMode(this.modes.PAIN_CONNECT);
                break;
            case 'escape':
                this.cancelCurrentDrawing();
                break;
        }
    }

    // Pointer interaction handlers
    handlePointerDown(event) {
        if (event.target !== this.canvas && !event.target.closest('.quantum-connectable')) {
            return;
        }

        event.preventDefault();
        const { x, y } = this.getRelativeCoordinates(event);

        switch (this.currentMode) {
            case this.modes.CLICK_TO_CONNECT:
                this.handleClickToConnect(x, y, event.target);
                break;
            case this.modes.DRAG_DRAW:
                this.startDragDraw(x, y);
                break;
            case this.modes.PIN_DRAW:
                this.startPinDraw(x, y);
                break;
            case this.modes.PAIN_CONNECT:
                this.handlePainConnect(x, y, event.target);
                break;
        }
    }

    handlePointerMove(event) {
        if (!this.isDrawing) return;

        const { x, y } = this.getRelativeCoordinates(event);

        switch (this.currentMode) {
            case this.modes.DRAG_DRAW:
                this.updateDragLine(x, y);
                break;
            case this.modes.PIN_DRAW:
                this.continuePinDraw(x, y);
                break;
        }
    }

    handlePointerUp(event) {
        if (!this.isDrawing && this.currentMode !== this.modes.CLICK_TO_CONNECT) return;

        const { x, y } = this.getRelativeCoordinates(event);

        switch (this.currentMode) {
            case this.modes.CLICK_TO_CONNECT:
                this.finishClickConnect(x, y, event.target);
                break;
            case this.modes.DRAG_DRAW:
                this.finishDragLine(x, y);
                break;
            case this.modes.PIN_DRAW:
                // Pin draw continues until cancelled
                break;
            case this.modes.PAIN_CONNECT:
                this.finishPainConnect(x, y, event.target);
                break;
        }
    }

    // Click-to-Connect Mode (Enhanced)
    handleClickToConnect(x, y, target) {
        if (!this.startPoint) {
            // First point of connection
            this.startPoint = { x, y, target };
            this.createTemporaryMarker(x, y);
            console.log('🔗 Connection started at:', { x, y });
        }
    }

    finishClickConnect(x, y, target) {
        if (this.startPoint) {
            // Create line between two clicked points
            const line = this.createLine(
                this.startPoint.x,
                this.startPoint.y,
                x,
                y
            );

            this.svg.appendChild(line);
            this.lines.push({
                element: line,
                start: { ...this.startPoint },
                end: { x, y, target },
                type: 'connection',
                metadata: {
                    created: Date.now(),
                    mode: this.currentMode
                }
            });

            // Track object connections if targets are available
            if (this.startPoint.target && target &&
                this.startPoint.target.dataset && target.dataset) {
                this.connections.set(
                    `${this.startPoint.target.dataset.id}-${target.dataset.id}`,
                    { start: this.startPoint.target, end: target, line }
                );
            }

            // Reset drawing state
            this.clearTemporaryMarker();
            this.startPoint = null;

            console.log('🔗 Connection completed');
        }
    }

    // Pain Connection Mode (AI Integration)
    handlePainConnect(x, y, target) {
        // Special handling for pain cluster connections
        const painElement = target.closest('.cluster-item, .pain-node');
        if (painElement) {
            const painLevel = this.getPainLevelFromElement(painElement);
            const style = this.painStyles[painLevel] || this.painStyles.low;

            if (!this.startPoint) {
                this.startPoint = { x, y, target: painElement, painLevel };
                this.createTemporaryMarker(x, y, style.stroke);
            }
        } else if (!this.startPoint) {
            this.startPoint = { x, y, target };
            this.createTemporaryMarker(x, y);
        }
    }

    finishPainConnect(x, y, target) {
        if (this.startPoint) {
            const endPainElement = target.closest('.cluster-item, .pain-node');
            const painLevel = this.startPoint.painLevel || 'low';
            const style = this.painStyles[painLevel] || this.painStyles.low;

            const line = this.createLine(
                this.startPoint.x,
                this.startPoint.y,
                x,
                y,
                style
            );

            // Add pain connection specific styling
            line.classList.add('pain-connection', `pain-${painLevel}`);

            this.svg.appendChild(line);
            this.lines.push({
                element: line,
                start: this.startPoint,
                end: { x, y, target: endPainElement },
                type: 'pain-connection',
                painLevel: painLevel,
                metadata: {
                    created: Date.now(),
                    mode: this.currentMode
                }
            });

            this.clearTemporaryMarker();
            this.startPoint = null;

            console.log(`🔥 Pain connection created (${painLevel} level)`);
        }
    }

    getPainLevelFromElement(element) {
        if (element.classList.contains('high-pain')) return 'high';
        if (element.classList.contains('medium-pain')) return 'medium';
        return 'low';
    }

    // Drag Draw Mode
    startDragDraw(x, y) {
        this.isDrawing = true;
        this.startPoint = { x, y };

        // Create temporary line
        this.currentLine = this.createLine(x, y, x, y);
        this.svg.appendChild(this.currentLine);
    }

    updateDragLine(x, y) {
        if (this.currentLine) {
            this.currentLine.setAttribute('x2', x);
            this.currentLine.setAttribute('y2', y);
        }
    }

    finishDragLine(x, y) {
        this.isDrawing = false;
        if (this.currentLine) {
            this.lines.push({
                element: this.currentLine,
                start: this.startPoint,
                end: { x, y },
                type: 'freeform',
                metadata: {
                    created: Date.now(),
                    mode: this.currentMode
                }
            });
        }
        this.currentLine = null;
    }

    // Pin Draw Mode (continuous line drawing)
    startPinDraw(x, y) {
        this.isDrawing = true;
        this.startPoint = { x, y };

        // Start a new path for continuous drawing
        this.currentLine = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        this.currentLine.setAttribute('points', `${x},${y}`);
        this.currentLine.setAttribute('fill', 'none');
        this.currentLine.setAttribute('stroke', this.lineStyle.stroke);
        this.currentLine.setAttribute('stroke-width', this.lineStyle.strokeWidth);
        this.currentLine.setAttribute('opacity', this.lineStyle.opacity);

        if (this.lineStyle.glowEffect) {
            this.currentLine.setAttribute('filter', 'url(#quantum-glow)');
        }

        this.svg.appendChild(this.currentLine);
    }

    continuePinDraw(x, y) {
        if (this.isDrawing && this.currentLine) {
            const currentPoints = this.currentLine.getAttribute('points');
            this.currentLine.setAttribute('points', `${currentPoints} ${x},${y}`);
        }
    }

    finishPinDraw() {
        this.isDrawing = false;
        if (this.currentLine) {
            this.lines.push({
                element: this.currentLine,
                type: 'path',
                metadata: {
                    created: Date.now(),
                    mode: this.currentMode
                }
            });
        }
        this.currentLine = null;
    }

    // Utility Methods
    getRelativeCoordinates(event) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }

    createTemporaryMarker(x, y, color = '#00ff7f') {
        this.clearTemporaryMarker();

        this.tempMarker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        this.tempMarker.setAttribute('cx', x);
        this.tempMarker.setAttribute('cy', y);
        this.tempMarker.setAttribute('r', 6);
        this.tempMarker.setAttribute('fill', color);
        this.tempMarker.setAttribute('opacity', '0.8');
        this.tempMarker.setAttribute('filter', 'url(#quantum-glow)');
        this.tempMarker.classList.add('quantum-line-pulse');

        this.svg.appendChild(this.tempMarker);
    }

    clearTemporaryMarker() {
        if (this.tempMarker) {
            this.svg.removeChild(this.tempMarker);
            this.tempMarker = null;
        }
    }

    cancelCurrentDrawing() {
        this.isDrawing = false;
        this.clearTemporaryMarker();

        if (this.currentLine) {
            this.svg.removeChild(this.currentLine);
            this.currentLine = null;
        }

        this.startPoint = null;
        console.log('🚫 Drawing cancelled');
    }

    // Line Interactivity
    addLineInteractivity(line) {
        line.style.pointerEvents = 'stroke';
        line.style.strokeWidth = parseInt(line.getAttribute('stroke-width')) + 4; // Invisible hit area

        line.addEventListener('pointerenter', () => {
            line.style.filter = 'url(#quantum-glow) brightness(1.3)';
        });

        line.addEventListener('pointerleave', () => {
            line.style.filter = 'url(#quantum-glow)';
        });

        line.addEventListener('click', (event) => {
            event.stopPropagation();
            this.selectLine(line);
        });

        line.addEventListener('dblclick', (event) => {
            event.stopPropagation();
            this.deleteLine(line);
        });
    }

    selectLine(lineElement) {
        // Clear previous selections
        this.lines.forEach(line => {
            line.element.classList.remove('selected');
        });

        // Select this line
        lineElement.classList.add('selected');
        lineElement.style.strokeWidth = parseInt(lineElement.getAttribute('stroke-width')) + 1;

        console.log('🔍 Line selected');
    }

    deleteLine(lineElement) {
        const lineIndex = this.lines.findIndex(line => line.element === lineElement);
        if (lineIndex !== -1) {
            this.svg.removeChild(lineElement);
            this.lines.splice(lineIndex, 1);
            console.log('🗑️ Line deleted');
        }
    }

    // Mode Switching
    setDrawingMode(mode) {
        this.cancelCurrentDrawing();
        this.currentMode = mode;

        // Update UI indicators
        document.dispatchEvent(new CustomEvent('quantumLineMode', {
            detail: { mode, availableModes: this.modes }
        }));

        console.log(`🎨 Drawing mode changed to: ${mode}`);
    }

    // Line Management
    clearAllLines() {
        this.lines.forEach(line => {
            if (line.element.parentNode) {
                this.svg.removeChild(line.element);
            }
        });
        this.lines = [];
        this.connections.clear();
        console.log('🧹 All lines cleared');
    }

    // Export and Import Lines
    exportLines() {
        return {
            lines: this.lines.map(line => ({
                type: line.type,
                start: line.start,
                end: line.end,
                painLevel: line.painLevel,
                metadata: line.metadata,
                elementData: this.getElementData(line.element)
            })),
            connections: Array.from(this.connections.entries())
        };
    }

    getElementData(element) {
        if (element.tagName === 'line') {
            return {
                type: 'line',
                x1: element.getAttribute('x1'),
                y1: element.getAttribute('y1'),
                x2: element.getAttribute('x2'),
                y2: element.getAttribute('y2'),
                stroke: element.getAttribute('stroke'),
                strokeWidth: element.getAttribute('stroke-width')
            };
        } else if (element.tagName === 'polyline') {
            return {
                type: 'polyline',
                points: element.getAttribute('points'),
                stroke: element.getAttribute('stroke'),
                strokeWidth: element.getAttribute('stroke-width')
            };
        }
    }

    importLines(lineData) {
        this.clearAllLines();

        lineData.lines.forEach(data => {
            let importedLine;

            if (data.elementData.type === 'line') {
                const ed = data.elementData;
                importedLine = this.createLine(ed.x1, ed.y1, ed.x2, ed.y2, {
                    stroke: ed.stroke,
                    strokeWidth: ed.strokeWidth,
                    opacity: this.lineStyle.opacity
                });
            } else if (data.elementData.type === 'polyline') {
                importedLine = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
                importedLine.setAttribute('points', data.elementData.points);
                importedLine.setAttribute('fill', 'none');
                importedLine.setAttribute('stroke', data.elementData.stroke);
                importedLine.setAttribute('stroke-width', data.elementData.strokeWidth);
                this.addLineInteractivity(importedLine);
            }

            if (importedLine) {
                this.svg.appendChild(importedLine);
                this.lines.push({
                    element: importedLine,
                    start: data.start,
                    end: data.end,
                    type: data.type,
                    painLevel: data.painLevel,
                    metadata: data.metadata
                });
            }
        });

        console.log(`📥 Imported ${lineData.lines.length} lines`);
    }

    // AI Integration Methods
    connectPainClusters(cluster1, cluster2, painLevel = 'medium') {
        const rect1 = cluster1.getBoundingClientRect();
        const rect2 = cluster2.getBoundingClientRect();
        const canvasRect = this.canvas.getBoundingClientRect();

        const start = {
            x: rect1.left + rect1.width / 2 - canvasRect.left,
            y: rect1.top + rect1.height / 2 - canvasRect.top
        };

        const end = {
            x: rect2.left + rect2.width / 2 - canvasRect.left,
            y: rect2.top + rect2.height / 2 - canvasRect.top
        };

        const style = this.painStyles[painLevel];
        const line = this.createLine(start.x, start.y, end.x, end.y, style);
        line.classList.add('auto-pain-connection', `pain-${painLevel}`);

        this.svg.appendChild(line);
        this.lines.push({
            element: line,
            start: { ...start, target: cluster1 },
            end: { ...end, target: cluster2 },
            type: 'auto-pain-connection',
            painLevel: painLevel,
            metadata: {
                created: Date.now(),
                automated: true
            }
        });

        console.log(`🤖 Auto-connected pain clusters (${painLevel} level)`);
    }

    // Visual feedback methods
    highlightConnection(connectionId, duration = 2000) {
        const connection = this.connections.get(connectionId);
        if (connection && connection.line) {
            connection.line.style.strokeWidth = '4';
            connection.line.style.filter = 'url(#quantum-glow) brightness(1.5)';

            setTimeout(() => {
                connection.line.style.strokeWidth = '2';
                connection.line.style.filter = 'url(#quantum-glow)';
            }, duration);
        }
    }

    pulseAllPainConnections() {
        this.lines
            .filter(line => line.type.includes('pain'))
            .forEach(line => {
                line.element.classList.add('quantum-line-pulse');
                setTimeout(() => {
                    line.element.classList.remove('quantum-line-pulse');
                }, 2000);
            });
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantumLineManager };
}

// Make available globally for integration
window.QuantumLineManager = QuantumLineManager;
