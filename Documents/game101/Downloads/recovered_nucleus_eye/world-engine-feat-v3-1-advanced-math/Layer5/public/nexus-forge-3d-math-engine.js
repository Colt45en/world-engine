/**
 * NEXUS Forge 3D Mathematical Visualization Engine
 * Connects Pythagorean algorithms to real-time 3D graphics
 * Integrates with Nucleus AI for advanced pattern learning
 * Version: 4.0.0 - Advanced 3D Mathematics & AI Integration
 */

class NexusForge3DMathEngine {
    constructor() {
        this.version = "4.0.0";
        this.canvas = null;
        this.ctx = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;

        // 3D Mathematical systems
        this.vectorMath = new Vector3DMath();
        this.distanceCalculator = new DistanceCalculator();
        this.collisionDetector = new CollisionDetector();
        this.lightingCalculator = new LightingCalculator();
        this.pythagoreanSolver = new PythagoreanSolver();

        // Nucleus integration
        this.nucleusIntegration = null;
        this.learningMetrics = {
            calculationsPerformed: 0,
            patternsDetected: 0,
            optimizationsFound: 0,
            aiInsights: []
        };

        // Real-time visualization
        this.visualizationModes = {
            'distance-visualization': this.visualizeDistance.bind(this),
            'vector-operations': this.visualizeVectors.bind(this),
            'collision-detection': this.visualizeCollisions.bind(this),
            'lighting-calculations': this.visualizeLighting.bind(this),
            'pythagorean-proof': this.visualizePythagorean.bind(this)
        };

        console.log("🎯 NEXUS 3D Math Engine v4.0.0 - Initializing...");
        this.initialize();
    }

    async initialize() {
        this.setupCanvas();
        this.initializeWebGL();
        this.setupNucleusIntegration();
        this.startRealtimeVisualization();

        console.log("✅ 3D Math Engine fully operational");
    }

    setupCanvas() {
        // Create main 3D canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = 1200;
        this.canvas.height = 800;
        this.canvas.id = 'nexus-3d-math-canvas';
        this.canvas.style.cssText = `
            border: 2px solid #00ffff;
            border-radius: 10px;
            background: linear-gradient(135deg, #000011 0%, #001122 100%);
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
        `;

        // Get WebGL context
        this.gl = this.canvas.getContext('webgl2') || this.canvas.getContext('webgl');
        if (!this.gl) {
            console.error("❌ WebGL not supported");
            return;
        }

        console.log("✅ WebGL context created");
    }

    initializeWebGL() {
        const gl = this.gl;

        // Set viewport
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        // Enable depth testing
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);

        // Clear color
        gl.clearColor(0.0, 0.05, 0.1, 1.0);

        // Initialize shaders
        this.initializeShaders();

        // Initialize camera
        this.camera = {
            position: [0, 0, 10],
            target: [0, 0, 0],
            up: [0, 1, 0],
            fov: 45,
            aspect: this.canvas.width / this.canvas.height,
            near: 0.1,
            far: 1000.0
        };
    }

    initializeShaders() {
        const gl = this.gl;

        // Vertex shader for 3D math visualization
        const vertexShaderSource = `
            attribute vec4 aPosition;
            attribute vec3 aNormal;
            attribute vec4 aColor;

            uniform mat4 uProjectionMatrix;
            uniform mat4 uModelViewMatrix;
            uniform mat4 uNormalMatrix;

            varying vec3 vNormal;
            varying vec4 vColor;
            varying vec3 vPosition;

            void main() {
                gl_Position = uProjectionMatrix * uModelViewMatrix * aPosition;
                vNormal = normalize((uNormalMatrix * vec4(aNormal, 0.0)).xyz);
                vColor = aColor;
                vPosition = (uModelViewMatrix * aPosition).xyz;
            }
        `;

        // Fragment shader with mathematical lighting
        const fragmentShaderSource = `
            precision mediump float;

            varying vec3 vNormal;
            varying vec4 vColor;
            varying vec3 vPosition;

            uniform vec3 uLightPosition;
            uniform vec3 uLightColor;
            uniform float uTime;

            void main() {
                // Calculate distance to light (Pythagorean theorem in 3D)
                vec3 lightDir = normalize(uLightPosition - vPosition);
                float distance = length(uLightPosition - vPosition);

                // Inverse square law attenuation
                float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);

                // Dot product for diffuse lighting
                float diffuse = max(dot(vNormal, lightDir), 0.0);

                // Mathematical pulse based on time
                float pulse = 0.5 + 0.5 * sin(uTime * 2.0);

                // Combine lighting with mathematical visualization
                vec3 finalColor = vColor.rgb * uLightColor * diffuse * attenuation * pulse;

                gl_FragColor = vec4(finalColor, vColor.a);
            }
        `;

        this.shaderProgram = this.createShaderProgram(vertexShaderSource, fragmentShaderSource);
        this.setupShaderAttributes();
    }

    createShaderProgram(vertexSource, fragmentSource) {
        const gl = this.gl;

        const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentSource);

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('❌ Shader program failed to link:', gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    }

    compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);

        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('❌ Shader compilation error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    setupShaderAttributes() {
        const gl = this.gl;

        this.programInfo = {
            attributes: {
                position: gl.getAttribLocation(this.shaderProgram, 'aPosition'),
                normal: gl.getAttribLocation(this.shaderProgram, 'aNormal'),
                color: gl.getAttribLocation(this.shaderProgram, 'aColor')
            },
            uniforms: {
                projectionMatrix: gl.getUniformLocation(this.shaderProgram, 'uProjectionMatrix'),
                modelViewMatrix: gl.getUniformLocation(this.shaderProgram, 'uModelViewMatrix'),
                normalMatrix: gl.getUniformLocation(this.shaderProgram, 'uNormalMatrix'),
                lightPosition: gl.getUniformLocation(this.shaderProgram, 'uLightPosition'),
                lightColor: gl.getUniformLocation(this.shaderProgram, 'uLightColor'),
                time: gl.getUniformLocation(this.shaderProgram, 'uTime')
            }
        };
    }

    setupNucleusIntegration() {
        // Connect to Nucleus integration system
        if (window.nucleusIntegration) {
            this.nucleusIntegration = window.nucleusIntegration;
            console.log("🧠 Connected to Nucleus Integration");

            // Send initialization message
            this.nucleusIntegration.sendDirectMessage(
                "3D Mathematical Visualization Engine connected - Ready for advanced geometric learning",
                "3d-math-engine",
                "high"
            );
        } else {
            console.log("⚠️ Nucleus Integration not available - operating in standalone mode");
        }
    }

    // Core mathematical functions with Nucleus integration

    async calculateDistance3D(point1, point2) {
        const dx = point2[0] - point1[0];
        const dy = point2[1] - point1[1];
        const dz = point2[2] - point1[2];

        // Pythagorean theorem in 3D
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        // Send to Nucleus for learning
        if (this.nucleusIntegration) {
            await this.nucleusIntegration.trainDirectly({
                operation: 'distance-3d',
                input: { point1, point2 },
                calculation: { dx, dy, dz, squaredDistance: dx * dx + dy * dy + dz * dz },
                result: distance,
                timestamp: Date.now()
            }, 'geometric-calculation');
        }

        this.learningMetrics.calculationsPerformed++;
        return distance;
    }

    async calculateVectorMagnitude(vector) {
        const magnitude = Math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);

        // Send pattern to Nucleus
        if (this.nucleusIntegration) {
            await this.nucleusIntegration.trainDirectly({
                operation: 'vector-magnitude',
                input: vector,
                result: magnitude,
                pattern: 'pythagorean-3d',
                timestamp: Date.now()
            }, 'mathematical-patterns');
        }

        this.learningMetrics.calculationsPerformed++;
        return magnitude;
    }

    async normalizeVector(vector) {
        const magnitude = await this.calculateVectorMagnitude(vector);

        if (magnitude === 0) return [0, 0, 0];

        const normalized = [
            vector[0] / magnitude,
            vector[1] / magnitude,
            vector[2] / magnitude
        ];

        // Train Nucleus with normalization pattern
        if (this.nucleusIntegration) {
            await this.nucleusIntegration.trainDirectly({
                operation: 'vector-normalization',
                input: vector,
                magnitude: magnitude,
                result: normalized,
                verification: await this.calculateVectorMagnitude(normalized), // Should be ~1.0
                timestamp: Date.now()
            }, 'algebraic-operations');
        }

        return normalized;
    }

    async checkSphereCollision(center1, radius1, center2, radius2) {
        // Use squared distance to avoid expensive sqrt operation
        const dx = center2[0] - center1[0];
        const dy = center2[1] - center1[1];
        const dz = center2[2] - center1[2];
        const distanceSquared = dx * dx + dy * dy + dz * dz;
        const radiusSumSquared = (radius1 + radius2) * (radius1 + radius2);

        const collision = distanceSquared <= radiusSumSquared;

        // Send collision data to Nucleus
        if (this.nucleusIntegration) {
            await this.nucleusIntegration.trainDirectly({
                operation: 'sphere-collision',
                spheres: { center1, radius1, center2, radius2 },
                calculation: { distanceSquared, radiusSumSquared },
                result: collision,
                optimization: 'avoided-sqrt-operation',
                timestamp: Date.now()
            }, 'physics-simulation');
        }

        this.learningMetrics.calculationsPerformed++;
        return collision;
    }

    async calculateLightAttenuation(lightPos, targetPos, intensity = 1.0) {
        const distance = await this.calculateDistance3D(lightPos, targetPos);

        // Inverse square law with small epsilon to prevent divide by zero
        const attenuation = intensity / (distance * distance + 0.0001);

        // Send lighting calculation to Nucleus
        if (this.nucleusIntegration) {
            await this.nucleusIntegration.trainDirectly({
                operation: 'light-attenuation',
                input: { lightPos, targetPos, intensity },
                distance: distance,
                result: attenuation,
                physicsLaw: 'inverse-square-law',
                timestamp: Date.now()
            }, 'physics-simulation');
        }

        return attenuation;
    }

    // Advanced visualization methods

    startRealtimeVisualization() {
        const animate = (timestamp) => {
            this.renderFrame(timestamp);
            this.updateLearningMetrics();
            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
        console.log("🎬 Real-time visualization started");
    }

    renderFrame(timestamp) {
        const gl = this.gl;

        // Clear the canvas
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // Use shader program
        gl.useProgram(this.shaderProgram);

        // Set up matrices
        const projectionMatrix = this.createPerspectiveMatrix();
        const modelViewMatrix = this.createModelViewMatrix(timestamp);
        const normalMatrix = this.createNormalMatrix(modelViewMatrix);

        // Set uniforms
        gl.uniformMatrix4fv(this.programInfo.uniforms.projectionMatrix, false, projectionMatrix);
        gl.uniformMatrix4fv(this.programInfo.uniforms.modelViewMatrix, false, modelViewMatrix);
        gl.uniformMatrix4fv(this.programInfo.uniforms.normalMatrix, false, normalMatrix);

        // Animated light position using circular motion
        const lightPos = [
            5 * Math.cos(timestamp * 0.001),
            3 * Math.sin(timestamp * 0.0007),
            5 * Math.sin(timestamp * 0.001)
        ];

        gl.uniform3fv(this.programInfo.uniforms.lightPosition, lightPos);
        gl.uniform3f(this.programInfo.uniforms.lightColor, 0.0, 1.0, 1.0); // Cyan light
        gl.uniform1f(this.programInfo.uniforms.time, timestamp * 0.001);

        // Render mathematical visualizations
        this.renderPythagoreanDemo(timestamp);
        this.renderDistanceVisualization(timestamp);
        this.renderVectorOperations(timestamp);
    }

    renderPythagoreanDemo(timestamp) {
        // Create a 3D representation of the Pythagorean theorem
        // Right triangle with sides a, b, and hypotenuse c
        const a = 3.0;
        const b = 4.0;
        const c = Math.sqrt(a * a + b * b); // Should be 5.0

        // Send this calculation to Nucleus
        if (this.nucleusIntegration && Math.floor(timestamp) % 1000 < 16) {
            this.nucleusIntegration.trainDirectly({
                operation: 'pythagorean-theorem',
                sides: { a, b },
                hypotenuse: c,
                verification: Math.abs(c - 5.0) < 0.0001,
                visual: '3d-demonstration',
                timestamp: Date.now()
            }, 'mathematical-patterns');
        }

        // Render the triangle in 3D space
        this.renderLine([0, 0, 0], [a, 0, 0], [1, 0, 0, 1]); // Red - side a
        this.renderLine([a, 0, 0], [a, b, 0], [0, 1, 0, 1]); // Green - side b
        this.renderLine([a, b, 0], [0, 0, 0], [0, 0, 1, 1]); // Blue - hypotenuse c

        // Add text labels (conceptual)
        this.addMathematicalLabel([a / 2, -0.3, 0], `a = ${a}`);
        this.addMathematicalLabel([a + 0.3, b / 2, 0], `b = ${b}`);
        this.addMathematicalLabel([a / 2, b / 2, 0.3], `c = √(${a}² + ${b}²) = ${c.toFixed(2)}`);
    }

    renderDistanceVisualization(timestamp) {
        // Demonstrate distance calculation between moving points
        const point1 = [
            2 * Math.sin(timestamp * 0.001),
            2 * Math.cos(timestamp * 0.001),
            0
        ];

        const point2 = [
            -2 * Math.sin(timestamp * 0.0007),
            1,
            2 * Math.cos(timestamp * 0.0007)
        ];

        // Calculate distance and send to Nucleus periodically
        if (Math.floor(timestamp) % 500 < 16) {
            this.calculateDistance3D(point1, point2);
        }

        // Render points and connection line
        this.renderSphere(point1, 0.2, [1, 0, 1, 1]); // Magenta sphere
        this.renderSphere(point2, 0.2, [1, 1, 0, 1]); // Yellow sphere
        this.renderLine(point1, point2, [1, 1, 1, 0.5]); // White connection line
    }

    renderVectorOperations(timestamp) {
        // Demonstrate vector operations
        const origin = [0, 0, 0];
        const vector1 = [3, 1, 2];
        const vector2 = [1, 3, -1];

        // Calculate dot product
        const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2];

        // Send vector operations to Nucleus
        if (Math.floor(timestamp) % 800 < 16) {
            this.nucleusIntegration?.trainDirectly({
                operation: 'vector-dot-product',
                vectors: { vector1, vector2 },
                result: dotProduct,
                magnitude1: Math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1] + vector1[2] * vector1[2]),
                magnitude2: Math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1] + vector2[2] * vector2[2]),
                timestamp: Date.now()
            }, 'mathematical-patterns');
        }

        // Render vectors
        this.renderVector(origin, vector1, [1, 0, 0, 1]); // Red vector
        this.renderVector(origin, vector2, [0, 1, 0, 1]); // Green vector
    }

    // Utility rendering functions

    renderLine(start, end, color) {
        // Create line geometry
        const vertices = new Float32Array([
            start[0], start[1], start[2],
            end[0], end[1], end[2]
        ]);

        const colors = new Float32Array([
            color[0], color[1], color[2], color[3],
            color[0], color[1], color[2], color[3]
        ]);

        // Render the line
        this.renderGeometry(vertices, colors, this.gl.LINES);
    }

    renderSphere(center, radius, color) {
        // Simple sphere representation using multiple triangles
        const vertices = [];
        const colors = [];
        const segments = 16;

        for (let i = 0; i <= segments; i++) {
            const lat = Math.PI * (-0.5 + i / segments);
            const y = radius * Math.sin(lat);
            const radiusAtLat = radius * Math.cos(lat);

            for (let j = 0; j <= segments; j++) {
                const lon = 2 * Math.PI * j / segments;
                const x = radiusAtLat * Math.cos(lon);
                const z = radiusAtLat * Math.sin(lon);

                vertices.push(center[0] + x, center[1] + y, center[2] + z);
                colors.push(color[0], color[1], color[2], color[3]);
            }
        }

        this.renderGeometry(new Float32Array(vertices), new Float32Array(colors), this.gl.POINTS);
    }

    renderVector(origin, vector, color) {
        const end = [
            origin[0] + vector[0],
            origin[1] + vector[1],
            origin[2] + vector[2]
        ];

        this.renderLine(origin, end, color);

        // Add arrowhead
        const magnitude = Math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
        if (magnitude > 0) {
            const normalized = [vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude];
            const arrowSize = 0.3;

            // Simple arrowhead representation
            this.renderSphere(end, arrowSize * 0.5, color);
        }
    }

    renderGeometry(vertices, colors, mode) {
        const gl = this.gl;

        // Create and bind vertex buffer
        const vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

        // Enable position attribute
        gl.enableVertexAttribArray(this.programInfo.attributes.position);
        gl.vertexAttribPointer(this.programInfo.attributes.position, 3, gl.FLOAT, false, 0, 0);

        // Create and bind color buffer
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);

        // Enable color attribute
        gl.enableVertexAttribArray(this.programInfo.attributes.color);
        gl.vertexAttribPointer(this.programInfo.attributes.color, 4, gl.FLOAT, false, 0, 0);

        // Draw
        gl.drawArrays(mode, 0, vertices.length / 3);

        // Clean up
        gl.deleteBuffer(vertexBuffer);
        gl.deleteBuffer(colorBuffer);
    }

    addMathematicalLabel(position, text) {
        // Store label for overlay rendering (conceptual)
        if (!this.labels) this.labels = [];
        this.labels.push({ position, text, timestamp: Date.now() });
    }

    // Matrix operations for 3D transformations

    createPerspectiveMatrix() {
        const fov = this.camera.fov * Math.PI / 180;
        const aspect = this.camera.aspect;
        const near = this.camera.near;
        const far = this.camera.far;

        const f = 1.0 / Math.tan(fov / 2);

        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0
        ]);
    }

    createModelViewMatrix(timestamp) {
        // Simple rotation around Y axis
        const angle = timestamp * 0.0005;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        return new Float32Array([
            cos, 0, sin, 0,
            0, 1, 0, 0,
            -sin, 0, cos, 0,
            0, 0, -this.camera.position[2], 1
        ]);
    }

    createNormalMatrix(modelViewMatrix) {
        // For simplicity, we'll use the upper 3x3 of the model-view matrix
        const m = modelViewMatrix;
        return new Float32Array([
            m[0], m[1], m[2], 0,
            m[4], m[5], m[6], 0,
            m[8], m[9], m[10], 0,
            0, 0, 0, 1
        ]);
    }

    updateLearningMetrics() {
        // Send periodic analytics to Nucleus
        if (this.nucleusIntegration && this.learningMetrics.calculationsPerformed % 100 === 0) {
            this.nucleusIntegration.sendDirectMessage(
                `3D Math Engine Analytics: ${this.learningMetrics.calculationsPerformed} calculations performed`,
                '3d-analytics',
                'normal'
            );
        }
    }

    // Public API methods

    attachToElement(elementId) {
        const container = document.getElementById(elementId);
        if (container) {
            container.appendChild(this.canvas);
            console.log(`✅ 3D Math Engine attached to ${elementId}`);
        }
    }

    setVisualizationMode(mode) {
        if (this.visualizationModes[mode]) {
            this.currentMode = mode;
            console.log(`🎯 Visualization mode set to: ${mode}`);
        }
    }

    getMetrics() {
        return { ...this.learningMetrics };
    }

    exportLearningData() {
        return {
            version: this.version,
            metrics: this.learningMetrics,
            timestamp: Date.now(),
            camera: this.camera
        };
    }
}

// Supporting mathematical classes

class Vector3DMath {
    static add(v1, v2) {
        return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]];
    }

    static subtract(v1, v2) {
        return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]];
    }

    static dot(v1, v2) {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    static cross(v1, v2) {
        return [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ];
    }
}

class DistanceCalculator {
    static euclidean3D(p1, p2) {
        const dx = p2[0] - p1[0];
        const dy = p2[1] - p1[1];
        const dz = p2[2] - p1[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    static manhattan3D(p1, p2) {
        return Math.abs(p2[0] - p1[0]) + Math.abs(p2[1] - p1[1]) + Math.abs(p2[2] - p1[2]);
    }
}

class CollisionDetector {
    static sphereSphere(center1, radius1, center2, radius2) {
        const dx = center2[0] - center1[0];
        const dy = center2[1] - center1[1];
        const dz = center2[2] - center1[2];
        const distanceSquared = dx * dx + dy * dy + dz * dz;
        const radiusSum = radius1 + radius2;
        return distanceSquared <= radiusSum * radiusSum;
    }
}

class LightingCalculator {
    static inverseSquareAttenuation(distance, intensity = 1.0) {
        return intensity / (distance * distance + 0.0001);
    }

    static lambertian(normal, lightDir) {
        return Math.max(Vector3DMath.dot(normal, lightDir), 0.0);
    }
}

class PythagoreanSolver {
    static solveForHypotenuse(a, b) {
        return Math.sqrt(a * a + b * b);
    }

    static solveForSide(hypotenuse, knownSide) {
        return Math.sqrt(hypotenuse * hypotenuse - knownSide * knownSide);
    }

    static verify(a, b, c) {
        return Math.abs(c * c - (a * a + b * b)) < 0.0001;
    }
}

// Global initialization
let nexus3DMathEngine = null;

window.addEventListener('load', async () => {
    console.log("🚀 Initializing NEXUS 3D Mathematical Engine...");
    nexus3DMathEngine = new NexusForge3DMathEngine();

    // Make available globally
    window.nexus3DMathEngine = nexus3DMathEngine;

    // Auto-attach if container exists
    if (document.getElementById('math-visualization-container')) {
        nexus3DMathEngine.attachToElement('math-visualization-container');
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForge3DMathEngine;
}
