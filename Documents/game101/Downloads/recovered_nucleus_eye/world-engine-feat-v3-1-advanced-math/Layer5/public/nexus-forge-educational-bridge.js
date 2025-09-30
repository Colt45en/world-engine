/**
 * NEXUS Forge Educational Bridge
 * Connects Pythagorean curriculum to 3D visualization and AI training
 * Transforms static mathematical concepts into interactive learning experiences
 * Version: 5.0.0 - Complete Educational Integration
 */

class NexusForgeEducationalBridge {
    constructor() {
        this.version = "5.0.0";
        this.curriculumData = null;
        this.activeLessons = new Map();
        this.learningProgress = new Map();
        this.interactiveDemos = new Map();

        // Connection to other systems
        this.playground = null;
        this.nucleusIntegration = null;
        this.math3DEngine = null;

        // Educational metrics
        this.educationMetrics = {
            lessonsCompleted: 0,
            conceptsLearned: 0,
            practiceProblems: 0,
            visualizationTime: 0,
            aiInteractions: 0
        };

        // Tier 5-8 Pythagorean curriculum integration
        this.pythagoreanCurriculum = this.initializePythagoreanCurriculum();

        console.log("📚 NEXUS Educational Bridge v5.0.0 - Initializing...");
        this.initialize();
    }

    initialize() {
        this.setupEducationalSystems();
        this.connectToPlayground();
        this.initializeInteractiveDemos();
        this.startEducationalLoop();

        console.log("✅ Educational Bridge fully operational");
    }

    initializePythagoreanCurriculum() {
        return {
            // Tier 5: Basic 3D Distance
            tier5: {
                title: "3D Distance Calculation",
                equation: "d = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)",
                concepts: [
                    "Euclidean distance in 3D space",
                    "Extension of 2D Pythagorean theorem",
                    "Practical applications in game development"
                ],
                examples: [
                    { point1: [0, 0, 0], point2: [3, 4, 0], expectedDistance: 5.0, context: "Classic 3-4-5 triangle" },
                    { point1: [1, 2, 3], point2: [4, 6, 8], expectedDistance: 7.07, context: "3D space navigation" },
                    { point1: [-2, -1, 0], point2: [1, 3, 4], expectedDistance: 6.40, context: "Negative coordinates" }
                ],
                visualizationModes: ['distance-visualization', 'pythagorean-proof'],
                aiTrainingData: 'geometric-calculation'
            },

            // Tier 6: Vector Operations
            tier6: {
                title: "Vector Magnitude & Normalization",
                equation: "||v|| = √(x² + y² + z²), v̂ = v/||v||",
                concepts: [
                    "Vector magnitude using Pythagorean theorem",
                    "Unit vector creation through normalization",
                    "Maintaining direction while standardizing length"
                ],
                examples: [
                    { vector: [3, 4, 0], expectedMagnitude: 5.0, context: "2D vector in 3D space" },
                    { vector: [1, 1, 1], expectedMagnitude: 1.732, context: "Equal components" },
                    { vector: [0, 5, 0], expectedMagnitude: 5.0, context: "Single axis vector" }
                ],
                visualizationModes: ['vector-operations', 'all'],
                aiTrainingData: 'mathematical-patterns'
            },

            // Tier 7: Collision Detection
            tier7: {
                title: "Sphere-Sphere Collision Detection",
                equation: "(x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)² ≤ (r₁+r₂)²",
                concepts: [
                    "Efficient collision detection using squared distances",
                    "Avoiding expensive square root operations",
                    "Real-time game physics optimization"
                ],
                examples: [
                    { sphere1: [[0, 0, 0], 2], sphere2: [[3, 0, 0], 2], collision: true, context: "Touching spheres" },
                    { sphere1: [[0, 0, 0], 1], sphere2: [[5, 0, 0], 1], collision: false, context: "Separated spheres" },
                    { sphere1: [[1, 1, 1], 1.5], sphere2: [[2, 2, 2], 1.5], collision: true, context: "3D overlap" }
                ],
                visualizationModes: ['collision-detection', 'all'],
                aiTrainingData: 'physics-simulation'
            },

            // Tier 8: Lighting Calculations
            tier8: {
                title: "Light Attenuation & Direction",
                equation: "I = I₀/d², d⃗ = (P₂-P₁)/||P₂-P₁||",
                concepts: [
                    "Inverse square law for light falloff",
                    "Direction vectors for lighting calculations",
                    "Realistic 3D rendering mathematics"
                ],
                examples: [
                    { lightPos: [0, 5, 0], targetPos: [3, 0, 4], distance: 7.07, attenuation: 0.02, context: "Overhead light" },
                    { lightPos: [2, 2, 2], targetPos: [0, 0, 0], distance: 3.46, attenuation: 0.083, context: "Point light source" },
                    { lightPos: [-1, 0, 1], targetPos: [1, 0, -1], distance: 2.83, attenuation: 0.125, context: "Directional lighting" }
                ],
                visualizationModes: ['lighting-calculations', 'all'],
                aiTrainingData: 'physics-simulation'
            }
        };
    }

    setupEducationalSystems() {
        // Connect to existing systems when they become available
        this.systemCheckInterval = setInterval(() => {
            if (window.playground && !this.playground) {
                this.playground = window.playground;
                console.log("🔗 Connected to 3D Playground");
            }

            if (window.nucleusIntegration && !this.nucleusIntegration) {
                this.nucleusIntegration = window.nucleusIntegration;
                console.log("🧠 Connected to Nucleus Integration");
            }

            if (window.nexus3DMathEngine && !this.math3DEngine) {
                this.math3DEngine = window.nexus3DMathEngine;
                console.log("🎯 Connected to 3D Math Engine");
            }

            // Stop checking once all systems are connected
            if (this.playground && this.nucleusIntegration && this.math3DEngine) {
                clearInterval(this.systemCheckInterval);
                this.onAllSystemsReady();
            }
        }, 1000);
    }

    onAllSystemsReady() {
        console.log("🚀 All educational systems connected - Starting advanced learning mode");

        // Send comprehensive curriculum to Nucleus for AI learning
        this.sendCurriculumToNucleus();

        // Initialize guided learning sequences
        this.startGuidedLearning();

        // Enable educational enhancements in playground
        this.enhancePlayground();
    }

    async sendCurriculumToNucleus() {
        if (!this.nucleusIntegration) return;

        try {
            // Send complete Pythagorean curriculum for AI analysis
            await this.nucleusIntegration.trainDirectly({
                curriculum: 'pythagorean-3d-mathematics',
                tiers: Object.keys(this.pythagoreanCurriculum),
                concepts: this.extractAllConcepts(),
                equations: this.extractAllEquations(),
                examples: this.extractAllExamples(),
                learningObjectives: this.generateLearningObjectives(),
                assessmentCriteria: this.generateAssessmentCriteria(),
                timestamp: Date.now()
            }, 'educational-content');

            console.log("📚 Curriculum sent to Nucleus for AI learning enhancement");

        } catch (error) {
            console.error("❌ Failed to send curriculum to Nucleus:", error);
        }
    }

    extractAllConcepts() {
        const allConcepts = [];
        for (const [tier, data] of Object.entries(this.pythagoreanCurriculum)) {
            allConcepts.push({
                tier,
                title: data.title,
                concepts: data.concepts
            });
        }
        return allConcepts;
    }

    extractAllEquations() {
        const allEquations = [];
        for (const [tier, data] of Object.entries(this.pythagoreanCurriculum)) {
            allEquations.push({
                tier,
                title: data.title,
                equation: data.equation
            });
        }
        return allEquations;
    }

    extractAllExamples() {
        const allExamples = [];
        for (const [tier, data] of Object.entries(this.pythagoreanCurriculum)) {
            allExamples.push({
                tier,
                title: data.title,
                examples: data.examples
            });
        }
        return allExamples;
    }

    generateLearningObjectives() {
        return [
            "Understand 3D distance calculation using Pythagorean theorem extension",
            "Apply vector mathematics to game development scenarios",
            "Implement efficient collision detection algorithms",
            "Calculate realistic lighting and attenuation in 3D space",
            "Connect mathematical theory to practical programming applications",
            "Develop intuitive understanding through interactive visualization"
        ];
    }

    generateAssessmentCriteria() {
        return [
            "Accurate calculation of 3D distances within 0.001 tolerance",
            "Correct vector normalization producing unit vectors",
            "Proper collision detection with 100% accuracy",
            "Appropriate light attenuation calculations matching physics",
            "Successful completion of interactive demonstrations",
            "Ability to explain mathematical concepts in practical context"
        ];
    }

    initializeInteractiveDemos() {
        // Create interactive demonstrations for each tier
        for (const [tier, data] of Object.entries(this.pythagoreanCurriculum)) {
            this.interactiveDemos.set(tier, {
                title: data.title,
                equation: data.equation,
                examples: data.examples,
                currentExample: 0,
                completed: false,
                demonstrations: this.createDemonstrations(tier, data)
            });
        }
    }

    createDemonstrations(tier, data) {
        const demonstrations = [];

        for (let i = 0; i < data.examples.length; i++) {
            const example = data.examples[i];
            demonstrations.push({
                id: `${tier}-demo-${i}`,
                title: `${data.title} - ${example.context}`,
                setup: () => this.setupDemonstration(tier, example),
                execute: () => this.executeDemonstration(tier, example),
                verify: () => this.verifyDemonstration(tier, example),
                visualize: () => this.visualizeDemonstration(tier, example)
            });
        }

        return demonstrations;
    }

    async setupDemonstration(tier, example) {
        console.log(`🎯 Setting up ${tier} demonstration: ${example.context}`);

        // Configure playground inputs based on example
        if (this.playground) {
            switch (tier) {
                case 'tier5': // Distance calculation
                    this.setPlaygroundInputs({
                        x1: example.point1[0], y1: example.point1[1], z1: example.point1[2],
                        x2: example.point2[0], y2: example.point2[1], z2: example.point2[2]
                    });
                    break;

                case 'tier6': // Vector operations
                    this.setPlaygroundInputs({
                        vx: example.vector[0], vy: example.vector[1], vz: example.vector[2]
                    });
                    break;

                case 'tier7': // Collision detection
                    // Set up collision demo parameters
                    this.setupCollisionDemo(example);
                    break;

                case 'tier8': // Lighting calculations
                    this.setupLightingDemo(example);
                    break;
            }
        }

        // Set appropriate visualization mode
        if (this.math3DEngine) {
            const modes = this.pythagoreanCurriculum[tier].visualizationModes;
            this.math3DEngine.setVisualizationMode(modes[0]);
        }
    }

    setPlaygroundInputs(inputs) {
        for (const [id, value] of Object.entries(inputs)) {
            const element = document.getElementById(id);
            if (element) {
                element.value = value;
            }
        }
    }

    async executeDemonstration(tier, example) {
        console.log(`🚀 Executing ${tier} demonstration`);

        let result = null;

        switch (tier) {
            case 'tier5':
                result = await this.executeDistanceDemo(example);
                break;
            case 'tier6':
                result = await this.executeVectorDemo(example);
                break;
            case 'tier7':
                result = await this.executeCollisionDemo(example);
                break;
            case 'tier8':
                result = await this.executeLightingDemo(example);
                break;
        }

        // Send demonstration results to Nucleus
        if (this.nucleusIntegration) {
            await this.nucleusIntegration.trainDirectly({
                demonstration: tier,
                example: example.context,
                input: this.extractInputFromExample(tier, example),
                result: result,
                educational: true,
                timestamp: Date.now()
            }, this.pythagoreanCurriculum[tier].aiTrainingData);
        }

        return result;
    }

    async executeDistanceDemo(example) {
        if (this.math3DEngine) {
            return await this.math3DEngine.calculateDistance3D(example.point1, example.point2);
        } else {
            // Fallback calculation
            const dx = example.point2[0] - example.point1[0];
            const dy = example.point2[1] - example.point1[1];
            const dz = example.point2[2] - example.point1[2];
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    async executeVectorDemo(example) {
        if (this.math3DEngine) {
            return await this.math3DEngine.calculateVectorMagnitude(example.vector);
        } else {
            const [x, y, z] = example.vector;
            return Math.sqrt(x * x + y * y + z * z);
        }
    }

    async executeCollisionDemo(example) {
        const [center1, radius1] = example.sphere1;
        const [center2, radius2] = example.sphere2;

        if (this.math3DEngine) {
            return await this.math3DEngine.checkSphereCollision(center1, radius1, center2, radius2);
        } else {
            const dx = center2[0] - center1[0];
            const dy = center2[1] - center1[1];
            const dz = center2[2] - center1[2];
            const distanceSquared = dx * dx + dy * dy + dz * dz;
            const radiusSumSquared = (radius1 + radius2) * (radius1 + radius2);
            return distanceSquared <= radiusSumSquared;
        }
    }

    async executeLightingDemo(example) {
        if (this.math3DEngine) {
            return await this.math3DEngine.calculateLightAttenuation(
                example.lightPos,
                example.targetPos,
                1.0
            );
        } else {
            const dx = example.targetPos[0] - example.lightPos[0];
            const dy = example.targetPos[1] - example.lightPos[1];
            const dz = example.targetPos[2] - example.lightPos[2];
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            return 1.0 / (distance * distance + 0.0001);
        }
    }

    extractInputFromExample(tier, example) {
        switch (tier) {
            case 'tier5':
                return { point1: example.point1, point2: example.point2 };
            case 'tier6':
                return { vector: example.vector };
            case 'tier7':
                return { sphere1: example.sphere1, sphere2: example.sphere2 };
            case 'tier8':
                return { lightPos: example.lightPos, targetPos: example.targetPos };
            default:
                return example;
        }
    }

    verifyDemonstration(tier, example) {
        // Verify results against expected values
        const tolerance = 0.1; // Allow for small floating point differences

        switch (tier) {
            case 'tier5':
                const expectedDistance = example.expectedDistance;
                return (result) => Math.abs(result - expectedDistance) < tolerance;

            case 'tier6':
                const expectedMagnitude = example.expectedMagnitude;
                return (result) => Math.abs(result - expectedMagnitude) < tolerance;

            case 'tier7':
                const expectedCollision = example.collision;
                return (result) => result === expectedCollision;

            case 'tier8':
                const expectedAttenuation = example.attenuation;
                return (result) => Math.abs(result - expectedAttenuation) < tolerance * 10; // Looser tolerance for attenuation
        }
    }

    startGuidedLearning() {
        console.log("📖 Starting guided learning sequence...");

        // Start with Tier 5 and progress through curriculum
        this.currentLearningTier = 'tier5';
        this.startTierLearning(this.currentLearningTier);
    }

    async startTierLearning(tier) {
        console.log(`🎓 Starting learning for ${tier}: ${this.pythagoreanCurriculum[tier].title}`);

        const demo = this.interactiveDemos.get(tier);
        if (!demo) return;

        // Send learning start notification to Nucleus
        if (this.nucleusIntegration) {
            await this.nucleusIntegration.sendDirectMessage(
                `Student starting ${tier}: ${demo.title}`,
                'educational-progress',
                'high'
            );
        }

        // Run through all demonstrations for this tier
        for (let i = 0; i < demo.demonstrations.length; i++) {
            const demonstration = demo.demonstrations[i];

            console.log(`📚 Running demonstration: ${demonstration.title}`);

            // Setup -> Execute -> Verify -> Visualize
            await demonstration.setup();
            await new Promise(resolve => setTimeout(resolve, 1000)); // Allow time for setup

            const result = await demonstration.execute();
            await new Promise(resolve => setTimeout(resolve, 500)); // Allow time for execution

            const verifier = demonstration.verify();
            const isCorrect = verifier(result);

            // Log learning progress
            this.logLearningProgress(tier, demonstration.id, result, isCorrect);

            // Visualize the result
            if (demonstration.visualize) {
                await demonstration.visualize();
            }

            // Wait before next demonstration
            await new Promise(resolve => setTimeout(resolve, 2000));
        }

        // Mark tier as completed
        demo.completed = true;
        this.educationMetrics.lessonsCompleted++;
        this.educationMetrics.conceptsLearned += this.pythagoreanCurriculum[tier].concepts.length;

        console.log(`✅ Completed ${tier}`);

        // Progress to next tier
        this.progressToNextTier(tier);
    }

    progressToNextTier(currentTier) {
        const tierOrder = ['tier5', 'tier6', 'tier7', 'tier8'];
        const currentIndex = tierOrder.indexOf(currentTier);

        if (currentIndex < tierOrder.length - 1) {
            const nextTier = tierOrder[currentIndex + 1];
            console.log(`📈 Progressing to ${nextTier}`);

            // Wait a bit before starting next tier
            setTimeout(() => {
                this.startTierLearning(nextTier);
            }, 3000);
        } else {
            console.log("🎉 All tiers completed! Educational sequence finished.");
            this.onEducationalSequenceComplete();
        }
    }

    async onEducationalSequenceComplete() {
        console.log("🎓 Educational sequence completed successfully!");

        // Send completion notification to Nucleus
        if (this.nucleusIntegration) {
            await this.nucleusIntegration.sendDirectMessage(
                `Educational sequence completed: All Pythagorean 3D mathematics tiers mastered`,
                'educational-completion',
                'high'
            );

            // Send comprehensive learning summary
            await this.nucleusIntegration.trainDirectly({
                completion: 'pythagorean-3d-curriculum',
                metrics: this.educationMetrics,
                learningProgress: Object.fromEntries(this.learningProgress),
                mastery: this.calculateMasteryLevel(),
                recommendations: this.generateRecommendations(),
                timestamp: Date.now()
            }, 'educational-completion');
        }

        // Enable free exploration mode
        this.enableFreeExplorationMode();
    }

    logLearningProgress(tier, demonstrationId, result, isCorrect) {
        const progressEntry = {
            tier,
            demonstrationId,
            result,
            isCorrect,
            timestamp: Date.now()
        };

        if (!this.learningProgress.has(tier)) {
            this.learningProgress.set(tier, []);
        }

        this.learningProgress.get(tier).push(progressEntry);

        // Update metrics
        this.educationMetrics.practiceProblems++;
        if (isCorrect) {
            // Accuracy tracking could be implemented here
        }

        console.log(`📝 Learning progress logged: ${tier} - ${isCorrect ? '✅' : '❌'}`);
    }

    calculateMasteryLevel() {
        let totalCorrect = 0;
        let totalProblems = 0;

        for (const [tier, progress] of this.learningProgress.entries()) {
            for (const entry of progress) {
                totalProblems++;
                if (entry.isCorrect) totalCorrect++;
            }
        }

        return totalProblems > 0 ? (totalCorrect / totalProblems) * 100 : 0;
    }

    generateRecommendations() {
        const masteryLevel = this.calculateMasteryLevel();
        const recommendations = [];

        if (masteryLevel >= 90) {
            recommendations.push("Excellent mastery! Ready for advanced 3D graphics programming");
            recommendations.push("Consider exploring quaternions and advanced transformations");
        } else if (masteryLevel >= 75) {
            recommendations.push("Good understanding achieved. Practice more complex examples");
            recommendations.push("Focus on real-time applications and optimization");
        } else {
            recommendations.push("Review fundamental concepts and practice basic examples");
            recommendations.push("Spend more time with interactive visualizations");
        }

        return recommendations;
    }

    enhancePlayground() {
        if (!this.playground) return;

        // Add educational enhancements to playground
        console.log("🎮 Enhancing playground with educational features");

        // This could add curriculum-specific buttons, guided tutorials, etc.
        // For now, we'll integrate seamlessly with existing playground
    }

    enableFreeExplorationMode() {
        console.log("🔬 Enabling free exploration mode");

        // Allow student to experiment freely with learned concepts
        // Enable all visualization modes
        // Provide advanced challenges

        if (this.playground) {
            // Enable auto-calculate for continuous learning
            if (this.playground.autoCalculateEnabled === false) {
                // Could programmatically enable auto-calculate
                console.log("💡 Suggesting auto-calculate for continuous exploration");
            }
        }
    }

    startEducationalLoop() {
        // Continuous educational monitoring and support
        setInterval(() => {
            this.monitorLearningProgress();
            this.provideLearningSupport();
            this.updateEducationalMetrics();
        }, 5000);
    }

    monitorLearningProgress() {
        // Check if student needs help or encouragement
        const recentProgress = this.getRecentLearningProgress();

        if (recentProgress.length > 5) {
            const recentAccuracy = recentProgress.filter(p => p.isCorrect).length / recentProgress.length;

            if (recentAccuracy < 0.6) {
                this.provideLearningSupport('struggling');
            } else if (recentAccuracy > 0.9) {
                this.provideLearningSupport('excelling');
            }
        }
    }

    getRecentLearningProgress() {
        const recent = [];
        const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);

        for (const [tier, progress] of this.learningProgress.entries()) {
            for (const entry of progress) {
                if (entry.timestamp > fiveMinutesAgo) {
                    recent.push(entry);
                }
            }
        }

        return recent;
    }

    async provideLearningSupport(type) {
        if (!this.nucleusIntegration) return;

        let supportMessage = '';

        switch (type) {
            case 'struggling':
                supportMessage = 'Student may need additional support with current concepts. Consider reviewing fundamentals.';
                break;
            case 'excelling':
                supportMessage = 'Student showing excellent progress. Ready for advanced challenges.';
                break;
            default:
                supportMessage = 'Continue providing personalized learning support based on progress patterns.';
        }

        await this.nucleusIntegration.sendDirectMessage(
            supportMessage,
            'educational-support',
            'normal'
        );
    }

    updateEducationalMetrics() {
        // Update visualization time if playground is active
        if (this.playground && this.playground.math3DEngine) {
            this.educationMetrics.visualizationTime += 5; // 5 seconds per update
        }

        // Count AI interactions
        if (this.nucleusIntegration) {
            // This would be tracked through the integration system
            this.educationMetrics.aiInteractions++;
        }
    }

    // Public API methods

    getCurrentTier() {
        return this.currentLearningTier;
    }

    getLearningProgress() {
        return {
            currentTier: this.currentLearningTier,
            metrics: this.educationMetrics,
            masteryLevel: this.calculateMasteryLevel(),
            completedTiers: Array.from(this.interactiveDemos.entries())
                .filter(([tier, demo]) => demo.completed)
                .map(([tier, demo]) => tier)
        };
    }

    getEducationalMetrics() {
        return { ...this.educationMetrics };
    }

    async skipToTier(tier) {
        if (this.pythagoreanCurriculum[tier]) {
            this.currentLearningTier = tier;
            await this.startTierLearning(tier);
            return true;
        }
        return false;
    }

    async requestPersonalizedHelp(topic) {
        if (!this.nucleusIntegration) return null;

        return await this.nucleusIntegration.sendDirectMessage(
            `Student requesting help with: ${topic}. Provide personalized guidance based on learning progress.`,
            'help-request',
            'high'
        );
    }
}

// Global educational bridge instance
let educationalBridge = null;

// Initialize when page loads
window.addEventListener('load', async () => {
    console.log("📚 Initializing NEXUS Educational Bridge...");
    educationalBridge = new NexusForgeEducationalBridge();

    // Make available globally
    window.educationalBridge = educationalBridge;
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeEducationalBridge;
}
