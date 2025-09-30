/**
 * NEXUS HOLY BEAT TRAINING INFRASTRUCTURE
 * ======================================
 *
 * Machine learning interface for system parameter optimization and pattern learning.
 * Collects data, trains models, and provides intelligent parameter suggestions.
 */

export class SystemTrainingInterface {
    constructor() {
        this.dataCollector = new DataCollector();
        this.parameterOptimizer = new ParameterOptimizer();
        this.patternLearner = new PatternLearner();
        this.trainingSession = null;
        this.isTraining = false;

        console.log('🤖 Training Interface initialized');
    }

    startTrainingSession(sessionName) {
        this.trainingSession = {
            name: sessionName,
            startTime: Date.now(),
            dataPoints: [],
            parameters: {},
            performance: {}
        };

        this.isTraining = true;
        this.dataCollector.start();
        console.log(`🎓 Started training session: ${sessionName}`);
    }

    stopTrainingSession() {
        if (!this.isTraining) return;

        this.dataCollector.stop();
        this.isTraining = false;

        const duration = Date.now() - this.trainingSession.startTime;
        console.log(`🎓 Completed training session: ${this.trainingSession.name} (${duration}ms)`);

        return this.trainingSession;
    }

    collectSystemState(clockState, audioState, artState, worldState) {
        if (!this.isTraining) return;

        const dataPoint = {
            timestamp: Date.now(),
            clock: clockState,
            audio: audioState,
            art: artState,
            world: worldState,
            crossModalFeatures: this.extractCrossModalFeatures(audioState, artState, worldState)
        };

        this.trainingSession.dataPoints.push(dataPoint);
        return dataPoint;
    }

    extractCrossModalFeatures(audioState, artState, worldState) {
        return {
            spectralCentroid: audioState.spectralCentroid,
            rmsEnergy: audioState.rmsEnergy,
            petalCount: artState.petalCount,
            terrainRoughness: worldState.terrainRoughness,
            correlations: this.calculateCorrelations(audioState, artState, worldState)
        };
    }

    calculateCorrelations(audioState, artState, worldState) {
        // Calculate cross-modal correlations for learning
        return {
            audioToArt: this.pearsonCorrelation(
                [audioState.spectralCentroid, audioState.rmsEnergy],
                [artState.petalCount, artState.strokeDensity]
            ),
            audioToWorld: this.pearsonCorrelation(
                [audioState.spectralCentroid, audioState.rmsEnergy],
                [worldState.terrainRoughness, worldState.cameraHeight]
            ),
            artToWorld: this.pearsonCorrelation(
                [artState.petalCount, artState.strokeDensity],
                [worldState.terrainRoughness, worldState.meshComplexity]
            )
        };
    }

    pearsonCorrelation(x, y) {
        if (x.length !== y.length) return 0;

        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        return denominator === 0 ? 0 : numerator / denominator;
    }

    getTrainingData() {
        return this.trainingSession;
    }

    exportTrainingData() {
        if (!this.trainingSession) return null;

        return {
            ...this.trainingSession,
            exportTime: Date.now(),
            version: '1.0.0'
        };
    }
}

export class DataCollector {
    constructor() {
        this.collecting = false;
        this.collectionRate = 100; // ms
        this.collectionInterval = null;
        this.dataBuffer = [];
    }

    start() {
        this.collecting = true;
        this.collectionInterval = setInterval(() => {
            this.collectFrame();
        }, this.collectionRate);

        console.log('📊 Data collection started');
    }

    stop() {
        this.collecting = false;
        if (this.collectionInterval) {
            clearInterval(this.collectionInterval);
            this.collectionInterval = null;
        }

        console.log('📊 Data collection stopped');
    }

    collectFrame() {
        // This would collect current system state
        // Implementation depends on access to system components
        const frame = {
            timestamp: Date.now(),
            // System state would be collected here
        };

        this.dataBuffer.push(frame);
    }

    getCollectedData() {
        return [...this.dataBuffer];
    }

    clearBuffer() {
        this.dataBuffer = [];
    }
}

export class ParameterOptimizer {
    constructor() {
        this.optimizationHistory = [];
        this.currentParameters = {};
        this.targetMetrics = {
            spectralCoherence: 0.8,
            crossModalBalance: 0.7,
            rhythmicStability: 0.9
        };
    }

    optimizeParameters(currentState, targetGoals) {
        const optimization = {
            timestamp: Date.now(),
            currentState: currentState,
            targetGoals: targetGoals,
            suggestions: this.generateParameterSuggestions(currentState, targetGoals),
            confidence: this.calculateOptimizationConfidence(currentState)
        };

        this.optimizationHistory.push(optimization);
        return optimization;
    }

    generateParameterSuggestions(currentState, targetGoals) {
        const suggestions = {};

        // Audio parameter optimization
        if (targetGoals.spectralBrightness && currentState.audio) {
            suggestions['synth.filterCutoff'] = this.optimizeFilterCutoff(
                currentState.audio.spectralCentroid,
                targetGoals.spectralBrightness
            );
        }

        // Cross-modal coupling optimization
        if (targetGoals.artCoherence && currentState.art) {
            suggestions['crossModal.alphaScaling'] = this.optimizeAlphaScaling(
                currentState.audio.spectralCentroid,
                currentState.art.petalCount,
                targetGoals.artCoherence
            );
        }

        // Temporal optimization
        if (targetGoals.rhythmicComplexity) {
            suggestions['lfo.amDivision'] = this.optimizeAmDivision(
                currentState.clock.bpm,
                targetGoals.rhythmicComplexity
            );
        }

        return suggestions;
    }

    optimizeFilterCutoff(currentCentroid, targetBrightness) {
        const baseCutoff = 2000;
        const centroidFactor = currentCentroid / 440; // Normalize around A4
        const brightnessFactor = targetBrightness * 2; // Scale target

        return Math.max(500, Math.min(8000, baseCutoff * centroidFactor * brightnessFactor));
    }

    optimizeAlphaScaling(spectralCentroid, currentPetals, targetCoherence) {
        const currentAlpha = currentPetals / spectralCentroid;
        const targetPetals = targetCoherence * 12; // Scale to reasonable petal range
        const targetAlpha = targetPetals / spectralCentroid;

        return Math.max(0.005, Math.min(0.02, targetAlpha));
    }

    optimizeAmDivision(bpm, complexity) {
        const baseDivision = 4;
        const complexityFactor = 1 + (complexity * 2); // 1x to 3x complexity

        return Math.max(1, Math.min(16, Math.round(baseDivision * complexityFactor)));
    }

    calculateOptimizationConfidence(currentState) {
        // Simple heuristic for optimization confidence
        const dataPoints = this.optimizationHistory.length;
        const baseConfidence = Math.min(0.9, dataPoints / 100);

        // Adjust based on system stability
        const stabilityFactor = currentState.safety?.stabilityMode ? 0.7 : 1.0;

        return baseConfidence * stabilityFactor;
    }

    getOptimizationHistory() {
        return [...this.optimizationHistory];
    }
}

export class PatternLearner {
    constructor() {
        this.learnedPatterns = [];
        this.patternBuffer = [];
        this.minPatternLength = 4; // beats
        this.maxPatternLength = 32; // beats
    }

    analyzePatterns(sequenceData) {
        const patterns = this.extractPatterns(sequenceData);
        const uniquePatterns = this.deduplicatePatterns(patterns);

        uniquePatterns.forEach(pattern => {
            const existingIndex = this.findSimilarPattern(pattern);
            if (existingIndex >= 0) {
                this.learnedPatterns[existingIndex].occurrences++;
                this.learnedPatterns[existingIndex].confidence = Math.min(1.0,
                    this.learnedPatterns[existingIndex].confidence + 0.1
                );
            } else {
                this.learnedPatterns.push({
                    pattern: pattern,
                    occurrences: 1,
                    confidence: 0.5,
                    discovered: Date.now()
                });
            }
        });

        return this.getTopPatterns(10);
    }

    extractPatterns(data) {
        const patterns = [];

        for (let length = this.minPatternLength; length <= this.maxPatternLength; length++) {
            for (let start = 0; start <= data.length - length; start++) {
                const pattern = data.slice(start, start + length);
                patterns.push({
                    sequence: pattern,
                    length: length,
                    startPosition: start
                });
            }
        }

        return patterns;
    }

    deduplicatePatterns(patterns) {
        const unique = [];
        const seen = new Set();

        patterns.forEach(pattern => {
            const signature = this.getPatternSignature(pattern.sequence);
            if (!seen.has(signature)) {
                seen.add(signature);
                unique.push(pattern);
            }
        });

        return unique;
    }

    getPatternSignature(sequence) {
        return sequence.map(item => JSON.stringify(item)).join('|');
    }

    findSimilarPattern(pattern) {
        const signature = this.getPatternSignature(pattern.sequence);
        return this.learnedPatterns.findIndex(learned =>
            this.getPatternSignature(learned.pattern.sequence) === signature
        );
    }

    getTopPatterns(count) {
        return this.learnedPatterns
            .sort((a, b) => (b.occurrences * b.confidence) - (a.occurrences * a.confidence))
            .slice(0, count);
    }

    predictNextState(currentSequence) {
        const matchingPatterns = this.learnedPatterns.filter(learned => {
            const pattern = learned.pattern.sequence;
            const matchLength = Math.min(pattern.length - 1, currentSequence.length);

            return this.sequencesMatch(
                pattern.slice(0, matchLength),
                currentSequence.slice(-matchLength)
            );
        });

        if (matchingPatterns.length === 0) return null;

        // Weight predictions by confidence and occurrence
        const weightedPrediction = this.combineWeightedPredictions(matchingPatterns, currentSequence);
        return weightedPrediction;
    }

    sequencesMatch(seq1, seq2) {
        if (seq1.length !== seq2.length) return false;

        return seq1.every((item, index) => {
            return JSON.stringify(item) === JSON.stringify(seq2[index]);
        });
    }

    combineWeightedPredictions(patterns, currentSequence) {
        const predictions = patterns.map(learned => {
            const pattern = learned.pattern.sequence;
            const matchIndex = this.findMatchIndex(pattern, currentSequence);

            if (matchIndex >= 0 && matchIndex < pattern.length - 1) {
                return {
                    nextState: pattern[matchIndex + 1],
                    weight: learned.confidence * learned.occurrences,
                    source: learned
                };
            }

            return null;
        }).filter(pred => pred !== null);

        if (predictions.length === 0) return null;

        // For simplicity, return highest weighted prediction
        // Could implement more sophisticated combination later
        return predictions.reduce((best, current) =>
            current.weight > best.weight ? current : best
        );
    }

    findMatchIndex(pattern, currentSequence) {
        const maxMatchLength = Math.min(pattern.length - 1, currentSequence.length);

        for (let matchLength = maxMatchLength; matchLength > 0; matchLength--) {
            const patternSlice = pattern.slice(0, matchLength);
            const currentSlice = currentSequence.slice(-matchLength);

            if (this.sequencesMatch(patternSlice, currentSlice)) {
                return matchLength - 1; // Return index of last matching element
            }
        }

        return -1;
    }

    getLearnedPatterns() {
        return [...this.learnedPatterns];
    }
}

console.log('🤖 NEXUS Holy Beat Training Infrastructure loaded');
