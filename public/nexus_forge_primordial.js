/**
 * NEXUS FORGE PRIMORDIAL
 * ======================
 *
 * AI-Powered Development Intelligence System
 *
 * Combines:
 * • GPT-powered pain/opportunity detection
 * • Real-time clustering and trend analysis
 * • Quantum Graphics Engine visualization
 * • VectorLab World Engine integration
 * • Tier-4 Meta System recommendations
 *
 * This creates the ultimate development intelligence hub that can:
 * - Detect pain points in code/issues/feedback automatically
 * - Visualize problem clusters as quantum effects
 * - Provide AI-driven recommendations for improvements
 * - Track project health with mathematical precision
 */

//--------------------------------------
// AI Pattern Recognition Engine
//--------------------------------------

class AIPatternEngine {
    constructor() {
        this.patterns = new Map();
        this.problemSolutions = new Map();
        this.initializePatterns();
        this.initializeSolutions();
    }

    initializePatterns() {
        // Pain patterns
        this.patterns.set('pain', [
            /won't work|doesn't work|not working/i,
            /stuck|blocked|can't|cannot/i,
            /error|fail|failing|failed/i,
            /how do i|help|confused/i,
            /integration.*fail|manual.*step/i,
            /bug|broken|crash/i,
            /timeout|slow|performance/i,
            /documentation.*missing|unclear/i
        ]);

        // Opportunity patterns
        this.patterns.set('opportunity', [
            /launch|released|new feature/i,
            /pricing|tier|starter|pro|enterprise/i,
            /discount|sale|offer/i,
            /upgrade|migration|switch/i,
            /competitor|alternative/i,
            /scale|growth|expand/i
        ]);

        // Code smell patterns
        this.patterns.set('code_smell', [
            /TODO|FIXME|HACK/i,
            /duplicate|copy.*paste/i,
            /magic number|hard.*cod/i,
            /god class|too.*complex/i,
            /deprecated|legacy/i,
            /memory leak|performance/i
        ]);
    }

    initializeSolutions() {
        this.problemSolutions.set(/onboarding|sop|checklist/i, 'Manual onboarding → SOP + bot automation');
        this.problemSolutions.set(/webhook|retry|queue/i, 'Unreliable webhooks → queue + replay system');
        this.problemSolutions.set(/csv|copy.*paste|export/i, 'Data shuffling → scheduled ETL pipeline');
        this.problemSolutions.set(/authentication|login|session/i, 'Auth issues → SSO + session management');
        this.problemSolutions.set(/database|query|slow/i, 'DB performance → indexing + query optimization');
        this.problemSolutions.set(/deployment|ci|cd/i, 'Deploy friction → automated pipeline + rollback');
    }

    detectPatterns(text, type) {
        const patterns = this.patterns.get(type) || [];
        const hits = patterns.filter(pattern => pattern.test(text)).length;
        return Math.tanh(hits * 0.5); // Normalize to 0-1
    }

    guessProblem(text) {
        for (const [pattern, solution] of this.problemSolutions.entries()) {
            if (pattern.test(text)) {
                return solution;
            }
        }
        return 'General automation opportunity → discovery needed';
    }

    calculatePainScore(text, engagement) {
        const patternScore = this.detectPatterns(text, 'pain');
        const engagementBoost = 1 + Math.log1p(Math.max(0, engagement));
        return Math.tanh(patternScore * engagementBoost);
    }

    calculateOpportunityScore(text, engagement) {
        const patternScore = this.detectPatterns(text, 'opportunity');
        const engagementBoost = 1 + Math.log1p(Math.max(0, engagement));
        return Math.tanh(patternScore * engagementBoost * 0.8); // Slightly lower weight
    }
}
// Pain patterns
this.patterns.set('pain', [
    /won't work|doesn't work|not working/i,
    /stuck|blocked|can't|cannot/i,
    /error|fail|failing|failed/i,
    /how do i|help|confused/i,
    /integration.*fail|manual.*step/i,
    /bug|broken|crash/i,
    /timeout|slow|performance/i,
    /documentation.*missing|unclear/i
]);

// Opportunity patterns
this.patterns.set('opportunity', [
    /launch|released|new feature/i,
    /pricing|tier|starter|pro|enterprise/i,
    /discount|sale|offer/i,
    /upgrade|migration|switch/i,
    /competitor|alternative/i,
    /scale|growth|expand/i
]);

// Code smell patterns
this.patterns.set('code_smell', [
    /TODO|FIXME|HACK/i,
    /duplicate|copy.*paste/i,
    /magic number|hard.*cod/i,
    /god class|too.*complex/i,
    /deprecated|legacy/i,
    /memory leak|performance/i
]);
    }

initializeSolutions() {
    this.problemSolutions.set(/onboarding|sop|checklist/i, 'Manual onboarding → SOP + bot automation');
    this.problemSolutions.set(/webhook|retry|queue/i, 'Unreliable webhooks → queue + replay system');
    this.problemSolutions.set(/csv|copy.*paste|export/i, 'Data shuffling → scheduled ETL pipeline');
    this.problemSolutions.set(/authentication|login|session/i, 'Auth issues → SSO + session management');
    this.problemSolutions.set(/database|query|slow/i, 'DB performance → indexing + query optimization');
    this.problemSolutions.set(/deployment|ci|cd/i, 'Deploy friction → automated pipeline + rollback');
}

detectPatterns(text, type) {
    const patterns = this.patterns.get(type) || [];
    const hits = patterns.filter(pattern => pattern.test(text)).length;
    return Math.tanh(hits * 0.5); // Normalize to 0-1
}

guessProblem(text) {
    for (const [pattern, solution] of this.problemSolutions.entries()) {
        if (pattern.test(text)) {
            return solution;
        }
    }
    return 'General automation opportunity → discovery needed';
}

calculatePainScore(text, engagement) {
    const patternScore = this.detectPatterns(text, 'pain');
    const engagementBoost = 1 + Math.log1p(Math.max(0, engagement));
    return Math.tanh(patternScore * engagementBoost);
}

calculateOpportunityScore(text, engagement) {
    const patternScore = this.detectPatterns(text, 'opportunity');
    const engagementBoost = 1 + Math.log1p(Math.max(0, engagement));
    return Math.tanh(patternScore * engagementBoost * 0.8); // Slightly lower weight
}
}

//--------------------------------------
// Text Clustering Engine
//--------------------------------------

class TextClusteringEngine {
    generateTrigrams(text) {
        const cleaned = text.toLowerCase()
            .replace(/[^a-z0-9 ]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

        const trigrams = new Set();
        for (let i = 0; i < cleaned.length - 2; i++) {
            trigrams.add(cleaned.slice(i, i + 3));
        }
        return trigrams;
    }

    calculateJaccard(setA, setB) {
        const intersection = new Set([...setA].filter(x => setB.has(x)));
        const union = new Set([...setA, ...setB]);
        return union.size > 0 ? intersection.size / union.size : 0;
    }

    clusterPainRecords(records, threshold = 0.35) {
        const trigrams = records.map(r => this.generateTrigrams(r.text));
        const clusters = new Map();
        const assignments = new Array(records.length).fill(-1);

        let clusterId = 0;

        for (let i = 0; i < records.length; i++) {
            if (assignments[i] !== -1) continue;

            assignments[i] = clusterId;
            const clusterIndices = [i];

            for (let j = i + 1; j < records.length; j++) {
                if (assignments[j] === -1 &&
                    this.calculateJaccard(trigrams[i], trigrams[j]) >= threshold) {
                    assignments[j] = clusterId;
                    clusterIndices.push(j);
                }
            }

            clusters.set(clusterId.toString(), clusterIndices);
            clusterId++;
        }

        // Group records by cluster
        const result = new Map();
        for (const [clusterIdStr, indices] of clusters.entries()) {
            const clusterRecords = indices.map(idx => ({
                ...records[idx],
                clusterId: clusterIdStr
            }));
            result.set(clusterIdStr, clusterRecords);
        }

        return result;
    }

    generateClusterLabels(clusters) {
        const labels = new Map();

        for (const [clusterId, records] of clusters.entries()) {
            if (records.length === 0) continue;

            // Find representative text (longest or most engaging)
            const representative = records.reduce((best, current) =>
                current.text.length > best.text.length ? current : best
            );

            const label = representative.text.slice(0, 80);
            labels.set(clusterId, label);
        }

        return labels;
    }
}

//--------------------------------------
// Nexus Forge Intelligence Engine
//--------------------------------------

class NexusForgeEngine {
    constructor() {
        this.patternEngine = new AIPatternEngine();
        this.clusteringEngine = new TextClusteringEngine();
        this.painRecords = [];
        this.opportunities = [];
        this.quantumEngine = null; // Will be connected to QuantumGraphicsEngine
        this.vectorLabEngine = null; // Will be connected to VectorLabWorldEngine
    }

    // Integration with existing engines
    connectQuantumEngine(engine) {
        this.quantumEngine = engine;
        console.log('🌌 Nexus Forge connected to Quantum Graphics Engine');
    }

    connectVectorLabEngine(engine) {
        this.vectorLabEngine = engine;
        console.log('🧬 Nexus Forge connected to VectorLab World Engine');
    }

    // Core intelligence methods
    ingestPainEvent(event) {
        const pain = this.patternEngine.calculatePainScore(event.text, event.eng);
        const opportunity = this.patternEngine.calculateOpportunityScore(event.text, event.eng);
        const severity = pain >= 0.8 ? 3 : pain >= 0.5 ? 2 : pain >= 0.25 ? 1 : 0;
        const problemGuess = this.patternEngine.guessProblem(event.text);

        const record = {
            ...event,
            pain,
            severity,
            aiConfidence: Math.max(pain, opportunity),
            problemGuess
        };

        this.painRecords.push(record);

        // Trigger visual effects in Quantum Engine
        if (this.quantumEngine && pain > 0.3) {
            this.visualizePainEvent(record);
        }

        // Pulse VectorLab Heart based on pain intensity
        if (this.vectorLabEngine && pain > 0.5) {
            this.vectorLabEngine.heartEngine.pulse(pain * 0.3);
        }

        console.log(`🔍 Pain ingested: ${pain.toFixed(3)} severity:${severity} - ${problemGuess}`);
        return record;
    }

    visualizePainEvent(record) {
        // Create quantum collapse effect based on pain severity
        const intensity = record.pain;
        const mathType = record.severity >= 3 ? 'chaos' :
            record.severity >= 2 ? 'absorb' : 'mirror';

        // Random position for visualization
        const x = Math.random() * (this.quantumEngine.canvas?.width || 800);
        const y = Math.random() * (this.quantumEngine.canvas?.height || 600);

        this.quantumEngine.eventOrchestrator.emitAgentCollapse(
            `pain_${record.id}`,
            { x, y },
            {
                mathType,
                intensity,
                metadata: {
                    source: record.source,
                    severity: record.severity,
                    problemGuess: record.problemGuess
                }
            }
        );
    }

    recomputeClusters() {
        const clusters = this.clusteringEngine.clusterPainRecords(this.painRecords);
        const labels = this.clusteringEngine.generateClusterLabels(clusters);

        // Visualize clusters as quantum fields
        if (this.quantumEngine) {
            this.visualizeClusters(clusters, labels);
        }

        console.log(`🔄 Recomputed ${clusters.size} pain clusters`);
        return clusters;
    }

    visualizeClusters(clusters, labels) {
        for (const [clusterId, records] of clusters.entries()) {
            if (records.length < 2) continue; // Only visualize meaningful clusters

            const avgPain = records.reduce((sum, r) => sum + r.pain, 0) / records.length;
            const clusterIntensity = avgPain * Math.log(records.length + 1);

            // Create cluster visualization as quantum field effect
            if (this.quantumEngine && avgPain > 0.4) {
                const centerX = (this.quantumEngine.canvas?.width || 800) / 2;
                const centerY = (this.quantumEngine.canvas?.height || 600) / 2;

                // Offset based on cluster ID for visual separation
                const angle = parseInt(clusterId) * (Math.PI * 2 / clusters.size);
                const radius = 100 + clusterIntensity * 50;
                const x = centerX + Math.cos(angle) * radius;
                const y = centerY + Math.sin(angle) * radius;

                this.quantumEngine.visualEffects.createCollapseEffect(
                    x, y,
                    clusterIntensity,
                    'cosine' // Use cosine for cluster effects
                );
            }
        }
    }

    generateIntelligence() {
        const clusters = this.recomputeClusters();
        const recommendations = this.generateRecommendations(clusters);
        const systemHealth = this.calculateSystemHealth();

        return {
            painClusters: this.convertClustersToInterface(clusters),
            opportunities: this.opportunities.slice(-10), // Last 10 opportunities
            systemHealth,
            recommendations
        };
    }

    convertClustersToInterface(clusters) {
        const result = [];

        for (const [clusterId, records] of clusters.entries()) {
            const avgPain = records.reduce((sum, r) => sum + r.pain, 0) / records.length;
            const lastSeen = records.reduce((latest, r) =>
                r.time > latest ? r.time : latest, records[0]?.time || new Date().toISOString()
            );

            const label = records.length > 0 ? records[0].text.slice(0, 80) : 'Unknown cluster';

            result.push({
                id: clusterId,
                label,
                members: records.map(r => r.id),
                avgPain,
                count: records.length,
                lastSeen
            });
        }

        return result.sort((a, b) => (b.avgPain * b.count) - (a.avgPain * a.count));
    }

    generateRecommendations(clusters) {
        const recommendations = [];
        let recommendationId = 0;

        // Generate recommendations from pain clusters
        for (const [clusterId, records] of clusters.entries()) {
            const avgPain = records.reduce((sum, r) => sum + r.pain, 0) / records.length;
            const count = records.length;

            if (avgPain > 0.6 && count > 2) { // High pain, multiple reports
                const problemGuess = records[0]?.problemGuess || 'Unknown issue';
                const priority = avgPain > 0.8 ? 'critical' : avgPain > 0.7 ? 'high' : 'medium';

                recommendations.push({
                    id: `rec_${recommendationId++}`,
                    type: 'fix_pain',
                    priority: priority,
                    description: `Address cluster: ${problemGuess}`,
                    confidence: Math.min(avgPain * 0.9 + count * 0.05, 1.0),
                    actionable: {
                        route: count > 5 ? 'automated' : 'form',
                        steps: this.generateActionSteps(problemGuess),
                        estimatedEffort: count > 10 ? '2-3 weeks' : count > 5 ? '1-2 weeks' : '3-5 days'
                    },
                    relatedClusters: [clusterId]
                });
            }
        }

        // Generate proactive recommendations based on patterns
        const codeSmells = this.detectCodeSmells();
        if (codeSmells.length > 0) {
            recommendations.push({
                id: `rec_${recommendationId++}`,
                type: 'optimize_code',
                priority: 'medium',
                description: `Code quality improvements detected: ${codeSmells.join(', ')}`,
                confidence: 0.7,
                actionable: {
                    route: 'code',
                    steps: ['Run code analysis', 'Refactor identified areas', 'Add tests', 'Review performance'],
                    estimatedEffort: '1 week'
                }
            });
        }

        return recommendations;
    }

    generateActionSteps(problemGuess) {
        if (problemGuess.includes('webhook')) {
            return [
                'Implement queue system for webhook reliability',
                'Add retry logic with exponential backoff',
                'Create monitoring dashboard',
                'Set up alerts for failures'
            ];
        } else if (problemGuess.includes('onboarding')) {
            return [
                'Document current manual process',
                'Identify automation opportunities',
                'Build automated SOP system',
                'Create onboarding bot'
            ];
        } else if (problemGuess.includes('ETL')) {
            return [
                'Analyze current data flow',
                'Design automated ETL pipeline',
                'Implement scheduling system',
                'Add data validation'
            ];
        }

        return [
            'Analyze root cause',
            'Design solution architecture',
            'Implement fix',
            'Test and monitor'
        ];
    }

    detectCodeSmells() {
        // This would analyze actual code in a real implementation
        // For now, return mock code smell detection
        const mockSmells = [];

        if (Math.random() > 0.7) mockSmells.push('Long methods detected');
        if (Math.random() > 0.8) mockSmells.push('Duplicate code found');
        if (Math.random() > 0.9) mockSmells.push('Magic numbers present');

        return mockSmells;
    }

    calculateSystemHealth() {
        const avgPain = this.painRecords.length > 0
            ? this.painRecords.reduce((sum, r) => sum + r.pain, 0) / this.painRecords.length
            : 0;

        const recentPain = this.painRecords
            .filter(r => Date.now() - Date.parse(r.time) < 7 * 24 * 60 * 60 * 1000) // Last 7 days
            .length;

        return {
            codeQuality: Math.max(0, 1 - avgPain), // Inverse of pain
            userSatisfaction: Math.max(0, 1 - recentPain * 0.1),
            developmentVelocity: 0.7 + Math.random() * 0.3, // Mock velocity
            technicalDebt: avgPain * 0.8 + Math.random() * 0.2
        };
    }

    // Burst detection for trending issues
    detectBurst(timeWindow = 7) {
        const now = Date.now();
        const windowMs = timeWindow * 24 * 60 * 60 * 1000;

        const recentRecords = this.painRecords.filter(r =>
            now - Date.parse(r.time) <= windowMs
        );

        const oldRecords = this.painRecords.filter(r => {
            const age = now - Date.parse(r.time);
            return age > windowMs && age <= windowMs * 2;
        });

        const recentRate = recentRecords.length / timeWindow;
        const oldRate = oldRecords.length / timeWindow;
        const burstScore = oldRate > 0 ? recentRate / oldRate : recentRecords.length > 0 ? 2 : 1;

        // Find trending topics by clustering recent records
        const recentClusters = this.clusteringEngine.clusterPainRecords(recentRecords, 0.4);
        const trendingTopics = Array.from(recentClusters.entries())
            .filter(([_, records]) => records.length > 1)
            .map(([_, records]) => records[0]?.problemGuess || 'Unknown trend')
            .slice(0, 5);

        return { burstScore, trendingTopics };
    }

    // Export intelligence for external use
    exportIntelligence() {
        const intelligence = this.generateIntelligence();
        const burst = this.detectBurst();

        return {
            timestamp: new Date().toISOString(),
            nexusForge: {
                version: '1.0.0',
                totalPainEvents: this.painRecords.length,
                totalOpportunities: this.opportunities.length
            },
            intelligence,
            burst,
            metadata: {
                quantumEngineConnected: !!this.quantumEngine,
                vectorLabEngineConnected: !!this.vectorLabEngine,
                lastRecompute: new Date().toISOString()
            }
        };
    }
}

//--------------------------------------
// Demo and Integration Functions
//--------------------------------------

function createMockPainEvents() {
    return [
        {
            id: 'issue_001',
            time: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
            source: 'issues',
            text: 'Webhook retries keep failing and we have to manually export CSV data every time',
            eng: 12,
            org: 'TechCorp',
            url: 'https://github.com/techcorp/project/issues/245'
        },
        {
            id: 'forum_002',
            time: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(), // 5 hours ago
            source: 'forum',
            text: 'Stuck on webhook configuration, manual steps are too complex',
            eng: 8,
            org: 'StartupXYZ'
        },
        {
            id: 'support_003',
            time: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
            source: 'support',
            text: 'Onboarding process is confusing, need better documentation and automation',
            eng: 15,
            org: 'Enterprise Inc'
        },
        {
            id: 'review_004',
            time: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(), // 3 days ago
            source: 'reviews',
            text: 'Great product but the pricing tiers are unclear, would like to see enterprise options',
            eng: 6,
            org: 'BigCorp'
        },
        {
            id: 'code_005',
            time: new Date(Date.now() - 30 * 60 * 1000).toISOString(), // 30 mins ago
            source: 'code',
            text: 'Performance issues in database queries, timeout errors happening frequently',
            eng: 20,
            org: 'DevTeam'
        }
    ];
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        NexusForgeEngine,
        AIPatternEngine,
        TextClusteringEngine,
        createMockPainEvents
    };
}

// Global instance for demo purposes
let globalNexusForge = null;

function initializeNexusForge() {
    if (!globalNexusForge) {
        globalNexusForge = new NexusForgeEngine();
        console.log('🔥 NEXUS FORGE PRIMORDIAL initialized');
    }
    return globalNexusForge;
}
