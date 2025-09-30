/**
 * NEXUS Forge AI Behavior System
 * • Intelligent NPC behavior with emergent AI
 * • Audio-reactive AI mood and decision making
 * • Advanced pathfinding and navigation
 * • Dynamic AI ecosystem interactions
 * • Machine learning-inspired behavior adaptation
 */

class NexusForgeAISystem {
    constructor() {
        this.entities = new Map();
        this.behaviorTrees = new Map();
        this.aiGroups = new Map();
        this.navigationMesh = null;
        this.knowledgeBase = new Map();

        this.audioInfluence = {
            moodModifiers: {
                calm: { aggression: -0.3, exploration: 0.2, social: 0.4 },
                energetic: { aggression: 0.1, exploration: 0.5, social: 0.3 },
                excited: { aggression: 0.4, exploration: 0.3, social: 0.6 },
                peaceful: { aggression: -0.5, exploration: 0.1, social: 0.2 }
            },
            beatResponse: {
                enabled: true,
                synchronization: 0.7,
                movementInfluence: 0.5
            }
        };

        this.aiParameters = {
            perceptionRadius: 25,
            memoryDuration: 30000, // 30 seconds
            decisionUpdateRate: 500, // ms
            pathfindingMaxNodes: 1000,
            socialInteractionRange: 10,
            audioResponseSensitivity: 0.8
        };

        this.behaviorTypes = this.initializeBehaviorTypes();
        this.emotionalStates = this.initializeEmotionalStates();

        console.log('🧠 AI Behavior System initialized');
    }

    initializeBehaviorTypes() {
        return {
            // Basic Behaviors
            idle: {
                priority: 1,
                duration: [2000, 8000],
                actions: ['stand', 'look_around', 'random_movement']
            },
            wander: {
                priority: 2,
                duration: [5000, 15000],
                actions: ['move_random', 'explore', 'investigate']
            },
            patrol: {
                priority: 3,
                duration: [10000, 30000],
                actions: ['follow_path', 'guard_area', 'scan_surroundings']
            },

            // Social Behaviors
            flock: {
                priority: 4,
                duration: [8000, 20000],
                actions: ['follow_group', 'maintain_distance', 'synchronize_movement']
            },
            interact: {
                priority: 5,
                duration: [3000, 12000],
                actions: ['approach_entity', 'communicate', 'share_resources']
            },

            // Survival Behaviors
            flee: {
                priority: 8,
                duration: [2000, 6000],
                actions: ['escape_danger', 'find_safety', 'call_for_help']
            },
            hunt: {
                priority: 7,
                duration: [5000, 25000],
                actions: ['track_target', 'pursue', 'attack']
            },
            forage: {
                priority: 4,
                duration: [8000, 20000],
                actions: ['search_resources', 'collect_items', 'store_resources']
            },

            // Audio-Reactive Behaviors
            dance: {
                priority: 3,
                duration: [4000, 16000],
                actions: ['rhythmic_movement', 'synchronized_dance', 'audio_interpretation'],
                audioTriggered: true
            },
            sing: {
                priority: 2,
                duration: [6000, 18000],
                actions: ['vocalize', 'harmonic_response', 'call_response'],
                audioTriggered: true
            },
            meditate: {
                priority: 1,
                duration: [10000, 30000],
                actions: ['still_pose', 'energy_gathering', 'audio_absorption'],
                audioTriggered: true
            }
        };
    }

    initializeEmotionalStates() {
        return {
            neutral: { energy: 0.5, happiness: 0.5, fear: 0.1, aggression: 0.2 },
            happy: { energy: 0.8, happiness: 0.9, fear: 0.1, aggression: 0.1 },
            excited: { energy: 0.9, happiness: 0.8, fear: 0.2, aggression: 0.4 },
            calm: { energy: 0.3, happiness: 0.7, fear: 0.1, aggression: 0.1 },
            fearful: { energy: 0.7, happiness: 0.2, fear: 0.9, aggression: 0.1 },
            aggressive: { energy: 0.8, happiness: 0.3, fear: 0.3, aggression: 0.9 },
            curious: { energy: 0.6, happiness: 0.6, fear: 0.3, aggression: 0.2 },
            tired: { energy: 0.2, happiness: 0.4, fear: 0.2, aggression: 0.1 }
        };
    }

    // Entity Management
    createAIEntity(config) {
        const entity = {
            id: this.generateEntityId(),
            type: config.type || 'generic',
            position: { x: config.x || 0, y: config.y || 0, z: config.z || 0 },
            rotation: { x: 0, y: Math.random() * Math.PI * 2, z: 0 },
            velocity: { x: 0, y: 0, z: 0 },

            // AI Properties
            behavior: config.behavior || 'idle',
            emotionalState: config.emotionalState || 'neutral',
            personality: this.generatePersonality(),
            memory: [],
            currentTarget: null,
            path: [],

            // Stats
            health: config.health || 100,
            energy: config.energy || 100,
            speed: config.speed || 2,
            perceptionRadius: config.perceptionRadius || this.aiParameters.perceptionRadius,

            // Audio Reactivity
            audioSensitivity: config.audioSensitivity || 0.5,
            beatSynchronization: config.beatSync || false,
            lastBeatResponse: 0,

            // Behavior State
            currentBehavior: null,
            behaviorStartTime: 0,
            behaviorDuration: 0,
            decisionCooldown: 0,

            // Social
            group: config.group || null,
            relationships: new Map(),

            created: Date.now()
        };

        this.entities.set(entity.id, entity);
        this.initializeEntityBehavior(entity);

        console.log(`🤖 AI Entity created: ${entity.id} (${entity.type})`);
        return entity.id;
    }

    generatePersonality() {
        return {
            openness: Math.random(),      // Exploration tendency
            conscientiousness: Math.random(), // Goal persistence
            extraversion: Math.random(),  // Social interaction
            agreeableness: Math.random(), // Cooperation
            neuroticism: Math.random(),   // Stress response
            creativity: Math.random(),    // Problem solving
            empathy: Math.random()        // Emotional response
        };
    }

    initializeEntityBehavior(entity) {
        const behaviorType = this.behaviorTypes[entity.behavior];
        if (behaviorType) {
            entity.currentBehavior = entity.behavior;
            entity.behaviorStartTime = Date.now();
            entity.behaviorDuration = this.randomBetween(behaviorType.duration[0], behaviorType.duration[1]);
        }
    }

    // Update System
    update(deltaTime, audioData, worldData) {
        const currentTime = Date.now();

        for (const [, entity] of this.entities) {
            // Update entity behavior
            this.updateEntityBehavior(entity, currentTime, deltaTime);

            // Update emotional state based on audio
            if (audioData) {
                this.updateAudioInfluence(entity, audioData);
            }

            // Update perception and decision making
            this.updatePerception(entity, worldData);

            // Update movement and physics
            this.updateMovement(entity, deltaTime);

            // Update memory and relationships
            this.updateMemory(entity, currentTime);
        }

        // Update group behaviors
        this.updateGroupBehaviors(currentTime, audioData);
    }

    updateEntityBehavior(entity, currentTime, deltaTime) {
        // Check if current behavior should change
        if (currentTime - entity.behaviorStartTime > entity.behaviorDuration ||
            this.shouldChangeBehavior(entity)) {

            const newBehavior = this.selectBehavior(entity);
            this.setBehavior(entity, newBehavior);
        }

        // Execute current behavior actions
        this.executeBehavior(entity, deltaTime);
    }

    shouldChangeBehavior(entity) {
        // Check for high priority interrupts
        const threats = this.detectThreats(entity);
        if (threats.length > 0) {
            return true;
        }

        // Check for social opportunities
        const socialOpportunities = this.detectSocialOpportunities(entity);
        if (socialOpportunities.length > 0 && entity.personality.extraversion > 0.6) {
            return true;
        }

        // Random behavior change based on personality
        const changeChance = entity.personality.openness * 0.01; // 1% max per update
        return Math.random() < changeChance;
    }

    selectBehavior(entity) {
        const availableBehaviors = Object.keys(this.behaviorTypes);
        const behaviorScores = new Map();

        for (const behavior of availableBehaviors) {
            const score = this.calculateBehaviorScore(entity, behavior);
            behaviorScores.set(behavior, score);
        }

        // Select behavior based on weighted probabilities
        return this.weightedRandomSelect(behaviorScores);
    }

    calculateBehaviorScore(entity, behaviorName) {
        const behavior = this.behaviorTypes[behaviorName];
        const emotional = this.emotionalStates[entity.emotionalState];
        let score = behavior.priority;

        // Personality influences
        switch (behaviorName) {
            case 'wander':
            case 'explore':
                score += entity.personality.openness * 3;
                break;
            case 'interact':
            case 'flock':
                score += entity.personality.extraversion * 3;
                break;
            case 'hunt':
            case 'attack':
                score += emotional.aggression * 2;
                break;
            case 'flee':
                score += emotional.fear * 4;
                break;
            case 'dance':
            case 'sing':
                score += (emotional.happiness + emotional.energy) * 2;
                break;
        }

        // Environmental factors
        const nearbyEntities = this.getNearbyEntities(entity);
        if (nearbyEntities.length > 0) {
            score += behaviorName === 'interact' ? 2 : 0;
            score += behaviorName === 'flee' && this.hasThreats(nearbyEntities) ? 5 : 0;
        }

        // Audio reactive behaviors
        if (behavior.audioTriggered && this.audioInfluence.beatResponse.enabled) {
            score += 3;
        }

        return Math.max(0, score);
    }

    setBehavior(entity, behaviorName) {
        const behavior = this.behaviorTypes[behaviorName];
        if (!behavior) return;

        entity.currentBehavior = behaviorName;
        entity.behaviorStartTime = Date.now();
        entity.behaviorDuration = this.randomBetween(behavior.duration[0], behavior.duration[1]);
        entity.path = []; // Clear current path

        console.log(`🤖 Entity ${entity.id} changed behavior to: ${behaviorName}`);
    }

    executeBehavior(entity, deltaTime) {
        switch (entity.currentBehavior) {
            case 'idle':
                this.executeBehaviorIdle(entity, deltaTime);
                break;
            case 'wander':
                this.executeBehaviorWander(entity, deltaTime);
                break;
            case 'patrol':
                this.executeBehaviorPatrol(entity, deltaTime);
                break;
            case 'flock':
                this.executeBehaviorFlock(entity, deltaTime);
                break;
            case 'interact':
                this.executeBehaviorInteract(entity, deltaTime);
                break;
            case 'flee':
                this.executeBehaviorFlee(entity, deltaTime);
                break;
            case 'hunt':
                this.executeBehaviorHunt(entity, deltaTime);
                break;
            case 'forage':
                this.executeBehaviorForage(entity, deltaTime);
                break;
            case 'dance':
                this.executeBehaviorDance(entity, deltaTime);
                break;
            case 'sing':
                this.executeBehaviorSing(entity, deltaTime);
                break;
            case 'meditate':
                this.executeBehaviorMeditate(entity, deltaTime);
                break;
        }
    }

    // Behavior Implementations
    executeBehaviorIdle(entity, deltaTime) {
        // Occasionally look around or make small movements
        if (Math.random() < 0.01) {
            entity.rotation.y += (Math.random() - 0.5) * 0.5;
        }

        // Slight random movement
        if (Math.random() < 0.005) {
            const direction = Math.random() * Math.PI * 2;
            entity.velocity.x = Math.cos(direction) * 0.5;
            entity.velocity.z = Math.sin(direction) * 0.5;
        }
    }

    executeBehaviorWander(entity, deltaTime) {
        // Generate random movement if no current path
        if (entity.path.length === 0) {
            const targetDistance = 5 + Math.random() * 15;
            const direction = Math.random() * Math.PI * 2;

            const target = {
                x: entity.position.x + Math.cos(direction) * targetDistance,
                z: entity.position.z + Math.sin(direction) * targetDistance
            };

            entity.path = [target];
        }

        this.moveTowardsTarget(entity, entity.path[0], deltaTime);

        // Remove reached targets
        if (this.distanceToTarget(entity, entity.path[0]) < 2) {
            entity.path.shift();
        }
    }

    executeBehaviorFlock(entity, deltaTime) {
        const nearbyEntities = this.getNearbyEntities(entity, 15);
        const flockmates = nearbyEntities.filter(e => e.type === entity.type);

        if (flockmates.length > 0) {
            // Calculate flocking forces
            const separation = this.calculateSeparation(entity, flockmates);
            const alignment = this.calculateAlignment(entity, flockmates);
            const cohesion = this.calculateCohesion(entity, flockmates);

            // Apply forces
            entity.velocity.x += (separation.x * 0.5 + alignment.x * 0.3 + cohesion.x * 0.2) * deltaTime * 0.001;
            entity.velocity.z += (separation.z * 0.5 + alignment.z * 0.3 + cohesion.z * 0.2) * deltaTime * 0.001;
        } else {
            // No flockmates, switch to wandering
            this.setBehavior(entity, 'wander');
        }
    }

    executeBehaviorDance(entity, deltaTime) {
        // Audio-reactive dancing behavior
        const audioState = this.audioInfluence.currentState;

        if (audioState && audioState.beatDetected) {
            const beatStrength = audioState.beatStrength || 0.5;

            // Rhythmic movement
            const time = Date.now() * 0.001;
            const beatTime = time * (audioState.bpm || 120) / 60;

            entity.position.y += Math.sin(beatTime * Math.PI * 2) * beatStrength * 0.5;
            entity.rotation.y += Math.sin(beatTime * Math.PI) * beatStrength * 0.2;

            // Dance movement pattern
            const danceRadius = 3 * beatStrength;
            entity.velocity.x = Math.cos(beatTime) * danceRadius * 0.1;
            entity.velocity.z = Math.sin(beatTime * 0.7) * danceRadius * 0.1;
        }
    }

    executeBehaviorSing(entity, deltaTime) {
        // Audio response behavior - creates harmonic responses
        if (Math.random() < 0.1) {
            this.createAudioEvent(entity, 'harmonic_vocalization');
        }

        // Gentle swaying motion
        const time = Date.now() * 0.001;
        entity.rotation.y += Math.sin(time * 0.5) * 0.05;
    }

    executeBehaviorMeditate(entity, deltaTime) {
        // Stationary behavior with energy restoration
        entity.velocity.x = 0;
        entity.velocity.z = 0;

        // Restore energy
        entity.energy = Math.min(100, entity.energy + deltaTime * 0.01);

        // Create peaceful aura effect
        if (Math.random() < 0.02) {
            this.createParticleEffect(entity, 'meditation_aura');
        }
    }

    // Audio Influence System
    updateAudioInfluence(entity, audioData) {
        const currentEmotion = audioData.heart?.emotionalState || 'neutral';
        const beatDetected = audioData.beat?.detected || false;
        const bassIntensity = audioData.frequencyBands?.bass?.value || 0;

        // Update current audio state
        this.audioInfluence.currentState = {
            emotionalState: currentEmotion,
            beatDetected: beatDetected,
            beatStrength: bassIntensity,
            bpm: audioData.beat?.bpm || 120
        };

        // Modify entity emotional state based on audio
        this.modifyEmotionalState(entity, currentEmotion, bassIntensity);

        // Beat synchronization
        if (beatDetected && entity.beatSynchronization) {
            this.synchronizeWithBeat(entity, audioData.beat);
        }
    }

    modifyEmotionalState(entity, audioEmotion, intensity) {
        const modifier = this.audioInfluence.moodModifiers[audioEmotion];
        if (!modifier) return;

        const currentState = this.emotionalStates[entity.emotionalState];
        const influence = entity.audioSensitivity * intensity;

        // Gradually shift emotional state
        Object.keys(modifier).forEach(trait => {
            if (currentState[trait] !== undefined) {
                currentState[trait] += modifier[trait] * influence * 0.01;
                currentState[trait] = Math.max(0, Math.min(1, currentState[trait]));
            }
        });

        // Determine new emotional state
        entity.emotionalState = this.determineEmotionalState(currentState);
    }

    synchronizeWithBeat(entity, beatData) {
        const now = Date.now();
        if (now - entity.lastBeatResponse < 200) return; // Minimum 200ms between responses

        entity.lastBeatResponse = now;

        // Beat-synchronized actions
        if (entity.currentBehavior === 'dance') {
            this.createParticleEffect(entity, 'beat_pulse');
        }

        // Movement synchronization
        if (this.audioInfluence.beatResponse.synchronization > 0.5) {
            const beatForce = this.audioInfluence.beatResponse.movementInfluence;
            entity.velocity.y += beatForce * 0.5;
        }
    }

    // Navigation and Pathfinding
    findPath(startPos, endPos) {
        // Simplified A* pathfinding
        if (!this.navigationMesh) {
            return [endPos]; // Direct path if no nav mesh
        }

        // Implement A* pathfinding algorithm here
        return this.aStar(startPos, endPos);
    }

    aStar(start, goal) {
        // Simplified A* implementation
        const openSet = [start];
        const cameFrom = new Map();
        const gScore = new Map();
        const fScore = new Map();

        gScore.set(this.positionKey(start), 0);
        fScore.set(this.positionKey(start), this.heuristic(start, goal));

        while (openSet.length > 0) {
            // Get node with lowest fScore
            const current = openSet.reduce((lowest, node) =>
                fScore.get(this.positionKey(node)) < fScore.get(this.positionKey(lowest)) ? node : lowest,
                openSet[0]
            ); if (this.distance(current, goal) < 2) {
                return this.reconstructPath(cameFrom, current);
            }

            openSet.splice(openSet.indexOf(current), 1);

            // Check neighbors (simplified 8-directional)
            const neighbors = this.getNeighbors(current);
            for (const neighbor of neighbors) {
                const tentativeGScore = gScore.get(this.positionKey(current)) + this.distance(current, neighbor);

                if (!gScore.has(this.positionKey(neighbor)) || tentativeGScore < gScore.get(this.positionKey(neighbor))) {
                    cameFrom.set(this.positionKey(neighbor), current);
                    gScore.set(this.positionKey(neighbor), tentativeGScore);
                    fScore.set(this.positionKey(neighbor), tentativeGScore + this.heuristic(neighbor, goal));

                    if (!openSet.includes(neighbor)) {
                        openSet.push(neighbor);
                    }
                }
            }
        }

        return [goal]; // Fallback direct path
    }

    // Movement and Physics
    moveTowardsTarget(entity, target, deltaTime) {
        const dx = target.x - entity.position.x;
        const dz = target.z - entity.position.z;
        const distance = Math.sqrt(dx * dx + dz * dz);

        if (distance > 0.1) {
            const speed = entity.speed * (deltaTime / 1000);
            entity.velocity.x = (dx / distance) * speed;
            entity.velocity.z = (dz / distance) * speed;

            // Update rotation to face movement direction
            entity.rotation.y = Math.atan2(dz, dx);
        }
    }

    updateMovement(entity, deltaTime) {
        // Apply velocity to position
        const dt = deltaTime / 1000;
        entity.position.x += entity.velocity.x * dt;
        entity.position.y += entity.velocity.y * dt;
        entity.position.z += entity.velocity.z * dt;

        // Apply friction
        entity.velocity.x *= 0.95;
        entity.velocity.z *= 0.95;
        entity.velocity.y *= 0.98; // Gravity-like effect

        // Keep entities above ground (simple terrain collision)
        if (entity.position.y < 0) {
            entity.position.y = 0;
            entity.velocity.y = 0;
        }
    }

    // Utility Functions
    getNearbyEntities(entity, radius = null) {
        const searchRadius = radius || entity.perceptionRadius;
        const nearby = [];

        for (const [id, other] of this.entities) {
            if (id === entity.id) continue;

            const distance = this.distance(entity.position, other.position);
            if (distance <= searchRadius) {
                nearby.push(other);
            }
        }

        return nearby;
    }

    distance(pos1, pos2) {
        const dx = pos1.x - pos2.x;
        const dy = (pos1.y || 0) - (pos2.y || 0);
        const dz = pos1.z - pos2.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    distanceToTarget(entity, target) {
        return this.distance(entity.position, target);
    }

    randomBetween(min, max) {
        return min + Math.random() * (max - min);
    }

    generateEntityId() {
        return 'ai-entity-' + Math.random().toString(36).substr(2, 9);
    }

    weightedRandomSelect(scoreMap) {
        const totalWeight = Array.from(scoreMap.values()).reduce((sum, weight) => sum + weight, 0);
        let random = Math.random() * totalWeight;

        for (const [item, weight] of scoreMap) {
            random -= weight;
            if (random <= 0) {
                return item;
            }
        }

        return Array.from(scoreMap.keys())[0]; // Fallback
    }

    // Event Creators
    createAudioEvent(entity, eventType) {
        console.log(`🎵 Audio event: ${entity.id} - ${eventType}`);
        // Trigger audio event in the main system
    }

    createParticleEffect(entity, effectType) {
        console.log(`✨ Particle effect: ${entity.id} - ${effectType}`);
        // Trigger particle effect in the main system
    }

    // Public API
    getEntityCount() {
        return this.entities.size;
    }

    getEntitiesByType(type) {
        return Array.from(this.entities.values()).filter(entity => entity.type === type);
    }

    removeEntity(entityId) {
        if (this.entities.has(entityId)) {
            this.entities.delete(entityId);
            console.log(`🗑️ Removed AI entity: ${entityId}`);
            return true;
        }
        return false;
    }

    setAudioReactivity(enabled) {
        this.audioInfluence.beatResponse.enabled = enabled;
        console.log(`🎵 AI audio reactivity: ${enabled ? 'enabled' : 'disabled'}`);
    }

    getAIStats() {
        const stats = {
            totalEntities: this.entities.size,
            behaviorDistribution: {},
            emotionalDistribution: {},
            averageEnergy: 0
        };

        let totalEnergy = 0;
        for (const entity of this.entities.values()) {
            // Count behaviors
            stats.behaviorDistribution[entity.currentBehavior] =
                (stats.behaviorDistribution[entity.currentBehavior] || 0) + 1;

            // Count emotional states
            stats.emotionalDistribution[entity.emotionalState] =
                (stats.emotionalDistribution[entity.emotionalState] || 0) + 1;

            totalEnergy += entity.energy;
        }

        stats.averageEnergy = this.entities.size > 0 ? totalEnergy / this.entities.size : 0;
        return stats;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeAISystem;
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    window.NexusForgeAISystem = NexusForgeAISystem;
    console.log('🧠 AI Behavior System available globally');
}
