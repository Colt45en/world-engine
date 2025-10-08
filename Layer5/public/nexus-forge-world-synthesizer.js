/**
 * NEXUS Forge World Synthesizer
 * • Advanced procedural world generation
 * • Biome-based ecosystem simulation
 * • Audio-reactive environmental changes
 * • Real-time world streaming and optimization
 * • Multi-layered noise generation for realistic terrain
 */

class NexusForgeWorldSynthesizer {
    constructor() {
        this.worldSeed = Math.random() * 1000000;
        this.chunkSize = 64;
        this.worldChunks = new Map();
        this.activeChunks = new Set();
        this.loadedRadius = 8;

        this.biomes = this.initializeBiomes();
        this.noiseGenerators = this.initializeNoiseGenerators();
        this.ecosystems = new Map();

        this.audioInfluence = {
            enabled: true,
            bassAmplitude: 0,
            midFreqShift: 0,
            trebleDetail: 0,
            beatPulse: 0,
            emotionalBias: 'neutral'
        };

        this.worldParameters = {
            seaLevel: 0,
            mountainHeight: 100,
            temperatureRange: [-20, 35],
            humidityRange: [0, 100],
            erosionStrength: 0.3,
            vegetationDensity: 0.7,
            resourceDistribution: 0.5
        };

        this.generationLayers = {
            heightmap: { scale: 0.01, octaves: 6, persistence: 0.5 },
            temperature: { scale: 0.005, octaves: 3, persistence: 0.6 },
            humidity: { scale: 0.007, octaves: 4, persistence: 0.4 },
            erosion: { scale: 0.02, octaves: 2, persistence: 0.3 },
            caves: { scale: 0.03, octaves: 3, persistence: 0.5, threshold: 0.6 },
            resources: { scale: 0.015, octaves: 2, persistence: 0.4 }
        };

        console.log('🌍 World Synthesizer initialized');
    }

    initializeBiomes() {
        return {
            ocean: {
                name: 'Ocean',
                color: [0, 100, 200],
                heightRange: [-50, -1],
                temperatureRange: [5, 25],
                humidityRange: [80, 100],
                vegetation: 0.1,
                resources: ['salt', 'fish', 'kelp']
            },
            beach: {
                name: 'Beach',
                color: [255, 235, 205],
                heightRange: [-1, 5],
                temperatureRange: [15, 30],
                humidityRange: [60, 85],
                vegetation: 0.2,
                resources: ['sand', 'shells', 'driftwood']
            },
            grassland: {
                name: 'Grassland',
                color: [100, 200, 100],
                heightRange: [5, 30],
                temperatureRange: [10, 25],
                humidityRange: [40, 70],
                vegetation: 0.8,
                resources: ['grass', 'flowers', 'herbs']
            },
            forest: {
                name: 'Forest',
                color: [50, 150, 50],
                heightRange: [20, 60],
                temperatureRange: [5, 20],
                humidityRange: [60, 90],
                vegetation: 0.95,
                resources: ['wood', 'berries', 'mushrooms', 'wildlife']
            },
            hills: {
                name: 'Hills',
                color: [120, 100, 80],
                heightRange: [30, 70],
                temperatureRange: [0, 15],
                humidityRange: [30, 60],
                vegetation: 0.6,
                resources: ['stone', 'clay', 'coal']
            },
            mountains: {
                name: 'Mountains',
                color: [150, 140, 130],
                heightRange: [70, 150],
                temperatureRange: [-10, 5],
                humidityRange: [20, 50],
                vegetation: 0.3,
                resources: ['stone', 'metals', 'gems', 'snow']
            },
            desert: {
                name: 'Desert',
                color: [255, 220, 150],
                heightRange: [5, 40],
                temperatureRange: [20, 45],
                humidityRange: [0, 20],
                vegetation: 0.1,
                resources: ['sand', 'cacti', 'oil', 'salt']
            },
            tundra: {
                name: 'Tundra',
                color: [200, 220, 255],
                heightRange: [10, 50],
                temperatureRange: [-25, 0],
                humidityRange: [30, 70],
                vegetation: 0.2,
                resources: ['ice', 'lichen', 'permafrost']
            }
        };
    }

    initializeNoiseGenerators() {
        return {
            heightmap: new ImprovedNoise(this.worldSeed),
            temperature: new ImprovedNoise(this.worldSeed + 1000),
            humidity: new ImprovedNoise(this.worldSeed + 2000),
            erosion: new ImprovedNoise(this.worldSeed + 3000),
            caves: new ImprovedNoise(this.worldSeed + 4000),
            resources: new ImprovedNoise(this.worldSeed + 5000),
            audio_bass: new ImprovedNoise(this.worldSeed + 6000),
            audio_harmony: new ImprovedNoise(this.worldSeed + 7000)
        };
    }

    generateWorldChunk(chunkX, chunkZ) {
        const chunk = {
            x: chunkX,
            z: chunkZ,
            heightmap: new Array(this.chunkSize * this.chunkSize),
            biomes: new Array(this.chunkSize * this.chunkSize),
            temperature: new Array(this.chunkSize * this.chunkSize),
            humidity: new Array(this.chunkSize * this.chunkSize),
            vegetation: new Array(this.chunkSize * this.chunkSize),
            resources: new Map(),
            structures: [],
            entities: [],
            audioModifications: new Array(this.chunkSize * this.chunkSize)
        };

        // Generate base terrain layers
        this.generateHeightmap(chunk);
        this.generateClimate(chunk);
        this.determineBiomes(chunk);
        this.generateVegetation(chunk);
        this.generateResources(chunk);
        this.generateCaves(chunk);

        // Apply audio influences
        if (this.audioInfluence.enabled) {
            this.applyAudioInfluences(chunk);
        }

        // Generate structures and entities
        this.generateStructures(chunk);
        this.populateEntities(chunk);

        return chunk;
    }

    generateHeightmap(chunk) {
        const heightLayer = this.generationLayers.heightmap;

        for (let z = 0; z < this.chunkSize; z++) {
            for (let x = 0; x < this.chunkSize; x++) {
                const worldX = chunk.x * this.chunkSize + x;
                const worldZ = chunk.z * this.chunkSize + z;

                let height = 0;
                let amplitude = 1;
                let frequency = heightLayer.scale;

                // Multi-octave noise generation
                for (let octave = 0; octave < heightLayer.octaves; octave++) {
                    height += this.noiseGenerators.heightmap.noise(
                        worldX * frequency,
                        worldZ * frequency,
                        0
                    ) * amplitude;

                    amplitude *= heightLayer.persistence;
                    frequency *= 2;
                }

                // Apply erosion
                const erosionValue = this.generateErosion(worldX, worldZ);
                height *= (1 - erosionValue * this.worldParameters.erosionStrength);

                // Scale to world height range
                height = height * this.worldParameters.mountainHeight;

                // Apply audio bass influence for dynamic terrain
                if (this.audioInfluence.enabled && this.audioInfluence.bassAmplitude > 0.1) {
                    const bassNoise = this.noiseGenerators.audio_bass.noise(worldX * 0.05, worldZ * 0.05, 0);
                    height += bassNoise * this.audioInfluence.bassAmplitude * 10;
                }

                chunk.heightmap[z * this.chunkSize + x] = height;
            }
        }
    }

    generateClimate(chunk) {
        const tempLayer = this.generationLayers.temperature;
        const humidLayer = this.generationLayers.humidity;

        for (let z = 0; z < this.chunkSize; z++) {
            for (let x = 0; x < this.chunkSize; x++) {
                const worldX = chunk.x * this.chunkSize + x;
                const worldZ = chunk.z * this.chunkSize + z;
                const height = chunk.heightmap[z * this.chunkSize + x];
                const index = z * this.chunkSize + x;

                // Temperature calculation with latitude and altitude effects
                let temperature = this.generateLayeredNoise(worldX, worldZ, tempLayer, this.noiseGenerators.temperature);
                temperature += (worldZ * 0.001); // Latitude effect
                temperature -= (height * 0.5); // Altitude effect
                temperature = this.mapToRange(temperature, this.worldParameters.temperatureRange);

                // Humidity calculation
                let humidity = this.generateLayeredNoise(worldX, worldZ, humidLayer, this.noiseGenerators.humidity);
                humidity += Math.max(0, -height * 0.3); // Lower areas more humid
                humidity = this.mapToRange(humidity, this.worldParameters.humidityRange);

                // Apply audio influence to climate
                if (this.audioInfluence.enabled) {
                    const audioClimate = this.calculateAudioClimateEffect(worldX, worldZ);
                    temperature += audioClimate.temperatureMod;
                    humidity += audioClimate.humidityMod;
                }

                chunk.temperature[index] = temperature;
                chunk.humidity[index] = humidity;
            }
        }
    }

    determineBiomes(chunk) {
        for (let z = 0; z < this.chunkSize; z++) {
            for (let x = 0; x < this.chunkSize; x++) {
                const index = z * this.chunkSize + x;
                const height = chunk.heightmap[index];
                const temperature = chunk.temperature[index];
                const humidity = chunk.humidity[index];

                let bestBiome = null;
                let bestScore = Infinity;

                // Find best matching biome
                for (const [biomeName, biome] of Object.entries(this.biomes)) {
                    const score = this.calculateBiomeScore(height, temperature, humidity, biome);
                    if (score < bestScore) {
                        bestScore = score;
                        bestBiome = biomeName;
                    }
                }

                chunk.biomes[index] = bestBiome;
            }
        }
    }

    calculateBiomeScore(height, temperature, humidity, biome) {
        let score = 0;

        // Height score
        if (height < biome.heightRange[0] || height > biome.heightRange[1]) {
            score += Math.abs(Math.min(height - biome.heightRange[0], biome.heightRange[1] - height)) * 2;
        }

        // Temperature score
        if (temperature < biome.temperatureRange[0] || temperature > biome.temperatureRange[1]) {
            score += Math.abs(Math.min(temperature - biome.temperatureRange[0], biome.temperatureRange[1] - temperature));
        }

        // Humidity score
        if (humidity < biome.humidityRange[0] || humidity > biome.humidityRange[1]) {
            score += Math.abs(Math.min(humidity - biome.humidityRange[0], biome.humidityRange[1] - humidity)) * 0.5;
        }

        return score;
    }

    generateVegetation(chunk) {
        for (let z = 0; z < this.chunkSize; z++) {
            for (let x = 0; x < this.chunkSize; x++) {
                const index = z * this.chunkSize + x;
                const biomeName = chunk.biomes[index];
                const biome = this.biomes[biomeName];
                const height = chunk.heightmap[index];
                const temperature = chunk.temperature[index];
                const humidity = chunk.humidity[index];

                // Base vegetation density from biome
                let vegetation = biome.vegetation;

                // Modify based on local conditions
                vegetation *= this.calculateVegetationModifier(temperature, humidity);

                // Add noise variation
                const worldX = chunk.x * this.chunkSize + x;
                const worldZ = chunk.z * this.chunkSize + z;
                const vegetationNoise = this.noiseGenerators.resources.noise(worldX * 0.1, worldZ * 0.1, 0);
                vegetation *= (0.8 + vegetationNoise * 0.4);

                // Audio influence on vegetation growth
                if (this.audioInfluence.enabled && this.audioInfluence.trebleDetail > 0.2) {
                    const harmonyNoise = this.noiseGenerators.audio_harmony.noise(worldX * 0.02, worldZ * 0.02, 0);
                    vegetation += harmonyNoise * this.audioInfluence.trebleDetail * 0.3;
                }

                chunk.vegetation[index] = Math.max(0, Math.min(1, vegetation));
            }
        }
    }

    generateResources(chunk) {
        const resourceLayer = this.generationLayers.resources;

        for (let z = 0; z < this.chunkSize; z++) {
            for (let x = 0; x < this.chunkSize; x++) {
                const index = z * this.chunkSize + x;
                const worldX = chunk.x * this.chunkSize + x;
                const worldZ = chunk.z * this.chunkSize + z;
                const biomeName = chunk.biomes[index];
                const biome = this.biomes[biomeName];
                const height = chunk.heightmap[index];

                // Generate resource deposits
                const resourceNoise = this.generateLayeredNoise(worldX, worldZ, resourceLayer, this.noiseGenerators.resources);

                if (Math.abs(resourceNoise) > 0.6) {
                    const availableResources = biome.resources;
                    const selectedResource = availableResources[Math.floor(Math.random() * availableResources.length)];
                    const abundance = Math.abs(resourceNoise) - 0.6;

                    if (!chunk.resources.has(selectedResource)) {
                        chunk.resources.set(selectedResource, []);
                    }

                    chunk.resources.get(selectedResource).push({
                        x: x,
                        z: z,
                        abundance: abundance,
                        depth: this.calculateResourceDepth(selectedResource, height)
                    });
                }
            }
        }
    }

    generateCaves(chunk) {
        const caveLayer = this.generationLayers.caves;

        for (let y = -20; y < 20; y += 2) {
            for (let z = 0; z < this.chunkSize; z++) {
                for (let x = 0; x < this.chunkSize; x++) {
                    const worldX = chunk.x * this.chunkSize + x;
                    const worldZ = chunk.z * this.chunkSize + z;

                    const caveNoise = this.noiseGenerators.caves.noise(
                        worldX * caveLayer.scale,
                        y * caveLayer.scale * 2,
                        worldZ * caveLayer.scale
                    );

                    if (Math.abs(caveNoise) < caveLayer.threshold) {
                        // Create cave opening
                        chunk.structures.push({
                            type: 'cave',
                            x: x,
                            y: y,
                            z: z,
                            size: (caveLayer.threshold - Math.abs(caveNoise)) * 5
                        });
                    }
                }
            }
        }
    }

    applyAudioInfluences(chunk) {
        for (let z = 0; z < this.chunkSize; z++) {
            for (let x = 0; x < this.chunkSize; x++) {
                const index = z * this.chunkSize + x;
                const worldX = chunk.x * this.chunkSize + x;
                const worldZ = chunk.z * this.chunkSize + z;

                // Create audio-reactive terrain modifications
                let audioMod = 0;

                // Bass creates rolling hills
                if (this.audioInfluence.bassAmplitude > 0.1) {
                    const bassWave = Math.sin(worldX * 0.02) * Math.cos(worldZ * 0.02);
                    audioMod += bassWave * this.audioInfluence.bassAmplitude * 5;
                }

                // Mid frequencies create detailed variations
                if (this.audioInfluence.midFreqShift > 0.1) {
                    const midNoise = this.noiseGenerators.audio_harmony.noise(worldX * 0.05, worldZ * 0.05, 0);
                    audioMod += midNoise * this.audioInfluence.midFreqShift * 3;
                }

                // Beat pulse creates temporary elevations
                if (this.audioInfluence.beatPulse > 0.5) {
                    const pulseDistance = Math.sqrt((worldX % 100 - 50) ** 2 + (worldZ % 100 - 50) ** 2);
                    if (pulseDistance < 20) {
                        audioMod += (20 - pulseDistance) * this.audioInfluence.beatPulse * 0.5;
                    }
                }

                chunk.audioModifications[index] = audioMod;
                chunk.heightmap[index] += audioMod;
            }
        }
    }

    generateStructures(chunk) {
        // Generate various structures based on biome and conditions
        for (let z = 5; z < this.chunkSize - 5; z += 10) {
            for (let x = 5; x < this.chunkSize - 5; x += 10) {
                const index = z * this.chunkSize + x;
                const biomeName = chunk.biomes[index];
                const vegetation = chunk.vegetation[index];
                const height = chunk.heightmap[index];

                // Structure generation probability
                if (Math.random() < 0.1) {
                    const structure = this.generateStructureForBiome(biomeName, x, z, height, vegetation);
                    if (structure) {
                        chunk.structures.push(structure);
                    }
                }
            }
        }
    }

    generateStructureForBiome(biomeName, x, z, height, vegetation) {
        const structures = {
            forest: ['tree_grove', 'wooden_shrine', 'mushroom_circle'],
            grassland: ['stone_circle', 'flower_field', 'ancient_mound'],
            mountains: ['crystal_formation', 'cave_entrance', 'peak_marker'],
            desert: ['oasis', 'sand_dune', 'ancient_ruins'],
            ocean: ['coral_reef', 'shipwreck', 'sea_mount'],
            tundra: ['ice_formation', 'frozen_lake', 'aurora_point']
        };

        const biomeStructures = structures[biomeName] || ['generic_landmark'];
        const structureType = biomeStructures[Math.floor(Math.random() * biomeStructures.length)];

        return {
            type: structureType,
            x: x,
            z: z,
            height: height,
            size: 2 + Math.random() * 4,
            orientation: Math.random() * 360,
            biomeSpecific: true
        };
    }

    populateEntities(chunk) {
        // Generate wildlife and other entities based on biome
        for (let i = 0; i < 20; i++) {
            const x = Math.floor(Math.random() * this.chunkSize);
            const z = Math.floor(Math.random() * this.chunkSize);
            const index = z * this.chunkSize + x;
            const biomeName = chunk.biomes[index];
            const vegetation = chunk.vegetation[index];

            if (Math.random() < vegetation * 0.5) {
                const entity = this.generateEntityForBiome(biomeName, x, z);
                if (entity) {
                    chunk.entities.push(entity);
                }
            }
        }
    }

    generateEntityForBiome(biomeName, x, z) {
        const entities = {
            forest: ['deer', 'rabbit', 'bird', 'squirrel'],
            grassland: ['horse', 'butterfly', 'bee', 'mouse'],
            ocean: ['fish', 'dolphin', 'whale', 'seagull'],
            mountains: ['eagle', 'goat', 'bear', 'falcon'],
            desert: ['lizard', 'snake', 'scorpion', 'cactus_flower'],
            tundra: ['penguin', 'seal', 'arctic_fox', 'polar_bear']
        };

        const biomeEntities = entities[biomeName] || ['generic_creature'];
        const entityType = biomeEntities[Math.floor(Math.random() * biomeEntities.length)];

        return {
            type: entityType,
            x: x,
            z: z,
            health: 100,
            behavior: 'wander',
            speed: 0.5 + Math.random() * 2,
            size: 0.5 + Math.random() * 1.5
        };
    }

    // Utility functions
    generateLayeredNoise(x, z, layer, generator) {
        let value = 0;
        let amplitude = 1;
        let frequency = layer.scale;

        for (let octave = 0; octave < layer.octaves; octave++) {
            value += generator.noise(x * frequency, z * frequency, 0) * amplitude;
            amplitude *= layer.persistence;
            frequency *= 2;
        }

        return value;
    }

    generateErosion(x, z) {
        const erosionLayer = this.generationLayers.erosion;
        return Math.abs(this.generateLayeredNoise(x, z, erosionLayer, this.noiseGenerators.erosion));
    }

    mapToRange(value, range) {
        const normalized = (value + 1) / 2; // Normalize from [-1,1] to [0,1]
        return range[0] + normalized * (range[1] - range[0]);
    }

    calculateVegetationModifier(temperature, humidity) {
        // Vegetation thrives in moderate temperature and high humidity
        const tempScore = 1 - Math.abs(temperature - 15) / 30;
        const humidScore = humidity / 100;
        return Math.max(0.1, tempScore * humidScore);
    }

    calculateResourceDepth(resourceType, height) {
        const depthMap = {
            'metals': Math.max(5, 20 - height * 0.2),
            'gems': Math.max(10, 30 - height * 0.1),
            'coal': Math.max(3, 15 - height * 0.3),
            'oil': Math.max(15, 40 - height * 0.1),
            'salt': 2,
            'sand': 1,
            'stone': Math.max(1, 10 - height * 0.1)
        };

        return depthMap[resourceType] || 5;
    }

    calculateAudioClimateEffect(x, z) {
        let temperatureMod = 0;
        let humidityMod = 0;

        // Emotional bias affects climate
        switch (this.audioInfluence.emotionalBias) {
            case 'excited':
                temperatureMod += 5;
                humidityMod -= 10;
                break;
            case 'calm':
                temperatureMod -= 2;
                humidityMod += 15;
                break;
            case 'energetic':
                temperatureMod += 3;
                humidityMod += 5;
                break;
        }

        return { temperatureMod, humidityMod };
    }

    // Public API
    updateAudioInfluence(audioData) {
        if (!audioData) return;

        this.audioInfluence.bassAmplitude = audioData.frequencyBands?.bass?.value || 0;
        this.audioInfluence.midFreqShift = audioData.frequencyBands?.mid?.value || 0;
        this.audioInfluence.trebleDetail = audioData.frequencyBands?.treble?.value || 0;
        this.audioInfluence.beatPulse = audioData.beat?.detected ? 1.0 : 0.0;
        this.audioInfluence.emotionalBias = audioData.heart?.emotionalState || 'neutral';
    }

    getChunk(chunkX, chunkZ) {
        const key = `${chunkX},${chunkZ}`;

        if (!this.worldChunks.has(key)) {
            const chunk = this.generateWorldChunk(chunkX, chunkZ);
            this.worldChunks.set(key, chunk);
            console.log(`🌍 Generated chunk [${chunkX}, ${chunkZ}]`);
        }

        return this.worldChunks.get(key);
    }

    updateActiveChunks(centerX, centerZ) {
        this.activeChunks.clear();

        for (let z = centerZ - this.loadedRadius; z <= centerZ + this.loadedRadius; z++) {
            for (let x = centerX - this.loadedRadius; x <= centerX + this.loadedRadius; x++) {
                const distance = Math.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2);
                if (distance <= this.loadedRadius) {
                    const key = `${x},${z}`;
                    this.activeChunks.add(key);

                    // Generate chunk if needed
                    this.getChunk(x, z);
                }
            }
        }

        // Clean up distant chunks
        for (const [key, chunk] of this.worldChunks) {
            if (!this.activeChunks.has(key)) {
                const [x, z] = key.split(',').map(Number);
                const distance = Math.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2);
                if (distance > this.loadedRadius * 2) {
                    this.worldChunks.delete(key);
                }
            }
        }
    }

    getBiomeAt(x, z) {
        const chunkX = Math.floor(x / this.chunkSize);
        const chunkZ = Math.floor(z / this.chunkSize);
        const chunk = this.getChunk(chunkX, chunkZ);

        const localX = x - chunkX * this.chunkSize;
        const localZ = z - chunkZ * this.chunkSize;
        const index = localZ * this.chunkSize + localX;

        return chunk.biomes[index];
    }

    getHeightAt(x, z) {
        const chunkX = Math.floor(x / this.chunkSize);
        const chunkZ = Math.floor(z / this.chunkSize);
        const chunk = this.getChunk(chunkX, chunkZ);

        const localX = x - chunkX * this.chunkSize;
        const localZ = z - chunkZ * this.chunkSize;
        const index = localZ * this.chunkSize + localX;

        return chunk.heightmap[index];
    }

    getWorldInfo() {
        return {
            seed: this.worldSeed,
            chunksGenerated: this.worldChunks.size,
            activeChunks: this.activeChunks.size,
            biomes: Object.keys(this.biomes).length,
            audioInfluenceEnabled: this.audioInfluence.enabled
        };
    }
}

// Improved Perlin Noise implementation
class ImprovedNoise {
    constructor(seed = 0) {
        this.seed = seed;
        this.p = new Array(512);
        this.permutation = [
            151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
            190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
            77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
            135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
            223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
            251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
            138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
        ];

        // Shuffle based on seed
        for (let i = 0; i < 256; i++) {
            const j = Math.floor(this.seededRandom() * (i + 1));
            [this.permutation[i], this.permutation[j]] = [this.permutation[j], this.permutation[i]];
        }

        // Duplicate the permutation array
        for (let i = 0; i < 256; i++) {
            this.p[256 + i] = this.p[i] = this.permutation[i];
        }
    }

    seededRandom() {
        this.seed = (this.seed * 9301 + 49297) % 233280;
        return this.seed / 233280;
    }

    noise(x, y, z) {
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y) & 255;
        const Z = Math.floor(z) & 255;

        x -= Math.floor(x);
        y -= Math.floor(y);
        z -= Math.floor(z);

        const u = this.fade(x);
        const v = this.fade(y);
        const w = this.fade(z);

        const A = this.p[X] + Y;
        const AA = this.p[A] + Z;
        const AB = this.p[A + 1] + Z;
        const B = this.p[X + 1] + Y;
        const BA = this.p[B] + Z;
        const BB = this.p[B + 1] + Z;

        return this.lerp(w,
            this.lerp(v,
                this.lerp(u, this.grad(this.p[AA], x, y, z),
                    this.grad(this.p[BA], x - 1, y, z)),
                this.lerp(u, this.grad(this.p[AB], x, y - 1, z),
                    this.grad(this.p[BB], x - 1, y - 1, z))),
            this.lerp(v,
                this.lerp(u, this.grad(this.p[AA + 1], x, y, z - 1),
                    this.grad(this.p[BA + 1], x - 1, y, z - 1)),
                this.lerp(u, this.grad(this.p[AB + 1], x, y - 1, z - 1),
                    this.grad(this.p[BB + 1], x - 1, y - 1, z - 1))));
    }

    fade(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    lerp(t, a, b) {
        return a + t * (b - a);
    }

    grad(hash, x, y, z) {
        const h = hash & 15;
        const u = h < 8 ? x : y;
        const v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeWorldSynthesizer;
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    window.NexusForgeWorldSynthesizer = NexusForgeWorldSynthesizer;
    console.log('🌍 World Synthesizer available globally');
}
