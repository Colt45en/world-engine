/**
 * World Engine Runtime
 * Core Nexus class with phonics, story, scripting, WebSocket integration
 */

import { WorldEngineLoader } from './loader.js';
import { LLEMath, GlyphCollationMap } from '../src/types/Math Pro.js';

export class WorldEngineRuntime {
    constructor(options = {}) {
        this.loader = new WorldEngineLoader(options.basePath);
        this.domains = {};
        this.services = {
            phonics: null,
            story: null,
            scripting: null,
            websocket: null,
            vault: null,
            mathpro: null
        };
        this.initialized = false;
    }

    async initialize() {
        console.log('🌍 Initializing World Engine...');

        // Load all domain data
        this.domains = await this.loader.loadAllDomains();

        // Initialize core services
        await this.initializeServices();

        this.initialized = true;
        console.log('✅ World Engine initialized');
        return this;
    }

    async initializeServices() {
        const manifest = this.loader.getManifest();

        // Initialize Math Pro service
        if (manifest.services?.mathpro?.enabled) {
            this.services.mathpro = {
                LLEMath,
                GlyphCollationMap: new GlyphCollationMap(),
                operators: this.domains.math?.operators || {},
                mappings: this.domains.math?.mappings || {}
            };
            console.log('🧮 Math Pro service initialized');
        }

        // Initialize Phonics service
        if (manifest.services?.phonics?.enabled) {
            this.services.phonics = new PhonicService(
                this.domains.english?.phonics_core,
                this.domains.english?.lexicon
            );
            console.log('🗣️ Phonics service initialized');
        }

        // Initialize Story service
        if (manifest.services?.story?.enabled) {
            this.services.story = new StoryService(
                this.domains.english?.grammar,
                this.domains.english?.lexicon
            );
            console.log('📖 Story service initialized');
        }

        // Initialize Scripting service
        if (manifest.services?.scripting?.enabled) {
            this.services.scripting = new ScriptingService(
                this.domains.gaming?.scripting,
                this.domains.math?.operators
            );
            console.log('⚡ Scripting service initialized');
        }

        // Initialize VectorLab Nexus service (AI Chatbot + Brain Integration)
        if (manifest.services?.vectorlab_nexus?.enabled) {
            this.services.vectorlab_nexus = {
                connection_bridge: 'nexus-merge-bridge',
                websocket_url: 'ws://localhost:9000',
                live_viewer: 'http://localhost:7777',
                brain_integration: true,
                chatbot_persistent: true,
                glyph_bridge: this.domains.tools?.vectorlab_nexus || {},
                ai_connection: this.domains.tools?.ai_chatbot_connection || {}
            };
            console.log('🧠 VectorLab Nexus service initialized with persistent AI chatbot connection');
        }
    }

    // Domain access methods
    getGaming() {
        return this.domains.gaming;
    }

    getEnglish() {
        return this.domains.english;
    }

    getMath() {
        return this.domains.math;
    }

    getGraphics() {
        return this.domains.graphics;
    }

    // Service access methods
    getPhonics() {
        return this.services.phonics;
    }

    getStory() {
        return this.services.story;
    }

    getScripting() {
        return this.services.scripting;
    }

    getMathPro() {
        return this.services.mathpro;
    }

    // VectorLab Nexus service accessor (AI Chatbot + Brain Integration)
    getVectorLabNexus() {
        return this.services.vectorlab_nexus;
    }

    // Utility methods
    getDomain(name) {
        return this.domains[name];
    }

    getService(name) {
        return this.services[name];
    }

    isInitialized() {
        return this.initialized;
    }
}

// Core service classes
class PhonicService {
    constructor(phonicsCore, lexicon) {
        this.phonicsCore = phonicsCore || {};
        this.lexicon = lexicon || [];
        this.phonemeMap = new Map();
        this.buildPhonemeMap();
    }

    buildPhonemeMap() {
        if (Array.isArray(this.lexicon)) {
            this.lexicon.forEach(entry => {
                if (entry.word && entry.phonetic) {
                    this.phonemeMap.set(entry.word, entry.phonetic);
                }
            });
        }
    }

    getPhonetic(word) {
        return this.phonemeMap.get(word.toLowerCase());
    }

    blendPhonemes(phonemes) {
        // Basic blending logic - can be enhanced
        return phonemes.join('');
    }
}

class StoryService {
    constructor(grammar, lexicon) {
        this.grammar = grammar || {};
        this.lexicon = lexicon || [];
        this.beats = this.grammar.beats || [];
        this.arcs = this.grammar.arcs || [];
    }

    generateStoryBeat(context = {}) {
        if (this.beats.length === 0) return null;

        const randomBeat = this.beats[Math.floor(Math.random() * this.beats.length)];
        return {
            ...randomBeat,
            context,
            timestamp: Date.now()
        };
    }

    getGrammarRules() {
        return this.grammar;
    }
}

class ScriptingService {
    constructor(scriptingPacks, mathOperators) {
        this.opcodes = scriptingPacks?.opcodes || {};
        this.eventHooks = scriptingPacks?.events || {};
        this.mathOperators = mathOperators || {};
        this.executionContext = new Map();
    }

    executeOpcode(opcode, args = []) {
        const opcodeDefinition = this.opcodes[opcode];
        if (!opcodeDefinition) {
            throw new Error(`Unknown opcode: ${opcode}`);
        }

        // Basic execution - can be enhanced with proper bytecode interpreter
        return {
            opcode,
            args,
            result: `executed_${opcode}`,
            timestamp: Date.now()
        };
    }

    registerEventHook(event, callback) {
        if (!this.eventHooks[event]) {
            this.eventHooks[event] = [];
        }
        this.eventHooks[event].push(callback);
    }

    triggerEvent(event, data) {
        const hooks = this.eventHooks[event];
        if (hooks && Array.isArray(hooks)) {
            hooks.forEach(hook => {
                try {
                    if (typeof hook === 'function') {
                        hook(data);
                    }
                } catch (error) {
                    console.error(`Event hook error for ${event}:`, error);
                }
            });
        }
    }
}

export { WorldEngineRuntime, PhonicService, StoryService, ScriptingService };
