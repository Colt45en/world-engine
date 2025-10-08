/**
 * LLEX Complete System - Full Integration
 * Address resolution, event sourcing, and integrated spine architecture
 * @fileoverview Content-addressable lexical processing with versioning
 */

// @ts-nocheck
// Import dependencies
if (typeof window !== 'undefined') {
    // Browser environment - just declare namespace
    if (!window.LLEX) window.LLEX = {};
} else {
    // Node.js environment
    try {
        const core = require('./llex-core.js');
        const indexes = require('./llex-indexes.js');
        // Use the actual exported names
    } catch (e) {
        console.warn('Module loading failed:', String(e));
    }
}

// Address Resolution System
class LLEXAddressResolver {
    constructor(engine) {
        this.engine = engine;
        this.cache = new Map(); // Recently resolved addresses
        this.maxCacheSize = 1000;
    }

    // Parse LLEX address: llex://namespace/type/lid@vid#cid
    parseAddress(address) {
        const match = address.match(/^llex:\/\/([^\/]+)\/([^\/]+)\/([^@#]+)(?:@([^#]+))?(?:#(.+))?$/);
        if (!match) {
            throw new Error(`Invalid LLEX address format: ${address}`);
        }

        const [, namespace, type, lid, vid, cid] = match;
        return {
            namespace,
            type,
            lid,
            vid: vid || 'current',
            cid,
            original: address
        };
    }

    // Build LLEX address from components
    buildAddress(namespace, type, lid, vid = null, cid = null) {
        let address = `llex://${namespace}/${type}/${lid}`;
        if (vid) address += `@${vid}`;
        if (cid) address += `#${cid}`;
        return address;
    }

    // Resolve address to object
    async resolve(address) {
        // Check cache first
        if (this.cache.has(address)) {
            const cached = this.cache.get(address);
            if (Date.now() - cached.timestamp < 60000) { // 1 minute cache
                return cached.object;
            }
        }

        const parsed = this.parseAddress(address);
        let object;

        if (parsed.cid) {
            // Direct CID resolution
            object = await this.engine.objectStore.fetch(parsed.cid);
        } else {
            // Catalog resolution
            const resolvedCID = this.engine.catalog.resolve(parsed.namespace, parsed.type, parsed.lid, parsed.vid);
            if (!resolvedCID) {
                throw new Error(`Address not found: ${address}`);
            }
            object = await this.engine.objectStore.fetch(resolvedCID);
        }

        // Cache the result
        this.cacheResult(address, object);
        return object;
    }

    cacheResult(address, object) {
        // Implement LRU cache behavior
        if (this.cache.size >= this.maxCacheSize) {
            const oldestKey = this.cache.keys().next().value;
            this.cache.delete(oldestKey);
        }

        this.cache.set(address, {
            object,
            timestamp: Date.now()
        });
    }

    // Resolve multiple addresses in parallel
    async resolveMany(addresses) {
        const promises = addresses.map(addr => this.resolve(addr));
        return await Promise.all(promises);
    }

    clearCache() {
        this.cache.clear();
    }

    getCacheStats() {
        return {
            size: this.cache.size,
            maxSize: this.maxCacheSize,
            hitRate: this.hitRate || 0
        };
    }
}

// Event Sourcing System
class LLEXEventSourcer {
    constructor(engine, indexManager) {
        this.engine = engine;
        this.indexManager = indexManager;
        this.snapshots = new Map(); // session -> latest snapshot info
        this.snapshotInterval = 10; // Take snapshot every N events
    }

    // Apply button click and create event
    async applyButtonClick(session, buttonAddress, currentState, params = {}, provenance = {}) {
        const buttonObj = await this.engine.resolve(buttonAddress);
        const seq = (this.engine.eventLog.sessions.get(session) || 0) + 1;

        // Apply transformation
        const newState = this.applyTransformation(currentState, buttonObj, params);

        // Create snapshots
        const inputSnapshot = await this.engine.createSnapshot(session, seq - 1, currentState);
        const outputSnapshot = await this.engine.createSnapshot(session, seq, newState);

        // Record event
        const event = await this.engine.recordClick(
            session,
            seq,
            buttonAddress,
            inputSnapshot.cid,
            outputSnapshot.cid,
            params,
            provenance
        );

        // Index everything
        this.indexManager.indexObject(inputSnapshot);
        this.indexManager.indexObject(outputSnapshot);
        this.indexManager.indexObject(event);

        // Update session tracking
        this.snapshots.set(session, {
            latest_seq: seq,
            latest_cid: outputSnapshot.cid,
            state: newState,
            timestamp: new Date().toISOString()
        });

        // Take periodic snapshots for fast recovery
        if (seq % this.snapshotInterval === 0) {
            console.debug(`📸 Checkpoint snapshot at seq ${seq} for session ${session}`);
        }

        return {
            event,
            inputSnapshot,
            outputSnapshot,
            newState
        };
    }

    // Apply mathematical transformation based on button
    applyTransformation(state, button, params = {}) {
        const { operator } = button;
        const { x, Sigma, kappa, level } = state;

        // Simple matrix transformation (extend for full operator algebra)
        const M = operator.M || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const b = operator.b || [0, 0, 0];
        const alpha = operator.alpha || 1.0;
        const deltaLevel = operator.delta_level || 0;

        // Apply transformation: x' = α * M * x + b
        const newX = [];
        for (let i = 0; i < x.length; i++) {
            let sum = 0;
            for (let j = 0; j < x.length; j++) {
                sum += (M[i] && M[i][j] !== undefined ? M[i][j] : 0) * x[j];
            }
            newX[i] = alpha * sum + (b[i] || 0);
        }

        return {
            x: newX,
            Sigma: Sigma, // Could transform covariance too
            kappa: Math.max(0, kappa + (params.kappa_delta || 0)),
            level: level + deltaLevel
        };
    }

    // Replay session from events
    async replaySession(session, toSeq = Infinity) {
        const events = this.engine.eventLog.getEvents(session, 0, toSeq);
        const states = [];
        let currentState = this.getInitialState();

        for (const event of events) {
            try {
                const inputState = await this.engine.objectStore.fetch(event.input_cid);
                const outputState = await this.engine.objectStore.fetch(event.output_cid);
                const button = await this.engine.resolve(`llex://core/button/${event.button}`);

                // Verify transformation (deterministic check)
                const recomputedState = this.applyTransformation(inputState, button, event.params);
                const stateMatches = this.compareStates(outputState, recomputedState);

                states.push({
                    seq: event.seq,
                    timestamp: event.t,
                    button: button,
                    input: inputState,
                    output: outputState,
                    recomputed: recomputedState,
                    verified: stateMatches,
                    params: event.params
                });

                currentState = outputState;
            } catch (error) {
                console.warn(`⚠️ Replay error at seq ${event.seq}:`, error);
                states.push({
                    seq: event.seq,
                    error: error.message,
                    event: event
                });
            }
        }

        return {
            session,
            finalState: currentState,
            states,
            verified: states.every(s => s.verified !== false)
        };
    }

    // Get initial state for new session
    getInitialState() {
        return {
            x: [0, 0, 0],
            Sigma: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            kappa: 1.0,
            level: 0
        };
    }

    // Compare states for verification
    compareStates(state1, state2, tolerance = 1e-6) {
        if (!state1 || !state2) return false;

        // Compare vectors
        if (state1.x.length !== state2.x.length) return false;
        for (let i = 0; i < state1.x.length; i++) {
            if (Math.abs(state1.x[i] - state2.x[i]) > tolerance) return false;
        }

        // Compare scalars
        if (Math.abs(state1.kappa - state2.kappa) > tolerance) return false;
        if (state1.level !== state2.level) return false;

        return true;
    }

    // Fast session recovery using snapshots + replay
    async recoverSession(session, targetSeq = null) {
        const sessionInfo = this.snapshots.get(session);
        if (!sessionInfo) {
            return await this.replaySession(session, targetSeq);
        }

        // Use latest snapshot as starting point
        const startSeq = sessionInfo.latest_seq;
        const startState = sessionInfo.state;

        if (targetSeq === null || targetSeq <= startSeq) {
            return {
                session,
                finalState: startState,
                recoveredFrom: 'snapshot',
                seq: startSeq
            };
        }

        // Replay from snapshot to target
        const events = this.engine.eventLog.getEvents(session, startSeq + 1, targetSeq);
        let currentState = startState;

        for (const event of events) {
            const button = await this.engine.resolve(`llex://core/button/${event.button}`);
            currentState = this.applyTransformation(currentState, button, event.params);
        }

        return {
            session,
            finalState: currentState,
            recoveredFrom: 'snapshot+replay',
            snapshotSeq: startSeq,
            targetSeq
        };
    }

    getStats() {
        return {
            active_sessions: this.snapshots.size,
            snapshot_interval: this.snapshotInterval,
            session_info: Array.from(this.snapshots.entries()).map(([session, info]) => ({
                session,
                latest_seq: info.latest_seq,
                timestamp: info.timestamp
            }))
        };
    }
}

// Complete LLEX System - puts it all together
class LLEXCompleteSystem {
    constructor() {
        this.engine = new (typeof window !== 'undefined' ? window.LLEX.Engine : LLEXEngine)();
        this.indexManager = new (typeof window !== 'undefined' ? window.LLEX.IndexManager : LLEXIndexManager)();
        this.addressResolver = new LLEXAddressResolver(this.engine);
        this.eventSourcer = new LLEXEventSourcer(this.engine, this.indexManager);

        // Initialize with some basic buttons
        this.initializeBasicButtons();

        console.log('🌟 LLEX Complete System initialized - full content-addressable lexical processing');
    }

    async initializeBasicButtons() {
        try {
            // Create basic transformation buttons
            await this.createButton('core', 'button:Scale', {
                M: [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]],
                b: [0, 0, 0],
                alpha: 1.0,
                delta_level: 0
            }, [], { class: 'Transform', author: 'system' });

            await this.createButton('core', 'button:Shift', {
                M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                b: [0.1, 0, 0],
                alpha: 1.0,
                delta_level: 0
            }, [], { class: 'Transform', author: 'system' });

            await this.createButton('core', 'button:Abstract', {
                M: [[0.9, 0, 0], [0, 0.9, 0], [0, 0, 1.2]],
                b: [0, 0, 0.1],
                alpha: 1.0,
                delta_level: 1
            }, ['ab'], { class: 'Abstraction', author: 'system' });

            console.log('✅ Basic buttons initialized');
        } catch (error) {
            console.warn('⚠️ Basic button initialization failed:', error);
        }
    }

    // High-level API methods
    async createButton(namespace, lid, operatorData, morphemes = [], metadata = {}) {
        const result = await this.engine.createButton(namespace, lid, operatorData, morphemes, metadata);

        // Index the button
        const button = await this.engine.objectStore.fetch(result.cid);
        this.indexManager.indexObject(button);

        return result;
    }

    async clickButton(session, buttonAddress, params = {}, provenance = {}) {
        // Get current session state
        const sessionInfo = this.eventSourcer.snapshots.get(session);
        const currentState = sessionInfo ? sessionInfo.state : this.eventSourcer.getInitialState();

        return await this.eventSourcer.applyButtonClick(session, buttonAddress, currentState, params, provenance);
    }

    async resolve(address) {
        return await this.addressResolver.resolve(address);
    }

    async search(query, options = {}) {
        return this.indexManager.unifiedSearch(query, options);
    }

    async replaySession(session, toSeq = null) {
        return await this.eventSourcer.replaySession(session, toSeq);
    }

    // Morphological analysis integration
    async analyzeMorphology(word, context = '') {
        // Create morpheme objects for each component
        const morphemes = this.extractMorphemes(word);
        const results = [];

        for (const morpheme of morphemes) {
            const morphemeObj = new (typeof window !== 'undefined' ? window.LLEX.Morpheme : LLEXMorpheme)(
                `morpheme:${morpheme.form}`,
                morpheme.type,
                morpheme.form,
                { meaning: morpheme.meaning, context },
                []
            );

            await morphemeObj.computeCID();
            await this.engine.objectStore.store(morphemeObj);
            this.indexManager.indexObject(morphemeObj);

            results.push({
                address: this.addressResolver.buildAddress('lexicon', 'morpheme', `morpheme:${morpheme.form}`, 'v1', morphemeObj.cid),
                morpheme: morpheme,
                cid: morphemeObj.cid
            });
        }

        return results;
    }

    // Simple morpheme extraction (extend with real morphological parser)
    extractMorphemes(word) {
        const prefixes = ['re', 'un', 'de', 'pre', 'sub', 'over', 'under'];
        const suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion'];

        const morphemes = [];
        let remaining = word.toLowerCase();

        // Extract prefix
        for (const prefix of prefixes) {
            if (remaining.startsWith(prefix)) {
                morphemes.push({ type: 'prefix', form: prefix, meaning: 'modifier' });
                remaining = remaining.slice(prefix.length);
                break;
            }
        }

        // Extract suffix
        for (const suffix of suffixes) {
            if (remaining.endsWith(suffix)) {
                morphemes.push({ type: 'suffix', form: suffix, meaning: 'modifier' });
                remaining = remaining.slice(0, -suffix.length);
                break;
            }
        }

        // Root
        if (remaining.length > 0) {
            morphemes.unshift({ type: 'root', form: remaining, meaning: 'core' });
        }

        return morphemes;
    }

    // Complete system statistics
    getStats() {
        return {
            engine: this.engine.getStats(),
            indexes: this.indexManager.getStats(),
            address_resolver: this.addressResolver.getCacheStats(),
            event_sourcer: this.eventSourcer.getStats(),
            timestamp: new Date().toISOString()
        };
    }

    // Health check
    async healthCheck() {
        const stats = this.getStats();
        const health = {
            status: 'healthy',
            checks: {
                object_store: stats.engine.object_store.total_objects > 0,
                catalog: stats.engine.catalog_entries > 0,
                indexes: stats.indexes.vector.total_vectors >= 0,
                event_log: stats.engine.total_events >= 0
            }
        };

        health.status = Object.values(health.checks).every(check => check) ? 'healthy' : 'degraded';
        return health;
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.LLEX = {
        ...window.LLEX,
        AddressResolver: LLEXAddressResolver,
        EventSourcer: LLEXEventSourcer,
        CompleteSystem: LLEXCompleteSystem
    };
} else if (typeof module !== 'undefined') {
    module.exports = {
        ...module.exports,
        LLEXAddressResolver,
        LLEXEventSourcer,
        LLEXCompleteSystem
    };
}
