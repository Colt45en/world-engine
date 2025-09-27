/**
 * LLEX Core Engine - Lexical Logic EXchange
 * Content-addressable storage with versioning and event sourcing
 *
 * Rule 1: Everything gets an ID
 * - Content ID (CID): cryptographic hash of canonical JSON (BLAKE3-like)
 * - Logical ID (LID): human/stable name (button:Rebuild, morpheme:re-)
 * - Version (VID): monotonic counter or semantic version
 * - Address: llex://<namespace>/<type>/<LID>@<VID>#<CID>
 */

// Simple hash function for CID generation (replace with BLAKE3 for production)
class CIDGenerator {
    static async hash(data) {
        const canonical = typeof data === 'string' ? data : JSON.stringify(data, null, 0);
        const encoder = new TextEncoder();
        const dataBuffer = encoder.encode(canonical);

        const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
        const hashArray = new Uint8Array(hashBuffer);
        const hashHex = Array.from(hashArray)
            .map(b => b.toString(16).padStart(2, '0'))
            .join('');

        return `cid-${hashHex.substring(0, 16)}`; // Truncated for demo
    }

    static canonicalize(obj) {
        // Sort keys recursively for deterministic hashing
        if (obj === null || typeof obj !== 'object') return obj;
        if (Array.isArray(obj)) return obj.map(item => this.canonicalize(item));

        const sorted = {};
        Object.keys(obj).sort().forEach(key => {
            sorted[key] = this.canonicalize(obj[key]);
        });
        return sorted;
    }
}

// Core object types following the spine architecture
class LLEXButton {
    constructor(lid, operatorData, morphemes = [], metadata = {}) {
        this.type = 'button';
        this.lid = lid;
        this.vid = null; // Set during storage
        this.cid = null; // Set during hashing
        this.class = metadata.class || 'Transform';
        this.morphemes = morphemes;
        this.operator = {
            M: operatorData.M || [[1]], // Transform matrix
            b: operatorData.b || [0],    // Bias vector
            C: operatorData.C || [[1]], // Constraint matrix
            alpha: operatorData.alpha || 1.0,
            beta: operatorData.beta || 0.0,
            delta_level: operatorData.delta_level || 0
        };
        this.meta = {
            created_at: new Date().toISOString(),
            author: metadata.author || 'system',
            ...metadata
        };
    }

    async computeCID() {
        const canonical = CIDGenerator.canonicalize(this);
        this.cid = await CIDGenerator.hash(canonical);
        return this.cid;
    }
}

class LLEXSnapshot {
    constructor(session, index, stationaryUnitState, prevCID = null) {
        this.type = 'snapshot';
        this.session = session;
        this.index = index;
        this.x = stationaryUnitState.x || [0, 0, 0];
        this.Sigma = stationaryUnitState.Sigma || [[1,0,0],[0,1,0],[0,0,1]];
        this.kappa = stationaryUnitState.kappa || 1.0;
        this.level = stationaryUnitState.level || 0;
        this.prev_cid = prevCID;
        this.timestamp = new Date().toISOString();
        this.cid = null;
    }

    async computeCID() {
        const canonical = CIDGenerator.canonicalize(this);
        this.cid = await CIDGenerator.hash(canonical);
        return this.cid;
    }
}

class LLEXEvent {
    constructor(session, seq, buttonLID, buttonVID, inputCID, outputCID, params = {}, provenance = {}) {
        this.type = 'event';
        this.session = session;
        this.seq = seq;
        this.t = new Date().toISOString();
        this.button = `${buttonLID}@${buttonVID}`;
        this.input_cid = inputCID;
        this.output_cid = outputCID;
        this.params = params;
        this.provenance = {
            user: provenance.user || 'system',
            client: provenance.client || 'llex-core',
            ...provenance
        };
        this.cid = null;
    }

    async computeCID() {
        const canonical = CIDGenerator.canonicalize(this);
        this.cid = await CIDGenerator.hash(canonical);
        return this.cid;
    }
}

class LLEXMorpheme {
    constructor(lid, type, form, semanticData = {}, relationships = []) {
        this.type = 'morpheme';
        this.lid = lid; // e.g., "morpheme:re-"
        this.vid = null;
        this.cid = null;
        this.morpheme_type = type; // 'prefix', 'root', 'suffix'
        this.form = form; // the actual text
        this.semantic = {
            meaning: semanticData.meaning || '',
            abstraction_level: semanticData.abstraction_level || 0,
            frequency: semanticData.frequency || 0,
            ...semanticData
        };
        this.relationships = relationships; // links to other morphemes
        this.meta = {
            created_at: new Date().toISOString(),
            author: 'system'
        };
    }

    async computeCID() {
        const canonical = CIDGenerator.canonicalize(this);
        this.cid = await CIDGenerator.hash(canonical);
        return this.cid;
    }
}

// Object Store - immutable blob storage
class LLEXObjectStore {
    constructor() {
        this.objects = new Map(); // In-memory store (use S3/GCS for production)
        this.stats = {
            total_objects: 0,
            size_bytes: 0
        };
    }

    async store(obj) {
        if (!obj.cid) {
            await obj.computeCID();
        }

        const blob = JSON.stringify(obj, null, 2);
        this.objects.set(obj.cid, blob);
        this.stats.total_objects++;
        this.stats.size_bytes += blob.length;

        console.debug(`📦 Stored object ${obj.cid} (${obj.type})`);
        return obj.cid;
    }

    async fetch(cid) {
        const blob = this.objects.get(cid);
        if (!blob) {
            throw new Error(`Object not found: ${cid}`);
        }

        return JSON.parse(blob);
    }

    exists(cid) {
        return this.objects.has(cid);
    }

    getStats() {
        return { ...this.stats };
    }
}

// Catalog - maps logical names to content IDs
class LLEXCatalog {
    constructor() {
        this.entries = new Map(); // namespace:type:lid:vid -> cid
        this.current_pointers = new Map(); // namespace:type:lid -> {vid, cid}
    }

    register(namespace, type, lid, vid, cid, isCurrent = false) {
        const key = `${namespace}:${type}:${lid}:${vid}`;
        const currentKey = `${namespace}:${type}:${lid}`;

        this.entries.set(key, {
            namespace, type, lid, vid, cid,
            created_at: new Date().toISOString()
        });

        if (isCurrent) {
            this.current_pointers.set(currentKey, { vid, cid });
        }

        console.debug(`📚 Catalogued ${namespace}/${type}/${lid}@${vid} → ${cid}`);
    }

    resolve(namespace, type, lid, vid = 'current') {
        if (vid === 'current') {
            const currentKey = `${namespace}:${type}:${lid}`;
            const current = this.current_pointers.get(currentKey);
            if (!current) return null;
            return current.cid;
        }

        const key = `${namespace}:${type}:${lid}:${vid}`;
        const entry = this.entries.get(key);
        return entry ? entry.cid : null;
    }

    list(namespace, type = null) {
        const results = [];
        for (const [key, entry] of this.entries) {
            if (entry.namespace === namespace && (type === null || entry.type === type)) {
                results.push(entry);
            }
        }
        return results;
    }

    getVersions(namespace, type, lid) {
        const results = [];
        const prefix = `${namespace}:${type}:${lid}:`;

        for (const [key, entry] of this.entries) {
            if (key.startsWith(prefix)) {
                results.push(entry);
            }
        }

        return results.sort((a, b) => a.vid.localeCompare(b.vid));
    }
}

// Event Log - append-only record of all changes
class LLEXEventLog {
    constructor() {
        this.events = []; // In-memory log (use Kafka/etc for production)
        this.sessions = new Map(); // session -> latest_seq
    }

    async append(event) {
        if (!event.cid) {
            await event.computeCID();
        }

        this.events.push(event);

        // Update session tracking
        const currentSeq = this.sessions.get(event.session) || 0;
        if (event.seq > currentSeq) {
            this.sessions.set(event.session, event.seq);
        }

        console.debug(`📝 Event logged: ${event.session}#${event.seq} - ${event.button}`);
        return event.cid;
    }

    getEvents(session, fromSeq = 0, toSeq = Infinity) {
        return this.events.filter(e =>
            e.session === session &&
            e.seq >= fromSeq &&
            e.seq <= toSeq
        ).sort((a, b) => a.seq - b.seq);
    }

    getAllEvents(fromTime = null, toTime = null) {
        let filtered = this.events;

        if (fromTime) {
            filtered = filtered.filter(e => new Date(e.t) >= fromTime);
        }

        if (toTime) {
            filtered = filtered.filter(e => new Date(e.t) <= toTime);
        }

        return filtered.sort((a, b) => new Date(a.t) - new Date(b.t));
    }

    getSessionInfo(session) {
        const events = this.getEvents(session);
        const latestSeq = this.sessions.get(session) || 0;

        return {
            session,
            total_events: events.length,
            latest_seq: latestSeq,
            first_event: events.length > 0 ? events[0].t : null,
            last_event: events.length > 0 ? events[events.length - 1].t : null
        };
    }
}

// Main LLEX Engine - coordinates all components
class LLEXEngine {
    constructor() {
        this.objectStore = new LLEXObjectStore();
        this.catalog = new LLEXCatalog();
        this.eventLog = new LLEXEventLog();
        this.sessions = new Map(); // session -> current state info

        console.log('🧠 LLEX Engine initialized - content-addressable lexical storage');
    }

    // Create and store a button
    async createButton(namespace, lid, operatorData, morphemes = [], metadata = {}) {
        const button = new LLEXButton(lid, operatorData, morphemes, metadata);
        await button.computeCID();

        // Auto-version (simple increment for demo)
        const versions = this.catalog.getVersions(namespace, 'button', lid);
        const nextVersion = `v${versions.length + 1}`;
        button.vid = nextVersion;

        await this.objectStore.store(button);
        this.catalog.register(namespace, 'button', lid, nextVersion, button.cid, true);

        return {
            address: `llex://${namespace}/button/${lid}@${nextVersion}#${button.cid}`,
            cid: button.cid,
            vid: nextVersion
        };
    }

    // Create session snapshot
    async createSnapshot(session, index, state, prevCID = null) {
        const snapshot = new LLEXSnapshot(session, index, state, prevCID);
        await snapshot.computeCID();
        await this.objectStore.store(snapshot);

        return snapshot;
    }

    // Record button click event
    async recordClick(session, seq, buttonAddress, inputCID, outputCID, params = {}, provenance = {}) {
        const [buttonLID, buttonVID] = this.parseButtonAddress(buttonAddress);
        const event = new LLEXEvent(session, seq, buttonLID, buttonVID, inputCID, outputCID, params, provenance);

        await this.eventLog.append(event);
        return event;
    }

    // Parse button address: "button:Rebuild@v3" -> ["button:Rebuild", "v3"]
    parseButtonAddress(address) {
        const parts = address.split('@');
        return [parts[0], parts[1]];
    }

    // Resolve llex:// address
    async resolve(address) {
        // Parse: llex://namespace/type/lid@vid#cid
        const match = address.match(/^llex:\/\/([^\/]+)\/([^\/]+)\/([^@#]+)(?:@([^#]+))?(?:#(.+))?$/);
        if (!match) throw new Error(`Invalid LLEX address: ${address}`);

        const [, namespace, type, lid, vid, cid] = match;

        if (cid) {
            // Direct CID reference
            return await this.objectStore.fetch(cid);
        } else {
            // Catalog lookup
            const resolvedCID = this.catalog.resolve(namespace, type, lid, vid || 'current');
            if (!resolvedCID) throw new Error(`Not found: ${address}`);
            return await this.objectStore.fetch(resolvedCID);
        }
    }

    // Get session replay capability
    async replaySession(session, toSeq = Infinity) {
        const events = this.eventLog.getEvents(session, 0, toSeq);
        const states = [];

        for (const event of events) {
            if (this.objectStore.exists(event.input_cid) && this.objectStore.exists(event.output_cid)) {
                const inputState = await this.objectStore.fetch(event.input_cid);
                const outputState = await this.objectStore.fetch(event.output_cid);
                const button = await this.resolve(`llex://core/button/${event.button}`);

                states.push({
                    seq: event.seq,
                    timestamp: event.t,
                    button: button,
                    input: inputState,
                    output: outputState,
                    params: event.params
                });
            }
        }

        return states;
    }

    // Statistics and health
    getStats() {
        return {
            object_store: this.objectStore.getStats(),
            catalog_entries: this.catalog.entries.size,
            current_pointers: this.catalog.current_pointers.size,
            total_events: this.eventLog.events.length,
            active_sessions: this.eventLog.sessions.size
        };
    }
}

// Export for use in web environment
if (typeof window !== 'undefined') {
    window.LLEX = {
        Engine: LLEXEngine,
        Button: LLEXButton,
        Snapshot: LLEXSnapshot,
        Event: LLEXEvent,
        Morpheme: LLEXMorpheme,
        CIDGenerator
    };
} else if (typeof module !== 'undefined') {
    module.exports = {
        LLEXEngine,
        LLEXButton,
        LLEXSnapshot,
        LLEXEvent,
        LLEXMorpheme,
        CIDGenerator
    };
}
