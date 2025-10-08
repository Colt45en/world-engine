/**
 * Upflow Automation - Sharded IndexedDB System
 * A drop-in ES module for indexing and linking words by morphological structure
 * Integrates with existing World Engine and Lexical Logic Engine systems
 */

// Simple FNV-1a 32-bit hash for deterministic sharding
function fnv1a32(str) {
    let hash = 2166136261;
    for (let i = 0; i < str.length; i++) {
        hash ^= str.charCodeAt(i);
        hash = (hash * 16777619) >>> 0;
    }
    return hash;
}

/**
 * IndexedDB Storage with synchronous facade
 */
export async function createIDBIndexStorage({
    dbName = 'UpflowDB',
    storeName = 'lexicon',
    version = 1,
    preloadKeys = []
}) {
    let db = null;
    const memoryCache = new Map();
    const pendingWrites = new Map();

    // Initialize IndexedDB
    try {
        db = await new Promise((resolve, reject) => {
            const request = indexedDB.open(dbName, version);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(storeName)) {
                    db.createObjectStore(storeName);
                }
            };
        });
    } catch (error) {
        console.warn('IndexedDB not available, falling back to localStorage:', error);
    }

    // Preload specified keys into memory
    if (db && preloadKeys.length > 0) {
        const transaction = db.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);

        await Promise.all(preloadKeys.map(async (key) => {
            try {
                const request = store.get(key);
                const value = await new Promise((resolve) => {
                    request.onsuccess = () => resolve(request.result);
                    request.onerror = () => resolve(null);
                });
                if (value !== null) {
                    memoryCache.set(key, value);
                }
            } catch (error) {
                console.warn(`Failed to preload key ${key}:`, error);
            }
        }));
    }

    // Storage interface
    const storage = {
        get(key) {
            // Check memory cache first
            if (memoryCache.has(key)) {
                return memoryCache.get(key);
            }

            // Fallback to localStorage if IndexedDB unavailable
            if (!db) {
                const stored = localStorage.getItem(`${dbName}.${key}`);
                return stored ? JSON.parse(stored) : null;
            }

            return null; // IDB lookup would be async, so return null for sync interface
        },

        set(key, value) {
            // Update memory cache immediately
            memoryCache.set(key, value);

            // Queue async write
            if (db) {
                pendingWrites.set(key, value);
                // Write to IndexedDB asynchronously
                setTimeout(() => this._flushKey(key), 0);
            } else {
                // Fallback to localStorage
                localStorage.setItem(`${dbName}.${key}`, JSON.stringify(value));
            }
        },

        delete(key) {
            memoryCache.delete(key);

            if (db) {
                const transaction = db.transaction([storeName], 'readwrite');
                const store = transaction.objectStore(storeName);
                store.delete(key);
            } else {
                localStorage.removeItem(`${dbName}.${key}`);
            }
        },

        async _flushKey(key) {
            if (!db || !pendingWrites.has(key)) return;

            const value = pendingWrites.get(key);
            pendingWrites.delete(key);

            try {
                const transaction = db.transaction([storeName], 'readwrite');
                const store = transaction.objectStore(storeName);
                store.put(value, key);
            } catch (error) {
                console.warn(`Failed to write key ${key}:`, error);
                // Put it back in pending writes for retry
                pendingWrites.set(key, value);
            }
        },

        async flush() {
            if (!db || pendingWrites.size === 0) return;

            const writes = Array.from(pendingWrites.entries());
            pendingWrites.clear();

            try {
                const transaction = db.transaction([storeName], 'readwrite');
                const store = transaction.objectStore(storeName);

                for (const [key, value] of writes) {
                    store.put(value, key);
                }

                await new Promise((resolve, reject) => {
                    transaction.oncomplete = resolve;
                    transaction.onerror = () => reject(transaction.error);
                });
            } catch (error) {
                console.warn('Batch flush failed:', error);
                // Restore pending writes
                for (const [key, value] of writes) {
                    pendingWrites.set(key, value);
                }
            }
        },

        async close() {
            await this.flush();
            if (db) {
                db.close();
            }
        }
    };

    return storage;
}

/**
 * Build shard keys for preloading
 */
export function buildShardKeys(indexKey, shards) {
    const keys = [];
    for (let i = 0; i < shards; i++) {
        keys.push(`${indexKey}.shard.${i}`);
    }
    keys.push(`${indexKey}.w2s`); // word-to-shard mapping
    return keys;
}

/**
 * Default abbreviation derivation
 */
function defaultAbbrs(word, english = '') {
    const abbrs = new Set();

    // Tri-grams from word
    if (word.length >= 3) {
        abbrs.add(word.slice(0, 3).toLowerCase());
        if (word.length > 3) {
            abbrs.add(word.slice(-3).toLowerCase());
        }
    }

    // Acronym from english description
    if (english) {
        const acronym = english.split(/\s+/)
            .map(w => w.charAt(0))
            .join('')
            .toLowerCase();
        if (acronym.length >= 2) {
            abbrs.add(acronym);
        }
    }

    // Leading chars from word
    for (let i = 2; i <= Math.min(4, word.length); i++) {
        abbrs.add(word.slice(0, i).toLowerCase());
    }

    return Array.from(abbrs);
}

/**
 * Default tokenization
 */
function defaultTokenize(text) {
    return text.toLowerCase()
        .split(/[^\w]+/)
        .filter(token => token.length > 2)
        .slice(0, 5); // Top 5 tokens
}

/**
 * Main upflow automation factory
 */
export function createUpflowAutomation({
    storage,
    indexKey = 'lexi.index',
    shards = 16,
    morph = null, // Inject existing morphology function
    deriveAbbrs = defaultAbbrs,
    tokenize = defaultTokenize
}) {
    // Default morphology if not provided
    const defaultMorph = (word) => {
        // Simple morphology - extract common prefixes/suffixes
        const prefixes = ['re', 'pre', 'un', 'dis', 'over', 'under', 'out', 'up', 'down'];
        const suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'ment', 'able', 'ful'];

        let prefix = '';
        let root = word.toLowerCase();
        let suffix = '';

        // Check for prefixes
        for (const pre of prefixes) {
            if (root.startsWith(pre) && root.length > pre.length + 2) {
                prefix = pre;
                root = root.slice(pre.length);
                break;
            }
        }

        // Check for suffixes
        for (const suf of suffixes) {
            if (root.endsWith(suf) && root.length > suf.length + 2) {
                suffix = suf;
                root = root.slice(0, -suf.length);
                break;
            }
        }

        return { prefix, root, suffix };
    };

    const morphoFn = morph || defaultMorph;

    // Get shard number for a root
    function getShardForRoot(root) {
        return fnv1a32(root) % shards;
    }

    // Load or initialize shard
    function loadShard(shardNum) {
        const key = `${indexKey}.shard.${shardNum}`;
        return storage.get(key) || {
            byRoot: {},
            byPrefix: {},
            bySuffix: {},
            byAbbrev: {},
            words: {}
        };
    }

    // Save shard
    function saveShard(shardNum, shard) {
        const key = `${indexKey}.shard.${shardNum}`;
        storage.set(key, shard);
    }

    // Load word-to-shard mapping
    function loadW2S() {
        return storage.get(`${indexKey}.w2s`) || {};
    }

    // Save word-to-shard mapping
    function saveW2S(w2s) {
        storage.set(`${indexKey}.w2s`, w2s);
    }

    // Add word to index
    function addWord(word, english = '', metadata = {}) {
        const morpho = morphoFn(word);
        const { prefix, root, suffix } = morpho;
        const abbrs = deriveAbbrs(word, english);

        const shardNum = getShardForRoot(root);
        const shard = loadShard(shardNum);

        // Update word-to-shard mapping
        const w2s = loadW2S();
        w2s[word] = shardNum;
        saveW2S(w2s);

        // Add to shard indices
        if (!shard.byRoot[root]) shard.byRoot[root] = new Set();
        shard.byRoot[root].add(word);

        if (prefix) {
            if (!shard.byPrefix[prefix]) shard.byPrefix[prefix] = new Set();
            shard.byPrefix[prefix].add(word);
        }

        if (suffix) {
            if (!shard.bySuffix[suffix]) shard.bySuffix[suffix] = new Set();
            shard.bySuffix[suffix].add(word);
        }

        for (const abbr of abbrs) {
            if (!shard.byAbbrev[abbr]) shard.byAbbrev[abbr] = new Set();
            shard.byAbbrev[abbr].add(word);
        }

        // Store full word record
        shard.words[word] = {
            word,
            english,
            root,
            prefix,
            suffix,
            abbrs,
            metadata,
            indexed: Date.now()
        };

        // Convert Sets to Arrays for serialization
        const serializedShard = JSON.parse(JSON.stringify(shard, (key, value) => {
            return value instanceof Set ? Array.from(value) : value;
        }));

        // Convert back to Sets
        for (const [key, val] of Object.entries(serializedShard)) {
            if (key !== 'words') {
                for (const [subkey, subval] of Object.entries(val)) {
                    shard[key][subkey] = new Set(subval);
                }
            }
        }

        saveShard(shardNum, serializedShard);

        return { word, root, prefix, suffix, abbrs };
    }

    // Query interface
    const query = {
        byRoot(root) {
            const shardNum = getShardForRoot(root);
            const shard = loadShard(shardNum);
            return shard.byRoot[root] ? Array.from(shard.byRoot[root]) : [];
        },

        byPrefix(prefix) {
            const results = new Set();
            for (let i = 0; i < shards; i++) {
                const shard = loadShard(i);
                if (shard.byPrefix[prefix]) {
                    Array.from(shard.byPrefix[prefix]).forEach(word => results.add(word));
                }
            }
            return Array.from(results);
        },

        bySuffix(suffix) {
            const results = new Set();
            for (let i = 0; i < shards; i++) {
                const shard = loadShard(i);
                if (shard.bySuffix[suffix]) {
                    Array.from(shard.bySuffix[suffix]).forEach(word => results.add(word));
                }
            }
            return Array.from(results);
        },

        byAbbrev(abbr) {
            const results = new Set();
            for (let i = 0; i < shards; i++) {
                const shard = loadShard(i);
                if (shard.byAbbrev[abbr]) {
                    Array.from(shard.byAbbrev[abbr]).forEach(word => results.add(word));
                }
            }
            return Array.from(results);
        },

        getWord(word) {
            const w2s = loadW2S();
            const shardNum = w2s[word];

            if (shardNum !== undefined) {
                const shard = loadShard(shardNum);
                return shard.words[word] || null;
            }

            // Fallback: scan all shards
            for (let i = 0; i < shards; i++) {
                const shard = loadShard(i);
                if (shard.words[word]) {
                    return shard.words[word];
                }
            }

            return null;
        }
    };

    // Helper interface
    const helper = {
        linkWord(word) {
            const record = query.getWord(word);
            if (!record) return null;

            const { root, prefix, suffix, abbrs } = record;

            return {
                ...record,
                rootLinked: query.byRoot(root).filter(w => w !== word),
                prefixLinked: prefix ? query.byPrefix(prefix).filter(w => w !== word) : [],
                suffixLinked: suffix ? query.bySuffix(suffix).filter(w => w !== word) : [],
                abbrLinked: abbrs.flatMap(abbr =>
                    query.byAbbrev(abbr).filter(w => w !== word)
                ).slice(0, 10) // Limit to prevent explosion
            };
        },

        linkMany(words) {
            return words.map(word => this.linkWord(word)).filter(Boolean);
        }
    };

    // Librarian interface
    const librarian = {
        verify() {
            let totalWords = 0;
            let totalLinks = 0;

            for (let i = 0; i < shards; i++) {
                const shard = loadShard(i);
                totalWords += Object.keys(shard.words).length;

                for (const set of Object.values(shard.byRoot)) {
                    totalLinks += Array.isArray(set) ? set.length : set.size;
                }
            }

            return {
                ok: true,
                shards,
                totalWords,
                totalLinks,
                avgWordsPerShard: totalWords / shards
            };
        },

        snapshot() {
            const snapshot = { shards: [], w2s: loadW2S() };

            for (let i = 0; i < shards; i++) {
                const shard = loadShard(i);
                const serialized = JSON.parse(JSON.stringify(shard, (key, value) => {
                    return value instanceof Set ? Array.from(value) : value;
                }));
                snapshot.shards.push(serialized);
            }

            return snapshot;
        },

        compact() {
            const snapshot = this.snapshot();
            let totalSize = 0;

            // Re-save all shards (compaction)
            for (let i = 0; i < snapshot.shards.length; i++) {
                saveShard(i, snapshot.shards[i]);
                totalSize += JSON.stringify(snapshot.shards[i]).length;
            }

            saveW2S(snapshot.w2s);
            totalSize += JSON.stringify(snapshot.w2s).length;

            return { compacted: true, totalSize, shards: snapshot.shards.length };
        }
    };

    // Main automation interface
    return {
        addWord,
        query,
        helper,
        librarian,

        async ingestFromLastRun(storageKey = 'wordEngine.lastRun') {
            try {
                const lastRun = localStorage.getItem(storageKey);
                if (!lastRun) {
                    throw new Error('No lastRun data found');
                }

                const data = JSON.parse(lastRun);

                // Extract word and english from various possible structures
                let word, english;

                if (data.word) {
                    word = data.word;
                    english = data.english || data.gloss || data.meaning || '';
                } else if (data.result && data.result.word) {
                    word = data.result.word;
                    english = data.result.english || data.result.gloss || '';
                } else if (Array.isArray(data) && data.length > 0) {
                    const first = data[0];
                    word = first.word || first.string || first.text || '';
                    english = first.english || first.gloss || first.meaning || '';
                } else {
                    throw new Error('Could not extract word from lastRun data');
                }

                if (!word) {
                    throw new Error('No word found in lastRun data');
                }

                // Add to index
                const result = addWord(word, english, { source: 'lastRun', data });

                return { ok: true, ...result };
            } catch (error) {
                console.warn('Failed to ingest from lastRun:', error);
                return { ok: false, error: error.message };
            }
        }
    };
}
