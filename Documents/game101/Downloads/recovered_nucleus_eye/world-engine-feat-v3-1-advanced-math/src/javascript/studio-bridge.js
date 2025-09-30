'use strict';

// ===== 1) Version & Install Guard =====
const NAME = 'studio-bridge';
const VERSION = '0.2.0-advanced';

function compareSemver(a, b) {
    try {
        const pa = a.split('-')[0].split('.').map(Number), pb = b.split('-')[0].split('.').map(Number);
        for (let i = 0; i < 3; i++) { const d = (pa[i] || 0) - (pb[i] || 0); if (d) return d; }
        return 0;
    } catch (_) { return 0; }
}

if (typeof window.__STUDIO_BRIDGE__ === 'object') {
    try {
        const existing = window.__STUDIO_BRIDGE__;
        if (existing && existing.version && compareSemver(existing.version, VERSION) >= 0) {
            if (location.search.includes('debug=studio')) {
                (console.info || console.log)('[Studio:init] skip ' + NAME + ' v' + VERSION + ' (existing ' + existing.version + ')');
            }
            // idempotent
        }
    } catch (_) { }
}

Object.defineProperty(window, '__STUDIO_BRIDGE__', {
    value: { version: VERSION, installedAt: Date.now() },
    writable: false, configurable: false, enumerable: true
});

// ===== 2) Config & Constants =====
const CONST = Object.freeze({
    BUS_NAME: 'studio-bus',
    NAMESPACE: 'studio:',
    MSG: Object.freeze({
        ENG_RUN: 'eng.run', ENG_TEST: 'eng.test', ENG_STATUS: 'eng.status', ENG_RESULT: 'eng.result',
        REC_START: 'rec.start', REC_STOP: 'rec.stop', REC_CLIP: 'rec.clip', REC_TRANSCRIPT: 'rec.transcript', REC_MARK: 'rec.mark',
        CHAT_CMD: 'chat.cmd', CHAT_ANN: 'chat.announce',
        // Advanced World Engine messages
        WE_ANALYZE: 'we.analyze', WE_SEARCH: 'we.search', WE_COMPREHENSIVE: 'we.comprehensive', WE_BENCHMARK: 'we.benchmark',
        WE_STATUS: 'we.status', WE_METRICS: 'we.metrics',
        // Audio/Beat System messages
        BEAT_START: 'beat.start', BEAT_STOP: 'beat.stop', BEAT_SYNC: 'beat.sync'
    })
});

// ===== 3) Bus (BroadcastChannel + fallback) =====
const bc = ('BroadcastChannel' in self) ? new BroadcastChannel(CONST.BUS_NAME) : null;
const listeners = new Set();
function _fan(msg) {
    if (!msg || typeof msg.type !== 'string') return;
    listeners.forEach(function (fn) { try { fn(msg); } catch (e) { console.warn('[Studio:bus-listener]', e); } });
}
if (bc) bc.onmessage = function (e) { _fan(e.data); };
else window.addEventListener('studio:msg', function (e) { _fan(e.detail); });

function onBus(fn) {
    if (typeof fn !== 'function') return function () { if (listeners.has(fn)) listeners.delete(fn); }; { (listeners.add)(fn); return function off() { listeners.delete(fn); }; }
    function sendBus(msg) {
        if (!msg || typeof msg.type !== 'string') return;
        try {
            if (bc) bc.postMessage(msg);
            else window.dispatchEvent(new CustomEvent('studio:msg', { detail: msg }));
        } catch (e) { console.warn('[Studio:bus-send]', e); }
    }

    // ===== 4) Store (namespaced, JSON-safe) =====
    const Store = Object.freeze({
        save: function (key, value) {
            try {
                if (window.externalStore && typeof window.externalStore.upsert === 'function') {
                    return Promise.resolve(window.externalStore.upsert(key, value));
                }
                if (window.localStorage) {
                    const serialized = JSON.stringify(value);
                    if (serialized && serialized.length > 1_000_000) console.warn('[Studio:store] large value for', key, (serialized.length / 1024).toFixed(1) + ' KB');
                    localStorage.setItem(CONST.NAMESPACE + key, serialized);
                    return Promise.resolve(value);
                }
            } catch (e) { return Promise.reject(e); }
            return Promise.reject(new Error('No storage available'));
        },
        load: function (key) {
            try {
                if (window.externalStore && typeof window.externalStore.get === 'function') {
                    return Promise.resolve(window.externalStore.get(key));
                }
                if (window.localStorage) {
                    const raw = localStorage.getItem(CONST.NAMESPACE + key);
                    if (!raw) return Promise.resolve(null);
                    try { return Promise.resolve(JSON.parse(raw)); }
                    catch (_) { console.warn('[Studio:store] JSON parse error for', key); return Promise.resolve(null); }
                }
            } catch (e) { return Promise.resolve(null); }
            return Promise.resolve(null);
        }
    });

    // ===== 5) Utils =====
    const Utils = Object.freeze({
        id: function () { try { return crypto.randomUUID(); } catch (_) { return String(Date.now()) + Math.random().toString(36).slice(2, 10); } },
        parseCommand: function (line) {
            const t = (String(line || '')).trim();
            if (t.startsWith('/run ')) return { type: 'run', args: t.slice(5) };
            if (t.startsWith('/test ')) return { type: 'test', args: t.slice(6) };
            if (t.startsWith('/analyze ')) return { type: 'analyze', args: t.slice(9) };
            if (t.startsWith('/search ')) return { type: 'search', args: t.slice(8) };
            if (t.startsWith('/comprehensive ')) return { type: 'comprehensive', args: t.slice(15) };
            if (t.startsWith('/benchmark')) return { type: 'benchmark', args: '' };
            if (t.startsWith('/beat ')) return { type: 'beat', args: t.slice(6) };
            if (t.startsWith('/rec ')) {
                const mode = t.split(/\s+/)[1] || 'start';
                return { type: 'rec', args: mode };
            }
            if (t.startsWith('/mark ')) return { type: 'mark', args: t.slice(6) };
            if (t === '/status') return { type: 'status', args: '' };
            if (t === '/metrics') return { type: 'metrics', args: '' };
            if (t === '/history') return { type: 'history', args: '' };
            if (t === '/clear') return { type: 'clear', args: '' };
            if (t === '/help') return { type: 'help', args: '' };
            return { type: 'run', args: t };
        },
        log: function (message, level) {
            const ts = new Date().toISOString(); const lv = level || 'info';
            if (location.search.includes('debug=studio')) (console[lv] || console.log)('[Studio:' + lv + '] ' + ts + ' - ' + message);
        }
    });

    // ===== 6) Engine Transport (same-origin iframe helper) =====
    function setupEngineTransport(engineFrame) {
        function isSameOrigin() { try { return !!engineFrame && !!engineFrame.contentWindow && !!engineFrame.contentWindow.document; } catch (e) { return false; } }
        function withEngine(fn) { if (!isSameOrigin()) { console.warn('[Studio:engine] cross-origin; use bus'); return; } try { fn(engineFrame.contentWindow.document); } catch (e) { console.warn('[Studio:engine] withEngine error', e); } }
        return { isSameOrigin: isSameOrigin, withEngine: withEngine };
    }

    // ===== 7) Public APIs (Recorder / Engine / Chat / WorldEngine / Beat) =====
    const RecorderAPI = Object.freeze({
        startMic: function (meta) { sendBus({ type: CONST.MSG.REC_START, mode: 'mic', meta: meta || {} }); },
        startScreen: function (meta) { sendBus({ type: CONST.MSG.REC_START, mode: 'screen', meta: meta || {} }); },
        startBoth: function (meta) { sendBus({ type: CONST.MSG.REC_START, mode: 'both', meta: meta || {} }); },
        stop: function () { sendBus({ type: CONST.MSG.REC_STOP }); },
        mark: function (tag, runId) { sendBus({ type: CONST.MSG.REC_MARK, tag: String(tag || 'mark'), runId: runId == null ? null : String(runId) }); }
    });

    const EngineAPI = Object.freeze({
        run: function (text) { sendBus({ type: CONST.MSG.ENG_RUN, text: String(text || '') }); },
        test: function (name) { sendBus({ type: CONST.MSG.ENG_TEST, name: String(name || '') }); },
        getStatus: function () { sendBus({ type: CONST.MSG.ENG_STATUS }); }
    });

    const WorldEngineAPI = Object.freeze({
        analyze: function (text, options) { sendBus({ type: CONST.MSG.WE_ANALYZE, text: String(text || ''), options: options || {} }); },
        search: function (query, options) { sendBus({ type: CONST.MSG.WE_SEARCH, query: String(query || ''), options: options || {} }); },
        comprehensive: function (input, options) { sendBus({ type: CONST.MSG.WE_COMPREHENSIVE, input: String(input || ''), options: options || {} }); },
        benchmark: function () { sendBus({ type: CONST.MSG.WE_BENCHMARK }); },
        getStatus: function () { sendBus({ type: CONST.MSG.WE_STATUS }); },
        getMetrics: function () { sendBus({ type: CONST.MSG.WE_METRICS }); }
    });

    const BeatSystemAPI = Object.freeze({
        start: function (config) { sendBus({ type: CONST.MSG.BEAT_START, config: config || {} }); },
        stop: function () { sendBus({ type: CONST.MSG.BEAT_STOP }); },
        sync: function (data) { sendBus({ type: CONST.MSG.BEAT_SYNC, data: data || {} }); }
    });

    const ChatAPI = Object.freeze({
        command: function (line) { sendBus({ type: CONST.MSG.CHAT_CMD, line: String(line || '') }); },
        announce: function (message, level) { sendBus({ type: CONST.MSG.CHAT_ANN, message: String(message || ''), level: String(level || 'info') }); }
    });

    // ===== 8) Bridge Export =====
    const Bridge = Object.freeze({
        onBus: onBus,
        sendBus: sendBus,
        RecorderAPI: RecorderAPI,
        EngineAPI: EngineAPI,
        WorldEngineAPI: WorldEngineAPI,
        BeatSystemAPI: BeatSystemAPI,
        ChatAPI: ChatAPI,
        Store: Store,
        Utils: Utils,
        setupEngineTransport: setupEngineTransport,
        CONST: CONST
    });

    Object.defineProperty(window, 'StudioBridge', { value: Bridge, writable: false, configurable: false, enumerable: true });
    if (typeof module !== 'undefined' && module.exports) { module.exports = Bridge; }

    // ===== 10) Diagnostics =====
    (console.info || console.log)('[Studio:init] ' + NAME + ' v' + VERSION + (bc ? ' [BroadcastChannel]' : ' [window-event]'));
    // Schema validation utilities
    export const Schemas = {
        // Run object validation
        Run: (x) => x &&
            typeof x.runId === 'string' &&
            typeof x.ts === 'number' &&
            typeof x.input === 'string' &&
            (x.outcome === null || typeof x.outcome === 'object') &&
            (x.clipId === null || typeof x.clipId === 'string') &&
            (x.meta === undefined || typeof x.meta === 'object'),
        // Clip object validation
        Clip: (x) => x &&
            typeof x.clipId === 'string' &&
            (x.meta === undefined || typeof x.meta === 'object'),
        // Marker object validation
        Mark: (x) => x &&
            typeof x.id === 'string' &&
            (x.meta === undefined || typeof x.meta === 'object'),
        // Command object validation
        Command: (x) => x &&
            typeof x.type === 'string' &&
            (x.args === undefined || typeof x.args === 'string'),
        // Engine result validation
        EngineResult: (x) => x &&
            typeof x.runId === 'string' &&
            typeof x.ts === 'number' &&
            (x.outcome === null || typeof x.outcome === 'object') &&
            (x.meta === undefined || typeof x.meta === 'object'),
        // Session export validation
        SessionExport: (x) => x &&
            typeof x.schema === 'string' &&
            (x.data === undefined || typeof x.data === 'object'),
    };

    // Type assertion helper
    export function assertShape(guard, obj, msg = 'Invalid shape') {
        if (!guard(obj)) {
            throw new Error(`${msg}: ${JSON.stringify(obj).slice(0, 100)}`);
        }
        return obj;
    } for (let index = 0; index < array.length; index++) {
        const element = array[index];

    }
