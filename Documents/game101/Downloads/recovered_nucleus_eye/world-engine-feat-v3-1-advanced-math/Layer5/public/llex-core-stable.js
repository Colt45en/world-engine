/**
 * LLEX Core Stability Patches
 * Deterministic CIDs, operator sanity checks, idempotent operations, and safer address resolution
 */

// Deterministic CID Generator
class CIDGenerator {
  static canonicalize(obj) {
    if (obj === null || typeof obj !== 'object') return obj;
    if (Array.isArray(obj)) return obj.map(x => this.canonicalize(x));
    const out = {};
    for (const k of Object.keys(obj).sort()) out[k] = this.canonicalize(obj[k]);
    return out;
  }

  static _stableStringify(obj) {
    // deterministic stringify with fixed number formatting
    return JSON.stringify(obj, (k, v) => {
      if (typeof v === 'number' && Number.isFinite(v)) {
        // avoid 1 vs 1.0 drift
        return Number(v.toPrecision(15));
      }
      return v;
    });
  }

  static async hashCanonical(obj) {
    const canonical = this.canonicalize(obj);
    const s = this._stableStringify(canonical);
    const encoder = new TextEncoder();
    const data = encoder.encode(s);

    // crypto.subtle exists in browser; Node: globalThis.crypto in >=18, else fallback
    if (globalThis.crypto?.subtle) {
      const buf = await crypto.subtle.digest('SHA-256', data);
      const hex = [...new Uint8Array(buf)].map(b=>b.toString(16).padStart(2,'0')).join('');
      return `cid-${hex.slice(0,16)}`;
    } else {
      // tiny fallback (node:crypto)
      const { createHash } = require('crypto');
      const hex = createHash('sha256').update(data).digest('hex');
      return `cid-${hex.slice(0,16)}`;
    }
  }
}

// Operator sanity checks
function assertMatrix(M, rows, cols, name='M') {
  if (!Array.isArray(M) || M.length !== rows) throw new Error(`${name} rows != ${rows}`);
  for (const r of M) if (!Array.isArray(r) || r.length !== cols) throw new Error(`${name} must be ${rows}x${cols}`);
}

// Enhanced LLEX Button with validation
class LLEXButton {
  constructor(lid, operatorData={}, morphemes=[], metadata={}) {
    this.type = 'button';
    this.lid = lid;
    this.vid = null;
    this.cid = null;
    this.class = metadata.class || 'Transform';
    this.morphemes = morphemes;

    const M = operatorData.M || [[1,0,0],[0,1,0],[0,0,1]];
    const b = operatorData.b || [0,0,0];
    const C = operatorData.C || [[1,0,0],[0,1,0],[0,0,1]];
    assertMatrix(M, 3, 3, 'M');
    if (!Array.isArray(b) || b.length !== 3) throw new Error('b must be length 3');
    assertMatrix(C, 3, 3, 'C');

    this.operator = {
      M, b, C,
      alpha: +operatorData.alpha || 1.0,
      beta:  +operatorData.beta  || 0.0,
      delta_level: operatorData.delta_level|0
    };

    this.meta = { created_at: new Date().toISOString(), author: metadata.author || 'system', ...metadata };
  }

  _hashable() {
    return {
      type: this.type,
      lid: this.lid,
      class: this.class,
      morphemes: this.morphemes,
      operator: this.operator
      // NOTE: no cid/vid/meta/timestamps
    };
  }

  async computeCID() {
    this.cid = await CIDGenerator.hashCanonical(this._hashable());
    return this.cid;
  }
}

// Enhanced Object Store with idempotency
class LLEXObjectStore {
  constructor() {
    this.objects = new Map();
    this.stats = { total_objects: 0, size_bytes: 0 };
  }

  async store(obj) {
    if (!obj.cid) await obj.computeCID();
    const blob = JSON.stringify(obj, null, 2);
    const existed = this.objects.has(obj.cid);
    const prevLen = existed ? this.objects.get(obj.cid).length : 0;

    this.objects.set(obj.cid, blob);
    if (!existed) {
      this.stats.total_objects++;
      this.stats.size_bytes += blob.length;
    } else {
      // adjust bytes if overwriting same CID (shouldn't happen if truly immutable, but safe)
      this.stats.size_bytes += (blob.length - prevLen);
    }
    return obj.cid;
  }

  async fetch(cid) {
    const blob = this.objects.get(cid);
    if (!blob) throw new Error(`Object not found: ${cid}`);
    return JSON.parse(blob);
  }

  exists(cid) {
    return this.objects.has(cid);
  }

  delete(cid) {
    const blob = this.objects.get(cid);
    if (blob) {
      this.objects.delete(cid);
      this.stats.total_objects--;
      this.stats.size_bytes -= blob.length;
      return true;
    }
    return false;
  }

  getStats() {
    return { ...this.stats };
  }
}

// Enhanced Catalog with VID validation
class LLEXCatalog {
  constructor() {
    this.entries = new Map();       // key: ns:type:lid:vid
    this.current_pointers = new Map(); // key: ns:type:lid -> {vid,cid}
  }

  static isValidVID(vid) { return /^v\d+$/.test(vid); } // simple; swap for semver if you want

  static buildAddress(ns, type, lid, vid, cid) {
    const at = vid ? `@${vid}` : '';
    const hash = cid ? `#${cid}` : '';
    return `llex://${ns}/${type}/${lid}${at}${hash}`;
  }

  register(namespace, type, lid, vid, cid, isCurrent=false) {
    if (!LLEXCatalog.isValidVID(vid)) throw new Error(`Invalid VID: ${vid}`);
    const key = `${namespace}:${type}:${lid}:${vid}`;
    if (this.entries.has(key)) {
      const existing = this.entries.get(key);
      if (existing.cid !== cid) throw new Error(`VID collision for ${key}: ${existing.cid} vs ${cid}`);
      // idempotent
    } else {
      this.entries.set(key, { namespace, type, lid, vid, cid, created_at: new Date().toISOString() });
    }
    if (isCurrent) this.current_pointers.set(`${namespace}:${type}:${lid}`, { vid, cid });
  }

  resolve(namespace, type, lid, vid = 'current') {
    if (vid === 'current') {
      const pointer = this.current_pointers.get(`${namespace}:${type}:${lid}`);
      return pointer ? pointer.cid : null;
    }
    const key = `${namespace}:${type}:${lid}:${vid}`;
    const entry = this.entries.get(key);
    return entry ? entry.cid : null;
  }

  list(namespace = '', type = '', lid = '') {
    const results = [];
    for (const [key, entry] of this.entries) {
      if (namespace && entry.namespace !== namespace) continue;
      if (type && entry.type !== type) continue;
      if (lid && entry.lid !== lid) continue;
      results.push(entry);
    }
    return results.sort((a, b) => a.created_at.localeCompare(b.created_at));
  }

  getVersions(namespace, type, lid) {
    return this.list(namespace, type, lid).map(e => e.vid);
  }
}

// Enhanced Event Log with idempotent append
class LLEXEventLog {
  constructor() {
    this.events = [];
    this.sessions = new Map();   // session -> latest_seq
    this._bySessionSeq = new Map(); // `${session}#${seq}` -> true
  }

  async append(event) {
    if (!event.cid) await event.computeCID();
    const key = `${event.session}#${event.seq}`;
    if (this._bySessionSeq.has(key)) return event.cid; // idempotent

    this._bySessionSeq.set(key, true);
    this.events.push(event);

    const latest = this.sessions.get(event.session) || 0;
    if (event.seq > latest) this.sessions.set(event.session, event.seq);
    return event.cid;
  }

  getEvents(session, fromSeq = 0, toSeq = Infinity) {
    return this.events.filter(e =>
      e.session === session &&
      e.seq >= fromSeq &&
      e.seq <= toSeq
    ).sort((a, b) => a.seq - b.seq);
  }

  getAllEvents() {
    return [...this.events].sort((a, b) => a.timestamp - b.timestamp);
  }

  getSessionInfo(session) {
    const events = this.getEvents(session);
    const latestSeq = this.sessions.get(session) || 0;
    return {
      session,
      total_events: events.length,
      latest_seq: latestSeq,
      first_event: events[0] || null,
      last_event: events[events.length - 1] || null
    };
  }
}

// Enhanced Engine with safer operations
class LLEXEngine {
  constructor() {
    this.objectStore = new LLEXObjectStore();
    this.catalog = new LLEXCatalog();
    this.eventLog = new LLEXEventLog();
    this.sessions = new Map();
  }

  _nextVid(namespace, type, lid) {
    const versions = this.catalog.getVersions(namespace, type, lid);
    const n = versions.length + 1;
    return `v${n}`;
  }

  static parseAddress(address) {
    const re = /^llex:\/\/([^\/]+)\/([^\/]+)\/([^@#]+)(?:@([^#]+))?(?:#(.+))?$/;
    const m = address.match(re);
    if (!m) throw new Error(`Invalid LLEX address: ${address}`);
    const [, namespace, type, lid, vid, cid] = m;
    return { namespace, type, lid, vid, cid };
  }

  async createButton(namespace, lid, operatorData, morphemes=[], metadata={}) {
    const button = new LLEXButton(lid, operatorData, morphemes, metadata);
    await button.computeCID();
    const vid = this._nextVid(namespace, 'button', lid);
    button.vid = vid;

    await this.objectStore.store(button);
    this.catalog.register(namespace, 'button', lid, vid, button.cid, true);

    return {
      address: LLEXCatalog.buildAddress(namespace, 'button', lid, vid, button.cid),
      cid: button.cid,
      vid
    };
  }

  async resolve(address) {
    const { namespace, type, lid, vid, cid } = LLEXEngine.parseAddress(address);
    if (cid) {
      // Optional: ensure catalog maps lid@vid → cid, if vid present
      if (vid) {
        const expect = this.catalog.resolve(namespace, type, lid, vid);
        if (expect && expect !== cid) throw new Error(`CID mismatch for ${lid}@${vid}: catalog=${expect} address=${cid}`);
      }
      return await this.objectStore.fetch(cid);
    }
    const resolvedCID = this.catalog.resolve(namespace, type, lid, vid || 'current');
    if (!resolvedCID) throw new Error(`Not found: ${address}`);
    return await this.objectStore.fetch(resolvedCID);
  }

  parseButtonAddress(address) {
    // allow "button:Rebuild@v3" OR full llex URI; normalize
    if (address.startsWith('llex://')) {
      const { type, lid, vid } = LLEXEngine.parseAddress(address);
      if (type !== 'button') throw new Error(`Not a button address: ${address}`);
      return [`${type}:${lid}`, vid];
    }
    const m = address.match(/^([^:@]+:[^@]+)@(.+)$/);
    if (!m) throw new Error(`Invalid button short address: ${address}`);
    return [m[1], m[2]];
  }

  setSessionHead(session, snapshotCID) {
    this.sessions.set(session, { head: snapshotCID, updated_at: Date.now() });
  }

  getSessionHead(session) {
    return this.sessions.get(session)?.head || null;
  }

  getStats() {
    return {
      object_store: this.objectStore.getStats(),
      catalog_entries: this.catalog.entries.size,
      total_events: this.eventLog.events.length,
      active_sessions: this.sessions.size
    };
  }
}

// Enhanced Address Resolver with true LRU cache
class LLEXAddressResolver {
  constructor(engine) {
    this.engine = engine;
    this.cache = new Map(); // preserves insertion order
    this.maxCacheSize = 1000;
    this.hits = 0; this.misses = 0;
  }

  _touch(key) {
    const v = this.cache.get(key);
    if (!v) return;
    this.cache.delete(key);        // move-to-end
    this.cache.set(key, v);
  }

  cacheResult(address, object) {
    if (this.cache.size >= this.maxCacheSize) {
      const oldestKey = this.cache.keys().next().value; // FIFO eviction
      this.cache.delete(oldestKey);
    }
    this.cache.set(address, { object, timestamp: Date.now() });
  }

  async resolve(address) {
    const cached = this.cache.get(address);
    if (cached && (Date.now() - cached.timestamp) < 60_000) {
      this.hits++; this._touch(address);
      return cached.object;
    }
    this.misses++;
    const parsed = this.engine.constructor.parseAddress(address);
    let object;
    if (parsed.cid) object = await this.engine.objectStore.fetch(parsed.cid);
    else {
      const resolvedCID = this.engine.catalog.resolve(parsed.namespace, parsed.type, parsed.lid, parsed.vid);
      if (!resolvedCID) throw new Error(`Address not found: ${address}`);
      object = await this.engine.objectStore.fetch(resolvedCID);
    }
    this.cacheResult(address, object);
    return object;
  }

  parseAddress(address) {
    return this.engine.constructor.parseAddress(address);
  }

  buildAddress(ns, type, lid, vid, cid) {
    return LLEXCatalog.buildAddress(ns, type, lid, vid, cid);
  }

  getCacheStats() {
    const lookups = this.hits + this.misses || 1;
    return {
      size: this.cache.size,
      maxSize: this.maxCacheSize,
      hitRate: this.hits / lookups
    };
  }
}

export {
  CIDGenerator,
  assertMatrix,
  LLEXButton,
  LLEXObjectStore,
  LLEXCatalog,
  LLEXEventLog,
  LLEXEngine,
  LLEXAddressResolver
};
