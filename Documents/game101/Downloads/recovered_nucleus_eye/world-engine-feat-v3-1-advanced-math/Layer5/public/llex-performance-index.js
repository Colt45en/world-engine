/**
 * High-Performance LLEX Index System
 * Faster vector operations, heap-based top-K, BM25 text search, and optimized neighbor lookups
 */

// 1) Vector index (faster cosine, heap top-K, squared distances)
class LLEXVectorIndex {
  constructor(dimension = 3) {
    this.vectors = new Map(); // cid -> {vector: Float32Array, norm: number, metadata}
    this.dimension = dimension;
  }

  setDimension(d) {
    if (this.vectors.size) throw new Error('Cannot change dimension after indexing vectors');
    if (!Number.isInteger(d) || d <= 0) throw new Error('Invalid dimension');
    this.dimension = d;
  }

  addVector(cid, vector, metadata = {}) {
    if (vector.length !== this.dimension) {
      throw new Error(`Vector dimension mismatch: expected ${this.dimension}, got ${vector.length}`);
    }
    const v = Float32Array.from(vector);
    let n = 0; for (let i = 0; i < v.length; i++) n += v[i]*v[i];
    const norm = Math.sqrt(n) || 1;

    this.vectors.set(cid, {
      vector: v,
      norm,
      metadata: { timestamp: new Date().toISOString(), ...metadata }
    });
  }

  removeVector(cid) { this.vectors.delete(cid); }

  // cosine using cached norms
  _cos(vecA, normA, vecB, normB) {
    let dot = 0;
    for (let i = 0; i < vecA.length; i++) dot += vecA[i]*vecB[i];
    const mag = normA * normB;
    return mag === 0 ? 0 : dot / mag;
  }

  // small min-heap for top-k
  kNearestNeighbors(queryVector, k = 5, filter = null) {
    if (queryVector.length !== this.dimension) {
      throw new Error(`Query vector dimension mismatch: expected ${this.dimension}, got ${queryVector.length}`);
    }
    const q = Float32Array.from(queryVector);
    let qn = 0; for (let i=0;i<q.length;i++) qn += q[i]*q[i];
    const qNorm = Math.sqrt(qn) || 1;

    const heap = []; // [{sim, cid, entry}]
    const push = (item) => {
      heap.push(item);
      heap.sort((a,b)=>a.sim-b.sim); // k is small; keep it simple
      if (heap.length > k) heap.shift();
    };

    for (const [cid, entry] of this.vectors) {
      if (filter && !filter(entry.metadata)) continue;
      const sim = this._cos(q, qNorm, entry.vector, entry.norm);
      if (heap.length < k || sim > heap[0].sim) {
        push({ sim, cid, vector: entry.vector, metadata: entry.metadata });
      }
    }
    // return high→low
    return heap.sort((a,b)=>b.sim-a.sim);
  }

  // squared distance (no sqrt)
  vectorsInRegion(center, radius, filter = null) {
    const c = Float32Array.from(center);
    const r2 = radius*radius;
    const out = [];
    for (const [cid, entry] of this.vectors) {
      if (filter && !filter(entry.metadata)) continue;
      let acc = 0;
      const v = entry.vector;
      for (let i=0;i<v.length;i++){ const d = v[i]-c[i]; acc += d*d; }
      if (acc <= r2) out.push({ cid, distance: Math.sqrt(acc), vector: v, metadata: entry.metadata });
    }
    out.sort((a,b)=>a.distance-b.distance);
    return out;
  }

  getStats() {
    return { total_vectors: this.vectors.size, dimension: this.dimension };
  }
}

// 2) Graph index (neighbor lookup in O(outdegree))
class LLEXGraphIndex {
  constructor() {
    this.nodes = new Map(); // cid -> {type, metadata, degree: {out:0,in:0}}
    this.edges = new Map(); // edge_id -> {from,to,type,weight,metadata}
    this.edgesByFrom = new Map(); // cid -> Map(edgeId -> edge)
    this.edgesByTo = new Map();   // cid -> Map(edgeId -> edge)
  }

  _ensureNode(cid, type='unknown', metadata={}) {
    if (!this.nodes.has(cid)) {
      this.nodes.set(cid, { type, metadata: { added_at: new Date().toISOString(), ...metadata }, degree:{out:0,in:0} });
    }
    if (!this.edgesByFrom.has(cid)) this.edgesByFrom.set(cid, new Map());
    if (!this.edgesByTo.has(cid)) this.edgesByTo.set(cid, new Map());
  }

  addNode(cid, type, metadata = {}) {
    this._ensureNode(cid, type, metadata);
  }

  addEdge(fromCID, toCID, type, weight = 1.0, metadata = {}) {
    this._ensureNode(fromCID); this._ensureNode(toCID);
    const edgeId = `${fromCID}->${toCID}:${type}`;
    if (this.edges.has(edgeId)) return edgeId; // idempotent

    const edge = { from: fromCID, to: toCID, type, weight,
      metadata: { created_at: new Date().toISOString(), ...metadata } };
    this.edges.set(edgeId, edge);
    this.edgesByFrom.get(fromCID).set(edgeId, edge);
    this.edgesByTo.get(toCID).set(edgeId, edge);
    this.nodes.get(fromCID).degree.out++;
    this.nodes.get(toCID).degree.in++;
    return edgeId;
  }

  removeEdge(edgeId) {
    const edge = this.edges.get(edgeId);
    if (!edge) return false;
    this.edges.delete(edgeId);
    this.edgesByFrom.get(edge.from)?.delete(edgeId);
    this.edgesByTo.get(edge.to)?.delete(edgeId);
    this.nodes.get(edge.from).degree.out--;
    this.nodes.get(edge.to).degree.in--;
    return true;
  }

  removeNode(cid) {
    if (!this.nodes.has(cid)) return false;
    // remove outgoing
    for (const [eid] of this.edgesByFrom.get(cid)) this.removeEdge(eid);
    // remove incoming
    for (const [eid] of this.edgesByTo.get(cid)) this.removeEdge(eid);
    this.edgesByFrom.delete(cid);
    this.edgesByTo.delete(cid);
    this.nodes.delete(cid);
    return true;
  }

  getNeighbors(cid, edgeType = null) {
    const out = [];
    const edges = this.edgesByFrom.get(cid) || new Map();
    for (const [edgeId, e] of edges) {
      if (edgeType && e.type !== edgeType) continue;
      out.push({ cid: e.to, node: this.nodes.get(e.to), edge: e, edgeId });
    }
    return out;
  }

  getReverseNeighbors(cid, edgeType = null) {
    const out = [];
    const edges = this.edgesByTo.get(cid) || new Map();
    for (const [edgeId, e] of edges) {
      if (edgeType && e.type !== edgeType) continue;
      out.push({ cid: e.from, node: this.nodes.get(e.from), edge: e, edgeId });
    }
    return out;
  }

  findPath(fromCID, toCID, maxDepth = 6) {
    if (fromCID === toCID) return [fromCID];
    const q = [[fromCID]];
    const seen = new Set([fromCID]);
    while (q.length) {
      const path = q.shift();
      const u = path[path.length-1];
      if (path.length > maxDepth) continue;
      for (const n of this.getNeighbors(u)) {
        if (n.cid === toCID) return [...path, toCID];
        if (!seen.has(n.cid)) { seen.add(n.cid); q.push([...path, n.cid]); }
      }
    }
    return null;
  }

  getConnectedComponent(start) {
    const comp = new Set();
    const stack = [start];
    while (stack.length) {
      const u = stack.pop();
      if (comp.has(u)) continue;
      comp.add(u);
      for (const n of this.getNeighbors(u)) if (!comp.has(n.cid)) stack.push(n.cid);
      for (const n of this.getReverseNeighbors(u)) if (!comp.has(n.cid)) stack.push(n.cid);
    }
    return Array.from(comp);
  }

  getNodeTypeDistribution() {
    const d = {};
    for (const [,node] of this.nodes) d[node.type] = (d[node.type]||0)+1;
    return d;
  }
  getEdgeTypeDistribution() {
    const d = {};
    for (const [,edge] of this.edges) d[edge.type] = (d[edge.type]||0)+1;
    return d;
  }

  getStats() {
    const degrees = { out: [], in: [] };
    for (const [,n] of this.nodes) { degrees.out.push(n.degree.out); degrees.in.push(n.degree.in); }
    const avg = a => a.reduce((x,y)=>x+y,0)/(a.length||1);
    return {
      total_nodes: this.nodes.size,
      total_edges: this.edges.size,
      node_types: this.getNodeTypeDistribution(),
      edge_types: this.getEdgeTypeDistribution(),
      avg_out_degree: avg(degrees.out).toFixed(2),
      avg_in_degree: avg(degrees.in).toFixed(2)
    };
  }
}

// 3) Text index (Unicode tokenizer + BM25)
class LLEXTextIndex {
  constructor() {
    this.invertedIndex = new Map(); // term -> Map(cid -> tf)
    this.docLen = new Map();        // cid -> length
    this.df = new Map();            // term -> document frequency
    this.documents = new Map();     // cid -> {tokens, metadata}
    this.morphemeIndex = new Map(); // morpheme -> Set(cid)
    this.classIndex = new Map();    // class -> Set(cid)
    this.N = 0; this.avgLen = 0;
  }

  tokenize(content) {
    // Unicode letters/numbers plus dashes; collapse whitespace
    return (content||'')
      .toLowerCase()
      .normalize('NFKC')
      .replace(/[^\p{L}\p{N}\- ]+/gu, ' ')
      .split(/\s+/)
      .filter(Boolean);
  }

  addDocument(cid, content, metadata = {}) {
    const tokens = this.tokenize(content);
    const morphemes = metadata.morphemes || [];
    const objectClass = metadata.class || 'unknown';

    this.documents.set(cid, { tokens, metadata:{ indexed_at:new Date().toISOString(), ...metadata } });

    // TF and DF
    const tf = new Map();
    for (const t of tokens) tf.set(t, (tf.get(t)||0)+1);
    for (const [t, count] of tf) {
      if (!this.invertedIndex.has(t)) this.invertedIndex.set(t, new Map());
      this.invertedIndex.get(t).set(cid, count);
      this.df.set(t, (this.df.get(t)||0)+1);
    }
    this.docLen.set(cid, tokens.length);
    this.N = this.documents.size;
    this.avgLen = [...this.docLen.values()].reduce((a,b)=>a+b,0)/(this.N||1);

    // Morphemes
    for (const m of morphemes) {
      if (!this.morphemeIndex.has(m)) this.morphemeIndex.set(m, new Set());
      this.morphemeIndex.get(m).add(cid);
    }
    // Class
    if (!this.classIndex.has(objectClass)) this.classIndex.set(objectClass, new Set());
    this.classIndex.get(objectClass).add(cid);
  }

  // BM25
  _bm25Score(queryTokens) {
    const k1 = 1.5, b = 0.75;
    const scores = new Map(); // cid -> score
    const uniq = Array.from(new Set(queryTokens));
    for (const q of uniq) {
      const postings = this.invertedIndex.get(q);
      if (!postings) continue;
      const df = this.df.get(q) || 0;
      const idf = Math.log(1 + (this.N - df + 0.5) / (df + 0.5));
      for (const [cid, tf] of postings) {
        const len = this.docLen.get(cid) || 1;
        const norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * (len / (this.avgLen||1))));
        scores.set(cid, (scores.get(cid)||0) + idf * norm);
      }
    }
    return scores;
  }

  search(query, options = {}) {
    const q = this.tokenize(query);
    const scores = this._bm25Score(q);
    const out = [];
    for (const [cid, score] of scores) {
      out.push({ cid, score, document: this.documents.get(cid) });
    }
    out.sort((a,b)=>b.score-a.score);
    const limit = options.limit ?? 10;
    return out.slice(0, limit);
  }

  searchByMorpheme(m) {
    const set = this.morphemeIndex.get(m) || new Set();
    return Array.from(set).map(cid => ({ cid, document: this.documents.get(cid) }));
  }

  searchByClass(className) {
    const set = this.classIndex.get(className) || new Set();
    return Array.from(set).map(cid => ({ cid, document: this.documents.get(cid) }));
  }

  getAllMorphemes() {
    const freq = {};
    for (const [m, docs] of this.morphemeIndex) freq[m] = docs.size;
    return freq;
  }

  getStats() {
    return {
      total_documents: this.documents.size,
      total_terms: this.invertedIndex.size,
      total_morphemes: this.morphemeIndex.size,
      total_classes: this.classIndex.size,
      avg_doc_len: Number(this.avgLen.toFixed(2)),
      morpheme_frequency: this.getAllMorphemes()
    };
  }
}

// 4) Unified search (score fusion + traces)
class LLEXIndexManager {
  constructor() {
    this.vectorIndex = new LLEXVectorIndex();
    this.graphIndex = new LLEXGraphIndex();
    this.textIndex = new LLEXTextIndex();
    this.fusionWeights = { text: 1.0, vector: 0.7, morpheme: 0.2, class: 0.2 };
  }

  setVectorDimension(d) { this.vectorIndex.setDimension(d); }

  indexObject(obj) {
    const cid = obj.cid || `obj-${Date.now()}-${Math.random()}`;

    // Vector indexing if coordinates present
    if (obj.x && Array.isArray(obj.x)) {
      this.vectorIndex.addVector(cid, obj.x, obj);
    }

    // Graph indexing
    this.graphIndex.addNode(cid, obj.type || 'object', obj);

    // Text indexing
    const content = [obj.word, obj.description, obj.english, obj.content].filter(Boolean).join(' ');
    this.textIndex.addDocument(cid, content, obj);

    return cid;
  }

  unifiedSearch(query, options = {}) {
    const { morpheme, vector, k=5, class: className, limit=20, weights } = options;
    const w = { ...this.fusionWeights, ...(weights||{}) };

    const text = this.textIndex.search(query, { limit: Math.max(limit, 50) });
    const morph = morpheme ? this.textIndex.searchByMorpheme(morpheme) : [];
    const cls = className ? this.textIndex.searchByClass(className) : [];
    const vec = vector ? this.vectorIndex.kNearestNeighbors(vector, k) : [];

    const combined = new Map(); // cid -> {score, sources:{}, parts:{}}
    const bump = (cid, amt, src, parts={}) => {
      if (!combined.has(cid)) combined.set(cid, { cid, score:0, sources:new Set(), parts:{} });
      const r = combined.get(cid);
      r.score += amt;
      r.sources.add(src);
      r.parts[src] = parts;
    };

    for (const r of text) bump(r.cid, w.text * r.score, 'text', { score:r.score });
    for (const r of vec)  bump(r.cid, w.vector * r.sim, 'vector', { sim:r.sim });
    for (const r of morph) bump(r.cid, w.morpheme * 1.0, 'morpheme');
    for (const r of cls)   bump(r.cid, w.class * 1.0, 'class');

    const results = Array.from(combined.values())
      .sort((a,b)=>b.score-a.score)
      .slice(0, limit)
      .map(r => ({ ...r, sources: Array.from(r.sources) }));

    return {
      results,
      breakdown: { text, morpheme: morph, vector: vec, class: cls },
      weights: w
    };
  }

  getStats() {
    return {
      vector: this.vectorIndex.getStats(),
      graph: this.graphIndex.getStats(),
      text: this.textIndex.getStats()
    };
  }
}

export {
  LLEXVectorIndex,
  LLEXGraphIndex,
  LLEXTextIndex,
  LLEXIndexManager
};
