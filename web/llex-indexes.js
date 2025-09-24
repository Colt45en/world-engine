/**
 * LLEX Index Layers - Search and retrieval on top of content-addressable storage
 * Implements vector (kNN), graph (relationships), and text (morpheme) indexes
 */

// Vector Index - k-nearest neighbors for semantic search
class LLEXVectorIndex {
    constructor() {
        this.vectors = new Map(); // cid -> {vector, metadata}
        this.dimension = 3; // Default for stationary unit x vector
    }

    // Add vector entry
    addVector(cid, vector, metadata = {}) {
        if (vector.length !== this.dimension) {
            throw new Error(`Vector dimension mismatch: expected ${this.dimension}, got ${vector.length}`);
        }

        this.vectors.set(cid, {
            vector: [...vector], // Copy to prevent mutation
            metadata: {
                timestamp: new Date().toISOString(),
                ...metadata
            }
        });

        console.debug(`🎯 Vector indexed: ${cid} at [${vector.join(', ')}]`);
    }

    // Remove vector entry
    removeVector(cid) {
        this.vectors.delete(cid);
    }

    // k-nearest neighbors search (cosine similarity)
    kNearestNeighbors(queryVector, k = 5, filter = null) {
        if (queryVector.length !== this.dimension) {
            throw new Error(`Query vector dimension mismatch: expected ${this.dimension}, got ${queryVector.length}`);
        }

        const candidates = [];

        for (const [cid, entry] of this.vectors) {
            // Apply filter if provided
            if (filter && !filter(entry.metadata)) continue;

            const similarity = this.cosineSimilarity(queryVector, entry.vector);
            candidates.push({
                cid,
                similarity,
                vector: entry.vector,
                metadata: entry.metadata
            });
        }

        // Sort by similarity (descending) and take top k
        candidates.sort((a, b) => b.similarity - a.similarity);
        return candidates.slice(0, k);
    }

    // Cosine similarity calculation
    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }

        const magnitude = Math.sqrt(normA * normB);
        return magnitude === 0 ? 0 : dotProduct / magnitude;
    }

    // Get all vectors in a region (euclidean distance)
    vectorsInRegion(center, radius, filter = null) {
        const results = [];

        for (const [cid, entry] of this.vectors) {
            if (filter && !filter(entry.metadata)) continue;

            const distance = this.euclideanDistance(center, entry.vector);
            if (distance <= radius) {
                results.push({
                    cid,
                    distance,
                    vector: entry.vector,
                    metadata: entry.metadata
                });
            }
        }

        results.sort((a, b) => a.distance - b.distance);
        return results;
    }

    // Euclidean distance
    euclideanDistance(vecA, vecB) {
        let sum = 0;
        for (let i = 0; i < vecA.length; i++) {
            sum += Math.pow(vecA[i] - vecB[i], 2);
        }
        return Math.sqrt(sum);
    }

    getStats() {
        return {
            total_vectors: this.vectors.size,
            dimension: this.dimension
        };
    }
}

// Graph Index - relationships between objects
class LLEXGraphIndex {
    constructor() {
        this.nodes = new Map(); // cid -> {type, metadata}
        this.edges = new Map(); // edge_id -> {from, to, type, weight, metadata}
        this.adjacency = new Map(); // cid -> Set of connected cids
        this.reverseAdjacency = new Map(); // cid -> Set of cids that point to this
    }

    // Add node to graph
    addNode(cid, type, metadata = {}) {
        this.nodes.set(cid, {
            type,
            metadata: {
                added_at: new Date().toISOString(),
                ...metadata
            }
        });

        // Initialize adjacency lists if not exist
        if (!this.adjacency.has(cid)) {
            this.adjacency.set(cid, new Set());
        }
        if (!this.reverseAdjacency.has(cid)) {
            this.reverseAdjacency.set(cid, new Set());
        }

        console.debug(`📊 Graph node added: ${cid} (${type})`);
    }

    // Add edge between nodes
    addEdge(fromCID, toCID, type, weight = 1.0, metadata = {}) {
        const edgeId = `${fromCID}->${toCID}:${type}`;

        // Ensure nodes exist
        if (!this.nodes.has(fromCID)) {
            this.addNode(fromCID, 'unknown');
        }
        if (!this.nodes.has(toCID)) {
            this.addNode(toCID, 'unknown');
        }

        this.edges.set(edgeId, {
            from: fromCID,
            to: toCID,
            type,
            weight,
            metadata: {
                created_at: new Date().toISOString(),
                ...metadata
            }
        });

        // Update adjacency lists
        this.adjacency.get(fromCID).add(toCID);
        this.reverseAdjacency.get(toCID).add(fromCID);

        console.debug(`🔗 Graph edge added: ${fromCID} -[${type}]-> ${toCID}`);
        return edgeId;
    }

    // Get neighbors (outgoing edges)
    getNeighbors(cid, edgeType = null) {
        const neighbors = [];
        const adjacentNodes = this.adjacency.get(cid) || new Set();

        for (const neighborCID of adjacentNodes) {
            for (const [edgeId, edge] of this.edges) {
                if (edge.from === cid && edge.to === neighborCID) {
                    if (edgeType === null || edge.type === edgeType) {
                        neighbors.push({
                            cid: neighborCID,
                            node: this.nodes.get(neighborCID),
                            edge: edge,
                            edgeId: edgeId
                        });
                    }
                }
            }
        }

        return neighbors;
    }

    // Get reverse neighbors (incoming edges)
    getReverseNeighbors(cid, edgeType = null) {
        const neighbors = [];
        const incomingNodes = this.reverseAdjacency.get(cid) || new Set();

        for (const sourceCID of incomingNodes) {
            for (const [edgeId, edge] of this.edges) {
                if (edge.to === cid && edge.from === sourceCID) {
                    if (edgeType === null || edge.type === edgeType) {
                        neighbors.push({
                            cid: sourceCID,
                            node: this.nodes.get(sourceCID),
                            edge: edge,
                            edgeId: edgeId
                        });
                    }
                }
            }
        }

        return neighbors;
    }

    // Path finding (simple BFS)
    findPath(fromCID, toCID, maxDepth = 6) {
        if (fromCID === toCID) return [fromCID];

        const queue = [[fromCID]];
        const visited = new Set([fromCID]);

        while (queue.length > 0) {
            const path = queue.shift();
            const currentCID = path[path.length - 1];

            if (path.length > maxDepth) continue;

            const neighbors = this.getNeighbors(currentCID);
            for (const neighbor of neighbors) {
                if (neighbor.cid === toCID) {
                    return [...path, toCID];
                }

                if (!visited.has(neighbor.cid)) {
                    visited.add(neighbor.cid);
                    queue.push([...path, neighbor.cid]);
                }
            }
        }

        return null; // No path found
    }

    // Get connected component
    getConnectedComponent(cid) {
        const component = new Set();
        const stack = [cid];

        while (stack.length > 0) {
            const currentCID = stack.pop();
            if (component.has(currentCID)) continue;

            component.add(currentCID);

            // Add both outgoing and incoming neighbors
            const outgoing = this.adjacency.get(currentCID) || new Set();
            const incoming = this.reverseAdjacency.get(currentCID) || new Set();

            for (const neighborCID of [...outgoing, ...incoming]) {
                if (!component.has(neighborCID)) {
                    stack.push(neighborCID);
                }
            }
        }

        return Array.from(component);
    }

    getStats() {
        return {
            total_nodes: this.nodes.size,
            total_edges: this.edges.size,
            node_types: this.getNodeTypeDistribution(),
            edge_types: this.getEdgeTypeDistribution()
        };
    }

    getNodeTypeDistribution() {
        const distribution = {};
        for (const [cid, node] of this.nodes) {
            distribution[node.type] = (distribution[node.type] || 0) + 1;
        }
        return distribution;
    }

    getEdgeTypeDistribution() {
        const distribution = {};
        for (const [edgeId, edge] of this.edges) {
            distribution[edge.type] = (distribution[edge.type] || 0) + 1;
        }
        return distribution;
    }
}

// Text Index - inverted index for morpheme and semantic search
class LLEXTextIndex {
    constructor() {
        this.invertedIndex = new Map(); // term -> Set of cids
        this.documents = new Map(); // cid -> {tokens, metadata}
        this.morphemeIndex = new Map(); // morpheme -> Set of cids
        this.classIndex = new Map(); // class -> Set of cids
    }

    // Add document to index
    addDocument(cid, content, metadata = {}) {
        const tokens = this.tokenize(content);
        const morphemes = metadata.morphemes || [];
        const objectClass = metadata.class || 'unknown';

        this.documents.set(cid, {
            tokens,
            morphemes,
            class: objectClass,
            metadata: {
                indexed_at: new Date().toISOString(),
                ...metadata
            }
        });

        // Build inverted index
        for (const token of tokens) {
            if (!this.invertedIndex.has(token)) {
                this.invertedIndex.set(token, new Set());
            }
            this.invertedIndex.get(token).add(cid);
        }

        // Build morpheme index
        for (const morpheme of morphemes) {
            if (!this.morphemeIndex.has(morpheme)) {
                this.morphemeIndex.set(morpheme, new Set());
            }
            this.morphemeIndex.get(morpheme).add(cid);
        }

        // Build class index
        if (!this.classIndex.has(objectClass)) {
            this.classIndex.set(objectClass, new Set());
        }
        this.classIndex.get(objectClass).add(cid);

        console.debug(`📝 Text indexed: ${cid} (${tokens.length} tokens, ${morphemes.length} morphemes)`);
    }

    // Simple tokenization
    tokenize(content) {
        return content
            .toLowerCase()
            .replace(/[^\w\s-]/g, ' ')
            .split(/\s+/)
            .filter(token => token.length > 0);
    }

    // Search by text query
    search(query, options = {}) {
        const tokens = this.tokenize(query);
        const results = new Map(); // cid -> score

        for (const token of tokens) {
            const matchingDocs = this.invertedIndex.get(token) || new Set();
            for (const cid of matchingDocs) {
                results.set(cid, (results.get(cid) || 0) + 1);
            }
        }

        // Convert to sorted array
        const sortedResults = Array.from(results.entries())
            .map(([cid, score]) => ({
                cid,
                score,
                document: this.documents.get(cid)
            }))
            .sort((a, b) => b.score - a.score);

        const limit = options.limit || 10;
        return sortedResults.slice(0, limit);
    }

    // Search by morpheme
    searchByMorpheme(morpheme) {
        const matchingDocs = this.morphemeIndex.get(morpheme) || new Set();
        return Array.from(matchingDocs).map(cid => ({
            cid,
            document: this.documents.get(cid)
        }));
    }

    // Search by class
    searchByClass(className) {
        const matchingDocs = this.classIndex.get(className) || new Set();
        return Array.from(matchingDocs).map(cid => ({
            cid,
            document: this.documents.get(cid)
        }));
    }

    // Get morpheme frequency
    getMorphemeFrequency(morpheme) {
        const docs = this.morphemeIndex.get(morpheme) || new Set();
        return docs.size;
    }

    // Get all morphemes with their frequencies
    getAllMorphemes() {
        const frequencies = {};
        for (const [morpheme, docs] of this.morphemeIndex) {
            frequencies[morpheme] = docs.size;
        }
        return frequencies;
    }

    getStats() {
        return {
            total_documents: this.documents.size,
            total_terms: this.invertedIndex.size,
            total_morphemes: this.morphemeIndex.size,
            total_classes: this.classIndex.size,
            morpheme_frequency: this.getAllMorphemes()
        };
    }
}

// Combined Index Manager - coordinates all index types
class LLEXIndexManager {
    constructor() {
        this.vectorIndex = new LLEXVectorIndex();
        this.graphIndex = new LLEXGraphIndex();
        this.textIndex = new LLEXTextIndex();
    }

    // Index a complete object across all index types
    indexObject(obj) {
        const cid = obj.cid;
        if (!cid) {
            throw new Error('Object must have CID for indexing');
        }

        // Add to graph
        this.graphIndex.addNode(cid, obj.type, {
            lid: obj.lid,
            vid: obj.vid,
            created_at: obj.meta?.created_at
        });

        // Type-specific indexing
        switch (obj.type) {
            case 'button':
                this.indexButton(obj);
                break;
            case 'snapshot':
                this.indexSnapshot(obj);
                break;
            case 'event':
                this.indexEvent(obj);
                break;
            case 'morpheme':
                this.indexMorpheme(obj);
                break;
        }

        console.debug(`🔍 Fully indexed object: ${cid} (${obj.type})`);
    }

    indexButton(button) {
        // Text index
        const content = `${button.lid} ${button.class} ${button.morphemes.join(' ')}`;
        this.textIndex.addDocument(button.cid, content, {
            morphemes: button.morphemes,
            class: button.class,
            lid: button.lid,
            vid: button.vid
        });

        // Graph relationships
        for (const morpheme of button.morphemes) {
            // This would link to morpheme objects when they exist
            this.graphIndex.addEdge(button.cid, `morpheme:${morpheme}`, 'contains', 1.0);
        }
    }

    indexSnapshot(snapshot) {
        // Vector index (stationary unit state)
        this.vectorIndex.addVector(snapshot.cid, snapshot.x, {
            session: snapshot.session,
            index: snapshot.index,
            level: snapshot.level,
            timestamp: snapshot.timestamp
        });

        // Graph relationships
        if (snapshot.prev_cid) {
            this.graphIndex.addEdge(snapshot.prev_cid, snapshot.cid, 'precedes', 1.0);
        }
    }

    indexEvent(event) {
        // Text index
        const content = `${event.button} ${event.session}`;
        this.textIndex.addDocument(event.cid, content, {
            button: event.button,
            session: event.session,
            user: event.provenance.user
        });

        // Graph relationships
        this.graphIndex.addEdge(event.input_cid, event.output_cid, 'transforms_to', 1.0, {
            button: event.button,
            seq: event.seq,
            timestamp: event.t
        });
    }

    indexMorpheme(morpheme) {
        // Text index
        const content = `${morpheme.form} ${morpheme.semantic.meaning} ${morpheme.morpheme_type}`;
        this.textIndex.addDocument(morpheme.cid, content, {
            morphemes: [morpheme.form],
            class: morpheme.morpheme_type,
            lid: morpheme.lid
        });
    }

    // Unified search across all indexes
    unifiedSearch(query, options = {}) {
        const results = {
            text: this.textIndex.search(query, options),
            morpheme: options.morpheme ? this.textIndex.searchByMorpheme(options.morpheme) : [],
            vector: options.vector ? this.vectorIndex.kNearestNeighbors(options.vector, options.k || 5) : [],
            class: options.class ? this.textIndex.searchByClass(options.class) : []
        };

        // Combine and deduplicate by CID
        const combined = new Map();
        const addResult = (result, source) => {
            if (combined.has(result.cid)) {
                combined.get(result.cid).sources.add(source);
            } else {
                combined.set(result.cid, { ...result, sources: new Set([source]) });
            }
        };

        results.text.forEach(r => addResult(r, 'text'));
        results.morpheme.forEach(r => addResult(r, 'morpheme'));
        results.vector.forEach(r => addResult(r, 'vector'));
        results.class.forEach(r => addResult(r, 'class'));

        return {
            results: Array.from(combined.values()),
            breakdown: results
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

// Export for use
if (typeof window !== 'undefined') {
    window.LLEX = {
        ...window.LLEX,
        VectorIndex: LLEXVectorIndex,
        GraphIndex: LLEXGraphIndex,
        TextIndex: LLEXTextIndex,
        IndexManager: LLEXIndexManager
    };
} else if (typeof module !== 'undefined') {
    module.exports = {
        ...module.exports,
        LLEXVectorIndex,
        LLEXGraphIndex,
        LLEXTextIndex,
        LLEXIndexManager
    };
}
