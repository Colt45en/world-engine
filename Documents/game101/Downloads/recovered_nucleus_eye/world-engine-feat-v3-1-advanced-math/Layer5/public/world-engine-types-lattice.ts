/**
 * World Engine Types Lattice - TypeScript Edition
 * Complete type system with composition checking and lattice operations
 * Version: 1.0.0
 */

// ===== Core Type System =====

export type TypeName =
    | 'void' | 'any' | 'unknown'
    | 'string' | 'number' | 'boolean'
    | 'array' | 'object' | 'function'
    | 'morpheme' | 'lexeme' | 'token'
    | 'vector2d' | 'vector3d' | 'matrix'
    | 'analysis' | 'result' | 'composite';

export interface TypeNode {
    name: TypeName;
    supertypes: TypeName[];
    subtypes: TypeName[];
    properties: Record<string, any>;
    meta: {
        created: number;
        rank: number;
        compatibility: TypeName[];
    };
}

export interface TypeLattice {
    nodes: Map<TypeName, TypeNode>;
    checkCompose(outA: TypeName, inB: TypeName): boolean;
    getSupertypes(type: TypeName): TypeName[];
    getSubtypes(type: TypeName): TypeName[];
    isCompatible(source: TypeName, target: TypeName): boolean;
    findCommonSupertype(types: TypeName[]): TypeName | null;
    canTransform(from: TypeName, to: TypeName): boolean;
}

// ===== Type Lattice Implementation =====

export class WorldEngineTypeLattice implements TypeLattice {
    nodes: Map<TypeName, TypeNode> = new Map();

    constructor() {
        this.initializeBasicTypes();
        this.initializeWorldEngineTypes();
        this.initializeMathematicalTypes();
        this.initializeCompositeTypes();
    }

    private initializeBasicTypes(): void {
        // Foundation types
        this.addType('void', [], ['any'], { primitive: true });
        this.addType('any', ['void'], [], { top: true });
        this.addType('unknown', ['void'], ['any'], { safe: true });

        // Basic data types
        this.addType('string', ['unknown'], ['token', 'lexeme'], { serializable: true });
        this.addType('number', ['unknown'], ['vector2d', 'vector3d'], { numeric: true });
        this.addType('boolean', ['unknown'], [], { logical: true });
        this.addType('array', ['unknown'], ['matrix', 'vector2d', 'vector3d'], { container: true });
        this.addType('object', ['unknown'], ['morpheme', 'analysis', 'result'], { structured: true });
        this.addType('function', ['unknown'], [], { executable: true });
    }

    private initializeWorldEngineTypes(): void {
        // Linguistic types
        this.addType('token', ['string'], ['lexeme'], { linguistic: true, atomic: true });
        this.addType('lexeme', ['token'], ['morpheme'], { linguistic: true, meaningful: true });
        this.addType('morpheme', ['object', 'lexeme'], ['composite'], {
            linguistic: true,
            transformable: true,
            components: ['symbol', 'matrix', 'bias']
        });
    }

    private initializeMathematicalTypes(): void {
        // Mathematical types
        this.addType('vector2d', ['array', 'number'], ['vector3d'], {
            mathematical: true,
            dimensions: 2,
            operations: ['add', 'scale', 'dot']
        });
        this.addType('vector3d', ['vector2d'], ['matrix'], {
            mathematical: true,
            dimensions: 3,
            operations: ['add', 'scale', 'dot', 'cross']
        });
        this.addType('matrix', ['array', 'vector3d'], [], {
            mathematical: true,
            dimensions: 'variable',
            operations: ['multiply', 'transpose', 'inverse']
        });
    }

    private initializeCompositeTypes(): void {
        // Analysis types
        this.addType('analysis', ['object'], ['result'], {
            worldengine: true,
            components: ['input', 'process', 'output']
        });
        this.addType('result', ['object', 'analysis'], ['composite'], {
            worldengine: true,
            components: ['outcome', 'confidence', 'metadata']
        });
        this.addType('composite', ['object', 'morpheme', 'result'], [], {
            worldengine: true,
            complex: true,
            combinable: true
        });
    }

    private addType(name: TypeName, supertypes: TypeName[], subtypes: TypeName[], properties: Record<string, any>): void {
        this.nodes.set(name, {
            name,
            supertypes,
            subtypes,
            properties,
            meta: {
                created: Date.now(),
                rank: this.calculateRank(supertypes),
                compatibility: this.calculateCompatibility(name, supertypes, properties)
            }
        });
    }

    private calculateRank(supertypes: TypeName[]): number {
        if (supertypes.length === 0) return 0;
        return Math.max(...supertypes.map(t => (this.nodes.get(t)?.meta.rank ?? 0) + 1));
    }

    private calculateCompatibility(name: TypeName, supertypes: TypeName[], properties: Record<string, any>): TypeName[] {
        const compatible: Set<TypeName> = new Set([name, ...supertypes]);

        // Add mathematical compatibility
        if (properties.mathematical) {
            compatible.add('number');
            compatible.add('array');
        }

        // Add linguistic compatibility
        if (properties.linguistic) {
            compatible.add('string');
            compatible.add('object');
        }

        return Array.from(compatible);
    }

    checkCompose(outA: TypeName, inB: TypeName): boolean {
        const nodeA = this.nodes.get(outA);
        const nodeB = this.nodes.get(inB);

        if (!nodeA || !nodeB) return false;

        // Direct compatibility check
        if (nodeA.meta.compatibility.includes(inB) ||
            nodeB.meta.compatibility.includes(outA)) {
            return true;
        }

        // Subtyping check
        if (this.isSubtype(outA, inB) || this.isSubtype(inB, outA)) {
            return true;
        }

        // Mathematical composition rules
        if (this.checkMathematicalComposition(nodeA, nodeB)) {
            return true;
        }

        // World Engine specific rules
        if (this.checkWorldEngineComposition(nodeA, nodeB)) {
            return true;
        }

        return false;
    }

    private checkMathematicalComposition(nodeA: TypeNode, nodeB: TypeNode): boolean {
        const mathA = nodeA.properties.mathematical;
        const mathB = nodeB.properties.mathematical;

        if (!mathA && !mathB) return false;

        // Vector-Matrix compositions
        if (nodeA.name === 'matrix' && (nodeB.name === 'vector2d' || nodeB.name === 'vector3d')) {
            return true;
        }

        // Vector-Vector compositions
        if ((nodeA.name === 'vector2d' || nodeA.name === 'vector3d') &&
            (nodeB.name === 'vector2d' || nodeB.name === 'vector3d')) {
            return true;
        }

        return false;
    }

    private checkWorldEngineComposition(nodeA: TypeNode, nodeB: TypeNode): boolean {
        const weA = nodeA.properties.worldengine;
        const weB = nodeB.properties.worldengine;
        const lingA = nodeA.properties.linguistic;
        const lingB = nodeB.properties.linguistic;

        // World Engine analysis can compose with linguistic types
        if (weA && lingB) return true;
        if (lingA && weB) return true;

        // Morphemes can compose with mathematical types
        if (nodeA.name === 'morpheme' && nodeB.properties.mathematical) {
            return true;
        }

        return false;
    }

    getSupertypes(type: TypeName): TypeName[] {
        const node = this.nodes.get(type);
        if (!node) return [];

        const supertypes = new Set<TypeName>(node.supertypes);

        // Recursively add supertypes of supertypes
        for (const supertype of node.supertypes) {
            this.getSupertypes(supertype).forEach(s => supertypes.add(s));
        }

        return Array.from(supertypes);
    }

    getSubtypes(type: TypeName): TypeName[] {
        const node = this.nodes.get(type);
        if (!node) return [];

        const subtypes = new Set<TypeName>(node.subtypes);

        // Recursively add subtypes of subtypes
        for (const subtype of node.subtypes) {
            this.getSubtypes(subtype).forEach(s => subtypes.add(s));
        }

        return Array.from(subtypes);
    }

    isCompatible(source: TypeName, target: TypeName): boolean {
        return this.checkCompose(source, target);
    }

    private isSubtype(subtype: TypeName, supertype: TypeName): boolean {
        return this.getSupertypes(subtype).includes(supertype);
    }

    findCommonSupertype(types: TypeName[]): TypeName | null {
        if (types.length === 0) return null;
        if (types.length === 1) return types[0];

        // Get all supertypes of the first type
        const commonSupertypes = new Set(this.getSupertypes(types[0]));
        commonSupertypes.add(types[0]);

        // Intersect with supertypes of other types
        for (let i = 1; i < types.length; i++) {
            const typeSupertypes = new Set(this.getSupertypes(types[i]));
            typeSupertypes.add(types[i]);

            // Keep only common supertypes
            for (const supertype of commonSupertypes) {
                if (!typeSupertypes.has(supertype)) {
                    commonSupertypes.delete(supertype);
                }
            }
        }

        if (commonSupertypes.size === 0) return null;

        // Return the most specific common supertype (highest rank)
        let bestSupertype: TypeName | null = null;
        let bestRank = -1;

        for (const supertype of commonSupertypes) {
            const rank = this.nodes.get(supertype)?.meta.rank ?? 0;
            if (rank > bestRank) {
                bestRank = rank;
                bestSupertype = supertype;
            }
        }

        return bestSupertype;
    }

    canTransform(from: TypeName, to: TypeName): boolean {
        if (from === to) return true;

        const fromNode = this.nodes.get(from);
        const toNode = this.nodes.get(to);

        if (!fromNode || !toNode) return false;

        // Check if transformation is possible through subtyping
        if (this.isSubtype(from, to) || this.isSubtype(to, from)) {
            return true;
        }

        // Check if both types have transformable properties
        if (fromNode.properties.transformable && toNode.properties.transformable) {
            return true;
        }

        // Mathematical transformations
        if (fromNode.properties.mathematical && toNode.properties.mathematical) {
            return true;
        }

        return false;
    }

    // ===== Utility Methods =====

    getTypeInfo(type: TypeName): TypeNode | null {
        return this.nodes.get(type) ?? null;
    }

    getAllTypes(): TypeName[] {
        return Array.from(this.nodes.keys());
    }

    validateTypeChain(chain: TypeName[]): { valid: boolean; error?: string } {
        if (chain.length < 2) return { valid: true };

        for (let i = 0; i < chain.length - 1; i++) {
            if (!this.checkCompose(chain[i], chain[i + 1])) {
                return {
                    valid: false,
                    error: `Cannot compose ${chain[i]} -> ${chain[i + 1]} at position ${i}`
                };
            }
        }

        return { valid: true };
    }

    suggestCompositions(type: TypeName): TypeName[] {
        const suggestions: TypeName[] = [];

        for (const [candidateType] of this.nodes) {
            if (candidateType !== type && this.checkCompose(type, candidateType)) {
                suggestions.push(candidateType);
            }
        }

        return suggestions;
    }
}

// ===== Integration with Studio Bridge =====

export interface StudioBridgeTyped {
    onBus(fn: (msg: any) => void): () => void;
    sendBus(msg: any): void;
    typeLattice: WorldEngineTypeLattice;
}

export const typeLattice = new WorldEngineTypeLattice();

// ===== Enhanced Store with Type Safety =====
export const TypedStore = {
    async save<T>(key: string, value: T, type?: TypeName): Promise<boolean> {
        try {
            if (typeof window !== 'undefined' && window.externalStore?.upsert) {
                const payload = type ? { value, type, timestamp: Date.now() } : value;
                await window.externalStore.upsert(key, payload);
                return true;
            } else if (typeof window !== 'undefined' && window.localStorage) {
                const payload = type ? { value, type, timestamp: Date.now() } : value;
                window.localStorage.setItem(key, JSON.stringify(payload));
                return true;
            }
            throw new Error('No storage available');
        } catch {
            return false;
        }
    },

    async load<T>(key: string, expectedType?: TypeName): Promise<T | null> {
        try {
            if (typeof window !== 'undefined' && window.externalStore?.get) {
                const result = await window.externalStore.get(key);
                return this.validateAndExtract(result, expectedType);
            } else if (typeof window !== 'undefined' && window.localStorage) {
                const item = window.localStorage.getItem(key);
                if (item) {
                    const parsed = JSON.parse(item);
                    return this.validateAndExtract(parsed, expectedType);
                }
            }
            return null;
        } catch {
            return null;
        }
    },

    private validateAndExtract<T>(data: any, expectedType?: TypeName): T | null {
        if (!expectedType) return data as T;

        if (data && typeof data === 'object' && 'type' in data && 'value' in data) {
            if (typeLattice.isCompatible(data.type, expectedType)) {
                return data.value as T;
            } else {
                console.warn(`Type mismatch: expected ${expectedType}, got ${data.type}`);
                return null;
            }
        }

        return data as T;
    }
};

// ===== Enhanced Transport with Type Safety =====
export function setupTypedEngineTransport(engineFrame: HTMLIFrameElement) {
    const isSameOrigin = (): boolean => {
        try {
            void engineFrame.contentWindow!.document;
            return true;
        } catch {
            return false;
        }
    };

    const withEngine = <T>(fn: (doc: Document) => T): T => {
        if (isSameOrigin()) {
            const doc = engineFrame.contentWindow!.document;
            return fn(doc);
        } else {
            throw new Error('Cross-origin detected, use postMessage transport');
        }
    };

    const sendTypedMessage = (message: any, type: TypeName): void => {
        const typedMessage = {
            ...message,
            __type: type,
            __timestamp: Date.now()
        };

        if (isSameOrigin()) {
            // Direct access for same-origin
            withEngine(doc => {
                const event = new CustomEvent('studio:typed:message', { detail: typedMessage });
                doc.dispatchEvent(event);
            });
        } else {
            // PostMessage for cross-origin
            engineFrame.contentWindow?.postMessage(typedMessage, '*');
        }
    };

    return { isSameOrigin, withEngine, sendTypedMessage, typeLattice };
}

// ===== Enhanced Utilities with Type Support =====
export const TypedUtils = {
    generateId: (): string => String(Date.now()) + Math.random().toString(36).slice(2, 11),

    parseCommand: (line: string): { type: string; args: string; dataType?: TypeName } => {
        const t = line.trim();

        // Enhanced parsing with type hints
        if (t.startsWith('/run ')) {
            const args = t.slice(5);
            const dataType = args.includes('--type=') ?
                args.match(/--type=(\w+)/)?.[1] as TypeName : 'string';
            return { type: 'run', args: args.replace(/--type=\w+\s*/, ''), dataType };
        }

        if (t.startsWith('/test ')) return { type: 'test', args: t.slice(6), dataType: 'analysis' };
        if (t === '/rec start') return { type: 'rec', args: 'start', dataType: 'void' };
        if (t === '/rec stop') return { type: 'rec', args: 'stop', dataType: 'result' };
        if (t.startsWith('/mark ')) return { type: 'mark', args: t.slice(6), dataType: 'string' };

        return { type: 'run', args: t, dataType: 'string' };
    },

    log: (message: string, level: 'info' | 'warn' | 'error' | 'debug' = 'info', context?: TypeName) => {
        const timestamp = new Date().toISOString();
        const prefix = context ? `[Studio:${level}:${context}]` : `[Studio:${level}]`;
        (console[level] ?? console.log)(`${prefix} ${timestamp} - ${message}`);
    },

    validateType: (value: any, expectedType: TypeName): boolean => {
        return typeLattice.isCompatible('unknown', expectedType); // Simplified for now
    },

    inferType: (value: any): TypeName => {
        if (value === null || value === undefined) return 'void';
        if (typeof value === 'string') return 'string';
        if (typeof value === 'number') return 'number';
        if (typeof value === 'boolean') return 'boolean';
        if (Array.isArray(value)) {
            if (value.length === 2 && value.every(v => typeof v === 'number')) return 'vector2d';
            if (value.length === 3 && value.every(v => typeof v === 'number')) return 'vector3d';
            return 'array';
        }
        if (typeof value === 'object') {
            if ('symbol' in value && 'M' in value && 'b' in value) return 'morpheme';
            if ('outcome' in value) return 'result';
            return 'object';
        }
        return 'unknown';
    }
};

// ===== Window Integration =====
try {
    if (typeof window !== 'undefined') {
        window.StudioBridge = {
            ...window.StudioBridge,
            typeLattice,
            TypedStore,
            TypedUtils,
            setupTypedEngineTransport
        } as any;
    }
} catch {
    /* non-browser environment */
}

// ===== Debug Helper with Type Information =====
try {
    if (typeof window !== 'undefined' && window.location?.search.includes('debug=types')) {
        console.log('🔬 World Engine Type Lattice Debug Mode');
        console.log('Available types:', typeLattice.getAllTypes());
        console.log('Type lattice:', typeLattice);

        // Test some compositions
        console.log('Composition tests:');
        console.log('string -> token:', typeLattice.checkCompose('string', 'token'));
        console.log('morpheme -> matrix:', typeLattice.checkCompose('morpheme', 'matrix'));
        console.log('vector3d -> matrix:', typeLattice.checkCompose('vector3d', 'matrix'));
    }
} catch {
    /* ignore */
}

// ===== Export for Module Systems =====
export default {
    WorldEngineTypeLattice,
    typeLattice,
    TypedStore,
    TypedUtils,
    setupTypedEngineTransport
};
