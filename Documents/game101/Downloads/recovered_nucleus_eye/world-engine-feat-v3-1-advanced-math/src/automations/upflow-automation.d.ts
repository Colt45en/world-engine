/**
 * IndexedDB Storage with synchronous facade
 */
export function createIDBIndexStorage({ dbName, storeName, version, preloadKeys }: {
    dbName?: string | undefined;
    storeName?: string | undefined;
    version?: number | undefined;
    preloadKeys?: never[] | undefined;
}): Promise<{
    get(key: any): any;
    set(key: any, value: any): void;
    delete(key: any): void;
    _flushKey(key: any): Promise<void>;
    flush(): Promise<void>;
    close(): Promise<void>;
}>;
/**
 * Build shard keys for preloading
 */
export function buildShardKeys(indexKey: any, shards: any): string[];
/**
 * Main upflow automation factory
 */
export function createUpflowAutomation({ storage, indexKey, shards, morph, deriveAbbrs, tokenize }: {
    storage: any;
    indexKey?: string | undefined;
    shards?: number | undefined;
    morph?: null | undefined;
    deriveAbbrs?: typeof defaultAbbrs | undefined;
    tokenize?: typeof defaultTokenize | undefined;
}): {
    addWord: (word: any, english?: string, metadata?: {}) => {
        word: any;
        root: any;
        prefix: string;
        suffix: string;
        abbrs: any[];
    };
    query: {
        byRoot(root: any): any[];
        byPrefix(prefix: any): any[];
        bySuffix(suffix: any): any[];
        byAbbrev(abbr: any): any[];
        getWord(word: any): any;
    };
    helper: {
        linkWord(word: any): any;
        linkMany(words: any): any;
    };
    librarian: {
        verify(): {
            ok: boolean;
            shards: number;
            totalWords: number;
            totalLinks: number;
            avgWordsPerShard: number;
        };
        snapshot(): {
            shards: never[];
            w2s: any;
        };
        compact(): {
            compacted: boolean;
            totalSize: number;
            shards: number;
        };
    };
    ingestFromLastRun(storageKey?: string): Promise<{
        word: any;
        root: any;
        prefix: string;
        suffix: string;
        abbrs: any[];
        ok: boolean;
        error?: never;
    } | {
        ok: boolean;
        error: any;
    }>;
};
/**
 * Default abbreviation derivation
 */
declare function defaultAbbrs(word: any, english?: string): any[];
/**
 * Default tokenization
 */
declare function defaultTokenize(text: any): any;
export {};
//# sourceMappingURL=upflow-automation.d.ts.map