/**
 * World Engine V3.1 - Enhanced Lexicon Explorer
 * Inspired by Lexicon Explorer.txt attachment
 *
 * Provides advanced morphological navigation, word linking, and semantic analysis
 * with reactive UI components and comprehensive lexicon management
 */

export class EnhancedLexiconExplorer {
  constructor(worldEngine, options = {}) {
    this.engine = worldEngine;
    this.options = {
      enableIndexedDB: true,
      enableCache: true,
      maxCacheSize: 1000,
      ...options
    };

    // Core data structures
    this.cache = new Map();
    this.morphologyIndex = new MorphologyIndex();
    this.wordLinks = new Map();
    this.snapshots = [];

    // Initialize subsystems
    this.initializeDataSources();
  }

  initializeDataSources() {
    // Initialize the morphology oracle and upflow index
    this.morphologyOracle = new MorphologyOracle();
    this.upflowIndex = new UpflowIndex();

    // Set up reactive data hooks
    this.queryAPI = new QueryAPI(this.upflowIndex);
    this.helperAPI = new HelperAPI(this);
  }

  // === CORE QUERY METHODS (following attachment API patterns) ===

  /**
   * Search by root morpheme
   * API: up.query.byRoot('stat') → ['state','status','static',...]
   */
  byRoot(root) {
    if (this.cache.has(`root:${root}`)) {
      return this.cache.get(`root:${root}`);
    }

    const results = this.morphologyIndex.findByRoot(root);
    const enrichedResults = results.map(word => this.enrichWordData(word));

    if (this.options.enableCache) {
      this.cache.set(`root:${root}`, enrichedResults);
    }

    return enrichedResults;
  }

  /**
   * Search by prefix
   * API: up.query.byPrefix('re') → ['rebuild', 'restore', 'restate', ...]
   */
  byPrefix(prefix) {
    const cacheKey = `prefix:${prefix}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const results = this.morphologyIndex.findByPrefix(prefix);
    const enrichedResults = results.map(word => ({
      word,
      prefix,
      root: this.extractRoot(word, prefix),
      suffixes: this.extractSuffixes(word, prefix),
      score: this.calculatePrefixScore(word, prefix)
    }));

    this.cache.set(cacheKey, enrichedResults);
    return enrichedResults;
  }

  /**
   * Search by suffix
   * API: up.query.bySuffix('tion') → ['creation', 'formation', 'action', ...]
   */
  bySuffix(suffix) {
    const cacheKey = `suffix:${suffix}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const results = this.morphologyIndex.findBySuffix(suffix);
    const enrichedResults = results.map(word => ({
      word,
      suffix,
      root: this.extractRoot(word, null, suffix),
      prefixes: this.extractPrefixes(word, suffix),
      score: this.calculateSuffixScore(word, suffix)
    }));

    this.cache.set(cacheKey, enrichedResults);
    return enrichedResults;
  }

  /**
   * Search by abbreviation
   * API: up.query.byAbbrev('sta') → words related to 'sta' abbreviation
   */
  byAbbrev(abbrev) {
    const cacheKey = `abbrev:${abbrev}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const results = this.morphologyIndex.findByAbbreviation(abbrev);
    const enrichedResults = results.map(word => ({
      word,
      abbrev,
      fullForm: this.expandAbbreviation(abbrev),
      context: this.getAbbreviationContext(abbrev),
      score: this.calculateAbbrevScore(word, abbrev)
    }));

    this.cache.set(cacheKey, enrichedResults);
    return enrichedResults;
  }

  /**
   * Get detailed word information
   * API: up.query.getWord('state') → { word, root, prefix, suffix, abbrs, ... }
   */
  getWord(word) {
    const cacheKey = `word:${word}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const analysis = this.analyzeMorphology(word);
    const wordDetail = {
      word,
      root: analysis.root,
      prefix: analysis.prefix,
      suffix: analysis.suffix,
      abbrs: this.findAbbreviations(word),
      morphemes: analysis.morphemes,
      etymology: this.getEtymology(word),
      semanticField: this.getSemanticField(word),
      relatedWords: this.findRelatedWords(word),
      usage: this.getUsagePatterns(word),
      score: this.calculateOverallScore(word)
    };

    this.cache.set(cacheKey, wordDetail);
    return wordDetail;
  }

  /**
   * Link words through morphological relationships
   * API: up.helper.linkWord('state') → { rootsLinked, abbrLinked, ... }
   */
  linkWord(word) {
    const analysis = this.getWord(word);
    const links = {
      word,
      root: analysis.root,
      prefix: analysis.prefix,
      suffix: analysis.suffix,
      abbrs: analysis.abbrs,
      links: []
    };

    // Find root-based links (highest score)
    if (analysis.root) {
      const rootLinks = this.byRoot(analysis.root)
        .filter(w => w.word !== word)
        .map(w => ({ word: w.word, kind: 'root', score: 1.0 }));
      links.links.push(...rootLinks);
    }

    // Find prefix-based links (medium score)
    if (analysis.prefix) {
      const prefixLinks = this.byPrefix(analysis.prefix)
        .filter(w => w.word !== word)
        .map(w => ({ word: w.word, kind: 'prefix', score: 0.7 }));
      links.links.push(...prefixLinks);
    }

    // Find suffix-based links (medium score)
    if (analysis.suffix) {
      const suffixLinks = this.bySuffix(analysis.suffix)
        .filter(w => w.word !== word)
        .map(w => ({ word: w.word, kind: 'suffix', score: 0.6 }));
      links.links.push(...suffixLinks);
    }

    // Find abbreviation-based links (lower score)
    analysis.abbrs.forEach(abbr => {
      const abbrLinks = this.byAbbrev(abbr)
        .filter(w => w.word !== word)
        .map(w => ({ word: w.word, kind: 'abbrev', score: 0.4 }));
      links.links.push(...abbrLinks);
    });

    // Sort by score and remove duplicates
    links.links = this.deduplicateAndSort(links.links);

    // Store the link relationship
    this.wordLinks.set(word, links);

    return links;
  }

  // === MORPHOLOGY ANALYSIS ===

  analyzeMorphology(word) {
    // Use existing World Engine morphology system if available
    if (this.engine.morphemeDiscovery) {
      const traditionalAnalysis = this.engine.morphemeDiscovery.analyze(word);

      // Enhance with our own analysis
      return this.enhanceMorphologicalAnalysis(word, traditionalAnalysis);
    }

    // Fallback to our own morphological analysis
    return this.performMorphologicalAnalysis(word);
  }

  enhanceMorphologicalAnalysis(word, traditionalAnalysis) {
    const prefixes = ['anti', 'auto', 'counter', 'de', 'dis', 'inter', 'multi', 'non', 'over', 'pre', 're', 'semi', 'sub', 'super', 'trans', 'ultra', 'un', 'under'];
    const suffixes = ['able', 'acy', 'age', 'al', 'ance', 'ation', 'dom', 'ed', 'en', 'ence', 'er', 'ery', 'est', 'ful', 'fy', 'hood', 'ible', 'ic', 'ing', 'ion', 'ism', 'ist', 'ity', 'ive', 'less', 'ly', 'ment', 'ness', 'or', 'ous', 'ship', 'sion', 'th', 'tion', 'ty', 'ward', 'wise', 'y'];

    let remaining = word.toLowerCase();
    let detectedPrefix = '';
    let detectedSuffix = '';

    // Enhanced prefix detection
    for (const prefix of prefixes.sort((a, b) => b.length - a.length)) {
      if (remaining.startsWith(prefix)) {
        detectedPrefix = prefix;
        remaining = remaining.slice(prefix.length);
        break;
      }
    }

    // Enhanced suffix detection
    for (const suffix of suffixes.sort((a, b) => b.length - a.length)) {
      if (remaining.endsWith(suffix)) {
        detectedSuffix = suffix;
        remaining = remaining.slice(0, -suffix.length);
        break;
      }
    }

    const morphemes = [];
    if (detectedPrefix) morphemes.push({ type: 'prefix', form: detectedPrefix, meaning: this.getPrefixMeaning(detectedPrefix) });
    if (remaining) morphemes.push({ type: 'root', form: remaining, meaning: this.getRootMeaning(remaining) });
    if (detectedSuffix) morphemes.push({ type: 'suffix', form: detectedSuffix, meaning: this.getSuffixMeaning(detectedSuffix) });

    return {
      word,
      root: remaining,
      prefix: detectedPrefix,
      suffix: detectedSuffix,
      morphemes,
      structure: morphemes.map(m => m.type).join('+'),
      complexity: morphemes.length,
      traditionalAnalysis
    };
  }

  // === UTILITY METHODS ===

  enrichWordData(word) {
    return {
      word,
      analysis: this.analyzeMorphology(word),
      score: this.calculateOverallScore(word),
      tags: this.generateTags(word)
    };
  }

  calculateOverallScore(word) {
    let score = 0;

    // Base score from word length and complexity
    score += Math.min(word.length / 10, 1) * 0.3;

    // Boost for morphological richness
    const analysis = this.analyzeMorphology(word);
    score += analysis.complexity * 0.2;

    // Boost for semantic field strength
    score += this.getSemanticFieldStrength(word) * 0.3;

    // Boost for usage frequency (simulated)
    score += Math.random() * 0.2;

    return Math.min(score, 1.0);
  }

  deduplicateAndSort(links) {
    const seen = new Set();
    const unique = links.filter(link => {
      const key = `${link.word}:${link.kind}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    return unique.sort((a, b) => b.score - a.score);
  }

  // === DATA MANAGEMENT ===

  /**
   * Create snapshot of current lexicon state
   * API: up.librarian.snapshot()
   */
  snapshot() {
    const snapshot = {
      timestamp: Date.now(),
      version: '3.1.0',
      morphologyIndex: this.morphologyIndex.export(),
      wordLinks: Array.from(this.wordLinks.entries()),
      cache: Array.from(this.cache.entries()),
      stats: this.getStats()
    };

    this.snapshots.push(snapshot);
    return snapshot;
  }

  /**
   * Compact and optimize data structures
   * API: up.librarian.compact()
   */
  compact() {
    // Clean up cache
    if (this.cache.size > this.options.maxCacheSize) {
      const entries = Array.from(this.cache.entries());
      this.cache.clear();

      // Keep most recently used entries
      entries.slice(-this.options.maxCacheSize).forEach(([key, value]) => {
        this.cache.set(key, value);
      });
    }

    // Optimize morphology index
    this.morphologyIndex.compact();

    // Remove old snapshots (keep last 10)
    if (this.snapshots.length > 10) {
      this.snapshots = this.snapshots.slice(-10);
    }

    return {
      cacheSize: this.cache.size,
      indexSize: this.morphologyIndex.size(),
      snapshotCount: this.snapshots.length
    };
  }

  /**
   * Verify data integrity
   * API: up.librarian.verify()
   */
  verify() {
    const issues = [];

    // Check cache consistency
    for (const [key, value] of this.cache.entries()) {
      if (!value || typeof value !== 'object') {
        issues.push(`Invalid cache entry: ${key}`);
      }
    }

    // Check word links consistency
    for (const [word, links] of this.wordLinks.entries()) {
      if (!links.links || !Array.isArray(links.links)) {
        issues.push(`Invalid word links for: ${word}`);
      }
    }

    // Check morphology index
    const indexIssues = this.morphologyIndex.verify();
    issues.push(...indexIssues);

    return {
      valid: issues.length === 0,
      issues,
      timestamp: Date.now()
    };
  }

  // === EXPORT/IMPORT ===

  exportJSONL() {
    const data = {
      metadata: {
        version: '3.1.0',
        timestamp: Date.now(),
        totalWords: this.morphologyIndex.size()
      },
      morphologyIndex: this.morphologyIndex.export(),
      wordLinks: Array.from(this.wordLinks.entries()),
      snapshots: this.snapshots.slice(-5) // Export last 5 snapshots
    };

    return JSON.stringify(data, null, 2);
  }

  importJSONL(jsonlData) {
    try {
      const data = JSON.parse(jsonlData);

      // Import morphology index
      this.morphologyIndex.import(data.morphologyIndex);

      // Import word links
      this.wordLinks.clear();
      data.wordLinks.forEach(([word, links]) => {
        this.wordLinks.set(word, links);
      });

      // Import snapshots
      if (data.snapshots) {
        this.snapshots.push(...data.snapshots);
      }

      // Clear cache to force refresh
      this.cache.clear();

      return {
        success: true,
        wordsImported: data.metadata?.totalWords || 0,
        timestamp: Date.now()
      };

    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  // === STATISTICS AND MONITORING ===

  getStats() {
    return {
      morphologyIndex: {
        totalWords: this.morphologyIndex.size(),
        uniqueRoots: this.morphologyIndex.getRootCount(),
        uniquePrefixes: this.morphologyIndex.getPrefixCount(),
        uniqueSuffixes: this.morphologyIndex.getSuffixCount()
      },
      cache: {
        size: this.cache.size,
        hitRate: this.calculateCacheHitRate(),
        maxSize: this.options.maxCacheSize
      },
      links: {
        totalLinks: this.wordLinks.size,
        averageLinksPerWord: this.calculateAverageLinks()
      },
      snapshots: {
        count: this.snapshots.length,
        latestTimestamp: this.snapshots[this.snapshots.length - 1]?.timestamp
      }
    };
  }

  // === HELPER METHODS ===

  getPrefixMeaning(prefix) {
    const meanings = {
      're': 'again, back',
      'pre': 'before',
      'un': 'not, reverse',
      'anti': 'against',
      'trans': 'across, through',
      'inter': 'between',
      'multi': 'many',
      'sub': 'under',
      'super': 'above, over'
    };
    return meanings[prefix] || 'unknown';
  }

  getSuffixMeaning(suffix) {
    const meanings = {
      'tion': 'action, process',
      'ing': 'ongoing action',
      'ed': 'past action',
      'er': 'one who does',
      'ly': 'in a manner',
      'ness': 'state of being',
      'able': 'capable of'
    };
    return meanings[suffix] || 'unknown';
  }

  getRootMeaning(root) {
    // Placeholder for semantic analysis
    return 'core meaning';
  }

  findAbbreviations(word) {
    // Generate plausible abbreviations
    const abbrevs = [];
    if (word.length >= 3) {
      abbrevs.push(word.slice(0, 3));
    }
    if (word.length >= 4) {
      abbrevs.push(word.slice(0, 2) + word.slice(-1));
    }
    return abbrevs;
  }

  getEtymology(word) {
    return { origin: 'unknown', language: 'english', notes: '' };
  }

  getSemanticField(word) {
    // Placeholder for semantic field classification
    if (word.includes('form')) return 'transformation';
    if (word.includes('struct')) return 'structure';
    if (word.includes('move')) return 'movement';
    return 'general';
  }

  getSemanticFieldStrength(word) {
    return Math.random(); // Placeholder
  }

  findRelatedWords(word) {
    // Use existing link system
    const links = this.linkWord(word);
    return links.links.slice(0, 5).map(l => l.word);
  }

  getUsagePatterns(word) {
    return {
      frequency: Math.random(),
      contexts: ['general', 'technical'],
      collocations: []
    };
  }

  calculateCacheHitRate() {
    // Placeholder for cache statistics
    return 0.85;
  }

  calculateAverageLinks() {
    if (this.wordLinks.size === 0) return 0;
    const totalLinks = Array.from(this.wordLinks.values())
      .reduce((sum, links) => sum + links.links.length, 0);
    return totalLinks / this.wordLinks.size;
  }

  generateTags(word) {
    const tags = [];
    const analysis = this.analyzeMorphology(word);

    if (analysis.prefix) tags.push(`prefix:${analysis.prefix}`);
    if (analysis.suffix) tags.push(`suffix:${analysis.suffix}`);
    if (analysis.complexity > 2) tags.push('complex');

    return tags;
  }

  // Placeholder extraction methods
  extractRoot(word, prefix = null, suffix = null) {
    let remaining = word.toLowerCase();
    if (prefix) remaining = remaining.slice(prefix.length);
    if (suffix) remaining = remaining.slice(0, -suffix.length);
    return remaining;
  }

  extractSuffixes(word, prefix) {
    const remaining = word.slice(prefix.length);
    return this.analyzeMorphology(remaining).suffix ? [this.analyzeMorphology(remaining).suffix] : [];
  }

  extractPrefixes(word, suffix) {
    const remaining = word.slice(0, -suffix.length);
    return this.analyzeMorphology(remaining).prefix ? [this.analyzeMorphology(remaining).prefix] : [];
  }

  calculatePrefixScore(word, prefix) {
    return prefix.length / word.length;
  }

  calculateSuffixScore(word, suffix) {
    return suffix.length / word.length;
  }

  calculateAbbrevScore(word, abbrev) {
    return abbrev.length / word.length * 0.5;
  }

  expandAbbreviation(abbrev) {
    // Placeholder for abbreviation expansion
    return `expanded_${abbrev}`;
  }

  getAbbreviationContext(abbrev) {
    return { domain: 'general', usage: 'common' };
  }

  performMorphologicalAnalysis(word) {
    // Simplified morphological analysis when World Engine system not available
    return {
      word,
      root: word,
      prefix: '',
      suffix: '',
      morphemes: [{ type: 'root', form: word, meaning: 'unknown' }],
      structure: 'root',
      complexity: 1
    };
  }
}

// === SUPPORTING CLASSES ===

class MorphologyIndex {
  constructor() {
    this.roots = new Map();
    this.prefixes = new Map();
    this.suffixes = new Map();
    this.abbreviations = new Map();
    this.words = new Set();
  }

  findByRoot(root) {
    return this.roots.get(root) || [];
  }

  findByPrefix(prefix) {
    return this.prefixes.get(prefix) || [];
  }

  findBySuffix(suffix) {
    return this.suffixes.get(suffix) || [];
  }

  findByAbbreviation(abbrev) {
    return this.abbreviations.get(abbrev) || [];
  }

  size() {
    return this.words.size;
  }

  getRootCount() {
    return this.roots.size;
  }

  getPrefixCount() {
    return this.prefixes.size;
  }

  getSuffixCount() {
    return this.suffixes.size;
  }

  export() {
    return {
      roots: Array.from(this.roots.entries()),
      prefixes: Array.from(this.prefixes.entries()),
      suffixes: Array.from(this.suffixes.entries()),
      abbreviations: Array.from(this.abbreviations.entries()),
      words: Array.from(this.words)
    };
  }

  import(data) {
    this.roots = new Map(data.roots || []);
    this.prefixes = new Map(data.prefixes || []);
    this.suffixes = new Map(data.suffixes || []);
    this.abbreviations = new Map(data.abbreviations || []);
    this.words = new Set(data.words || []);
  }

  compact() {
    // Remove empty entries and optimize storage
    for (const [key, value] of this.roots.entries()) {
      if (!value || value.length === 0) {
        this.roots.delete(key);
      }
    }
    // Similar for other maps...
  }

  verify() {
    const issues = [];
    // Add verification logic
    return issues;
  }
}

class MorphologyOracle {
  // Placeholder for morphology oracle functionality
}

class UpflowIndex {
  // Placeholder for upflow index functionality
}

class QueryAPI {
  constructor(upflowIndex) {
    this.upflowIndex = upflowIndex;
  }
}

class HelperAPI {
  constructor(explorer) {
    this.explorer = explorer;
  }
}

// === DATA SERVICE WRAPPER (from attachment) ===

export function makeDataService(explorer) {
  return {
    search(term) {
      return {
        roots: explorer.byRoot(term),
        prefixes: explorer.byPrefix(term),
        suffixes: explorer.bySuffix(term),
        abbrevs: explorer.byAbbrev(term),
        word: explorer.getWord(term)
      };
    },
    link(word) {
      return explorer.linkWord(word);
    },
    ingest() {
      return explorer.ingestFromLastRun();
    },
    snapshot() {
      return explorer.snapshot();
    },
    compact() {
      return explorer.compact();
    },
    verify() {
      return explorer.verify();
    }
  };
}

// Export for integration with World Engine V3.1
export function createEnhancedLexiconExplorer(worldEngine, options = {}) {
  return new EnhancedLexiconExplorer(worldEngine, options);
}
