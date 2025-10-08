/**
 * Morpheme Discovery System for World Engine V3.1
 * Learns new morphemes from user interactions with frequency tracking
 * Automatically promotes high-frequency patterns to permanent vocabulary
 */

export class MorphemeDiscovery {
  constructor(options = {}) {
    this.patterns = new Map(); // Pattern -> frequency count
    this.sequences = []; // Raw interaction sequences
    this.morphemes = new Map(); // Discovered morphemes -> metadata
    this.config = {
      minFrequency: options.minFrequency || 3,
      maxSequenceLength: options.maxSequenceLength || 100,
      promotionThreshold: options.promotionThreshold || 5,
      decayFactor: options.decayFactor || 0.95,
      ...options
    };
    this.listeners = new Map();
  }

  /**
   * Record a user interaction sequence
   * @param {string[]} sequence - Array of button abbreviations
   * @param {Object} context - Interaction context
   */
  recordInteraction(sequence, context = {}) {
    if (!Array.isArray(sequence) || sequence.length === 0) return;

    const interaction = {
      sequence: [...sequence],
      timestamp: new Date().toISOString(),
      context: { ...context },
      sessionId: context.sessionId || 'default'
    };

    this.sequences.push(interaction);

    // Extract patterns of various lengths
    this.extractPatterns(sequence);

    // Check for promotions
    this.checkForPromotions();

    // Emit discovery events
    this.emit('interactionRecorded', interaction);

    // Maintain sequence history size
    if (this.sequences.length > this.config.maxSequenceLength) {
      this.sequences.shift();
    }
  }

  /**
   * Extract patterns from a sequence
   * @param {string[]} sequence - Button sequence
   */
  extractPatterns(sequence) {
    // Extract n-grams of length 2 to 4
    for (let n = 2; n <= Math.min(4, sequence.length); n++) {
      for (let i = 0; i <= sequence.length - n; i++) {
        const pattern = sequence.slice(i, i + n);
        const patternKey = pattern.join('→');

        // Update frequency
        const currentCount = this.patterns.get(patternKey) || 0;
        this.patterns.set(patternKey, currentCount + 1);

        // Check if this is a new significant pattern
        if (currentCount + 1 === this.config.minFrequency) {
          this.emit('patternDiscovered', {
            pattern: pattern,
            frequency: currentCount + 1,
            key: patternKey
          });
        }
      }
    }
  }

  /**
   * Check if any patterns should be promoted to morphemes
   */
  checkForPromotions() {
    for (const [patternKey, frequency] of this.patterns.entries()) {
      if (frequency >= this.config.promotionThreshold && !this.morphemes.has(patternKey)) {
        const pattern = patternKey.split('→');
        this.promoteMorpheme(pattern, frequency);
      }
    }
  }

  /**
   * Promote a pattern to a permanent morpheme
   * @param {string[]} pattern - Button pattern to promote
   * @param {number} frequency - Current frequency count
   */
  promoteMorpheme(pattern, frequency) {
    const morphemeId = this.generateMorphemeId(pattern);

    const morpheme = {
      id: morphemeId,
      pattern: [...pattern],
      frequency: frequency,
      discoveredAt: new Date().toISOString(),
      usageCount: frequency,
      effectiveness: this.calculateEffectiveness(pattern),
      contexts: this.getPatternContexts(pattern),
      suggested_label: this.suggestLabel(pattern),
      strength: this.calculateMorphemeStrength(frequency)
    };

    this.morphemes.set(morphemeId, morpheme);

    this.emit('morphemePromoted', morpheme);

    console.log(`🔬 Morpheme discovered: "${morpheme.suggested_label}" (${pattern.join('→')}) - frequency: ${frequency}`);
  }

  /**
   * Generate a unique ID for a morpheme
   * @param {string[]} pattern - Button pattern
   * @returns {string} Unique morpheme ID
   */
  generateMorphemeId(pattern) {
    const base = pattern.join('_').toLowerCase();
    const timestamp = Date.now().toString(36);
    return `morph_${base}_${timestamp}`;
  }

  /**
   * Calculate effectiveness of a pattern based on state transitions
   * @param {string[]} pattern - Button pattern
   * @returns {number} Effectiveness score (0-1)
   */
  calculateEffectiveness(pattern) {
    // Find interactions that used this pattern
    const patternStr = pattern.join('→');
    const relevantInteractions = this.sequences.filter(interaction => {
      const seqStr = interaction.sequence.join('→');
      return seqStr.includes(patternStr);
    });

    if (relevantInteractions.length === 0) return 0.5;

    // Estimate effectiveness based on context improvements
    // This is a simplified heuristic - could be enhanced with actual state analysis
    const avgContextScore = relevantInteractions.reduce((sum, interaction) => {
      const contextValue = interaction.context.improvement || 0.5;
      return sum + contextValue;
    }, 0) / relevantInteractions.length;

    return Math.max(0, Math.min(1, avgContextScore));
  }

  /**
   * Get contexts where a pattern was used
   * @param {string[]} pattern - Button pattern
   * @returns {Object[]} Array of usage contexts
   */
  getPatternContexts(pattern) {
    const patternStr = pattern.join('→');
    return this.sequences
      .filter(interaction => interaction.sequence.join('→').includes(patternStr))
      .map(interaction => ({
        timestamp: interaction.timestamp,
        sessionId: interaction.sessionId,
        context: interaction.context
      }))
      .slice(-5); // Keep recent 5 contexts
  }

  /**
   * Suggest a human-readable label for a pattern
   * @param {string[]} pattern - Button pattern
   * @returns {string} Suggested label
   */
  suggestLabel(pattern) {
    // Simple label generation based on pattern characteristics
    const verbs = ['enhance', 'adjust', 'refine', 'optimize', 'transform', 'balance'];
    const objects = ['flow', 'structure', 'harmony', 'focus', 'energy', 'clarity'];

    const verb = verbs[pattern.length % verbs.length];
    const object = objects[pattern[0].charCodeAt(0) % objects.length];

    return `${verb}_${object}`;
  }

  /**
   * Calculate morpheme strength based on frequency and effectiveness
   * @param {number} frequency - Usage frequency
   * @returns {number} Strength score (0-1)
   */
  calculateMorphemeStrength(frequency) {
    // Logarithmic scaling for strength
    return Math.min(1, Math.log10(frequency + 1) / Math.log10(11));
  }

  /**
   * Get discovered morphemes matching criteria
   * @param {Object} criteria - Filter criteria
   * @returns {Object[]} Matching morphemes
   */
  getMorphemes(criteria = {}) {
    const morphemes = Array.from(this.morphemes.values());

    return morphemes.filter(morpheme => {
      if (criteria.minFrequency && morpheme.frequency < criteria.minFrequency) return false;
      if (criteria.minStrength && morpheme.strength < criteria.minStrength) return false;
      if (criteria.pattern && !morpheme.pattern.includes(criteria.pattern)) return false;
      if (criteria.since && new Date(morpheme.discoveredAt) < new Date(criteria.since)) return false;
      return true;
    }).sort((a, b) => b.frequency - a.frequency);
  }

  /**
   * Get the most effective morphemes
   * @param {number} count - Number of morphemes to return
   * @returns {Object[]} Top morphemes by effectiveness
   */
  getTopMorphemes(count = 10) {
    return Array.from(this.morphemes.values())
      .sort((a, b) => (b.effectiveness * b.strength) - (a.effectiveness * a.strength))
      .slice(0, count);
  }

  /**
   * Apply decay to pattern frequencies to prioritize recent patterns
   */
  applyDecay() {
    for (const [key, frequency] of this.patterns.entries()) {
      const decayedFreq = frequency * this.config.decayFactor;

      if (decayedFreq < 1) {
        this.patterns.delete(key);
      } else {
        this.patterns.set(key, decayedFreq);
      }
    }

    this.emit('decayApplied', { remainingPatterns: this.patterns.size });
  }

  /**
   * Export discovered morphemes to a format suitable for button creation
   * @returns {Object[]} Morpheme export data
   */
  exportMorphemes() {
    return Array.from(this.morphemes.values()).map(morpheme => ({
      abbr: morpheme.id.split('_')[1] || morpheme.id,
      label: morpheme.suggested_label,
      sequence: morpheme.pattern,
      frequency: morpheme.frequency,
      effectiveness: morpheme.effectiveness,
      discovered_at: morpheme.discoveredAt,
      auto_generated: true,
      morpheme_data: {
        strength: morpheme.strength,
        contexts: morpheme.contexts,
        usage_count: morpheme.usageCount
      }
    }));
  }

  /**
   * Import morphemes from external source
   * @param {Object[]} morphemeData - External morpheme data
   */
  importMorphemes(morphemeData) {
    for (const data of morphemeData) {
      if (data.sequence && Array.isArray(data.sequence)) {
        const morpheme = {
          id: data.id || this.generateMorphemeId(data.sequence),
          pattern: data.sequence,
          frequency: data.frequency || 1,
          discoveredAt: data.discovered_at || new Date().toISOString(),
          usageCount: data.usage_count || data.frequency || 1,
          effectiveness: data.effectiveness || 0.5,
          contexts: data.contexts || [],
          suggested_label: data.label || this.suggestLabel(data.sequence),
          strength: data.strength || this.calculateMorphemeStrength(data.frequency || 1),
          imported: true
        };

        this.morphemes.set(morpheme.id, morpheme);
      }
    }

    this.emit('morphemesImported', { count: morphemeData.length });
  }

  /**
   * Analyze learning progress and provide insights
   * @returns {Object} Learning analytics
   */
  getAnalytics() {
    const morphemeArray = Array.from(this.morphemes.values());
    const patternArray = Array.from(this.patterns.entries());

    return {
      discovery_stats: {
        total_morphemes: morphemeArray.length,
        total_patterns: patternArray.length,
        total_interactions: this.sequences.length,
        avg_pattern_frequency: patternArray.length > 0
          ? patternArray.reduce((sum, [, freq]) => sum + freq, 0) / patternArray.length
          : 0
      },
      morpheme_quality: {
        avg_effectiveness: morphemeArray.length > 0
          ? morphemeArray.reduce((sum, m) => sum + m.effectiveness, 0) / morphemeArray.length
          : 0,
        avg_strength: morphemeArray.length > 0
          ? morphemeArray.reduce((sum, m) => sum + m.strength, 0) / morphemeArray.length
          : 0,
        high_quality_count: morphemeArray.filter(m => m.effectiveness > 0.7 && m.strength > 0.5).length
      },
      learning_velocity: {
        recent_discoveries: morphemeArray.filter(m => {
          const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
          return new Date(m.discoveredAt) > dayAgo;
        }).length,
        discovery_rate: this.sequences.length > 0 ? morphemeArray.length / this.sequences.length : 0
      },
      top_patterns: patternArray
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([pattern, freq]) => ({ pattern, frequency: freq }))
    };
  }

  /**
   * Clear all discovery data
   */
  clear() {
    this.patterns.clear();
    this.sequences = [];
    this.morphemes.clear();
    this.emit('dataCleared');
  }

  /**
   * Add event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function to remove
   */
  off(event, callback) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit event to listeners
   * @param {string} event - Event name
   * @param {*} data - Event data
   */
  emit(event, data) {
    const callbacks = this.listeners.get(event) || [];
    callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Morpheme discovery event error (${event}):`, error);
      }
    });
  }
}

/**
 * Enhanced morpheme discovery with semantic clustering
 */
export class SemanticMorphemeDiscovery extends MorphemeDiscovery {
  constructor(options = {}) {
    super(options);
    this.semanticClusters = new Map();
    this.config = {
      ...this.config,
      clusterThreshold: options.clusterThreshold || 0.7,
      semanticWindowSize: options.semanticWindowSize || 3
    };
  }

  /**
   * Cluster morphemes by semantic similarity
   */
  clusterMorphemes() {
    const morphemes = Array.from(this.morphemes.values());
    const clusters = new Map();

    for (const morpheme of morphemes) {
      const clusterId = this.findSemanticCluster(morpheme, clusters);

      if (clusterId) {
        clusters.get(clusterId).morphemes.push(morpheme);
      } else {
        // Create new cluster
        const newClusterId = `cluster_${clusters.size + 1}`;
        clusters.set(newClusterId, {
          id: newClusterId,
          morphemes: [morpheme],
          centroid: this.calculateCentroid([morpheme]),
          label: this.generateClusterLabel([morpheme])
        });
      }
    }

    this.semanticClusters = clusters;
    this.emit('clustersUpdated', { clusters: clusters.size });

    return clusters;
  }

  /**
   * Find the best semantic cluster for a morpheme
   * @param {Object} morpheme - Morpheme to cluster
   * @param {Map} clusters - Existing clusters
   * @returns {string|null} Cluster ID or null
   */
  findSemanticCluster(morpheme, clusters) {
    let bestCluster = null;
    let bestSimilarity = 0;

    for (const [clusterId, cluster] of clusters.entries()) {
      const similarity = this.calculateSemanticSimilarity(morpheme, cluster.centroid);

      if (similarity > this.config.clusterThreshold && similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestCluster = clusterId;
      }
    }

    return bestCluster;
  }

  /**
   * Calculate semantic similarity between morpheme and cluster centroid
   * @param {Object} morpheme - Morpheme object
   * @param {Object} centroid - Cluster centroid
   * @returns {number} Similarity score (0-1)
   */
  calculateSemanticSimilarity(morpheme, centroid) {
    // Simple similarity based on pattern overlap and effectiveness
    const patternSimilarity = this.calculatePatternSimilarity(morpheme.pattern, centroid.pattern || []);
    const effectivenessSimilarity = 1 - Math.abs(morpheme.effectiveness - (centroid.effectiveness || 0.5));

    return (patternSimilarity + effectivenessSimilarity) / 2;
  }

  /**
   * Calculate pattern similarity between two sequences
   * @param {string[]} pattern1 - First pattern
   * @param {string[]} pattern2 - Second pattern
   * @returns {number} Similarity score (0-1)
   */
  calculatePatternSimilarity(pattern1, pattern2) {
    if (pattern1.length === 0 || pattern2.length === 0) return 0;

    // Use Jaccard similarity on pattern elements
    const set1 = new Set(pattern1);
    const set2 = new Set(pattern2);
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);

    return intersection.size / union.size;
  }

  /**
   * Calculate centroid for a cluster of morphemes
   * @param {Object[]} morphemes - Morphemes in cluster
   * @returns {Object} Cluster centroid
   */
  calculateCentroid(morphemes) {
    if (morphemes.length === 0) return {};

    const avgEffectiveness = morphemes.reduce((sum, m) => sum + m.effectiveness, 0) / morphemes.length;
    const avgStrength = morphemes.reduce((sum, m) => sum + m.strength, 0) / morphemes.length;

    // Find most common pattern elements
    const allElements = morphemes.flatMap(m => m.pattern);
    const elementCounts = allElements.reduce((counts, elem) => {
      counts[elem] = (counts[elem] || 0) + 1;
      return counts;
    }, {});

    const commonPattern = Object.entries(elementCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([elem]) => elem);

    return {
      pattern: commonPattern,
      effectiveness: avgEffectiveness,
      strength: avgStrength,
      morpheme_count: morphemes.length
    };
  }

  /**
   * Generate label for semantic cluster
   * @param {Object[]} morphemes - Morphemes in cluster
   * @returns {string} Cluster label
   */
  generateClusterLabel(morphemes) {
    if (morphemes.length === 0) return 'empty_cluster';

    // Use the most effective morpheme's label as base
    const topMorpheme = morphemes.reduce((best, current) =>
      current.effectiveness > best.effectiveness ? current : best
    );

    return `${topMorpheme.suggested_label}_family`;
  }

  /**
   * Get cluster analytics
   * @returns {Object} Cluster analysis
   */
  getClusterAnalytics() {
    const clusters = Array.from(this.semanticClusters.values());

    if (clusters.length === 0) {
      return { no_clusters: true };
    }

    return {
      total_clusters: clusters.length,
      avg_cluster_size: clusters.reduce((sum, c) => sum + c.morphemes.length, 0) / clusters.length,
      largest_cluster: Math.max(...clusters.map(c => c.morphemes.length)),
      cluster_distribution: clusters.map(cluster => ({
        id: cluster.id,
        label: cluster.label,
        size: cluster.morphemes.length,
        avg_effectiveness: cluster.centroid.effectiveness,
        top_patterns: cluster.morphemes
          .sort((a, b) => b.frequency - a.frequency)
          .slice(0, 3)
          .map(m => m.pattern.join('→'))
      }))
    };
  }
}
