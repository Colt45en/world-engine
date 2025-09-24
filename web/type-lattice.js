/**
 * Type Lattice System for World Engine V3.1
 * Implements hierarchical type relationships: State ⊑ Property ⊑ Structure ⊑ Concept
 * Provides composition validation and type safety for mathematical operations
 */

export class TypeLattice {
  constructor(hierarchy = ['State', 'Property', 'Structure', 'Concept']) {
    // Map type names to their hierarchical levels
    this.level = new Map(hierarchy.map((t, i) => [t, i]));
    
    // Map each type to its parent (immediate supertype)
    this.parents = new Map();
    for (let i = 1; i < hierarchy.length; i++) {
      this.parents.set(hierarchy[i], hierarchy[i - 1]);
    }
    
    // Store the original hierarchy for introspection
    this.hierarchy = [...hierarchy];
  }

  /**
   * Check if type 'a' is a subtype of or equal to type 'b' (a ⊑ b)
   * @param {string} a - The subtype
   * @param {string} b - The supertype
   * @returns {boolean} True if a ⊑ b
   */
  leq(a, b) {
    if (!this.level.has(a) || !this.level.has(b)) return false;
    return this.level.get(a) <= this.level.get(b);
  }

  /**
   * Compute the join (least upper bound) of two types: a ⊔ b
   * Returns the most specific common supertype
   * @param {string} a - First type
   * @param {string} b - Second type
   * @returns {string|null} The join type or null if incompatible
   */
  join(a, b) {
    if (!this.level.has(a) || !this.level.has(b)) return null;
    return this.level.get(a) > this.level.get(b) ? a : b;
  }

  /**
   * Compute the meet (greatest lower bound) of two types: a ⊓ b
   * Returns the most general common subtype
   * @param {string} a - First type
   * @param {string} b - Second type
   * @returns {string|null} The meet type or null if incompatible
   */
  meet(a, b) {
    if (!this.level.has(a) || !this.level.has(b)) return null;
    return this.level.get(a) < this.level.get(b) ? a : b;
  }

  /**
   * Check if two operations can be composed: A produces outA, B accepts inB
   * Composition is valid if outA ⊑ inB (output type can be used as input type)
   * @param {string} outA - Output type of first operation
   * @param {string} inB - Input type of second operation
   * @returns {boolean} True if composition is valid
   */
  checkCompose(outA, inB) {
    return outA === inB || this.leq(outA, inB);
  }

  /**
   * Get the parent (immediate supertype) of a given type
   * @param {string} type - The type to find parent of
   * @returns {string|null} Parent type or null if at top level
   */
  getParent(type) {
    return this.parents.get(type) || null;
  }

  /**
   * Get all supertypes of a given type (transitive closure of parent relation)
   * @param {string} type - The type to find supertypes of
   * @returns {string[]} Array of all supertypes, ordered from immediate parent to root
   */
  getSupertypes(type) {
    const supertypes = [];
    let current = this.getParent(type);
    while (current) {
      supertypes.push(current);
      current = this.getParent(current);
    }
    return supertypes;
  }

  /**
   * Get all subtypes of a given type
   * @param {string} type - The type to find subtypes of
   * @returns {string[]} Array of all subtypes
   */
  getSubtypes(type) {
    if (!this.level.has(type)) return [];
    const level = this.level.get(type);
    return this.hierarchy.filter(t => this.level.get(t) < level);
  }

  /**
   * Check if the lattice contains a specific type
   * @param {string} type - The type to check
   * @returns {boolean} True if type exists in lattice
   */
  hasType(type) {
    return this.level.has(type);
  }

  /**
   * Get the level (depth in hierarchy) of a type
   * @param {string} type - The type to get level of
   * @returns {number|null} Numeric level or null if type not found
   */
  getLevel(type) {
    return this.level.get(type) ?? null;
  }

  /**
   * Add a new type to the lattice at a specific position
   * @param {string} newType - Name of the new type
   * @param {string|null} parentType - Parent type, or null for root
   * @returns {boolean} True if successfully added
   */
  addType(newType, parentType = null) {
    if (this.hasType(newType)) return false;
    
    if (parentType === null) {
      // Adding new root type
      const newLevel = Math.max(...this.level.values()) + 1;
      this.level.set(newType, newLevel);
      this.hierarchy.push(newType);
    } else if (this.hasType(parentType)) {
      // Adding as child of existing type
      const parentLevel = this.level.get(parentType);
      this.level.set(newType, parentLevel - 0.5); // Insert between parent and its current children
      this.parents.set(newType, parentType);
      
      // Update hierarchy array
      const insertIndex = this.hierarchy.indexOf(parentType);
      this.hierarchy.splice(insertIndex, 0, newType);
    } else {
      return false; // Parent type doesn't exist
    }
    
    return true;
  }

  /**
   * Create a visual representation of the type lattice
   * @returns {string} ASCII art representation
   */
  toString() {
    const lines = [];
    lines.push('Type Lattice:');
    
    for (let i = 0; i < this.hierarchy.length; i++) {
      const type = this.hierarchy[i];
      const indent = '  '.repeat(i);
      const level = this.getLevel(type);
      lines.push(`${indent}${type} (level ${level})`);
    }
    
    return lines.join('\n');
  }

  /**
   * Export lattice configuration for serialization
   * @returns {Object} Serializable lattice configuration
   */
  toJSON() {
    return {
      hierarchy: this.hierarchy,
      levels: Object.fromEntries(this.level),
      parents: Object.fromEntries(this.parents)
    };
  }

  /**
   * Create lattice from serialized configuration
   * @param {Object} config - Serialized lattice configuration
   * @returns {TypeLattice} Restored type lattice
   */
  static fromJSON(config) {
    const lattice = new TypeLattice(config.hierarchy);
    // Levels and parents are automatically computed from hierarchy
    return lattice;
  }
}

/**
 * Factory for common type lattice configurations
 */
export class TypeLatticeFactory {
  /**
   * Create standard linguistic type lattice
   * @returns {TypeLattice} Standard lattice with State ⊑ Property ⊑ Structure ⊑ Concept
   */
  static createStandard() {
    return new TypeLattice(['State', 'Property', 'Structure', 'Concept']);
  }

  /**
   * Create extended linguistic lattice with more fine-grained types
   * @returns {TypeLattice} Extended lattice with additional intermediate types
   */
  static createExtended() {
    return new TypeLattice([
      'State',        // Momentary states, conditions
      'Process',      // Ongoing actions, changes
      'Property',     // Attributes, qualities
      'Relation',     // Relationships, connections
      'Structure',    // Complex entities, systems
      'Category',     // Abstract classifications
      'Concept'       // Pure abstract concepts
    ]);
  }

  /**
   * Create minimal type lattice for testing
   * @returns {TypeLattice} Simple two-level lattice
   */
  static createMinimal() {
    return new TypeLattice(['Concrete', 'Abstract']);
  }

  /**
   * Create mathematical type lattice for numerical operations
   * @returns {TypeLattice} Math-focused lattice
   */
  static createMathematical() {
    return new TypeLattice([
      'Scalar',       // Individual numbers
      'Vector',       // Arrays of numbers  
      'Matrix',       // 2D arrays
      'Tensor'        // N-dimensional arrays
    ]);
  }
}