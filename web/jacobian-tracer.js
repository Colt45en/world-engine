/**
 * Jacobian Tracing System for World Engine V3.1  
 * Implements mathematical explanation for "why did x move there?"
 * Computes local Jacobian matrices and tracks transformation effects
 */

export class JacobianTracer {
  /**
   * Compute the Jacobian matrix for a button transformation
   * For x' = α M x + b, the Jacobian ∂x'/∂x = α M
   * @param {Object} button - Button with M matrix and alpha scalar
   * @returns {number[][]} Jacobian matrix
   */
  static jacobian(button) {
    const M = button.M || [[1,0,0],[0,1,0],[0,0,1]]; // Default to identity
    const alpha = button.alpha ?? 1.0;
    
    // J = α M (element-wise multiplication of scalar with matrix)
    return M.map(row => row.map(value => alpha * value));
  }

  /**
   * Compute the effect vector (difference between states)
   * @param {number[]} prevX - Previous state vector
   * @param {number[]} nextX - Next state vector  
   * @returns {number[]} Difference vector Δx = x' - x
   */
  static effect(prevX, nextX) {
    return nextX.map((value, i) => value - prevX[i]);
  }

  /**
   * Compute the magnitude of the effect
   * @param {number[]} deltaX - Effect vector
   * @returns {number} Euclidean norm of effect
   */
  static effectMagnitude(deltaX) {
    return Math.sqrt(deltaX.reduce((sum, dx) => sum + dx * dx, 0));
  }

  /**
   * Analyze directional impact of transformation
   * @param {number[][]} jacobian - Jacobian matrix
   * @returns {Object} Directional analysis
   */
  static directionalAnalysis(jacobian) {
    const dimensions = jacobian.length;
    const analysis = {
      expansive: [],    // Dimensions that expand
      contractive: [],  // Dimensions that contract
      rotational: [],   // Dimensions with cross-coupling
      preserving: []    // Dimensions unchanged
    };

    for (let i = 0; i < dimensions; i++) {
      const diagonalElement = jacobian[i][i];
      const offDiagonalSum = jacobian[i].reduce((sum, val, j) => 
        i !== j ? sum + Math.abs(val) : sum, 0);

      if (Math.abs(diagonalElement - 1) < 1e-6 && offDiagonalSum < 1e-6) {
        analysis.preserving.push(i);
      } else if (diagonalElement > 1) {
        analysis.expansive.push(i);
      } else if (diagonalElement < 1 && diagonalElement > 0) {
        analysis.contractive.push(i);
      }

      if (offDiagonalSum > 1e-6) {
        analysis.rotational.push(i);
      }
    }

    return analysis;
  }

  /**
   * Create a comprehensive trace log for a transformation
   * @param {Object} params - Transformation parameters
   * @param {Object} params.button - Button that was applied
   * @param {Object} params.before - State before transformation
   * @param {Object} params.after - State after transformation
   * @param {Object} params.metadata - Additional metadata
   * @returns {Object} Comprehensive trace log
   */
  static log({ button, before, after, metadata = {} }) {
    const jacobian = JacobianTracer.jacobian(button);
    const deltaX = JacobianTracer.effect(before.x, after.x);
    const effectMag = JacobianTracer.effectMagnitude(deltaX);
    const directional = JacobianTracer.directionalAnalysis(jacobian);
    
    // Compute eigenvalue approximations for stability analysis
    const trace = jacobian.reduce((sum, row, i) => sum + row[i], 0);
    const determinant = JacobianTracer.computeDeterminant(jacobian);
    
    return {
      timestamp: new Date().toISOString(),
      operation: {
        button: { 
          abbr: button.abbr, 
          label: button.label, 
          deltaLevel: button.deltaLevel,
          morphemes: button.morphemes || []
        },
        jacobian,
        determinant,
        trace,
        stability: Math.abs(trace) < 2 && Math.abs(determinant) < 10 ? 'stable' : 'unstable'
      },
      transformation: {
        delta_x: deltaX,
        effect_magnitude: effectMag,
        directional_impact: directional
      },
      states: {
        before: {
          x: [...before.x],
          kappa: before.kappa,
          level: before.level,
          timestamp: before.timestamp
        },
        after: {
          x: [...after.x], 
          kappa: after.kappa,
          level: after.level,
          timestamp: after.timestamp
        }
      },
      explanation: JacobianTracer.generateExplanation(jacobian, deltaX, button),
      metadata
    };
  }

  /**
   * Generate human-readable explanation of transformation
   * @param {number[][]} jacobian - Jacobian matrix
   * @param {number[]} deltaX - Effect vector
   * @param {Object} button - Button that caused transformation
   * @returns {string} Human-readable explanation
   */
  static generateExplanation(jacobian, deltaX, button) {
    const effects = [];
    const threshold = 1e-3;

    for (let i = 0; i < deltaX.length; i++) {
      const change = deltaX[i];
      if (Math.abs(change) > threshold) {
        const direction = change > 0 ? 'increased' : 'decreased';
        const magnitude = Math.abs(change).toFixed(3);
        effects.push(`Dimension ${i} ${direction} by ${magnitude}`);
      }
    }

    if (effects.length === 0) {
      return `Button '${button.label}' applied with minimal change`;
    }

    const morphemeInfo = button.morphemes && button.morphemes.length > 0 
      ? ` (composed from morphemes: ${button.morphemes.join(', ')})`
      : '';

    return `Button '${button.label}'${morphemeInfo} caused: ${effects.join('; ')}`;
  }

  /**
   * Compute matrix determinant (up to 3x3 for now)
   * @param {number[][]} matrix - Square matrix
   * @returns {number} Determinant value
   */
  static computeDeterminant(matrix) {
    const n = matrix.length;
    
    if (n === 1) {
      return matrix[0][0];
    } else if (n === 2) {
      return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    } else if (n === 3) {
      return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
           - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
           + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    } else {
      // For higher dimensions, return trace as approximation
      return matrix.reduce((sum, row, i) => sum + row[i], 0);
    }
  }

  /**
   * Check if transformation is volume-preserving (determinant ≈ 1)
   * @param {number[][]} jacobian - Jacobian matrix
   * @returns {boolean} True if transformation preserves volume
   */
  static isVolumePreserving(jacobian) {
    const det = JacobianTracer.computeDeterminant(jacobian);
    return Math.abs(det - 1) < 1e-6;
  }

  /**
   * Check if transformation is contractive (all eigenvalues < 1)
   * Approximation using Frobenius norm
   * @param {number[][]} jacobian - Jacobian matrix
   * @returns {boolean} True if likely contractive
   */
  static isContractive(jacobian) {
    const frobeniusNorm = Math.sqrt(
      jacobian.reduce((sum, row) => 
        sum + row.reduce((rowSum, val) => rowSum + val * val, 0), 0)
    );
    return frobeniusNorm < 1;
  }

  /**
   * Predict effect of applying transformation to a hypothetical state
   * @param {Object} button - Button to apply
   * @param {number[]} stateVector - Hypothetical state
   * @returns {Object} Predicted transformation result
   */
  static predictEffect(button, stateVector) {
    const jacobian = JacobianTracer.jacobian(button);
    const bias = button.b || Array(stateVector.length).fill(0);
    
    // Linear approximation: x' ≈ J x + b
    const predicted = stateVector.map((x, i) => 
      jacobian[i].reduce((sum, jij, j) => sum + jij * stateVector[j], 0) + bias[i]
    );
    
    return {
      predicted_state: predicted,
      jacobian,
      linearization_valid: true // Could add nonlinearity checks here
    };
  }

  /**
   * Analyze sequence of transformations for cumulative effects
   * @param {Object[]} traceSequence - Array of trace logs
   * @returns {Object} Cumulative analysis
   */
  static analyzeCumulativeEffects(traceSequence) {
    if (traceSequence.length === 0) return { total_effect: 0, stability: 'unknown' };

    const totalDelta = traceSequence.reduce((acc, trace) => {
      trace.transformation.delta_x.forEach((dx, i) => {
        acc[i] = (acc[i] || 0) + dx;
      });
      return acc;
    }, []);

    const totalMagnitude = JacobianTracer.effectMagnitude(Object.values(totalDelta));
    
    const stabilityTrend = traceSequence.map(t => t.transformation.effect_magnitude);
    const isConverging = stabilityTrend.length > 1 && 
      stabilityTrend[stabilityTrend.length - 1] < stabilityTrend[0];

    return {
      total_delta: totalDelta,
      total_magnitude: totalMagnitude,
      sequence_length: traceSequence.length,
      convergence_trend: isConverging ? 'converging' : 'diverging',
      average_effect: totalMagnitude / traceSequence.length,
      stability_assessment: totalMagnitude < 1 ? 'stable' : 'potentially_unstable'
    };
  }

  /**
   * Format trace for console output
   * @param {Object} trace - Trace log object
   * @returns {string} Formatted trace string
   */
  static formatTrace(trace) {
    const lines = [
      `[${trace.timestamp}] ${trace.operation.button.label} (${trace.operation.stability})`,
      `  Effect: ${trace.explanation}`,
      `  Magnitude: ${trace.transformation.effect_magnitude.toFixed(4)}`,
      `  Jacobian determinant: ${trace.operation.determinant.toFixed(4)}`
    ];
    
    if (trace.transformation.directional_impact.expansive.length > 0) {
      lines.push(`  Expansive dims: ${trace.transformation.directional_impact.expansive.join(', ')}`);
    }
    
    if (trace.transformation.directional_impact.contractive.length > 0) {
      lines.push(`  Contractive dims: ${trace.transformation.directional_impact.contractive.join(', ')}`);
    }
    
    return lines.join('\n');
  }
}

/**
 * Utility class for managing trace histories and analysis
 */
export class TraceManager {
  constructor(maxHistory = 100) {
    this.traces = [];
    this.maxHistory = maxHistory;
    this.listeners = new Map();
  }

  /**
   * Add a new trace to the history
   * @param {Object} trace - Trace log object
   */
  addTrace(trace) {
    this.traces.push(trace);
    
    // Maintain history size limit
    if (this.traces.length > this.maxHistory) {
      this.traces.shift();
    }
    
    // Notify listeners
    this.emit('traceAdded', trace);
  }

  /**
   * Get recent traces
   * @param {number} count - Number of recent traces to retrieve
   * @returns {Object[]} Array of recent traces
   */
  getRecentTraces(count = 10) {
    return this.traces.slice(-count);
  }

  /**
   * Get traces matching criteria
   * @param {Object} criteria - Filter criteria
   * @returns {Object[]} Filtered traces
   */
  getTraces(criteria = {}) {
    return this.traces.filter(trace => {
      if (criteria.button && trace.operation.button.abbr !== criteria.button) return false;
      if (criteria.stability && trace.operation.stability !== criteria.stability) return false;
      if (criteria.minMagnitude && trace.transformation.effect_magnitude < criteria.minMagnitude) return false;
      if (criteria.maxMagnitude && trace.transformation.effect_magnitude > criteria.maxMagnitude) return false;
      return true;
    });
  }

  /**
   * Clear trace history
   */
  clearTraces() {
    this.traces = [];
    this.emit('tracesCleared');
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
        console.error(`Trace manager event error (${event}):`, error);
      }
    });
  }

  /**
   * Generate summary statistics
   * @returns {Object} Summary statistics
   */
  getSummary() {
    if (this.traces.length === 0) {
      return { total: 0, empty: true };
    }

    const magnitudes = this.traces.map(t => t.transformation.effect_magnitude);
    const stabilities = this.traces.map(t => t.operation.stability);
    
    return {
      total: this.traces.length,
      average_magnitude: magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length,
      max_magnitude: Math.max(...magnitudes),
      min_magnitude: Math.min(...magnitudes),
      stability_distribution: {
        stable: stabilities.filter(s => s === 'stable').length,
        unstable: stabilities.filter(s => s === 'unstable').length
      },
      cumulative_analysis: JacobianTracer.analyzeCumulativeEffects(this.traces)
    };
  }
}