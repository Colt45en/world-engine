/**
 * World Engine Tier 4 - IDE Demo
 * ================================
 *
 * Interactive demonstration of World Engine Tier 4 capabilities
 * with IDE integration features.
 */

// Import the World Engine (in practice, this would be from npm or local module)
// For this demo, we'll include simplified versions of the core functions

// ============================= Core Types =============================

class WorldEngineTier4Demo {
  constructor() {
    this.state = { p: 0.0, i: 0.5, g: 0.3, c: 0.6 };
    this.snapshots = [];
    this.eventLog = [];
    this.ideContext = {};

    // Audio processing
    this.emaEnergy = 1e-3;
    this.emaCentroid = 0.5;
    this.emaZCR = 0.05;
    this.bpm = 120;
    this.phase = 0;

    // Integration enhancements
    this.hud = { op: "-", dt: 0, dx: 0, mu: 0, level: 0, kappa: 0.6, lastError: null };
    this.macroHistory = [];
    this.debugMode = false;

    // Three Ides Macros (no more 2D collapse)
    this.MACROS = {
      IDE_A: ["ST", "SEL", "PRV"],        // Analysis path
      IDE_B: ["CNV", "PRV", "RB"],        // Constraint path
      IDE_C: ["EDT", "UP", "ST"],         // Build path
      ALIGN_IDES: ["CNV", "CNV"],         // Dimensional stability
      MERGE_ABC: ["ALIGN_IDES", "IDE_A", "IDE_B", "IDE_C"],
      OPTIMIZE: ["ST", "SEL", "PRV", "RB"], // Status → Select → Prevent → Rebuild
      DEBUG: ["EDT", "SEL", "RST"],        // Edit → Select → Restore
      STABILIZE: ["PRV", "RB", "EDT"]      // Prevent → Rebuild → Edit
    };

    this.setupOperators();
    this.initializeDemo();
    this.setupHotkeys();
    this.loadSession();
  }

  // ============================= Operator Definitions =============================

  setupOperators() {
    this.operators = {
      'RB': {
        name: 'REBUILD',
        D: [1, 1.2, 1.2, 0.95],
        b: [0, 0.02, 0.03, -0.01],
        ideMapping: 'Refactor code, rebuild project',
        color: '#ff6b6b'
      },
      'UP': {
        name: 'UPDATE',
        D: [1, 1.05, 1, 1.05],
        b: [0, 0.01, 0, 0.01],
        ideMapping: 'Save file, incremental update',
        color: '#4ecdc4'
      },
      'ST': {
        name: 'SNAPSHOT',
        snapshot: true,
        ideMapping: 'Git commit, save checkpoint',
        color: '#45b7d1'
      },
      'PRV': {
        name: 'PREVENT',
        D: [1, 0.9, 1, 1.1],
        b: [0, -0.02, 0, 0.02],
        guard: (s) => ({ ...s, i: Math.min(s.i, 2.0) }),
        ideMapping: 'Apply linting rules, type checking',
        color: '#f7b731'
      },
      'EDT': {
        name: 'EDITOR',
        D: [1, 0.95, 1.08, 1],
        b: [0, 0, 0.01, 0.01],
        ideMapping: 'Format code, auto-fix issues',
        color: '#5f27cd'
      },
      'RST': {
        name: 'RESTORE',
        restore: true,
        ideMapping: 'Git reset, undo changes',
        color: '#ff9ff3'
      },
      'CNV': {
        name: 'CONVERT',
        D: [0.95, 1, 1.1, 1],
        ideMapping: 'Refactor, change paradigm',
        color: '#54a0ff'
      },
      'SEL': {
        name: 'SELECT',
        D: [1, 0.85, 1, 1],
        b: [0, 0, 0, 0.005],
        ideMapping: 'Select text, narrow focus',
        color: '#26de81'
      }
    };
  }

  // ============================= Core Engine Methods =============================

  // Content-addressable ID generation (browser-safe FNV hash)
  cidOf(obj) {
    const s = JSON.stringify(obj, Object.keys(obj).sort());
    let h = 2166136261 >>> 0; // FNV-1a seed
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return "cid_" + (h >>> 0).toString(16);
  }

  // Determinism validation
  assertDeterministic(opId) {
    const op = this.operators[opId];
    if (!op) return;

    const stateA = this.applyOperatorDirect(this.state, opId);
    const stateB = this.applyOperatorDirect(this.state, opId);

    if (this.cidOf(stateA) !== this.cidOf(stateB)) {
      throw new Error(`Non-deterministic operator: ${opId}`);
    }
  }

  // Prevent dimension loss
  assertNoSilentLoss(prev, next, opId) {
    const dropped = [];
    ['p', 'i', 'g', 'c'].forEach((key, idx) => {
      if (prev[key] !== 0 && next[key] === 0) {
        dropped.push(key);
      }
    });

    if (dropped.length && opId !== 'PRV') {
      throw new Error(`Axis dropped without Prevent: ${dropped.join(',')}`);
    }
  }

  // Direct operator application (no side effects)
  applyOperatorDirect(state, opId) {
    const op = this.operators[opId];
    if (!op) return state;

    let newState = { ...state };

    if (op.D && op.b) {
      // Apply transformation: newState = D * state + b
      newState.p = op.D[0] * state.p + op.b[0];
      newState.i = op.D[1] * state.i + op.b[1];
      newState.g = op.D[2] * state.g + op.b[2];
      newState.c = op.D[3] * state.c + op.b[3];
    }

    // Apply guard if present
    if (op.guard) {
      newState = op.guard(newState);
    }

    return newState;
  }

  applyOperator(opId, strength = 1.0) {
    const op = this.operators[opId];
    if (!op) return;

    const t0 = performance.now();
    const prevState = { ...this.state };

    console.log(`[WE4] Applying ${opId} (${op.name}) - strength: ${strength}`);

    try {
      // Validation checks
      this.assertDeterministic(opId);

      if (op.snapshot) {
        this.takeSnapshot(`Snapshot via ${op.name}`);
        this.log(`📸 ${op.name}: Snapshot taken`);
        return;
      }

      if (op.restore && this.snapshots.length > 0) {
        const latest = this.snapshots[this.snapshots.length - 1];
        this.state = { ...latest.state };
        this.log(`🔄 ${op.name}: Restored to snapshot ${latest.id}`);
        return;
      }

      // Apply transformation
      const newState = this.applyOperatorDirect(this.state, opId);

      // Dimension loss check
      this.assertNoSilentLoss(prevState, newState, opId);

      // Update state
      this.state = newState;

      // Update HUD
      const dt = performance.now() - t0;
      const dx = Math.hypot(
        newState.p - prevState.p,
        newState.i - prevState.i,
        newState.g - prevState.g,
        newState.c - prevState.c
      );

      this.hud = {
        op: opId,
        dt: +dt.toFixed(2),
        dx: +dx.toFixed(4),
        mu: +(Math.abs(newState.p) + newState.i + newState.g + newState.c).toFixed(3),
        level: this.computeLevel(),
        kappa: newState.c,
        lastError: null
      };

      this.log(`⚡ ${op.name}: ${op.ideMapping} (Δ=${dx.toFixed(3)}, t=${dt.toFixed(1)}ms)`);
      this.autoSave();

    } catch (error) {
      this.hud.lastError = error.message;
      console.error(`Operator ${opId} failed:`, error);
      this.log(`❌ ${op.name}: ${error.message}`);
    }
  }

  // Run macro sequences
  runMacro(macroName) {
    const sequence = this.MACROS[macroName];
    if (!sequence) {
      console.error(`Unknown macro: ${macroName}`);
      return;
    }

    this.log(`🎯 Running macro: ${macroName}`);
    this.macroHistory.push(macroName);

    // Expand nested macros
    const expandedSequence = [];
    for (const step of sequence) {
      if (this.MACROS[step]) {
        expandedSequence.push(...this.MACROS[step]);
      } else {
        expandedSequence.push(step);
      }
    }

    // Execute sequence
    for (const opId of expandedSequence) {
      if (this.operators[opId]) {
        this.applyOperator(opId, 0.8); // Slightly reduced strength for macros
      }
    }

    this.log(`✅ Macro complete: ${expandedSequence.join(' → ')}`);
  }

  // Auto-planner (Tier-4 autonomous decisions)
  planNextMove() {
    const candidates = ['ST', 'SEL', 'PRV', 'RB', 'CNV', 'EDT', 'UP', 'RST'];
    let best = { opId: '', score: -Infinity, reasoning: '' };

    const currentScore = this.scoreState(this.state);

    for (const opId of candidates) {
      if (!this.operators[opId]) continue;

      try {
        const futureState = this.applyOperatorDirect(this.state, opId);
        const futureScore = this.scoreState(futureState);
        const improvement = futureScore - currentScore - 0.02; // Action cost

        if (improvement > best.score) {
          best = {
            opId,
            score: improvement,
            reasoning: `${opId}: score ${currentScore.toFixed(3)} → ${futureScore.toFixed(3)}`
          };
        }
      } catch (error) {
        continue; // Skip operators that would fail
      }
    }

    if (best.opId) {
      this.log(`🧠 Auto-plan: ${best.reasoning}`);
      this.applyOperator(best.opId);
    } else {
      this.log(`🤔 Auto-plan: No beneficial moves found`);
    }
  }

  // State scoring for auto-planner
  scoreState(state) {
    return 0.6 * state.c + 0.4 * Math.tanh(this.computeLevel() / 3.0);
  }

  computeLevel() {
    // Compute abstraction level based on state complexity
    return Math.floor(Math.abs(this.state.p) + this.state.i + this.state.g);
  }

  // Session persistence
  saveSession() {
    const sessionData = {
      state: this.state,
      snapshots: this.snapshots,
      eventLog: this.eventLog,
      macroHistory: this.macroHistory,
      hud: this.hud,
      timestamp: Date.now(),
      version: "1.0.0"
    };

    try {
      localStorage.setItem("tier4_world_engine_session", JSON.stringify(sessionData));
      this.log("💾 Session saved to localStorage");
    } catch (error) {
      console.error("Failed to save session:", error);
    }
  }

  loadSession() {
    try {
      const saved = localStorage.getItem("tier4_world_engine_session");
      if (!saved) return;

      const data = JSON.parse(saved);
      if (data.state) {
        this.state = data.state;
        this.snapshots = data.snapshots || [];
        this.eventLog = data.eventLog || [];
        this.macroHistory = data.macroHistory || [];
        this.hud = data.hud || this.hud;
        this.log(`📂 Session loaded: ${this.eventLog.length} events`);
      }
    } catch (error) {
      console.error("Failed to load session:", error);
    }
  }

  autoSave() {
    // Auto-save every 10 operations
    if (this.eventLog.length % 10 === 0) {
      this.saveSession();
    }
  }

  // Setup debug hotkeys
  setupHotkeys() {
    document.addEventListener('keydown', (event) => {
      // Skip if input is focused
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
      }

      switch (event.key.toLowerCase()) {
        case 'r':
          if (!event.ctrlKey && !event.metaKey) {
            this.resetToInitial();
            this.log("🔄 Reset triggered via hotkey");
          }
          break;

        case 'p':
          this.planNextMove();
          break;

        case 's':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            this.saveSession();
          }
          break;

        case 'l':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            this.loadSession();
            this.updateDisplay();
          }
          break;

        case 'd':
          this.debugMode = !this.debugMode;
          this.log(`🐛 Debug mode: ${this.debugMode ? 'ON' : 'OFF'}`);
          this.updateHUD();
          break;
      }
    });
  }

  resetToInitial() {
    this.state = { p: 0.0, i: 0.5, g: 0.3, c: 0.6 };
    this.snapshots = [];
    this.eventLog = [];
    this.macroHistory = [];
    this.hud = { op: "-", dt: 0, dx: 0, mu: 0, level: 0, kappa: 0.6, lastError: null };
    this.updateDisplay();
  }

    if (op.snapshot) {
      this.snapshots.push({ ...this.state });
      this.log(`📸 Snapshot taken (${this.snapshots.length} total)`);
      return;
    }

    if (op.restore && this.snapshots.length > 0) {
      this.state = { ...this.snapshots[this.snapshots.length - 1] };
      this.log('🔄 Restored from snapshot');
      return;
    }

    // Apply affine transformation
    let newState = { ...this.state };

    if (op.D) {
      newState.p *= op.D[0];
      newState.i *= op.D[1];
      newState.g *= op.D[2];
      newState.c *= op.D[3];
    }

    if (op.b) {
      newState.p += op.b[0] * strength;
      newState.i += op.b[1] * strength;
      newState.g += op.b[2] * strength;
      newState.c += op.b[3] * strength;
    }

    if (op.guard) {
      newState = op.guard(newState);
    }

    // Apply constraints
    newState.p = this.clamp(newState.p, -1, 1);
    newState.i = this.clamp(newState.i, 0, 2.5);
    newState.g = this.clamp(newState.g, 0, 2.5);
    newState.c = this.clamp(newState.c, 0, 1);

    this.state = newState;

    this.eventLog.push({
      timestamp: Date.now(),
      operator: opId,
      state: { ...this.state },
      mu: this.getMu()
    });

    this.updateDisplay();
    this.log(`✨ ${op.name}: μ=${this.getMu().toFixed(3)}`);
  }

  getMu() {
    return Math.abs(this.state.p) + this.state.i + this.state.g + this.state.c;
  }

  clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  parseWord(word) {
    const w = word.toLowerCase();
    const ops = [];

    if (w.includes('rebuild') || w.includes('refactor')) ops.push('RB');
    if (w.includes('update') || w.includes('save')) ops.push('UP');
    if (w.includes('snapshot') || w.includes('commit')) ops.push('ST');
    if (w.includes('prevent') || w.includes('lint')) ops.push('PRV');
    if (w.includes('format') || w.includes('edit')) ops.push('EDT');
    if (w.includes('restore') || w.includes('undo')) ops.push('RST');
    if (w.includes('convert') || w.includes('transform')) ops.push('CNV');
    if (w.includes('select') || w.includes('focus')) ops.push('SEL');

    ops.forEach(op => this.applyOperator(op));
    return ops;
  }

  // ============================= IDE Integration =============================

  updateIDEContext(context) {
    this.ideContext = { ...this.ideContext, ...context };

    if (context.buildStatus === 'error') {
      this.applyOperator('PRV', 0.8);
      this.log('🚨 Build error - applying constraints');
    } else if (context.buildStatus === 'success') {
      this.applyOperator('UP', 0.6);
      this.log('✅ Build success - updating state');
    }

    if (context.testResults === false) {
      this.applyOperator('RST', 1.0);
      this.log('❌ Tests failed - rolling back');
    } else if (context.testResults === true) {
      this.applyOperator('ST', 1.0);
      this.log('✅ Tests passed - taking snapshot');
    }
  }

  getRecommendations() {
    const recommendations = [];
    const mu = this.getMu();

    if (this.state.c < 0.3) {
      recommendations.push({
        op: 'RST',
        reason: 'Low confidence - consider rolling back',
        urgency: 'high'
      });
    }

    if (this.state.i > 1.8) {
      recommendations.push({
        op: 'PRV',
        reason: 'High intensity - apply constraints',
        urgency: 'medium'
      });
    }

    if (mu > 3.0 && this.snapshots.length === 0) {
      recommendations.push({
        op: 'ST',
        reason: 'High complexity - take snapshot',
        urgency: 'low'
      });
    }

    return recommendations;
  }

  // ============================= Demo Interface =============================

  initializeDemo() {
    this.setupUI();
    this.startDemo();
  }

  setupUI() {
    // Create demo container
    const container = document.createElement('div');
    container.id = 'world-engine-demo';
    container.innerHTML = `
      <div class="we-header">
        <h1>🌍 World Engine Tier 4 - IDE Demo</h1>
        <div class="we-mu">μ (Stack Height): <span id="mu-display">0.000</span></div>
      </div>

      <div class="we-main">
        <div class="we-state">
          <h3>State Vector [p,i,g,c]</h3>
          <div class="state-bars">
            <div class="state-bar">
              <label>Polarity (p)</label>
              <div class="bar"><div class="fill" id="p-bar"></div></div>
              <span id="p-value">0.000</span>
            </div>
            <div class="state-bar">
              <label>Intensity (i)</label>
              <div class="bar"><div class="fill" id="i-bar"></div></div>
              <span id="i-value">0.500</span>
            </div>
            <div class="state-bar">
              <label>Generality (g)</label>
              <div class="bar"><div class="fill" id="g-bar"></div></div>
              <span id="g-value">0.300</span>
            </div>
            <div class="state-bar">
              <label>Confidence (c)</label>
              <div class="bar"><div class="fill" id="c-bar"></div></div>
              <span id="c-value">0.600</span>
            </div>
          </div>
        </div>

        <div class="we-controls">
          <h3>Operators</h3>
          <div class="operator-grid" id="operator-grid"></div>

          <div class="we-word-input">
            <h4>Natural Language</h4>
            <input type="text" id="word-input" placeholder="e.g., 'rebuild component'" />
            <button onclick="demo.handleWordInput()">Parse & Apply</button>
          </div>
        </div>

        <div class="we-ide">
          <h3>IDE Simulation</h3>
          <div class="ide-controls">
            <button onclick="demo.simulateBuildSuccess()">✅ Build Success</button>
            <button onclick="demo.simulateBuildError()">❌ Build Error</button>
            <button onclick="demo.simulateTestPass()">🧪 Tests Pass</button>
            <button onclick="demo.simulateTestFail()">🧪 Tests Fail</button>
          </div>
          <div id="ide-context"></div>
        </div>
      </div>

      <div class="we-log">
        <h3>Event Log</h3>
        <div id="event-log"></div>
      </div>

      <style>
        #world-engine-demo {
          font-family: 'Segoe UI', sans-serif;
          max-width: 1200px;
          margin: 20px auto;
          padding: 20px;
          background: #1e1e1e;
          color: #ffffff;
          border-radius: 10px;
        }

        .we-header {
          text-align: center;
          margin-bottom: 30px;
          border-bottom: 2px solid #333;
          padding-bottom: 20px;
        }

        .we-mu {
          font-size: 18px;
          margin-top: 10px;
        }

        .we-main {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 20px;
          margin-bottom: 20px;
        }

        .we-state, .we-controls, .we-ide {
          background: #2d2d2d;
          padding: 20px;
          border-radius: 8px;
        }

        .state-bar {
          margin-bottom: 15px;
        }

        .state-bar label {
          display: block;
          margin-bottom: 5px;
          font-weight: bold;
        }

        .bar {
          width: 100%;
          height: 20px;
          background: #444;
          border-radius: 10px;
          overflow: hidden;
          margin-bottom: 5px;
        }

        .fill {
          height: 100%;
          background: linear-gradient(90deg, #4ecdc4, #45b7d1);
          transition: width 0.3s ease;
        }

        .operator-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
          gap: 10px;
          margin-bottom: 20px;
        }

        .operator-btn {
          padding: 10px;
          border: none;
          border-radius: 5px;
          color: white;
          cursor: pointer;
          font-weight: bold;
          transition: transform 0.2s ease;
        }

        .operator-btn:hover {
          transform: scale(1.05);
        }

        .operator-btn:active {
          transform: scale(0.95);
        }

        .we-word-input input {
          width: 70%;
          padding: 8px;
          margin-right: 10px;
          background: #444;
          border: 1px solid #666;
          color: white;
          border-radius: 4px;
        }

        .ide-controls button {
          margin: 5px;
          padding: 8px 12px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          background: #444;
          color: white;
        }

        .we-log {
          background: #2d2d2d;
          padding: 20px;
          border-radius: 8px;
          max-height: 200px;
          overflow-y: auto;
        }

        #event-log {
          font-family: monospace;
          font-size: 12px;
          line-height: 1.4;
        }

        .log-entry {
          margin-bottom: 5px;
          opacity: 0.8;
        }

        .log-entry.recent {
          opacity: 1;
          font-weight: bold;
        }
      </style>
    `;

    document.body.appendChild(container);
    this.createOperatorButtons();
  }

  createOperatorButtons() {
    const grid = document.getElementById('operator-grid');

    Object.entries(this.operators).forEach(([id, op]) => {
      const btn = document.createElement('button');
      btn.className = 'operator-btn';
      btn.textContent = id;
      btn.title = `${op.name}: ${op.ideMapping}`;
      btn.style.backgroundColor = op.color || '#666';
      btn.onclick = () => this.applyOperator(id);
      grid.appendChild(btn);
    });
  }

  updateDisplay() {
    // Update state bars
    const s = this.state;

    document.getElementById('p-value').textContent = s.p.toFixed(3);
    document.getElementById('i-value').textContent = s.i.toFixed(3);
    document.getElementById('g-value').textContent = s.g.toFixed(3);
    document.getElementById('c-value').textContent = s.c.toFixed(3);

    // Update bar widths (normalize for display)
    document.getElementById('p-bar').style.width = `${((s.p + 1) / 2) * 100}%`;
    document.getElementById('i-bar').style.width = `${(s.i / 2.5) * 100}%`;
    document.getElementById('g-bar').style.width = `${(s.g / 2.5) * 100}%`;
    document.getElementById('c-bar').style.width = `${s.c * 100}%`;

    // Update mu
    document.getElementById('mu-display').textContent = this.getMu().toFixed(3);

    // Update recommendations
    const recommendations = this.getRecommendations();
    if (recommendations.length > 0) {
      const rec = recommendations[0];
      this.log(`💡 Recommendation: ${rec.op} - ${rec.reason}`);
    }
  }

  log(message) {
    const logDiv = document.getElementById('event-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry recent';
    entry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;

    logDiv.insertBefore(entry, logDiv.firstChild);

    // Remove 'recent' class after animation
    setTimeout(() => entry.classList.remove('recent'), 2000);

    // Keep only last 50 entries
    while (logDiv.children.length > 50) {
      logDiv.removeChild(logDiv.lastChild);
    }
  }

  handleWordInput() {
    const input = document.getElementById('word-input');
    const word = input.value.trim();
    if (word) {
      const ops = this.parseWord(word);
      this.log(`🗣️ Parsed "${word}" → [${ops.join(', ')}]`);
      input.value = '';
    }
  }

  // IDE simulation methods
  simulateBuildSuccess() {
    this.updateIDEContext({ buildStatus: 'success' });
  }

  simulateBuildError() {
    this.updateIDEContext({ buildStatus: 'error' });
  }

  simulateTestPass() {
    this.updateIDEContext({ testResults: true });
  }

  simulateTestFail() {
    this.updateIDEContext({ testResults: false });
  }

  startDemo() {
    this.updateDisplay();
    this.log('🚀 World Engine Tier 4 initialized');
    this.log('🎯 Try clicking operators or typing commands');

    // Simulate some development activity
    setTimeout(() => {
      this.log('📝 Simulating development activity...');
      this.applyOperator('RB', 0.5);
    }, 2000);

    setTimeout(() => {
      this.applyOperator('UP', 0.3);
    }, 4000);

    setTimeout(() => {
      this.applyOperator('ST');
      this.log('🎉 Demo sequence complete - try the controls!');
    }, 6000);
  }
}

// Initialize demo when page loads
let demo;
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    demo = new WorldEngineTier4Demo();
    window.demo = demo; // Make globally accessible for button clicks
  });
}

// Export for Node.js usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { WorldEngineTier4Demo };
}
