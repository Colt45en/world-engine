/**
 * World Engine Initialization System
 *
 * Coordinates startup, configuration, and integration of all World Engine components
 * with the Studio Bridge system. Handles proper sequencing, error recovery, and
 * status reporting throughout the initialization process.
 */

(function() {
  'use strict';

  // Version and metadata
  const WE_INIT_VERSION = '3.1.0';
  const WE_INIT_NAME = 'WorldEngine Initializer';

  // Installation guard
  if (window.__WORLD_ENGINE_INIT__) {
    console.warn(`${WE_INIT_NAME} already installed`);
    return;
  }
  window.__WORLD_ENGINE_INIT__ = { version: WE_INIT_VERSION, installedAt: Date.now() };

  // Configuration constants
  const CONFIG = Object.freeze({
    INIT_TIMEOUT_MS: 10000,
    COMPONENT_TIMEOUT_MS: 5000,
    STATUS_POLL_INTERVAL_MS: 500,
    MAX_RETRY_ATTEMPTS: 3,
    REQUIRED_COMPONENTS: ['engine', 'recorder', 'chat', 'lexicon'],
    PYTHON_ENGINE_PATH: 'world_engine.py',
    LEXICON_ENGINE_PATH: 'tri_circ_square.html'
  });

  // Initialization state
  const initState = {
    phase: 'starting',
    components: new Map(),
    errors: [],
    startedAt: Date.now(),
    completedAt: null,
    retryCount: 0
  };

  // Component registry
  const components = new Map();

  /**
   * Initialize the World Engine system with full integration
   */
  async function initializeWorldEngine(options = {}) {
    const opts = {
      autoStart: true,
      enablePython: true,
      enableLexicon: true,
      statusCallback: null,
      ...options
    };

    try {
      updateStatus('initializing', 'Starting World Engine initialization...');

      // Phase 1: Safety and Bridge Setup
      await initializeBridge();

      // Phase 2: Component Discovery
      await discoverComponents();

      // Phase 3: Component Initialization
      await initializeComponents(opts);

      // Phase 4: Integration and Wiring
      await wireComponents();

      // Phase 5: Health Check and Validation
      await validateSystem();

      updateStatus('ready', 'World Engine system ready');
      initState.completedAt = Date.now();

      // Final announcement
      announceReady();

      return {
        success: true,
        components: Array.from(components.keys()),
        initTime: initState.completedAt - initState.startedAt,
        version: WE_INIT_VERSION
      };

    } catch (error) {
      updateStatus('error', `Initialization failed: ${error.message}`);
      initState.errors.push(error);

      // Retry logic
      if (initState.retryCount < CONFIG.MAX_RETRY_ATTEMPTS) {
        initState.retryCount++;
        console.warn(`World Engine init failed, retrying (${initState.retryCount}/${CONFIG.MAX_RETRY_ATTEMPTS})...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
        return initializeWorldEngine(options);
      }

      throw error;
    }
  }

  /**
   * Initialize Studio Bridge and safety systems
   */
  async function initializeBridge() {
    updateStatus('bridge', 'Setting up Studio Bridge...');

    // Ensure Studio Bridge is available
    if (!window.StudioBridge) {
      throw new Error('Studio Bridge not found - load studio-bridge.js first');
    }

    // Initialize Safety Harness if available
    if (window.SafetyHarness) {
      window.SafetyHarness.setStatus('run');
      window.SafetyHarness.log('World Engine initialization starting');
    }

    // Set up bus listeners for initialization events
    window.StudioBridge.onBus((msg) => {
      handleBusMessage(msg);
    });

    components.set('bridge', {
      name: 'Studio Bridge',
      status: 'ready',
      api: window.StudioBridge,
      health: () => !!window.StudioBridge
    });
  }

  /**
   * Discover available components in the system
   */
  async function discoverComponents() {
    updateStatus('discovery', 'Discovering components...');

    const discoveries = [
      discoverPythonEngine(),
      discoverLexiconEngine(),
      discoverRecorderController(),
      discoverChatController(),
      discoverEngineController()
    ];

    await Promise.allSettled(discoveries);
  }

  /**
   * Discover Python World Engine
   */
  async function discoverPythonEngine() {
    try {
      // Check if Python engine file exists
      const response = await fetch(CONFIG.PYTHON_ENGINE_PATH, { method: 'HEAD' });
      if (response.ok) {
        components.set('python-engine', {
          name: 'Python World Engine',
          status: 'discovered',
          path: CONFIG.PYTHON_ENGINE_PATH,
          health: () => true // Will be updated after initialization
        });
      }
    } catch (error) {
      console.warn('Python engine not found:', error.message);
    }
  }

  /**
   * Discover Lexicon Engine
   */
  async function discoverLexiconEngine() {
    try {
      // Check if the lexicon engine HTML exists or is embedded
      const lexiconFrame = document.querySelector('iframe[src*="lexicon"], iframe[src*="tri_circ_square"]');
      const lexiconContainer = document.getElementById('lexicon-engine');

      if (lexiconFrame || lexiconContainer || document.getElementById('viz')) {
        components.set('lexicon-engine', {
          name: 'Lexicon Engine',
          status: 'discovered',
          element: lexiconFrame || lexiconContainer,
          health: () => !!document.getElementById('viz')
        });
      }
    } catch (error) {
      console.warn('Lexicon engine not found:', error.message);
    }
  }

  /**
   * Discover Recorder Controller
   */
  async function discoverRecorderController() {
    if (window.RecorderController) {
      components.set('recorder', {
        name: 'Recorder Controller',
        status: 'discovered',
        constructor: window.RecorderController,
        health: () => !!window.RecorderController
      });
    }
  }

  /**
   * Discover Chat Controller
   */
  async function discoverChatController() {
    if (window.ChatController) {
      components.set('chat', {
        name: 'Chat Controller',
        status: 'discovered',
        constructor: window.ChatController,
        health: () => !!window.ChatController
      });
    }
  }

  /**
   * Discover Engine Controller
   */
  async function discoverEngineController() {
    if (window.EngineController) {
      components.set('engine', {
        name: 'Engine Controller',
        status: 'discovered',
        constructor: window.EngineController,
        health: () => !!window.EngineController
      });
    }
  }

  /**
   * Initialize all discovered components
   */
  async function initializeComponents(opts) {
    updateStatus('components', 'Initializing components...');

    const initPromises = [];

    for (const [key, component] of components.entries()) {
      if (component.status === 'discovered') {
        initPromises.push(initializeComponent(key, component, opts));
      }
    }

    await Promise.allSettled(initPromises);
  }

  /**
   * Initialize a single component
   */
  async function initializeComponent(key, component, opts) {
    try {
      updateStatus('component', `Initializing ${component.name}...`);

      switch (key) {
      case 'lexicon-engine':
        await initializeLexiconEngine(component);
        break;
      case 'recorder':
        await initializeRecorder(component, opts);
        break;
      case 'chat':
        await initializeChat(component, opts);
        break;
      case 'engine':
        await initializeEngine(component, opts);
        break;
      default:
        component.status = 'ready';
      }

      component.status = 'ready';
      console.log(`✓ ${component.name} initialized`);

    } catch (error) {
      component.status = 'error';
      component.error = error.message;
      console.error(`✗ ${component.name} failed:`, error.message);
    }
  }

  /**
   * Initialize Lexicon Engine
   */
  async function initializeLexiconEngine(component) {
    // Check if lexicon engine is already running
    const vizElement = document.getElementById('viz');
    if (vizElement) {
      // Engine is embedded, check if it's functional
      const canvas = vizElement.getContext?.('2d');
      if (canvas) {
        component.api = {
          run: (text) => {
            const input = document.getElementById('input');
            const runBtn = document.getElementById('run');
            if (input && runBtn) {
              input.value = text;
              runBtn.click();
              return true;
            }
            return false;
          },
          getResults: () => {
            const output = document.getElementById('out');
            try {
              return output ? JSON.parse(output.textContent) : null;
            } catch {
              return { error: 'Parse failed' };
            }
          }
        };
      }
    }
  }

  /**
   * Initialize Recorder
   */
  async function initializeRecorder(component, opts) {
    component.instance = new component.constructor({
      transcription: opts.enableTranscription || false,
      autoMarkRuns: true,
      chunkSize: 200
    });

    component.api = {
      startMic: () => component.instance.startMic?.(),
      startScreen: () => component.instance.startScreen?.(),
      stop: () => component.instance.stop?.(),
      getStatus: () => component.instance.getStatus?.()
    };
  }

  /**
   * Initialize Chat Controller
   */
  async function initializeChat(component, opts) {
    component.instance = new component.constructor({
      autoLinkClips: true,
      transcriptCommands: opts.enableTranscription || false,
      commandPrefix: '/'
    });

    component.api = {
      execute: (cmd) => component.instance.execute?.(cmd),
      announce: (msg, level) => component.instance.announce?.(msg, level),
      getStats: () => component.instance.getStats?.()
    };
  }

  /**
   * Initialize Engine Controller
   */
  async function initializeEngine(component, opts) {
    // Look for engine iframe
    const engineFrame = document.querySelector('iframe[src*="engine"], iframe[src*="lexicon"]');
    if (engineFrame) {
      component.instance = new component.constructor(engineFrame);

      component.api = {
        runText: (text) => component.instance.runText?.(text),
        loadTest: (name) => component.instance.loadTest?.(name),
        getStatus: () => component.instance.getStatus?.()
      };
    }
  }

  /**
   * Wire components together for integrated operation
   */
  async function wireComponents() {
    updateStatus('wiring', 'Wiring components together...');

    // Set up cross-component communication
    if (components.has('chat') && components.has('engine')) {
      // Chat can control engine
      const chat = components.get('chat');
      const engine = components.get('engine');

      chat.engineAPI = engine.api;
    }

    if (components.has('chat') && components.has('recorder')) {
      // Chat can control recorder
      const chat = components.get('chat');
      const recorder = components.get('recorder');

      chat.recorderAPI = recorder.api;
    }

    // Set up status reporting
    for (const [key, component] of components.entries()) {
      component.reportStatus = () => {
        const health = component.health ? component.health() : component.status === 'ready';
        window.StudioBridge?.sendBus({
          type: 'component.status',
          component: key,
          name: component.name,
          status: component.status,
          healthy: health,
          timestamp: Date.now()
        });
      };
    }
  }

  /**
   * Validate the complete system
   */
  async function validateSystem() {
    updateStatus('validation', 'Validating system...');

    const validations = [];

    for (const [key, component] of components.entries()) {
      validations.push(validateComponent(key, component));
    }

    const results = await Promise.allSettled(validations);

    const failures = results.filter(r => r.status === 'rejected');
    if (failures.length > 0) {
      throw new Error(`System validation failed: ${failures.length} components unhealthy`);
    }
  }

  /**
   * Validate a single component
   */
  async function validateComponent(key, component) {
    try {
      const isHealthy = component.health ? component.health() : component.status === 'ready';
      if (!isHealthy) {
        throw new Error(`${component.name} failed health check`);
      }

      // Component-specific validation
      switch (key) {
      case 'lexicon-engine':
        // Test basic functionality
        if (component.api?.run) {
          component.api.run('test');
          await new Promise(resolve => setTimeout(resolve, 100));
          const results = component.api.getResults();
          if (!results) {
            throw new Error('Lexicon engine not responding');
          }
        }
        break;
      case 'bridge':
        // Test bus communication
        let busWorking = false;
        const testHandler = () => { busWorking = true; };
        window.StudioBridge.onBus(testHandler);
        window.StudioBridge.sendBus({ type: 'test.ping' });
        await new Promise(resolve => setTimeout(resolve, 50));
        if (!busWorking) {
          throw new Error('Studio Bridge bus not working');
        }
        break;
      }

      console.log(`✓ ${component.name} validation passed`);
      return true;

    } catch (error) {
      console.error(`✗ ${component.name} validation failed:`, error.message);
      throw error;
    }
  }

  /**
   * Handle bus messages during initialization
   */
  function handleBusMessage(msg) {
    switch (msg.type) {
    case 'init.status':
      // Status request
      window.StudioBridge.sendBus({
        type: 'init.status.result',
        state: initState,
        components: Object.fromEntries(components),
        timestamp: Date.now()
      });
      break;
    case 'component.health':
      // Health check request
      if (msg.component && components.has(msg.component)) {
        const component = components.get(msg.component);
        component.reportStatus?.();
      }
      break;
    }
  }

  /**
   * Update initialization status
   */
  function updateStatus(phase, message) {
    initState.phase = phase;

    console.log(`[WorldEngine:${phase}] ${message}`);

    // Update safety harness if available
    if (window.SafetyHarness) {
      const statusMap = {
        'starting': 'run',
        'initializing': 'run',
        'bridge': 'run',
        'discovery': 'run',
        'components': 'run',
        'component': 'run',
        'wiring': 'run',
        'validation': 'warn',
        'ready': 'good',
        'error': 'err'
      };

      window.SafetyHarness.setStatus(statusMap[phase] || 'run');
      window.SafetyHarness.log(message);
    }

    // Send bus message
    if (window.StudioBridge) {
      window.StudioBridge.sendBus({
        type: 'init.progress',
        phase,
        message,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Announce system ready
   */
  function announceReady() {
    const summary = {
      version: WE_INIT_VERSION,
      components: components.size,
      initTime: initState.completedAt - initState.startedAt,
      healthy: Array.from(components.values()).filter(c => c.status === 'ready').length
    };

    console.log(`🚀 World Engine v${WE_INIT_VERSION} ready!`);
    console.log(`   Components: ${summary.healthy}/${summary.components} healthy`);
    console.log(`   Init time: ${summary.initTime}ms`);

    // Announce to chat if available
    const chat = components.get('chat');
    if (chat?.api?.announce) {
      chat.api.announce(
        `🚀 World Engine v${WE_INIT_VERSION} ready - ${summary.healthy}/${summary.components} components healthy`,
        'system'
      );
    }

    // Send ready message
    window.StudioBridge.sendBus({
      type: 'we.ready',
      summary,
      components: Array.from(components.keys()),
      timestamp: Date.now()
    });
  }

  /**
   * Public API
   */
  const WorldEngineInit = {
    version: WE_INIT_VERSION,
    initialize: initializeWorldEngine,
    getComponents: () => new Map(components),
    getState: () => ({ ...initState }),
    getComponent: (name) => components.get(name),
    healthCheck: async () => {
      const results = new Map();
      for (const [key, component] of components.entries()) {
        try {
          const healthy = component.health ? component.health() : component.status === 'ready';
          results.set(key, { healthy, status: component.status, error: component.error });
        } catch (error) {
          results.set(key, { healthy: false, status: 'error', error: error.message });
        }
      }
      return results;
    }
  };

  // Export
  window.WorldEngineInit = WorldEngineInit;

  // Auto-initialize if not explicitly disabled
  if (!window.WE_MANUAL_INIT) {
    // Small delay to let other components load
    setTimeout(() => {
      initializeWorldEngine().catch(error => {
        console.error('Auto-initialization failed:', error);
      });
    }, 100);
  }

  console.log(`${WE_INIT_NAME} v${WE_INIT_VERSION} loaded`);

})();
