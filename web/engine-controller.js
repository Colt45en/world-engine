/**
 * Engine Controller - Connects World Engine to the Studio Bridge
 *
 * Handles engine operations, runs, tests, and result processing.
 * Designed to work with worldengine.html as an iframe.
 */

class EngineController {
  constructor(engineFrame) {
    this.engineFrame = engineFrame;
    this.transport = null;
    this.isReady = false;
    this.setupTransport();
    this.bindEvents();
  }

  setupTransport() {
    this.transport = setupEngineTransport(this.engineFrame);

    // Wait for iframe to load
    this.engineFrame.addEventListener('load', () => {
      setTimeout(() => {
        this.isReady = true;
        Utils.log('Engine iframe loaded and ready');
        sendBus({ type: 'eng.ready' });
      }, 500); // Give engine time to initialize
    });
  }

  bindEvents() {
    onBus(async (msg) => {
      if (!this.isReady) {
        Utils.log('Engine not ready, queuing message', 'warn');
        setTimeout(() => onBus(msg), 100);
        return;
      }

      try {
        switch (msg.type) {
          case 'eng.run':
            await this.handleRun(msg);
            break;
          case 'eng.test':
            await this.handleTest(msg);
            break;
          case 'eng.status':
            await this.handleStatus(msg);
            break;
        }
      } catch (error) {
        Utils.log(`Engine error: ${error.message}`, 'error');
        sendBus({
          type: 'eng.error',
          error: error.message,
          originalMessage: msg
        });
      }
    });
  }

  async handleRun(msg) {
    const runId = Utils.generateId();
    Utils.log(`Starting engine run: ${runId}`);

    // Mark start for recorder
    sendBus({ type: 'rec.mark', tag: 'run-start', runId });

    this.transport.withEngine((doc) => {
      const input = doc.getElementById('input');
      const runBtn = doc.getElementById('run');

      if (!input || !runBtn) {
        throw new Error('Engine input/run elements not found');
      }

      // Set input and trigger run
      input.value = msg.text.trim();
      runBtn.click();

      // Wait for results with timeout
      const checkResults = () => {
        setTimeout(() => {
          try {
            const output = doc.getElementById('out');
            if (!output) {
              throw new Error('Engine output element not found');
            }

            const raw = output.textContent || '{}';
            let outcome = {};

            try {
              outcome = JSON.parse(raw);
            } catch (parseError) {
              // If not JSON, wrap as text result
              outcome = {
                type: 'text',
                result: raw,
                input: msg.text,
                timestamp: Date.now()
              };
            }

            // Save to store
            Store.save('wordEngine.lastRun', outcome);
            Store.save(`runs.${runId}`, {
              runId,
              ts: Date.now(),
              input: msg.text,
              outcome,
              clipId: null // Will be linked by chat controller
            });

            // Mark end and announce result
            sendBus({ type: 'rec.mark', tag: 'run-end', runId });
            sendBus({ type: 'eng.result', runId, outcome, input: msg.text });

            Utils.log(`Engine run completed: ${runId}`);

          } catch (error) {
            Utils.log(`Engine result processing error: ${error.message}`, 'error');
            sendBus({
              type: 'eng.error',
              runId,
              error: error.message
            });
          }
        }, 150); // Give engine time to process
      };

      checkResults();
    });
  }

  async handleTest(msg) {
    Utils.log(`Loading test: ${msg.name}`);

    this.transport.withEngine((doc) => {
      const testsContainer = doc.getElementById('tests');
      if (!testsContainer) {
        throw new Error('Tests container not found');
      }

      const buttons = testsContainer.querySelectorAll('button');
      const testName = msg.name.toLowerCase();

      let testFound = false;
      for (const btn of buttons) {
        const btnText = (btn.textContent || '').trim().toLowerCase();
        if (btnText === testName || btnText.includes(testName)) {
          btn.click();
          testFound = true;
          Utils.log(`Test loaded: ${msg.name}`);

          // Chain into a run after test loads
          setTimeout(() => {
            const input = doc.getElementById('input');
            if (input && input.value.trim()) {
              sendBus({ type: 'eng.run', text: input.value.trim() });
            }
          }, 100);
          break;
        }
      }

      if (!testFound) {
        throw new Error(`Test not found: ${msg.name}`);
      }
    });
  }

  async handleStatus(msg) {
    this.transport.withEngine((doc) => {
      const input = doc.getElementById('input');
      const output = doc.getElementById('out');

      const status = {
        ready: this.isReady,
        hasInput: !!input,
        hasOutput: !!output,
        inputValue: input?.value || '',
        lastOutput: output?.textContent || '',
        transport: this.transport.isSameOrigin() ? 'same-origin' : 'cross-origin'
      };

      sendBus({ type: 'eng.status.result', status });
      Utils.log(`Engine status: ${JSON.stringify(status)}`);
    });
  }

  // Public methods for direct control
  async runText(text) {
    return new Promise((resolve) => {
      const runId = Utils.generateId();
      const handler = (msg) => {
        if (msg.type === 'eng.result' && msg.runId === runId) {
          resolve(msg.outcome);
        }
      };

      onBus(handler);
      sendBus({ type: 'eng.run', text });
    });
  }

  async loadTest(name) {
    sendBus({ type: 'eng.test', name });
  }

  getStatus() {
    return {
      ready: this.isReady,
      transport: this.transport?.isSameOrigin() ? 'same-origin' : 'cross-origin'
    };
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = EngineController;
} else {
  window.EngineController = EngineController;
}
