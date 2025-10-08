// synthetic/studio-integration.js
/**
 * Studio Integration for Synthetic Directive System
 * Connects synthetic test results to World Engine Studio UI
 */

class SyntheticStudioIntegration {
  constructor(studioBridge) {
    this.studioBridge = studioBridge;
    this.resultsPanel = null;
    this.setupUI();
    this.bindEvents();
  }

  setupUI() {
    // Create synthetic results panel
    const container = document.createElement('div');
    container.id = 'synthetic-results-panel';
    container.innerHTML = `
      <div class="panel-header">
        <h3>🧮 Synthetic Tests</h3>
        <button id="run-synthetic-btn" class="btn-primary">Run Tests</button>
        <button id="watch-synthetic-btn" class="btn-secondary">Watch</button>
      </div>
      <div class="panel-content">
        <div id="synthetic-results">
          <p>Click "Run Tests" to execute synthetic directives...</p>
        </div>
      </div>
    `;

    // Add to studio layout
    const rightPanel = document.querySelector('#right-panels') || document.body;
    rightPanel.appendChild(container);

    this.resultsPanel = container;
  }

  bindEvents() {
    document.getElementById('run-synthetic-btn')?.addEventListener('click', () => {
      this.runSyntheticTests();
    });

    document.getElementById('watch-synthetic-btn')?.addEventListener('click', () => {
      this.startWatching();
    });

    // Listen for studio bridge messages
    if (this.studioBridge) {
      this.studioBridge.onBus('synthetic.results', (data) => {
        this.displayResults(data.results);
      });
    }
  }

  async runSyntheticTests() {
    const btn = document.getElementById('run-synthetic-btn');
    const originalText = btn?.textContent;
    if (btn) btn.textContent = 'Running...';

    try {
      // Fetch from our synthetic server
      const response = await fetch('http://localhost:7077/synthetic');
      if (response.ok) {
        const html = await response.text();
        this.displayResultsHTML(html);
      } else {
        // Fallback: trigger synthetic CLI via studio bridge
        if (this.studioBridge) {
          this.studioBridge.sendBus({
            type: 'synthetic.run',
            payload: { root: window.location.pathname }
          });
        }
      }
    } catch (error) {
      this.displayError(`Failed to run synthetic tests: ${error.message}`);
    } finally {
      if (btn) btn.textContent = originalText;
    }
  }

  displayResultsHTML(html) {
    const resultsDiv = document.getElementById('synthetic-results');
    if (resultsDiv) {
      // Extract the body content from the HTML
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');
      const content = doc.body.innerHTML;
      resultsDiv.innerHTML = content;
    }
  }

  displayResults(results) {
    const resultsDiv = document.getElementById('synthetic-results');
    if (!resultsDiv) return;

    const passing = results.filter(r => r.ok).length;
    const total = results.length;

    let html = `<div class="results-header">Results: ${passing}/${total} passing</div>`;

    results.forEach(result => {
      const statusClass = result.ok ? 'pass' : 'fail';
      const statusIcon = result.ok ? '✅' : '❌';

      html += `
        <div class="result-row ${statusClass}">
          <div class="result-header">
            ${statusIcon} <strong>[${result.kind}]</strong> ${result.name}
            <span class="file-location">${result.file}:${result.line}</span>
          </div>
          ${result.msg ? `<div class="result-message">${result.msg}</div>` : ''}
        </div>
      `;
    });

    resultsDiv.innerHTML = html;

    // Notify studio of results
    if (this.studioBridge) {
      this.studioBridge.sendBus({
        type: 'synthetic.complete',
        payload: { passing, total, results }
      });
    }
  }

  displayError(message) {
    const resultsDiv = document.getElementById('synthetic-results');
    if (resultsDiv) {
      resultsDiv.innerHTML = `<div class="error">⚠️ ${message}</div>`;
    }
  }

  startWatching() {
    // This would integrate with your file watching system
    console.log('🔍 Synthetic watching started (integrate with your file watcher)');

    // Example integration point
    if (this.studioBridge) {
      this.studioBridge.sendBus({
        type: 'synthetic.watch.start',
        payload: { patterns: ['**/*.js', '**/*.ts'] }
      });
    }
  }
}

// Auto-initialize if studio bridge exists
if (typeof window !== 'undefined' && window.studioBridge) {
  window.syntheticIntegration = new SyntheticStudioIntegration(window.studioBridge);
}

export { SyntheticStudioIntegration };
