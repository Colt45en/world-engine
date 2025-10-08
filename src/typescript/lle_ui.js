// Minimal UI wiring in the webview: palette → click → state → timeline
(function () {
    const vscode = acquireVsCodeApi();
    const el = (id) => document.getElementById(id);

    // Simple DOM build
    const bar = document.createElement('div');
    bar.id = 'lle-bar';

    const palette = document.createElement('div');
    palette.id = 'lle-palette';

    const state = document.createElement('pre');
    state.id = 'lle-state';

    const timeline = document.createElement('div');
    timeline.id = 'lle-timeline';

    const container = document.getElementById('pad-container');
    if (container) {
        container.prepend(bar, palette, state, timeline);
    }

    function renderState(su) {
        if (state) {
            state.textContent = `x: ${JSON.stringify(su.x)}
level: ${su.level}
κ: ${su.kappa.toFixed(3)}`;
        }
    }

    function addButton(spec) {
        const b = document.createElement('button');
        b.textContent = spec.abbr;
        b.title = spec.label;
        b.onclick = () => vscode.postMessage({ type: 'lle.click', button: spec.abbr });
        palette.appendChild(b);
    }

    const defaultButtons = [
        { abbr: 'MD', label: 'Module' },
        { abbr: 'CP', label: 'Component' },
        { abbr: 'PR', label: 'Prevent' },
        { abbr: 'CV', label: 'Convert' },
        { abbr: 'RB', label: 'Rebuild' },
        { abbr: 'UP', label: 'Update' }
    ];

    defaultButtons.forEach(addButton);

    window.addEventListener('message', (e) => {
        const msg = e.data;
        if (msg.type === 'lle.state') renderState(msg.su);
        if (msg.type === 'lle.timeline') {
            if (timeline) {
                timeline.textContent = msg.events.map(ev => `${ev.seq} ${ev.button} → ${ev.outputCid}`).join('\n');
            }
        }
    });
})();
