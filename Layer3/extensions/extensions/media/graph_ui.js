/* eslint-disable linebreak-style */
/* global acquireVsCodeApi, cytoscape */

(function () {
    const vscode = acquireVsCodeApi();
    const container = document.getElementById('pad-container');
    if (!container) {
        vscode.postMessage({ type: 'lle.graph.error', error: 'container-missing' });
        return;
    }

    const gdiv = document.createElement('div');
    gdiv.id = 'lle-graph';
    container.appendChild(gdiv);

    let cy;
    function renderGraph(graph) {
        if (typeof cytoscape !== 'function') {
            vscode.postMessage({ type: 'lle.graph.error', error: 'cytoscape-missing' });
            return;
        }
        if (cy) cy.destroy();
        cy = cytoscape({
            container: gdiv,
            elements: [
                ...graph.nodes.map((n) => ({ data: { id: n.id, label: n.label, type: n.type } })),
                ...graph.edges.map((e) => ({ data: { source: e.from, target: e.to, label: e.label } }))
            ],
            style: [
                { selector: 'node', style: { label: 'data(label)', 'background-color': '#999', 'text-valign': 'center', 'font-size': '10px' } },
                { selector: 'node[type="SU"]', style: { 'background-color': '#3b82f6' } },
                { selector: 'node[type="Event"]', style: { 'background-color': '#f59e0b' } },
                { selector: 'node[type="Button"]', style: { 'background-color': '#10b981' } },
                { selector: 'edge', style: { 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#888', 'target-arrow-color': '#888', 'font-size': '8px', label: 'data(label)' } }
            ],
            layout: { name: 'cose', animate: false }
        });
    }

    window.addEventListener('message', (event) => {
        const msg = event.data;
        if (msg?.type === 'lle.graph') renderGraph(msg.graph);
    });
})();
