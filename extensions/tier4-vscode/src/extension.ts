/**
 * VS Code Extension for Tier-4 Meta System
 * =========================================
 *
 * Native VS Code integration for the Tier-4 Meta Reasoning System.
 * Provides real-time state visualization, operator commands, and IDE integration.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

// Import Tier-4 system (would need proper module resolution)
interface StationaryUnit {
    x: number[];
    kappa: number;
    level: number;
}

interface HUDState {
    op: string;
    dt: number;
    dx: number;
    mu: number;
    level: number;
    kappa: number;
    lastError?: string;
}

class Tier4Extension {
    private context: vscode.ExtensionContext;
    private statusBarItem: vscode.StatusBarItem;
    private hudWebviewPanel: vscode.WebviewPanel | undefined;
    private currentState: StationaryUnit;
    private hud: HUDState;
    private isActive = false;
    private sessionPath: string;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
        this.currentState = { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 };
        this.hud = { op: '-', dt: 0, dx: 0, mu: 0, level: 0, kappa: 0.6 };
        this.sessionPath = path.join(context.globalStorageUri.fsPath, 'tier4-sessions');

        this.setupCommands();
        this.setupStatusBar();
        this.setupViews();

        // Auto-activate if configured
        const config = vscode.workspace.getConfiguration('tier4');
        if (config.get('autoActivate')) {
            this.activate();
        }
    }

    // ============================= Command Setup =============================

    private setupCommands() {
        const commands = [
            vscode.commands.registerCommand('tier4.activate', () => this.activate()),
            vscode.commands.registerCommand('tier4.demo', () => this.openDemo()),
            vscode.commands.registerCommand('tier4.applyOperator', () => this.showOperatorPicker()),
            vscode.commands.registerCommand('tier4.runMacro', () => this.showMacroPicker()),
            vscode.commands.registerCommand('tier4.showHUD', () => this.toggleHUD()),
            vscode.commands.registerCommand('tier4.saveSession', () => this.saveSession()),
            vscode.commands.registerCommand('tier4.loadSession', () => this.loadSession()),
            vscode.commands.registerCommand('tier4.autoPlanner', () => this.toggleAutoPlanner()),
            vscode.commands.registerCommand('tier4.validate', () => this.validateIntegration()),
        ];

        this.context.subscriptions.push(...commands);
    }

    private setupStatusBar() {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            100
        );
        this.statusBarItem.command = 'tier4.showHUD';
        this.updateStatusBar();
        this.context.subscriptions.push(this.statusBarItem);
    }

    private setupViews() {
        // Register tree data providers for custom views
        const stateProvider = new Tier4StateProvider(this);
        vscode.window.createTreeView('tier4-state', { treeDataProvider: stateProvider });

        const operatorProvider = new Tier4OperatorProvider();
        vscode.window.createTreeView('tier4-operators', { treeDataProvider: operatorProvider });

        const macroProvider = new Tier4MacroProvider();
        vscode.window.createTreeView('tier4-macros', { treeDataProvider: macroProvider });
    }

    // ============================= Core Operations =============================

    private activate() {
        this.isActive = true;
        vscode.commands.executeCommand('setContext', 'tier4.active', true);

        this.statusBarItem.show();
        this.updateStatusBar();

        vscode.window.showInformationMessage(
            '🧠 Tier-4 Meta System activated! Use Ctrl+Shift+T for operators.',
            'Show HUD', 'Open Demo'
        ).then(selection => {
            if (selection === 'Show HUD') {
                this.toggleHUD();
            } else if (selection === 'Open Demo') {
                this.openDemo();
            }
        });

        // Setup file watchers for Git integration
        this.setupGitIntegration();
    }

    private async showOperatorPicker() {
        const operators = [
            { label: 'RB - Rebuild', detail: 'Recompose from parts (concretize)', value: 'RB' },
            { label: 'UP - Update', detail: 'Move along current manifold', value: 'UP' },
            { label: 'ST - Snapshot', detail: 'Save current state', value: 'ST' },
            { label: 'PR - Prevent', detail: 'Apply constraints and safeguards', value: 'PR' },
            { label: 'ED - Edit', detail: 'Structural modifications', value: 'ED' },
            { label: 'RS - Restore', detail: 'Revert to previous state', value: 'RS' },
            { label: 'CV - Convert', detail: 'Change representation', value: 'CV' },
            { label: 'SL - Select', detail: 'Focus on specific aspects', value: 'SL' },
            { label: 'CH - Channel', detail: 'Direct information flow', value: 'CH' },
            { label: 'MD - Module', detail: 'Package into components', value: 'MD' }
        ];

        const selected = await vscode.window.showQuickPick(operators, {
            placeHolder: 'Select Tier-4 operator to apply...',
            matchOnDetail: true
        });

        if (selected) {
            this.applyOperator(selected.value);
        }
    }

    private async showMacroPicker() {
        const macros = [
            { label: 'IDE_A - Analysis Path', detail: 'ST → SL → CP', value: 'IDE_A' },
            { label: 'IDE_B - Constraint Path', detail: 'CV → PR → RC', value: 'IDE_B' },
            { label: 'IDE_C - Build Path', detail: 'TL → RB → MD', value: 'IDE_C' },
            { label: 'MERGE_ABC - Full Integration', detail: 'Combine all three ides safely', value: 'MERGE_ABC' },
            { label: 'OPTIMIZE - Standard Flow', detail: 'ST → CP → PR → RC', value: 'OPTIMIZE' },
            { label: 'DEBUG - Debug Mode', detail: 'TL → SL → ED → RS', value: 'DEBUG' },
            { label: 'STABILIZE - Stabilize State', detail: 'PR → RC → TL', value: 'STABILIZE' }
        ];

        const selected = await vscode.window.showQuickPick(macros, {
            placeHolder: 'Select Three Ides macro to execute...',
            matchOnDetail: true
        });

        if (selected) {
            this.runMacro(selected.value);
        }
    }

    private applyOperator(operatorId: string) {
        try {
            const prevState = { ...this.currentState };
            const startTime = Date.now();

            // Apply operator transformation (simplified)
            this.currentState = this.transformState(this.currentState, operatorId);

            const endTime = Date.now();
            const dt = endTime - startTime;

            // Update HUD
            this.hud = this.updateHUD(this.hud, operatorId, prevState, this.currentState, dt);
            this.updateStatusBar();
            this.updateHUDWebview();

            // Show result
            vscode.window.showInformationMessage(
                `✅ Applied ${operatorId}: Δx=${this.hud.dx.toFixed(3)}, κ=${this.hud.kappa.toFixed(2)}`
            );

            // Auto-save if configured
            const config = vscode.workspace.getConfiguration('tier4');
            if (config.get('sessionAutoSave')) {
                this.saveSession();
            }

            // Git integration
            this.mapOperatorToGit(operatorId);

        } catch (error) {
            vscode.window.showErrorMessage(`❌ Operator ${operatorId} failed: ${error}`);
            this.hud.lastError = String(error);
            this.updateStatusBar();
        }
    }

    private runMacro(macroId: string) {
        const macros: Record<string, string[]> = {
            IDE_A: ['ST', 'SL', 'CP'],
            IDE_B: ['CV', 'PR', 'RC'],
            IDE_C: ['TL', 'RB', 'MD'],
            MERGE_ABC: ['CV', 'CV', 'ST', 'SL', 'CP', 'CV', 'PR', 'RC', 'TL', 'RB', 'MD'],
            OPTIMIZE: ['ST', 'CP', 'PR', 'RC'],
            DEBUG: ['TL', 'SL', 'ED', 'RS'],
            STABILIZE: ['PR', 'RC', 'TL']
        };

        const sequence = macros[macroId];
        if (!sequence) {
            vscode.window.showErrorMessage(`Unknown macro: ${macroId}`);
            return;
        }

        vscode.window.showInformationMessage(
            `🔥 Running macro ${macroId}: ${sequence.join(' → ')}`
        );

        // Execute sequence with delays for visualization
        let delay = 0;
        sequence.forEach((op, index) => {
            setTimeout(() => {
                this.applyOperator(op);
                if (index === sequence.length - 1) {
                    vscode.window.showInformationMessage(`✅ Macro ${macroId} completed!`);
                }
            }, delay);
            delay += 200; // 200ms between operations
        });
    }

    // ============================= State Management =============================

    private transformState(state: StationaryUnit, operatorId: string): StationaryUnit {
        // Simplified operator transformations
        const operators: Record<string, { M: number[][], b: number[], alpha: number }> = {
            RB: { M: [[1, 0, 0, 0], [0, 1.05, 0, 0], [0, 0, 1.05, 0], [0, 0, 0, 0.95]], b: [0, 0.02, 0.03, -0.01], alpha: 1 },
            UP: { M: [[1, 0, 0, 0], [0, 1.05, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.05]], b: [0, 0.01, 0, 0.01], alpha: 1 },
            ST: { M: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], b: [0, 0, 0, 0], alpha: 1 }, // Snapshot
            PR: { M: [[1, 0, 0, 0], [0, 0.9, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.1]], b: [0, -0.02, 0, 0.02], alpha: 1 },
            CV: { M: [[0.95, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1.1, 0], [0, 0, 0, 1]], b: [0, 0, 0.01, 0.01], alpha: 1 }
        };

        const op = operators[operatorId];
        if (!op) {
            throw new Error(`Unknown operator: ${operatorId}`);
        }

        // Matrix-vector multiplication
        const newX = new Array(4);
        for (let i = 0; i < 4; i++) {
            let sum = 0;
            for (let j = 0; j < 4; j++) {
                sum += op.M[i][j] * state.x[j];
            }
            newX[i] = op.alpha * sum + op.b[i];
        }

        // Update kappa based on operation
        let newKappa = state.kappa;
        if (operatorId === 'PR') newKappa = Math.min(1, newKappa + 0.1);
        if (operatorId === 'CV') newKappa = Math.max(0, newKappa - 0.05);

        return {
            x: newX,
            kappa: newKappa,
            level: state.level + (operatorId === 'UP' ? 1 : operatorId === 'RB' ? -1 : 0)
        };
    }

    private updateHUD(hud: HUDState, op: string, prev: StationaryUnit, next: StationaryUnit, dt: number): HUDState {
        const dx = Math.sqrt(next.x.reduce((sum, v, i) => sum + (v - prev.x[i]) ** 2, 0));
        const mu = Math.abs(next.x[0]) + next.x[1] + next.x[2] + next.x[3];

        return {
            op,
            dt: Math.round(dt),
            dx: Math.round(dx * 1000) / 1000,
            mu: Math.round(mu * 100) / 100,
            level: next.level,
            kappa: Math.round(next.kappa * 100) / 100,
            lastError: undefined
        };
    }

    private updateStatusBar() {
        if (!this.isActive) {
            this.statusBarItem.hide();
            return;
        }

        const κ = (this.hud.kappa * 100).toFixed(0);
        const level = this.hud.level >= 0 ? `+${this.hud.level}` : `${this.hud.level}`;

        this.statusBarItem.text = `🧠 T4: ${this.hud.op} | κ=${κ}% | L=${level} | Δx=${this.hud.dx}`;
        this.statusBarItem.tooltip = `Tier-4 Meta System\nLast: ${this.hud.op} (${this.hud.dt}ms)\nConfidence: ${κ}%\nLevel: ${level}\nΔx: ${this.hud.dx}`;

        if (this.hud.lastError) {
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            this.statusBarItem.tooltip += `\nError: ${this.hud.lastError}`;
        } else {
            this.statusBarItem.backgroundColor = undefined;
        }

        this.statusBarItem.show();
    }

    // ============================= HUD Webview =============================

    private toggleHUD() {
        if (this.hudWebviewPanel) {
            this.hudWebviewPanel.dispose();
            this.hudWebviewPanel = undefined;
        } else {
            this.createHUDWebview();
        }
    }

    private createHUDWebview() {
        this.hudWebviewPanel = vscode.window.createWebviewPanel(
            'tier4-hud',
            'Tier-4 Developer HUD',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        this.hudWebviewPanel.webview.html = this.getHUDWebviewContent();

        this.hudWebviewPanel.onDidDispose(() => {
            this.hudWebviewPanel = undefined;
        }, null, this.context.subscriptions);

        this.updateHUDWebview();
    }

    private getHUDWebviewContent(): string {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tier-4 HUD</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #1e1e1e; color: #cccccc; }
                .hud-container { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
                .hud-metric { background: #2d2d30; padding: 15px; border-radius: 8px; border-left: 4px solid #007acc; }
                .metric-label { font-size: 12px; color: #999; margin-bottom: 5px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #fff; }
                .state-vector { background: #2d2d30; padding: 15px; border-radius: 8px; margin-top: 20px; }
                .vector-component { display: inline-block; margin: 5px 10px; padding: 8px; background: #404040; border-radius: 4px; }
                .controls { margin-top: 20px; display: flex; gap: 10px; }
                .btn { background: #007acc; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
                .btn:hover { background: #005a9e; }
                #status { margin-top: 10px; padding: 10px; background: #0e1e0e; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h2>🧠 Tier-4 Developer HUD</h2>

            <div class="hud-container">
                <div class="hud-metric">
                    <div class="metric-label">Last Operation</div>
                    <div class="metric-value" id="last-op">-</div>
                </div>
                <div class="hud-metric">
                    <div class="metric-label">Execution Time (ms)</div>
                    <div class="metric-value" id="exec-time">0</div>
                </div>
                <div class="hud-metric">
                    <div class="metric-label">State Change (Δx)</div>
                    <div class="metric-value" id="delta-x">0.000</div>
                </div>
                <div class="hud-metric">
                    <div class="metric-label">Confidence (κ)</div>
                    <div class="metric-value" id="confidence">60%</div>
                </div>
                <div class="hud-metric">
                    <div class="metric-label">Abstraction Level</div>
                    <div class="metric-value" id="level">0</div>
                </div>
                <div class="hud-metric">
                    <div class="metric-label">System Magnitude (μ)</div>
                    <div class="metric-value" id="magnitude">0.00</div>
                </div>
            </div>

            <div class="state-vector">
                <div class="metric-label">Current State Vector [p, i, g, c]</div>
                <div>
                    <span class="vector-component">p: <span id="p-val">0.000</span></span>
                    <span class="vector-component">i: <span id="i-val">0.500</span></span>
                    <span class="vector-component">g: <span id="g-val">0.400</span></span>
                    <span class="vector-component">c: <span id="c-val">0.600</span></span>
                </div>
            </div>

            <div class="controls">
                <button class="btn" onclick="sendCommand('applyOperator')">Apply Operator</button>
                <button class="btn" onclick="sendCommand('runMacro')">Run Macro</button>
                <button class="btn" onclick="sendCommand('saveSession')">Save Session</button>
                <button class="btn" onclick="sendCommand('validate')">Validate</button>
            </div>

            <div id="status">Ready - Tier-4 system operational</div>

            <script>
                const vscode = acquireVsCodeApi();

                function sendCommand(command) {
                    vscode.postMessage({ command: command });
                }

                function updateHUD(data) {
                    document.getElementById('last-op').textContent = data.op || '-';
                    document.getElementById('exec-time').textContent = data.dt || 0;
                    document.getElementById('delta-x').textContent = (data.dx || 0).toFixed(3);
                    document.getElementById('confidence').textContent = ((data.kappa || 0) * 100).toFixed(0) + '%';
                    document.getElementById('level').textContent = data.level || 0;
                    document.getElementById('magnitude').textContent = (data.mu || 0).toFixed(2);
                }

                function updateState(state) {
                    document.getElementById('p-val').textContent = (state.x[0] || 0).toFixed(3);
                    document.getElementById('i-val').textContent = (state.x[1] || 0).toFixed(3);
                    document.getElementById('g-val').textContent = (state.x[2] || 0).toFixed(3);
                    document.getElementById('c-val').textContent = (state.x[3] || 0).toFixed(3);
                }

                window.addEventListener('message', event => {
                    const message = event.data;
                    if (message.type === 'hud-update') {
                        updateHUD(message.hud);
                        updateState(message.state);
                    }
                });
            </script>
        </body>
        </html>`;
    }

    private updateHUDWebview() {
        if (this.hudWebviewPanel) {
            this.hudWebviewPanel.webview.postMessage({
                type: 'hud-update',
                hud: this.hud,
                state: this.currentState
            });
        }
    }

    // ============================= Session Management =============================

    private async saveSession() {
        try {
            if (!fs.existsSync(this.sessionPath)) {
                fs.mkdirSync(this.sessionPath, { recursive: true });
            }

            const session = {
                state: this.currentState,
                hud: this.hud,
                timestamp: new Date().toISOString(),
                version: '1.0.0'
            };

            const sessionFile = path.join(this.sessionPath, `session-${Date.now()}.json`);
            fs.writeFileSync(sessionFile, JSON.stringify(session, null, 2));

            vscode.window.showInformationMessage(`💾 Session saved: ${path.basename(sessionFile)}`);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to save session: ${error}`);
        }
    }

    private async loadSession() {
        try {
            if (!fs.existsSync(this.sessionPath)) {
                vscode.window.showWarningMessage('No saved sessions found.');
                return;
            }

            const files = fs.readdirSync(this.sessionPath)
                .filter(f => f.endsWith('.json'))
                .sort()
                .reverse()
                .slice(0, 10); // Show last 10 sessions

            if (files.length === 0) {
                vscode.window.showWarningMessage('No saved sessions found.');
                return;
            }

            const selected = await vscode.window.showQuickPick(
                files.map(f => ({
                    label: f.replace('.json', ''),
                    detail: new Date(parseInt(f.split('-')[1].split('.')[0])).toLocaleString(),
                    value: f
                })),
                { placeHolder: 'Select session to load...' }
            );

            if (selected) {
                const sessionFile = path.join(this.sessionPath, selected.value);
                const sessionData = JSON.parse(fs.readFileSync(sessionFile, 'utf8'));

                this.currentState = sessionData.state;
                this.hud = sessionData.hud;

                this.updateStatusBar();
                this.updateHUDWebview();

                vscode.window.showInformationMessage(`📂 Loaded session: ${selected.label}`);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to load session: ${error}`);
        }
    }

    // ============================= Git Integration =============================

    private setupGitIntegration() {
        const config = vscode.workspace.getConfiguration('tier4');
        if (!config.get('gitIntegration')) return;

        // Watch for Git operations
        const gitWatcher = vscode.workspace.createFileSystemWatcher('**/.git/**');

        gitWatcher.onDidChange(() => {
            // Map Git operations to Tier-4 operators
            this.detectGitOperation();
        });

        this.context.subscriptions.push(gitWatcher);
    }

    private mapOperatorToGit(operatorId: string) {
        const config = vscode.workspace.getConfiguration('tier4');
        if (!config.get('gitIntegration')) return;

        // Map operators to Git commands
        const gitMappings: Record<string, string> = {
            ST: 'add .',
            RB: 'reset HEAD~1',
            CV: 'checkout -b',
            RS: 'checkout HEAD~1'
        };

        const gitCommand = gitMappings[operatorId];
        if (gitCommand && vscode.workspace.workspaceFolders) {
            // This would execute git command (simplified for demo)
            console.log(`Git mapping: ${operatorId} → git ${gitCommand}`);
        }
    }

    private detectGitOperation() {
        // Detect recent git operations and suggest corresponding Tier-4 operators
        // This is a placeholder for real git integration
    }

    // ============================= Demo and Validation =============================

    private openDemo() {
        const panel = vscode.window.createWebviewPanel(
            'tier4-demo',
            'Tier-4 Meta System Demo',
            vscode.ViewColumn.One,
            { enableScripts: true }
        );

        // Load demo HTML from file system
        const demoPath = path.join(this.context.extensionPath, 'resources', 'demo.html');
        if (fs.existsSync(demoPath)) {
            panel.webview.html = fs.readFileSync(demoPath, 'utf8');
        } else {
            panel.webview.html = '<h1>Demo HTML not found</h1><p>Please check extension installation.</p>';
        }
    }

    private async validateIntegration() {
        const results = [];

        try {
            // Test 1: State transformations
            const testState = { x: [0.5, 0.4, 0.3, 0.6], kappa: 0.6, level: 0 };
            const transformed = this.transformState(testState, 'RB');
            results.push(`✅ State transformation: OK`);
        } catch (error) {
            results.push(`❌ State transformation: ${error}`);
        }

        try {
            // Test 2: HUD updates
            const testHUD = this.updateHUD(this.hud, 'TEST', this.currentState, this.currentState, 100);
            results.push(`✅ HUD updates: OK`);
        } catch (error) {
            results.push(`❌ HUD updates: ${error}`);
        }

        // Show results
        const message = results.join('\n');
        if (results.every(r => r.startsWith('✅'))) {
            vscode.window.showInformationMessage(`🎉 Validation passed!\n${message}`);
        } else {
            vscode.window.showErrorMessage(`❌ Validation failed!\n${message}`);
        }
    }

    private toggleAutoPlanner() {
        const config = vscode.workspace.getConfiguration('tier4');
        const current = config.get('autoPlannerEnabled');
        config.update('autoPlannerEnabled', !current, vscode.ConfigurationTarget.Global);

        vscode.window.showInformationMessage(
            `🤖 Auto-planner ${!current ? 'ENABLED' : 'DISABLED'}`
        );
    }

    // ============================= Getters for Views =============================

    getCurrentState(): StationaryUnit {
        return this.currentState;
    }

    getHUD(): HUDState {
        return this.hud;
    }

    isSystemActive(): boolean {
        return this.isActive;
    }
}

// ============================= Tree Data Providers =============================

class Tier4StateProvider implements vscode.TreeDataProvider<string> {
    constructor(private extension: Tier4Extension) { }

    getTreeItem(element: string): vscode.TreeItem {
        const state = this.extension.getCurrentState();
        const hud = this.extension.getHUD();

        switch (element) {
            case 'confidence':
                return new vscode.TreeItem(`κ: ${(state.kappa * 100).toFixed(0)}%`, vscode.TreeItemCollapsibleState.None);
            case 'level':
                return new vscode.TreeItem(`Level: ${state.level}`, vscode.TreeItemCollapsibleState.None);
            case 'vector':
                return new vscode.TreeItem(`[${state.x.map(v => v.toFixed(3)).join(', ')}]`, vscode.TreeItemCollapsibleState.None);
            case 'lastOp':
                return new vscode.TreeItem(`Last: ${hud.op} (${hud.dt}ms)`, vscode.TreeItemCollapsibleState.None);
            default:
                return new vscode.TreeItem(element, vscode.TreeItemCollapsibleState.None);
        }
    }

    getChildren(element?: string): string[] {
        if (!element) {
            return ['confidence', 'level', 'vector', 'lastOp'];
        }
        return [];
    }
}

class Tier4OperatorProvider implements vscode.TreeDataProvider<string> {
    private operators = ['RB', 'UP', 'ST', 'PR', 'ED', 'RS', 'CV', 'SL', 'CH', 'MD'];

    getTreeItem(element: string): vscode.TreeItem {
        const item = new vscode.TreeItem(element, vscode.TreeItemCollapsibleState.None);
        item.command = {
            command: 'tier4.applyOperator',
            title: 'Apply Operator',
            arguments: [element]
        };
        return item;
    }

    getChildren(): string[] {
        return this.operators;
    }
}

class Tier4MacroProvider implements vscode.TreeDataProvider<string> {
    private macros = ['IDE_A', 'IDE_B', 'IDE_C', 'MERGE_ABC', 'OPTIMIZE', 'DEBUG', 'STABILIZE'];

    getTreeItem(element: string): vscode.TreeItem {
        const item = new vscode.TreeItem(element, vscode.TreeItemCollapsibleState.None);
        item.command = {
            command: 'tier4.runMacro',
            title: 'Run Macro',
            arguments: [element]
        };
        return item;
    }

    getChildren(): string[] {
        return this.macros;
    }
}

// ============================= Extension Activation =============================

export function activate(context: vscode.ExtensionContext) {
    console.log('🧠 Tier-4 Meta System extension is now active!');

    const tier4Extension = new Tier4Extension(context);

    // Store reference for other commands
    context.globalState.update('tier4Extension', tier4Extension);
}

export function deactivate() {
    console.log('🧠 Tier-4 Meta System extension deactivated');
}
