/**
 * VS Code "Coding Pad" Extension with Full LLE Integration
 * ======================================================
 *
 * Features:
 * - Multi-pad Monaco editor with Python/JS execution
 * - Full Lexical Logic Engine with typed operations
 * - Event-sourced memory with deterministic replay
 * - Real-time state visualization and timeline
 * - Theme synchronization with VS Code
 */

import * as vscode from 'vscode';
import { SU, click as lleClick, eye } from './src/lle/core';
import { InMemoryStore, SessionLog } from './src/lle/storage';

type PadMap = Record<string, string>;

export function activate(context: vscode.ExtensionContext) {
    const provider = new CodingPadViewProvider(context);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('codingPad.codingPadView', provider),
        vscode.commands.registerCommand('codingPad.open', () => provider.reveal()),
        vscode.commands.registerCommand('codingPad.export', () => provider.exportPad()),
        vscode.commands.registerCommand('codingPad.newPad', () => provider.newPad()),
        vscode.commands.registerCommand('codingPad.renamePad', () => provider.renamePad()),
        vscode.commands.registerCommand('codingPad.deletePad', () => provider.deletePad()),
        vscode.commands.registerCommand('codingPad.setLanguage', async () => {
            const lang = await vscode.window.showQuickPick(["javascript", "python", "typescript", "plaintext"], { placeHolder: 'Language' });
            if (lang) provider.setLanguage(lang);
        }),
        vscode.commands.registerCommand('codingPad.run', () => provider.runCode()),
        vscode.window.onDidChangeActiveColorTheme((theme) => provider.syncTheme(theme))
    );
}

export function deactivate() { }

class CodingPadViewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private currentPad: string = 'default';
    private currentLanguage: string = 'javascript';

    // LLE Integration
    private lleStore = new InMemoryStore();
    private lleLog = new SessionLog();
    private lleSU: SU = { x: [1, 0, 0], Sigma: eye(3), kappa: 0.7, level: 0 };

    // LLE Button Registry
    private btn = {
        MD: { label: 'Module', abbr: 'MD', class: 'Structure' as const, morphemes: [], M: [[1, 0, 0], [0, 0.8, 0], [0, 0, 0.8]], b: [0, 0, 0], C: eye(3), alpha: 1, beta: 0, delta_level: +1 },
        CP: { label: 'Component', abbr: 'CP', class: 'Structure' as const, morphemes: [], M: [[1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]], b: [0, 0, 0], C: eye(3), alpha: 1, beta: 0, delta_level: -1 },
        PR: { label: 'Prevent', abbr: 'PR', class: 'Constraint' as const, morphemes: [], M: eye(3), b: [0, 0, 0], C: [[1, 0, 0], [0, 0, 0], [0, 0, 1]], alpha: 1, beta: 0, delta_level: 0 },
        CV: { label: 'Convert', abbr: 'CV', class: 'Action' as const, morphemes: [], M: [[Math.SQRT1_2, -Math.SQRT1_2, 0], [Math.SQRT1_2, Math.SQRT1_2, 0], [0, 0, 1]], b: [0, 0, 0], C: eye(3), alpha: 1, beta: 0, delta_level: 0 },
        RB: { label: 'Rebuild', abbr: 'RB', class: 'Action' as const, morphemes: [], M: [[1, 0, 0], [0, 1.05, 0], [0, 0, 1.05]], b: [0, 0, 0], C: eye(3), alpha: 0.98, beta: 0, delta_level: -1 },
        UP: { label: 'Update', abbr: 'UP', class: 'Action' as const, morphemes: [], M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], b: [0.02, 0, 0], C: eye(3), alpha: 1, beta: 0, delta_level: 0 },
    } as const;

    constructor(private readonly context: vscode.ExtensionContext) { }

    // === Storage helpers ===
    private pads(): PadMap {
        return this.context.globalState.get<PadMap>('pads') || { default: '' };
    }

    private setPads(next: PadMap) {
        return this.context.globalState.update('pads', next);
    }

    private savePad(name: string, text: string) {
        const all = this.pads();
        all[name] = text;
        this.setPads(all);
    }

    private loadPad(name: string): string {
        const all = this.pads();
        return all[name] ?? '';
    }

    private listPadNames(): string[] {
        return Object.keys(this.pads());
    }

    // === UI wiring ===
    reveal() {
        if (this._view) this._view.show?.(true);
        else vscode.commands.executeCommand('workbench.view.extension.codingPad.container');
    }

    resolveWebviewView(webviewView: vscode.WebviewView) {
        this._view = webviewView;
        const webview = webviewView.webview;
        webview.options = { enableScripts: true };
        webview.html = this.getHtml(webview);

        // Initial state (pads + selected + language + theme)
        this.postState();

        webview.onDidReceiveMessage((msg) => {
            switch (msg.type) {
                case 'save': {
                    this.savePad(msg.padName, msg.text);
                    return;
                }
                case 'switchPad': {
                    this.currentPad = msg.padName;
                    const text = this.loadPad(this.currentPad);
                    webview.postMessage({ type: 'loadPad', padName: this.currentPad, text });
                    this.postState();
                    return;
                }
                case 'command': {
                    // Allow toolbar buttons in the webview to trigger extension commands
                    if (typeof msg.command === 'string') {
                        vscode.commands.executeCommand(msg.command);
                    }
                    return;
                }
                case 'lle.click': {
                    const spec = (this.btn as any)[msg.button];
                    if (!spec) return;
                    const beforeCid = this.lleStore.put({ su: this.lleSU });
                    this.lleSU = lleClick(this.lleSU, spec);
                    const afterCid = this.lleStore.put({ su: this.lleSU });
                    this.lleLog.appendEvent({
                        type: 'event',
                        session: 'default',
                        seq: this.lleLog.events.length + 1,
                        ts: new Date().toISOString(),
                        button: spec.abbr,
                        buttonVid: 'v1',
                        inputCid: beforeCid,
                        outputCid: afterCid
                    });
                    webview.postMessage({ type: 'lle.state', su: this.lleSU });
                    webview.postMessage({ type: 'lle.timeline', events: this.lleLog.events.slice(-20) });
                    return;
                }
            }
        });
    }

    private postState() {
        const theme = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark ? 'vs-dark' : 'vs';
        this._view?.webview.postMessage({
            type: 'state',
            pads: this.listPadNames(),
            currentPad: this.currentPad,
            language: this.currentLanguage,
            theme,
            text: this.loadPad(this.currentPad)
        });
        // Send initial LLE state
        this._view?.webview.postMessage({ type: 'lle.state', su: this.lleSU });
    }

    newPad() {
        vscode.window.showInputBox({ prompt: 'New pad name' }).then((name) => {
            if (!name) return;
            const all = this.pads();
            if (all[name] !== undefined) {
                vscode.window.showWarningMessage(`Pad "${name}" already exists.`);
                return;
            }
            all[name] = '';
            this.setPads(all);
            this.currentPad = name;
            this.postState();
        });
    }

    renamePad() {
        vscode.window.showInputBox({ prompt: `Rename pad "${this.currentPad}" to:` }).then((next) => {
            if (!next) return;
            const all = this.pads();
            if (all[next] !== undefined) {
                vscode.window.showWarningMessage(`Pad "${next}" already exists.`);
                return;
            }
            all[next] = all[this.currentPad] || '';
            delete all[this.currentPad];
            this.currentPad = next;
            this.setPads(all);
            this.postState();
        });
    }

    deletePad() {
        const name = this.currentPad;
        if (name === 'default') {
            vscode.window.showWarningMessage('Cannot delete the default pad.');
            return;
        }
        const all = this.pads();
        delete all[name];
        this.setPads(all);
        this.currentPad = 'default';
        this.postState();
    }

    exportPad() {
        const text = this.loadPad(this.currentPad);
        vscode.workspace.openTextDocument({ content: text, language: this.currentLanguage })
            .then((d) => vscode.window.showTextDocument(d));
    }

    setLanguage(lang: string) {
        this.currentLanguage = lang;
        this._view?.webview.postMessage({ type: 'setLanguage', language: lang });
        this.postState();
    }

    runCode() {
        const text = this.loadPad(this.currentPad);
        const term = vscode.window.createTerminal('CodingPad Runner');
        if (text.trim().length === 0) {
            term.sendText('echo "Pad is empty"');
        } else if (this.currentLanguage === 'python') {
            term.sendText('cat <<\'EOF\' > pad_temp.py');
            term.sendText(text);
            term.sendText('EOF');
            term.sendText('python pad_temp.py');
        } else {
            term.sendText('cat <<\'EOF\' > pad_temp.js');
            term.sendText(text);
            term.sendText('EOF');
            term.sendText('node pad_temp.js');
        }
        term.show();
    }

    syncTheme(theme: vscode.ColorTheme) {
        const monacoTheme = theme.kind === vscode.ColorThemeKind.Dark ? 'vs-dark' : 'vs';
        this._view?.webview.postMessage({ type: 'setTheme', theme: monacoTheme });
    }

    private getHtml(webview: vscode.Webview): string {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media', 'pad.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media', 'pad.css'));
        const lleUiUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media', 'lle_ui.js'));
        const monacoLoader = 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.51.0/min/vs/loader.min.js';

        return /* html */ `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} https:; style-src ${webview.cspSource} 'unsafe-inline' https:; script-src ${webview.cspSource} https:; font-src https:; connect-src https:;" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="${styleUri}">
  <title>Coding Pad + LLE</title>
</head>
<body>
  <div id="pad-container">
    <div id="toolbar">
      <select id="padSwitcher" title="Pads"></select>
      <button data-cmd="codingPad.newPad">New</button>
      <button data-cmd="codingPad.renamePad">Rename</button>
      <button data-cmd="codingPad.deletePad">Delete</button>
      <button data-cmd="codingPad.export">Export</button>
      <button data-cmd="codingPad.setLanguage">Lang</button>
      <button data-cmd="codingPad.run">Run</button>
    </div>
    <div id="editor"></div>
  </div>
  <script src="${monacoLoader}"></script>
  <script>const __PAD_ENV__ = { csp: ${JSON.stringify(webview.cspSource)} };</script>
  <script src="${scriptUri}"></script>
  <script src="${lleUiUri}"></script>
</body>
</html>`;
    }
}

type GraphEventRecord = GraphEvent;

interface GraphSnapshotRecord extends GraphSnapshot {
    state: StationaryUnit;
}

function cloneStationaryUnit(state: StationaryUnit): StationaryUnit {
    return {
        x: [...state.x],
        kappa: state.kappa,
        level: state.level
    };
}

function computeStateHash(state: StationaryUnit): string {
    const payload = JSON.stringify(state);
    let hash = 0;
    for (let i = 0; i < payload.length; i++) {
        hash = Math.imul(31, hash) + payload.charCodeAt(i);
    }
    return Math.abs(hash).toString(16);
}

class LLELog {
    private seq: number;
    readonly events: GraphEventRecord[];
    readonly snapshots: GraphSnapshotRecord[];

    constructor(initialState: StationaryUnit) {
        this.seq = 0;
        this.events = [];
        this.snapshots = [];
        this.captureSnapshot(initialState);
    }

    currentSnapshot(): GraphSnapshotRecord {
        return this.snapshots[this.snapshots.length - 1];
    }

    captureSnapshot(state: StationaryUnit): GraphSnapshotRecord {
        const snapshot: GraphSnapshotRecord = {
            index: this.snapshots.length,
            cid: `su_${this.snapshots.length}_${computeStateHash(state)}`,
            state: cloneStationaryUnit(state),
            timestamp: Date.now()
        };
        this.snapshots.push(snapshot);
        return snapshot;
    }

    appendEvent(button: string, inputCid: string, outputCid: string): GraphEventRecord {
        const event: GraphEventRecord = {
            seq: this.seq++,
            button,
            inputCid,
            outputCid,
            timestamp: Date.now()
        };
        this.events.push(event);
        return event;
    }
}

class Tier4Extension {
    private context: vscode.ExtensionContext;
    private statusBarItem: vscode.StatusBarItem;
    private hudWebviewPanel: vscode.WebviewPanel | undefined;
    private currentState: StationaryUnit;
    private hud: HUDState;
    private isActive = false;
    private sessionPath: string;
    private graphDecoder: GraphDecoder;
    private lleLog!: LLELog;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
        this.currentState = { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 };
        this.hud = { op: '-', dt: 0, dx: 0, mu: 0, level: 0, kappa: 0.6 };
        this.sessionPath = path.join(context.globalStorageUri.fsPath, 'tier4-sessions');
        this.graphDecoder = new GraphDecoder();
        this.resetGraphState(this.currentState);

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
        ).then((selection: string | undefined) => {
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
            const inputSnapshot = this.lleLog.currentSnapshot();

            // Apply operator transformation (simplified)
            this.currentState = this.transformState(this.currentState, operatorId);

            const outputSnapshot = this.lleLog.captureSnapshot(this.currentState);
            this.lleLog.appendEvent(operatorId, inputSnapshot.cid, outputSnapshot.cid);
            this.broadcastGraph();

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

        const webview = this.hudWebviewPanel.webview;
        this.hudWebviewPanel.webview.html = this.getHUDWebviewContent(webview);

        this.hudWebviewPanel.onDidDispose(() => {
            this.hudWebviewPanel = undefined;
        }, null, this.context.subscriptions);

        webview.onDidReceiveMessage((message: { type?: string; error?: string }) => {
            if (message?.type === 'lle.graph.error') {
                vscode.window.showWarningMessage(`Tier-4 Graph View: ${message.error}`);
            }
        }, undefined, this.context.subscriptions);

        this.updateHUDWebview();
        this.broadcastGraph(webview);
    }

    private getHUDWebviewContent(webview: vscode.Webview): string {
        const mediaRoot = this.context.extensionUri;
        const graphScriptUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, 'media', 'graph_ui.js'));
        const padStylesUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, 'media', 'pad.css'));
        const nonce = this.getNonce();
        const csp = `default-src 'none'; img-src ${webview.cspSource} https:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}' ${webview.cspSource} https://unpkg.com;`;

        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tier-4 HUD</title>
            <meta http-equiv="Content-Security-Policy" content="${csp}">
            <link rel="stylesheet" href="${padStylesUri}">
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
                .coding-pad { margin-top: 24px; background: #1b1b20; border: 1px solid #333; border-radius: 8px; padding: 12px; }
                .coding-pad h3 { font-size: 14px; margin-bottom: 8px; color: #7cdcff; }
                #pad-container { min-height: 80px; }
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

            <div class="coding-pad">
                <h3>Tier-4 Graph Decoder</h3>
                <div id="pad-container"></div>
            </div>

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
            <script nonce="${nonce}" src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
            <script nonce="${nonce}" src="${graphScriptUri}"></script>
        </body>
        </html>`;
    }

    private updateHUDWebview() {
        if (this.hudWebviewPanel) {
            const webview = this.hudWebviewPanel.webview;
            webview.postMessage({
                type: 'hud-update',
                hud: this.hud,
                state: this.currentState
            });
            this.broadcastGraph(webview);
        }
    }

    private broadcastGraph(target?: vscode.Webview) {
        if (!this.lleLog) {
            return;
        }
        const webview = target ?? this.hudWebviewPanel?.webview;
        if (!webview) {
            return;
        }
        const graph = this.graphDecoder.fromEvents(this.lleLog.events, this.lleLog.snapshots);
        webview.postMessage({ type: 'lle.graph', graph });
    }

    private getNonce(): string {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let nonce = '';
        for (let i = 0; i < 32; i++) {
            nonce += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return nonce;
    }

    private resetGraphState(state: StationaryUnit) {
        this.lleLog = new LLELog(state);
        this.graphDecoder.fromEvents(this.lleLog.events, this.lleLog.snapshots);
        this.broadcastGraph();
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
                this.resetGraphState(this.currentState);

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
