/**
 * VS Code "Coding Pad" Extension with Full LLE Integration + Engine Assets
 * =======================================================================
 *
 * Features:
 * - Multi-pad Monaco editor with Python/JS execution
 * - Full Lexical Logic Engine with typed operations
 * - Event-sourced memory with deterministic replay
 * - Real-time state visualization and timeline
 * - Theme synchronization with VS Code
 * - AssetResourceBridge integration for asset management
 * - Assets daemon for NDJSON messaging
 * - Tier4AudioBridge integration for audio feedback
 */

import * as vscode from 'vscode';
import { SU, click as lleClick } from './src/lle/core';
import { eye } from './src/lle/algebra';
import { InMemoryStore, SessionLog } from './src/lle/storage';
import { AssetRegistry, CODING_PAD_ASSETS } from './src/assets/AssetRegistry';
import { LLEAudioBridge, AudioFeatures, LLEAudioEvent } from './src/audio/LLEAudioBridge';

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
        vscode.commands.registerCommand('codingPad.toggleAudio', () => provider.toggleAudio()),
        vscode.commands.registerCommand('codingPad.preloadAssets', () => provider.preloadAssets()),
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

    // Engine Integration
    private assetRegistry: AssetRegistry;
    private audiobridge: LLEAudioBridge;
    private isAudioActive = false;

    // LLE Button Registry
    private btn = {
        MD: { label: 'Module', abbr: 'MD', class: 'Structure' as const, morphemes: [], M: [[1, 0, 0], [0, 0.8, 0], [0, 0, 0.8]], b: [0, 0, 0], C: eye(3), alpha: 1, beta: 0, delta_level: +1 },
        CP: { label: 'Component', abbr: 'CP', class: 'Structure' as const, morphemes: [], M: [[1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]], b: [0, 0, 0], C: eye(3), alpha: 1, beta: 0, delta_level: -1 },
        PR: { label: 'Prevent', abbr: 'PR', class: 'Constraint' as const, morphemes: [], M: eye(3), b: [0, 0, 0], C: [[1, 0, 0], [0, 0, 0], [0, 0, 1]], alpha: 1, beta: 0, delta_level: 0 },
        CV: { label: 'Convert', abbr: 'CV', class: 'Action' as const, morphemes: [], M: [[Math.SQRT1_2, -Math.SQRT1_2, 0], [Math.SQRT1_2, Math.SQRT1_2, 0], [0, 0, 1]], b: [0, 0, 0], C: eye(3), alpha: 1, beta: 0, delta_level: 0 },
        RB: { label: 'Rebuild', abbr: 'RB', class: 'Action' as const, morphemes: [], M: [[1, 0, 0], [0, 1.05, 0], [0, 0, 1.05]], b: [0, 0, 0], C: eye(3), alpha: 0.98, beta: 0, delta_level: -1 },
        UP: { label: 'Update', abbr: 'UP', class: 'Action' as const, morphemes: [], M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], b: [0.02, 0, 0], C: eye(3), alpha: 1, beta: 0, delta_level: 0 },
    } as const;

    constructor(private readonly context: vscode.ExtensionContext) {
        // Initialize engines
        this.assetRegistry = new AssetRegistry(context);
        this.audiobridge = new LLEAudioBridge();

        // Setup asset event handling
        this.assetRegistry.on('assetLoaded', (event) => {
            console.log(`Asset loaded: ${event.type}:${event.id}`);
            this.sendWebviewMessage({ type: 'assetLoaded', payload: event });
        });

        this.assetRegistry.on('assetError', (event) => {
            console.error(`Asset error: ${event.type}:${event.id} - ${event.reason}`);
            this.sendWebviewMessage({ type: 'assetError', payload: event });
        });

        // Setup audio event handling
        this.audiobridge.onAudioEvent((event: LLEAudioEvent) => {
            this.handleAudioEvent(event);
        });

        // Initialize engines asynchronously
        this.initializeEngines();
    }

    private async initializeEngines(): Promise<void> {
        try {
            await this.assetRegistry.initialize();
            await this.audiobridge.initialize();
            console.log('All engines initialized successfully');
        } catch (error) {
            console.error('Failed to initialize engines:', error);
        }
    }

    private handleAudioEvent(event: LLEAudioEvent): void {
        if (event.type === 'audio_trigger' && event.operator) {
            // Apply LLE operation triggered by audio
            const btnKey = event.operator as keyof typeof this.btn;
            if (this.btn[btnKey]) {
                this.applyLLEButton(btnKey);
                this.sendWebviewMessage({
                    type: 'audioTrigger',
                    payload: { operator: event.operator, features: event.features }
                });
            }
        }
    }

    async preloadAssets(): Promise<void> {
        try {
            await this.assetRegistry.preloadAssets(CODING_PAD_ASSETS);
            vscode.window.showInformationMessage('Assets preloaded successfully');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to preload assets: ${error}`);
        }
    }

    toggleAudio(): void {
        if (this.isAudioActive) {
            this.audiobridge.stopListening();
            this.isAudioActive = false;
            vscode.window.showInformationMessage('Audio feedback disabled');
        } else {
            this.audiobridge.startListening();
            this.isAudioActive = true;
            vscode.window.showInformationMessage('Audio feedback enabled');
        }

        this.sendWebviewMessage({
            type: 'audioStateChanged',
            payload: { active: this.isAudioActive }
        });
    }

    private sendWebviewMessage(message: any): void {
        if (this._view?.webview) {
            this._view.webview.postMessage(message);
        }
    }

    private applyLLEButton(btnKey: keyof typeof this.btn): void {
        const button = this.btn[btnKey];
        const newSU = lleClick(this.lleSU, button);

        // Store the event
        const event = {
            type: 'lleOperation' as const,
            operator: btnKey,
            prevSU: { ...this.lleSU },
            newSU: { ...newSU },
            timestamp: Date.now()
        };

        this.lleStore.store('events', event);
        this.lleLog.log(event);

        // Update current state
        this.lleSU = newSU;

        // Generate audio feedback
        if (this.isAudioActive) {
            this.audiobridge.generateLLEFeedback(newSU, btnKey);
        }

        // Send state to webview
        this.sendWebviewMessage({
            type: 'lleStateUpdate',
            payload: {
                su: newSU,
                operation: btnKey,
                event
            }
        });

        console.log(`LLE ${btnKey} operation applied:`, newSU);
    }

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

                    // Use the enhanced LLE button handling
                    this.applyLLEButton(msg.button);
                    return;
                }
                case 'audioFeatures': {
                    // Process audio features from webview
                    if (this.isAudioActive && msg.features) {
                        this.audiobridge.processAudioFeatures(msg.features);
                    }
                    return;
                }
                case 'requestAsset': {
                    // Handle asset requests from webview
                    if (msg.asset) {
                        this.assetRegistry.requestAsset(msg.asset);
                    }
                    return;
                }
                case 'audioToggle': {
                    // Handle audio toggle from webview
                    this.toggleAudio();
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
