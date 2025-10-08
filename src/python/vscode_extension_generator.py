# VS CODE EXTENSION - CONVERSATIONAL IDE INTEGRATION
# TypeScript extension that bridges the Python conversational engine with VS Code

# package.json for the extension
PACKAGE_JSON = '''
{
    "name": "conversational-ide-assistant",
    "displayName": "Conversational IDE Assistant",
    "description": "AI-powered conversational assistant for VS Code with INGEST → UNDERSTAND → PLAN → RESPOND pipeline",
    "version": "1.0.0",
    "engines": {
        "vscode": "^1.74.0"
    },
    "categories": ["Other", "Machine Learning", "Education"],
    "keywords": ["ai", "assistant", "conversation", "natural-language", "ide", "nexus"],
    "activationEvents": [
        "onCommand:conversationalIDE.startChat",
        "onLanguage:python",
        "onLanguage:typescript",
        "onLanguage:javascript",
        "onLanguage:csharp"
    ],
    "main": "./out/extension.js",
    "contributes": {
        "commands": [
            {
                "command": "conversationalIDE.startChat",
                "title": "Start Conversational Assistant",
                "category": "Conversational IDE",
                "icon": "$(comment-discussion)"
            },
            {
                "command": "conversationalIDE.processSelection",
                "title": "Process Selected Text",
                "category": "Conversational IDE",
                "icon": "$(gear)"
            },
            {
                "command": "conversationalIDE.explainCode",
                "title": "Explain This Code",
                "category": "Conversational IDE",
                "icon": "$(question)"
            },
            {
                "command": "conversationalIDE.generateCode",
                "title": "Generate Code from Description",
                "category": "Conversational IDE",
                "icon": "$(code)"
            },
            {
                "command": "conversationalIDE.debugAssist",
                "title": "Debug Assistance",
                "category": "Conversational IDE",
                "icon": "$(debug)"
            }
        ],
        "menus": {
            "editor/context": [
                {
                    "command": "conversationalIDE.explainCode",
                    "when": "editorHasSelection",
                    "group": "conversational@1"
                },
                {
                    "command": "conversationalIDE.processSelection",
                    "when": "editorHasSelection",
                    "group": "conversational@2"
                }
            ],
            "editor/title": [
                {
                    "command": "conversationalIDE.startChat",
                    "when": "resourceExtname == .py || resourceExtname == .ts || resourceExtname == .js || resourceExtname == .cs",
                    "group": "navigation"
                }
            ]
        },
        "views": {
            "explorer": [
                {
                    "id": "conversationalIDE.chatView",
                    "name": "Conversational Assistant",
                    "when": "conversationalIDE.activated"
                }
            ]
        },
        "viewsContainers": {
            "panel": [
                {
                    "id": "conversationalIDE.panel",
                    "title": "Conversational IDE",
                    "icon": "$(comment-discussion)"
                }
            ]
        },
        "configuration": {
            "title": "Conversational IDE",
            "properties": {
                "conversationalIDE.pythonPath": {
                    "type": "string",
                    "default": "python",
                    "description": "Path to Python executable for the conversational engine"
                },
                "conversationalIDE.enginePath": {
                    "type": "string",
                    "default": "",
                    "description": "Path to the conversational engine directory"
                },
                "conversationalIDE.enableTooltips": {
                    "type": "boolean",
                    "default": true,
                    "description": "Enable intelligent tooltips with conversation context"
                },
                "conversationalIDE.enableDiagnostics": {
                    "type": "boolean",
                    "default": true,
                    "description": "Enable conversational diagnostics and suggestions"
                },
                "conversationalIDE.autoActivate": {
                    "type": "boolean",
                    "default": true,
                    "description": "Automatically activate the assistant when opening supported files"
                },
                "conversationalIDE.responseStyle": {
                    "type": "string",
                    "enum": ["concise", "detailed", "teaching"],
                    "default": "detailed",
                    "description": "Preferred response style from the assistant"
                }
            }
        }
    },
    "scripts": {
        "vscode:prepublish": "npm run compile",
        "compile": "tsc -p ./",
        "watch": "tsc -watch -p ./"
    },
    "devDependencies": {
        "@types/vscode": "^1.74.0",
        "@types/node": "16.x",
        "typescript": "^4.9.4"
    },
    "dependencies": {
        "axios": "^1.4.0",
        "ws": "^8.13.0"
    }
}
'''

# Main extension TypeScript code
EXTENSION_TS = '''
// src/extension.ts - Main extension entry point
import * as vscode from 'vscode';
import { ConversationalAssistant } from './conversationalAssistant';
import { ConversationPanel } from './conversationPanel';
import { PythonEngineClient } from './pythonEngineClient';
import { DiagnosticsProvider } from './diagnosticsProvider';
import { TooltipProvider } from './tooltipProvider';

let assistant: ConversationalAssistant;
let panel: ConversationPanel | undefined;
let engineClient: PythonEngineClient;
let diagnosticsProvider: DiagnosticsProvider;
let tooltipProvider: TooltipProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('Conversational IDE Assistant is activating...');

    // Initialize the Python engine client
    engineClient = new PythonEngineClient(context);

    // Initialize core components
    assistant = new ConversationalAssistant(engineClient, context);
    diagnosticsProvider = new DiagnosticsProvider(engineClient);
    tooltipProvider = new TooltipProvider(engineClient);

    // Register commands
    const startChatCommand = vscode.commands.registerCommand('conversationalIDE.startChat', async () => {
        if (!panel) {
            panel = new ConversationPanel(context.extensionUri, assistant);
        }
        panel.show();
    });

    const processSelectionCommand = vscode.commands.registerCommand('conversationalIDE.processSelection', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);

        if (!selectedText) {
            vscode.window.showWarningMessage('No text selected');
            return;
        }

        // Process through conversation pipeline
        const result = await assistant.processText(selectedText, {
            context: 'selection',
            filePath: editor.document.uri.fsPath,
            language: editor.document.languageId
        });

        // Show result in panel
        if (!panel) {
            panel = new ConversationPanel(context.extensionUri, assistant);
        }
        panel.show();
        panel.addConversationTurn(selectedText, result.response);
    });

    const explainCodeCommand = vscode.commands.registerCommand('conversationalIDE.explainCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);

        if (!selectedText) {
            vscode.window.showWarningMessage('Please select code to explain');
            return;
        }

        const prompt = `Explain this ${editor.document.languageId} code: \\n\\n${selectedText}`;

        const result = await assistant.processText(prompt, {
            intent: 'explain',
            context: 'code_explanation',
            filePath: editor.document.uri.fsPath,
            language: editor.document.languageId
        });

        // Show explanation in hover or panel
        vscode.window.showInformationMessage(result.response.substring(0, 100) + '...', 'Show Full Explanation')
            .then(selection => {
                if (selection) {
                    if (!panel) {
                        panel = new ConversationPanel(context.extensionUri, assistant);
                    }
                    panel.show();
                    panel.addConversationTurn(prompt, result.response);
                }
            });
    });

    const generateCodeCommand = vscode.commands.registerCommand('conversationalIDE.generateCode', async () => {
        const description = await vscode.window.showInputBox({
            prompt: 'Describe the code you want to generate',
            placeHolder: 'e.g., "Create a function to validate email addresses"'
        });

        if (!description) return;

        const editor = vscode.window.activeTextEditor;
        const language = editor?.document.languageId || 'python';

        const prompt = `Generate ${language} code: ${description}`;

        const result = await assistant.processText(prompt, {
            intent: 'code_generation',
            context: 'code_request',
            language: language
        });

        if (editor) {
            // Insert generated code at cursor position
            const position = editor.selection.active;
            editor.edit(editBuilder => {
                editBuilder.insert(position, result.response);
            });
        } else {
            // Show in panel if no editor
            if (!panel) {
                panel = new ConversationPanel(context.extensionUri, assistant);
            }
            panel.show();
            panel.addConversationTurn(prompt, result.response);
        }
    });

    const debugAssistCommand = vscode.commands.registerCommand('conversationalIDE.debugAssist', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const errorDescription = await vscode.window.showInputBox({
            prompt: 'Describe the error or issue you\'re experiencing',
            placeHolder: 'e.g., "Getting TypeError on line 25"'
        });

        if (!errorDescription) return;

        const currentCode = editor.document.getText();
        const prompt = `Debug assistance needed: ${errorDescription}\\n\\nCurrent code:\\n${currentCode}`;

        const result = await assistant.processText(prompt, {
            intent: 'debug',
            context: 'debug_assistance',
            filePath: editor.document.uri.fsPath,
            language: editor.document.languageId
        });

        if (!panel) {
            panel = new ConversationPanel(context.extensionUri, assistant);
        }
        panel.show();
        panel.addConversationTurn(prompt, result.response);
    });

    // Register providers
    const tooltipDisposable = vscode.languages.registerHoverProvider(
        ['python', 'typescript', 'javascript', 'csharp'],
        tooltipProvider
    );

    // Register diagnostics for supported languages
    const diagnosticsDisposable = vscode.languages.createDiagnosticCollection('conversationalIDE');
    diagnosticsProvider.setDiagnosticCollection(diagnosticsDisposable);

    // Auto-activate on supported file types
    const config = vscode.workspace.getConfiguration('conversationalIDE');
    if (config.get('autoActivate')) {
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor && ['python', 'typescript', 'javascript', 'csharp'].includes(editor.document.languageId)) {
                vscode.commands.executeCommand('setContext', 'conversationalIDE.activated', true);
            }
        });
    }

    // Add all disposables to context
    context.subscriptions.push(
        startChatCommand,
        processSelectionCommand,
        explainCodeCommand,
        generateCodeCommand,
        debugAssistCommand,
        tooltipDisposable,
        diagnosticsDisposable
    );

    console.log('Conversational IDE Assistant activated successfully!');
}

export function deactivate() {
    if (panel) {
        panel.dispose();
    }
    if (engineClient) {
        engineClient.dispose();
    }
}
'''

# Conversation panel implementation
CONVERSATION_PANEL_TS = '''
// src/conversationPanel.ts - Web view panel for conversations
import * as vscode from 'vscode';
import { ConversationalAssistant } from './conversationalAssistant';

export class ConversationPanel {
    public static currentPanel: ConversationPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private _disposables: vscode.Disposable[] = [];

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _assistant: ConversationalAssistant
    ) {
        this._panel = vscode.window.createWebviewPanel(
            'conversationalIDE',
            'Conversational IDE Assistant',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(this._extensionUri, 'media'),
                    vscode.Uri.joinPath(this._extensionUri, 'out/compiled')
                ]
            }
        );

        this._panel.webview.html = this._getHtmlForWebview();
        this._setWebviewMessageListener();

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        ConversationPanel.currentPanel = this;
    }

    public show() {
        this._panel.reveal(vscode.ViewColumn.Beside);
    }

    public dispose() {
        ConversationPanel.currentPanel = undefined;
        this._panel.dispose();

        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }

    public addConversationTurn(userInput: string, assistantResponse: string) {
        this._panel.webview.postMessage({
            command: 'addTurn',
            userInput,
            assistantResponse,
            timestamp: new Date().toISOString()
        });
    }

    private _setWebviewMessageListener() {
        this._panel.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.command) {
                    case 'sendMessage':
                        const result = await this._assistant.processText(message.text, {
                            context: 'panel_conversation'
                        });

                        this.addConversationTurn(message.text, result.response);
                        break;

                    case 'clearConversation':
                        this._assistant.clearContext();
                        this._panel.webview.postMessage({ command: 'cleared' });
                        break;
                }
            },
            null,
            this._disposables
        );
    }

    private _getHtmlForWebview(): string {
        const scriptUri = this._panel.webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'conversationPanel.js')
        );
        const styleUri = this._panel.webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'conversationPanel.css')
        );

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Conversational IDE Assistant</title>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🧠 Conversational IDE Assistant</h2>
                    <p>INGEST → UNDERSTAND → PLAN → RESPOND</p>
                    <button id="clearButton" class="clear-btn">Clear Conversation</button>
                </div>

                <div id="conversation" class="conversation"></div>

                <div class="input-area">
                    <textarea id="messageInput" placeholder="Ask me anything about your code, request explanations, generate code, or get debugging help..."></textarea>
                    <button id="sendButton" class="send-btn">Send</button>
                </div>

                <div class="status-bar">
                    <span id="statusText">Ready for conversation...</span>
                </div>
            </div>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}
'''

# Python engine client
PYTHON_ENGINE_CLIENT_TS = '''
// src/pythonEngineClient.ts - Client for communicating with Python conversational engine
import * as vscode from 'vscode';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';

export interface ConversationResult {
    success: boolean;
    response: string;
    turn_id: string;
    understanding: {
        act: string;
        intents: string[];
        entities: any[];
        confidence: number;
    };
    plan: {
        style: string;
        sections: number;
        followup?: string;
    };
    error?: string;
}

export class PythonEngineClient {
    private _pythonProcess: ChildProcess | undefined;
    private _isInitialized = false;
    private _enginePath: string;
    private _pythonPath: string;

    constructor(private readonly _context: vscode.ExtensionContext) {
        const config = vscode.workspace.getConfiguration('conversationalIDE');
        this._pythonPath = config.get('pythonPath') || 'python';
        this._enginePath = config.get('enginePath') || this._getDefaultEnginePath();

        this._initialize();
    }

    private _getDefaultEnginePath(): string {
        // Try to find the engine relative to extension
        const extensionPath = this._context.extensionPath;
        return path.join(extensionPath, '..', 'conversational_ide_engine.py');
    }

    private async _initialize(): Promise<void> {
        if (this._isInitialized) return;

        try {
            // Test if Python and engine are available
            const testResult = await this._runPythonCommand(['-c', 'print("Python OK")']);
            if (!testResult.success) {
                throw new Error('Python not available');
            }

            // Test engine import
            const engineTest = await this._runPythonCommand([
                '-c',
                `import sys; sys.path.insert(0, "${path.dirname(this._enginePath)}"); ` +
                'from conversational_ide_engine import ConversationalIDEEngine; print("Engine OK")'
            ]);

            if (!engineTest.success) {
                vscode.window.showWarningMessage(
                    'Conversational IDE engine not found. Please configure the engine path in settings.'
                );
                return;
            }

            this._isInitialized = true;
            console.log('Python conversational engine initialized successfully');

        } catch (error) {
            console.error('Failed to initialize Python engine:', error);
            vscode.window.showErrorMessage('Failed to initialize conversational engine');
        }
    }

    public async processConversation(input: string, context?: any): Promise<ConversationResult> {
        if (!this._isInitialized) {
            await this._initialize();
            if (!this._isInitialized) {
                return {
                    success: false,
                    response: 'Conversational engine not available',
                    turn_id: '',
                    understanding: { act: '', intents: [], entities: [], confidence: 0 },
                    plan: { style: '', sections: 0 },
                    error: 'Engine not initialized'
                };
            }
        }

        try {
            const script = `
import sys
import json
sys.path.insert(0, "${path.dirname(this._enginePath)}")

from conversational_ide_engine import ConversationalIDEEngine

# Create engine instance
engine = ConversationalIDEEngine()

# Process input
input_text = """${input.replace(/"/g, '\\"')}"""
result = engine.plan_and_respond(input_text)

# Output JSON result
print("RESULT_START")
print(json.dumps(result))
print("RESULT_END")
            `;

            const result = await this._runPythonCommand(['-c', script]);

            if (!result.success) {
                return {
                    success: false,
                    response: 'Failed to process conversation',
                    turn_id: '',
                    understanding: { act: '', intents: [], entities: [], confidence: 0 },
                    plan: { style: '', sections: 0 },
                    error: result.error
                };
            }

            // Extract JSON result from output
            const output = result.output;
            const startMarker = 'RESULT_START';
            const endMarker = 'RESULT_END';

            const startIndex = output.indexOf(startMarker);
            const endIndex = output.indexOf(endMarker);

            if (startIndex === -1 || endIndex === -1) {
                throw new Error('Could not find result markers in output');
            }

            const jsonStr = output.substring(startIndex + startMarker.length, endIndex).trim();
            const conversationResult = JSON.parse(jsonStr);

            return conversationResult;

        } catch (error) {
            console.error('Error processing conversation:', error);
            return {
                success: false,
                response: 'Error processing your request',
                turn_id: '',
                understanding: { act: '', intents: [], entities: [], confidence: 0 },
                plan: { style: '', sections: 0 },
                error: error.toString()
            };
        }
    }

    private _runPythonCommand(args: string[]): Promise<{success: boolean, output: string, error?: string}> {
        return new Promise((resolve) => {
            const process = spawn(this._pythonPath, args, {
                cwd: path.dirname(this._enginePath)
            });

            let output = '';
            let errorOutput = '';

            process.stdout?.on('data', (data) => {
                output += data.toString();
            });

            process.stderr?.on('data', (data) => {
                errorOutput += data.toString();
            });

            process.on('close', (code) => {
                if (code === 0) {
                    resolve({ success: true, output });
                } else {
                    resolve({ success: false, output, error: errorOutput });
                }
            });

            process.on('error', (error) => {
                resolve({ success: false, output, error: error.message });
            });
        });
    }

    public dispose(): void {
        if (this._pythonProcess) {
            this._pythonProcess.kill();
        }
    }
}
'''

# CSS and JavaScript for the webview
WEBVIEW_CSS = '''
/* media/conversationPanel.css */
body {
    font-family: var(--vscode-font-family);
    color: var(--vscode-foreground);
    background-color: var(--vscode-editor-background);
    margin: 0;
    padding: 0;
    height: 100vh;
}

.container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 16px;
}

.header {
    border-bottom: 1px solid var(--vscode-panel-border);
    padding-bottom: 16px;
    margin-bottom: 16px;
}

.header h2 {
    margin: 0;
    color: var(--vscode-foreground);
}

.header p {
    margin: 4px 0 0 0;
    color: var(--vscode-descriptionForeground);
    font-size: 0.9em;
}

.clear-btn {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    border: none;
    padding: 6px 12px;
    border-radius: 3px;
    cursor: pointer;
    margin-top: 8px;
}

.clear-btn:hover {
    background: var(--vscode-button-secondaryHoverBackground);
}

.conversation {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 16px;
    padding: 8px;
    border: 1px solid var(--vscode-panel-border);
    border-radius: 4px;
}

.message {
    margin-bottom: 16px;
    padding: 12px;
    border-radius: 6px;
}

.message.user {
    background: var(--vscode-inputValidation-infoBackground);
    border-left: 4px solid var(--vscode-inputValidation-infoBorder);
}

.message.assistant {
    background: var(--vscode-textCodeBlock-background);
    border-left: 4px solid var(--vscode-focusBorder);
}

.message-header {
    font-weight: bold;
    margin-bottom: 8px;
    font-size: 0.9em;
}

.message-content {
    white-space: pre-wrap;
    font-family: var(--vscode-editor-font-family);
    line-height: 1.4;
}

.message-content pre {
    background: var(--vscode-textPreformat-background);
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    margin: 8px 0;
}

.input-area {
    display: flex;
    gap: 8px;
}

#messageInput {
    flex: 1;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border);
    padding: 8px;
    border-radius: 3px;
    font-family: var(--vscode-font-family);
    resize: vertical;
    min-height: 60px;
}

#messageInput:focus {
    outline: none;
    border-color: var(--vscode-focusBorder);
}

.send-btn {
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    border: none;
    padding: 8px 16px;
    border-radius: 3px;
    cursor: pointer;
    align-self: flex-end;
}

.send-btn:hover {
    background: var(--vscode-button-hoverBackground);
}

.send-btn:disabled {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    cursor: not-allowed;
}

.status-bar {
    margin-top: 8px;
    font-size: 0.8em;
    color: var(--vscode-descriptionForeground);
    text-align: center;
}
'''

WEBVIEW_JS = '''
// media/conversationPanel.js
(function() {
    const vscode = acquireVsCodeApi();

    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');
    const conversation = document.getElementById('conversation');
    const statusText = document.getElementById('statusText');

    let isProcessing = false;

    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

        const headerDiv = document.createElement('div');
        headerDiv.className = 'message-header';
        headerDiv.textContent = isUser ? '👤 You' : '🧠 Assistant';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(contentDiv);

        conversation.appendChild(messageDiv);
        conversation.scrollTop = conversation.scrollHeight;
    }

    function setProcessing(processing) {
        isProcessing = processing;
        sendButton.disabled = processing;
        sendButton.textContent = processing ? 'Processing...' : 'Send';
        statusText.textContent = processing ? 'Processing your request...' : 'Ready for conversation...';
    }

    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message || isProcessing) return;

        addMessage(message, true);
        messageInput.value = '';
        setProcessing(true);

        vscode.postMessage({
            command: 'sendMessage',
            text: message
        });
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    clearButton.addEventListener('click', () => {
        conversation.innerHTML = '';
        vscode.postMessage({ command: 'clearConversation' });
        statusText.textContent = 'Conversation cleared. Ready for new conversation...';
    });

    // Handle messages from extension
    window.addEventListener('message', event => {
        const message = event.data;

        switch (message.command) {
            case 'addTurn':
                if (isProcessing) {
                    addMessage(message.assistantResponse, false);
                    setProcessing(false);
                }
                break;

            case 'cleared':
                conversation.innerHTML = '';
                statusText.textContent = 'Ready for conversation...';
                break;
        }
    });

    // Focus input on load
    messageInput.focus();
})();
'''

def create_vscode_extension():
    """Create VS Code extension files"""

    print("🚀 Creating VS Code Extension for Conversational IDE...")
    print("=" * 60)

    # Extension files to create
    files = {
        'vscode-extension/package.json': PACKAGE_JSON,
        'vscode-extension/src/extension.ts': EXTENSION_TS,
        'vscode-extension/src/conversationPanel.ts': CONVERSATION_PANEL_TS,
        'vscode-extension/src/pythonEngineClient.ts': PYTHON_ENGINE_CLIENT_TS,
        'vscode-extension/media/conversationPanel.css': WEBVIEW_CSS,
        'vscode-extension/media/conversationPanel.js': WEBVIEW_JS
    }

    print("📁 Extension Structure:")
    for filepath in files.keys():
        print(f"   • {filepath}")

    print("\n✅ Extension files prepared!")
    print("\n🔧 Setup Instructions:")
    print("   1. Create the extension directory structure")
    print("   2. Copy the files to their respective locations")
    print("   3. Run 'npm install' in the extension directory")
    print("   4. Run 'npm run compile' to build TypeScript")
    print("   5. Press F5 in VS Code to test the extension")

    print("\n🎯 Extension Features:")
    print("   • Conversational chat panel with INGEST → UNDERSTAND → PLAN → RESPOND")
    print("   • Context menu integration for code explanation")
    print("   • Smart tooltips with conversation context")
    print("   • Code generation from natural language")
    print("   • Debug assistance with conversational AI")
    print("   • Integration with Python conversational engine")

    return files

if __name__ == "__main__":
    extension_files = create_vscode_extension()

    print("\n" + "="*80)
    print("🎉 VS CODE CONVERSATIONAL IDE EXTENSION READY!")
    print("="*80)
    print("The extension integrates the conversational AI pipeline directly into VS Code")
    print("with natural language understanding, context awareness, and intelligent responses!")
