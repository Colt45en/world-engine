"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = require("vscode");
function activate(context) {
    console.log('Tier-4 Meta System extension is now active!');
    const disposable = vscode.commands.registerCommand('tier4.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from Tier-4 Meta System!');
    });
    context.subscriptions.push(disposable);
}
exports.activate = activate;
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map