import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('Tier-4 Meta System extension is now active!');

    const disposable = vscode.commands.registerCommand('tier4.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from Tier-4 Meta System!');
    });

    context.subscriptions.push(disposable);
}

export function deactivate() { }
