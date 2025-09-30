/**
 * Asset Registry - Unified asset management using AssetResourceBridge + Assets Daemon
 * ==================================================================================
 *
 * Integrates:
 * - AssetResourceBridge (C++ engine) via Python bindings
 * - assets_daemon.py for NDJSON messaging
 * - Asset priority system for Monaco editor, LLE visualizations, media
 */

import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';

export interface AssetDefinition {
    type: 'monaco' | 'lle' | 'media' | 'theme' | 'audio';
    id: string;
    basePath?: string;
    priority?: number;
    memoryHint?: number; // MB
}

export interface AssetEvent {
    type: 'ASSET_EVENT';
    kind: 'loaded' | 'error';
    payload: {
        type: string;
        id: string;
        reason?: string;
    };
}

export interface AssetMessage {
    type: 'ASSET_REGISTER' | 'ASSET_MEMORY' | 'ASSET_PRELOAD' | 'ASSET_REQUEST';
    payload: any;
}

export class AssetRegistry extends EventEmitter {
    private daemon: ChildProcess | null = null;
    private isActive = false;
    private pendingRequests = new Map<string, AssetDefinition>();

    constructor(private context: vscode.ExtensionContext) {
        super();
    }

    async initialize(): Promise<void> {
        if (this.isActive) return;

        try {
            // Start assets_daemon.py subprocess
            const daemonPath = vscode.Uri.joinPath(
                this.context.extensionUri,
                '../src/optimization/assets_daemon.py'
            ).fsPath;

            this.daemon = spawn('python', [daemonPath], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: vscode.Uri.joinPath(this.context.extensionUri, '..').fsPath
            });

            if (!this.daemon.stdout || !this.daemon.stdin) {
                throw new Error('Failed to create daemon pipes');
            }

            // Setup message handling
            this.daemon.stdout.on('data', (data) => {
                const lines = data.toString().split('\n').filter(Boolean);
                for (const line of lines) {
                    try {
                        const message: AssetEvent = JSON.parse(line);
                        this.handleAssetEvent(message);
                    } catch (err) {
                        console.error('AssetRegistry: Invalid message', err);
                    }
                }
            });

            this.daemon.stderr?.on('data', (data) => {
                console.error('AssetDaemon stderr:', data.toString());
            });

            this.daemon.on('close', (code) => {
                console.log('AssetDaemon closed with code:', code);
                this.isActive = false;
            });

            this.isActive = true;

            // Register default asset paths
            await this.registerDefaultPaths();

            console.log('AssetRegistry initialized successfully');
        } catch (error) {
            console.error('Failed to initialize AssetRegistry:', error);
            throw error;
        }
    }

    private async registerDefaultPaths(): Promise<void> {
        const extensionPath = this.context.extensionUri.fsPath;

        // Register Monaco editor assets
        await this.sendMessage({
            type: 'ASSET_REGISTER',
            payload: {
                type: 'monaco',
                basePath: vscode.Uri.joinPath(this.context.extensionUri, 'media').fsPath
            }
        });

        // Register LLE visualization assets
        await this.sendMessage({
            type: 'ASSET_REGISTER',
            payload: {
                type: 'lle',
                basePath: vscode.Uri.joinPath(this.context.extensionUri, 'media').fsPath
            }
        });

        // Register media assets
        await this.sendMessage({
            type: 'ASSET_REGISTER',
            payload: {
                type: 'media',
                basePath: vscode.Uri.joinPath(this.context.extensionUri, 'media').fsPath
            }
        });
    }

    async preloadAssets(assets: AssetDefinition[]): Promise<void> {
        if (!this.isActive) await this.initialize();

        const items = assets.map(asset => ({
            type: asset.type,
            id: asset.id
        }));

        await this.sendMessage({
            type: 'ASSET_PRELOAD',
            payload: { items }
        });
    }

    async requestAsset(asset: AssetDefinition): Promise<void> {
        if (!this.isActive) await this.initialize();

        const key = `${asset.type}:${asset.id}`;
        this.pendingRequests.set(key, asset);

        await this.sendMessage({
            type: 'ASSET_REQUEST',
            payload: {
                type: asset.type,
                id: asset.id,
                priority: asset.priority || 0
            }
        });
    }

    private async sendMessage(message: AssetMessage): Promise<void> {
        if (!this.daemon?.stdin) {
            throw new Error('AssetDaemon not initialized');
        }

        return new Promise((resolve, reject) => {
            this.daemon!.stdin!.write(JSON.stringify(message) + '\n', (err) => {
                if (err) reject(err);
                else resolve();
            });
        });
    }

    private handleAssetEvent(event: AssetEvent): void {
        const key = `${event.payload.type}:${event.payload.id}`;
        const asset = this.pendingRequests.get(key);

        if (event.kind === 'loaded') {
            this.emit('assetLoaded', {
                asset,
                type: event.payload.type,
                id: event.payload.id
            });
        } else if (event.kind === 'error') {
            this.emit('assetError', {
                asset,
                type: event.payload.type,
                id: event.payload.id,
                reason: event.payload.reason
            });
        }

        this.pendingRequests.delete(key);
    }

    dispose(): void {
        if (this.daemon) {
            this.daemon.kill();
            this.daemon = null;
        }
        this.isActive = false;
    }
}

// Predefined asset definitions for Coding Pad
export const CODING_PAD_ASSETS: AssetDefinition[] = [
    // Monaco Editor Core
    { type: 'monaco', id: 'editor-core', priority: 10 },
    { type: 'monaco', id: 'python-lang', priority: 8 },
    { type: 'monaco', id: 'javascript-lang', priority: 8 },
    { type: 'monaco', id: 'typescript-lang', priority: 7 },

    // LLE Visualization
    { type: 'lle', id: 'button-palette', priority: 9 },
    { type: 'lle', id: 'state-renderer', priority: 8 },
    { type: 'lle', id: 'timeline-viz', priority: 6 },

    // Media Assets
    { type: 'media', id: 'pad-styles', priority: 9 },
    { type: 'media', id: 'lle-ui-js', priority: 8 },
    { type: 'media', id: 'pad-js', priority: 10 },

    // Theme Assets
    { type: 'theme', id: 'vscode-dark', priority: 5 },
    { type: 'theme', id: 'vscode-light', priority: 5 }
];
