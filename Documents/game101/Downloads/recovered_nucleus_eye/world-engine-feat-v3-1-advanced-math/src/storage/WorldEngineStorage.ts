// OPFS → ZIP → Manifest workflow as specified in master plan
export interface StorageConfig {
    opfsRoot: string;
    zipBundleSize: number; // MB
    manifestVersion: string;
}

export interface ManifestEntry {
    id: string;
    path: string;
    size: number;
    created: number;
    modified: number;
    checksum: string;
    tags: string[];
}

export interface StorageManifest {
    version: string;
    created: number;
    entries: ManifestEntry[];
    bundles: {
        id: string;
        path: string;
        entries: string[];
        compressed: boolean;
    }[];
}

export class WorldEngineStorage {
    private opfsRoot?: FileSystemDirectoryHandle;
    private manifest: StorageManifest | null = null;

    constructor(private config: StorageConfig = {
        opfsRoot: 'world-engine-data',
        zipBundleSize: 10, // 10MB bundles
        manifestVersion: '1.0.0'
    }) { }

    async initialize(): Promise<void> {
        try {
            // Get OPFS root
            this.opfsRoot = await navigator.storage.getDirectory();
            const dataDir = await this.opfsRoot.getDirectoryHandle(this.config.opfsRoot, {
                create: true
            });

            // Load or create manifest
            await this.loadManifest();

            console.log('WorldEngineStorage initialized:', {
                opfs: !!this.opfsRoot,
                manifest: !!this.manifest,
                fsaSupport: 'showDirectoryPicker' in window
            });
        } catch (error) {
            console.error('Storage initialization failed:', error);
            throw error;
        }
    }

    private async loadManifest(): Promise<void> {
        try {
            if (!this.opfsRoot) throw new Error('OPFS not initialized');

            const dataDir = await this.opfsRoot.getDirectoryHandle(this.config.opfsRoot);
            const manifestFile = await dataDir.getFileHandle('manifest.json');
            const file = await manifestFile.getFile();
            const text = await file.text();

            this.manifest = JSON.parse(text);
        } catch (error) {
            // Create new manifest if none exists
            this.manifest = {
                version: this.config.manifestVersion,
                created: Date.now(),
                entries: [],
                bundles: []
            };
            await this.saveManifest();
        }
    }

    private async saveManifest(): Promise<void> {
        if (!this.opfsRoot || !this.manifest) return;

        try {
            const dataDir = await this.opfsRoot.getDirectoryHandle(this.config.opfsRoot);
            const manifestFile = await dataDir.getFileHandle('manifest.json', { create: true });
            const writable = await manifestFile.createWritable();

            await writable.write(JSON.stringify(this.manifest, null, 2));
            await writable.close();
        } catch (error) {
            console.error('Failed to save manifest:', error);
        }
    }

    async storeFile(path: string, data: Blob | ArrayBuffer | string, tags: string[] = []): Promise<string> {
        if (!this.opfsRoot || !this.manifest) throw new Error('Storage not initialized');

        try {
            const dataDir = await this.opfsRoot.getDirectoryHandle(this.config.opfsRoot);

            // Convert data to blob
            const blob = data instanceof Blob ? data : new Blob([data]);

            // Generate unique ID
            const id = crypto.randomUUID();
            const filePath = `files/${id}`;

            // Create file path directories
            const pathParts = filePath.split('/');
            let currentDir = dataDir;

            for (let i = 0; i < pathParts.length - 1; i++) {
                currentDir = await currentDir.getDirectoryHandle(pathParts[i], { create: true });
            }

            // Write file
            const fileHandle = await currentDir.getFileHandle(pathParts[pathParts.length - 1], {
                create: true
            });
            const writable = await fileHandle.createWritable();
            await writable.write(blob);
            await writable.close();

            // Calculate checksum (simple hash of content)
            const arrayBuffer = await blob.arrayBuffer();
            const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
            const checksum = Array.from(new Uint8Array(hashBuffer))
                .map(b => b.toString(16).padStart(2, '0'))
                .join('');

            // Add to manifest
            const entry: ManifestEntry = {
                id,
                path,
                size: blob.size,
                created: Date.now(),
                modified: Date.now(),
                checksum,
                tags
            };

            this.manifest.entries.push(entry);
            await this.saveManifest();

            return id;
        } catch (error) {
            console.error('Failed to store file:', error);
            throw error;
        }
    }

    async retrieveFile(id: string): Promise<Blob | null> {
        if (!this.opfsRoot || !this.manifest) throw new Error('Storage not initialized');

        try {
            const entry = this.manifest.entries.find(e => e.id === id);
            if (!entry) return null;

            const dataDir = await this.opfsRoot.getDirectoryHandle(this.config.opfsRoot);
            const fileHandle = await dataDir.getFileHandle(entry.path);
            const file = await fileHandle.getFile();

            return file;
        } catch (error) {
            console.error('Failed to retrieve file:', error);
            return null;
        }
    }

    async createZipBundle(entryIds: string[]): Promise<string> {
        if (!this.manifest) throw new Error('Storage not initialized');

        // This would use a ZIP library like JSZip in practice
        // For now, creating a mock bundle entry
        const bundleId = crypto.randomUUID();
        const bundlePath = `bundles/${bundleId}.zip`;

        this.manifest.bundles.push({
            id: bundleId,
            path: bundlePath,
            entries: entryIds,
            compressed: true
        });

        await this.saveManifest();
        return bundleId;
    }

    async exportToFSA(): Promise<void> {
        if (!('showDirectoryPicker' in window)) {
            throw new Error('File System Access API not supported');
        }

        try {
            const dirHandle = await (window as any).showDirectoryPicker();

            if (!this.manifest) throw new Error('No manifest to export');

            // Export manifest
            const manifestFile = await dirHandle.getFileHandle('manifest.json', { create: true });
            const manifestWritable = await manifestFile.createWritable();
            await manifestWritable.write(JSON.stringify(this.manifest, null, 2));
            await manifestWritable.close();

            console.log('Data exported to file system');
        } catch (error) {
            console.error('Export failed:', error);
            throw error;
        }
    }

    getStorageStats(): {
        totalFiles: number;
        totalSize: number;
        bundles: number;
        opfsSupported: boolean;
        fsaSupported: boolean;
    } {
        return {
            totalFiles: this.manifest?.entries.length || 0,
            totalSize: this.manifest?.entries.reduce((sum, entry) => sum + entry.size, 0) || 0,
            bundles: this.manifest?.bundles.length || 0,
            opfsSupported: 'storage' in navigator && 'getDirectory' in navigator.storage,
            fsaSupported: 'showDirectoryPicker' in window
        };
    }

    async cleanup(): Promise<void> {
        // Perform cleanup tasks like removing old files, compressing bundles, etc.
        if (!this.manifest) return;

        const now = Date.now();
        const oldEntries = this.manifest.entries.filter(
            entry => now - entry.created > 30 * 24 * 60 * 60 * 1000 // 30 days
        );

        // In practice, would also clean up actual files
        console.log(`Cleanup: found ${oldEntries.length} old entries`);
    }
}

// Global instance
export const worldEngineStorage = new WorldEngineStorage();
