/**
 * World Engine Loader
 * Loads and registers knowledge packs from the manifest
 */

import { readFile } from 'fs/promises';
import { join } from 'path';

export class WorldEngineLoader {
    constructor(basePath = '.') {
        this.basePath = basePath;
        this.manifest = null;
        this.loadedPacks = new Map();
        this.services = new Map();
    }

    async loadManifest() {
        try {
            const manifestPath = join(this.basePath, 'world-engine', 'index.json');
            const manifestData = await readFile(manifestPath, 'utf8');
            this.manifest = JSON.parse(manifestData);
            console.log(`✅ Loaded manifest: ${this.manifest.name} v${this.manifest.version}`);
            return this.manifest;
        } catch (error) {
            throw new Error(`Failed to load manifest: ${error.message}`);
        }
    }

    async loadDomain(domainName) {
        if (!this.manifest) {
            await this.loadManifest();
        }

        const domain = this.manifest.domains[domainName];
        if (!domain) {
            throw new Error(`Domain '${domainName}' not found`);
        }

        console.log(`📦 Loading domain: ${domainName}`);
        const domainPacks = {};

        for (const pack of domain.packs) {
            try {
                const packData = await this.loadPack(domain.path, pack);
                domainPacks[pack.id] = packData;
                this.loadedPacks.set(`${domainName}.${pack.id}`, packData);
                console.log(`  ✓ ${pack.id}: ${pack.description}`);
            } catch (error) {
                console.warn(`  ⚠️ Failed to load ${pack.id}: ${error.message}`);
            }
        }

        return domainPacks;
    }

    async loadPack(domainPath, pack) {
        const fullPath = join(this.basePath, domainPath.substring(1)); // Remove leading slash

        if (pack.file) {
            // Single file pack
            const filePath = join(fullPath, pack.file);
            const content = await readFile(filePath, 'utf8');

            if (pack.file.endsWith('.jsonl')) {
                // JSONL format - split lines and parse each
                return content.split('\n')
                    .filter(line => line.trim())
                    .map(line => JSON.parse(line));
            } else {
                // Regular JSON
                return JSON.parse(content);
            }
        } else if (pack.path) {
            // Directory pack - load all files
            const { readdir } = await import('fs/promises');
            const packPath = join(fullPath, pack.path);
            const files = await readdir(packPath);
            const packData = {};

            for (const file of files) {
                if (file.endsWith('.json') || file.endsWith('.jsonl')) {
                    const fileName = file.replace(/\.(json|jsonl)$/, '');
                    const fileContent = await readFile(join(packPath, file), 'utf8');
                    packData[fileName] = file.endsWith('.jsonl')
                        ? fileContent.split('\n').filter(line => line.trim()).map(line => JSON.parse(line))
                        : JSON.parse(fileContent);
                }
            }

            return packData;
        }

        throw new Error(`Invalid pack format: ${pack.id}`);
    }

    async loadAllDomains() {
        if (!this.manifest) {
            await this.loadManifest();
        }

        const allDomains = {};
        const domainNames = Object.keys(this.manifest.domains);

        console.log(`🌐 Loading ${domainNames.length} domains...`);

        for (const domainName of domainNames) {
            try {
                allDomains[domainName] = await this.loadDomain(domainName);
            } catch (error) {
                console.error(`Failed to load domain ${domainName}:`, error);
            }
        }

        return allDomains;
    }

    getPack(packId) {
        return this.loadedPacks.get(packId);
    }

    getService(serviceName) {
        return this.services.get(serviceName);
    }

    registerService(name, service) {
        this.services.set(name, service);
        console.log(`🔌 Registered service: ${name}`);
    }

    getManifest() {
        return this.manifest;
    }

    getLoadedPacks() {
        return Array.from(this.loadedPacks.keys());
    }
}
