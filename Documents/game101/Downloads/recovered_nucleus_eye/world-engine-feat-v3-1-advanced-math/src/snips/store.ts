import * as fs from "fs";
import * as fsp from "fs/promises";
import * as path from "path";
import * as crypto from "crypto";
import * as readline from "readline";
import { Snippet, validateSnippet } from "./schema.js";

export type AddInput = {
    title?: string;
    tags?: string[];
    lang?: string;
    source?: string;
    notes?: string;
    code: string;
    id?: string;
};

export type SearchQuery = {
    text?: string;   // substring match across title/notes/code (lowercased)
    tags?: string[]; // all must be contained
    lang?: string;   // exact match
    limit?: number;  // default 20
};

/** Simple cross-platform lock using an atomic directory create. */
class FileLock {
    constructor(private dir: string) { }

    async acquire(timeoutMs = 2000, stepMs = 50): Promise<void> {
        const start = Date.now();
        // eslint-disable-next-line no-constant-condition
        while (true) {
            try {
                await fsp.mkdir(this.dir, { recursive: false });
                return;
            } catch {
                if (Date.now() - start > timeoutMs) {
                    throw new Error(`Lock timeout: ${this.dir}`);
                }
                await new Promise(r => setTimeout(r, stepMs));
            }
        }
    }

    async release(): Promise<void> {
        await fsp.rm(this.dir, { recursive: true, force: true });
    }
}

export class SnipStore {
    private file: string;
    private lock: FileLock;

    constructor(rootDir = process.cwd()) {
        const dir = path.resolve(rootDir, "snips");
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        this.file = path.join(dir, "snippets.ndjson");
        this.lock = new FileLock(path.join(dir, ".lock"));
        if (!fs.existsSync(this.file)) fs.writeFileSync(this.file, "");
    }

    static normalizeCode(code: string): string {
        // normalize line endings + strip trailing whitespace per line + ensure trailing newline
        return code.replace(/\r\n/g, "\n").replace(/[ \t]+\n/g, "\n").trim() + "\n";
    }

    static sha256(text: string): string {
        return crypto.createHash("sha256").update(text).digest("hex");
    }

    /** Streaming iterator over all snippets; tolerant of bad lines. */
    async *iterAll(): AsyncGenerator<Snippet> {
        const stream = fs.createReadStream(this.file, { encoding: "utf8" });
        const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

        for await (const line of rl) {
            if (!line) continue;
            try {
                const obj = JSON.parse(line);
                yield validateSnippet(obj);
            } catch {
                // skip malformed rows
            }
        }
    }

    /** Read all snippets into memory (only used where sorting is required). */
    private async readAll(): Promise<Snippet[]> {
        const out: Snippet[] = [];
        for await (const s of this.iterAll()) out.push(s);
        return out;
    }

    /** Atomic write of the whole file. */
    private async writeAllAtomic(snips: Snippet[]): Promise<void> {
        const tmp = `${this.file}.tmp-${process.pid}-${Math.random().toString(36).slice(2)}`;
        await fsp.writeFile(
            tmp,
            snips.map(s => JSON.stringify(s)).join("\n") + (snips.length ? "\n" : ""),
            "utf8"
        );
        await fsp.rename(tmp, this.file);
    }

    /** Add snippet with dedupe (hash + title) and return the stored record. */
    async add(input: AddInput): Promise<Snippet> {
        const now = new Date().toISOString();
        const codeN = SnipStore.normalizeCode(input.code);
        const hash = SnipStore.sha256(codeN);
        const id = input.id ?? `${Date.now().toString(36)}-${hash.slice(0, 8)}`;

        const candidate: Snippet = validateSnippet({
            id,
            title: input.title?.trim() || "snippet",
            tags: (input.tags ?? []).map(t => t.trim()).filter(Boolean),
            lang: input.lang?.trim(),
            source: input.source?.trim(),
            notes: input.notes?.trim(),
            code: codeN,
            hash,
            createdAt: now,
            updatedAt: now
        });

        await this.lock.acquire();
        try {
            // check for exact dupe
            for await (const s of this.iterAll()) {
                if (s.hash === candidate.hash && s.title === candidate.title) {
                    return s;
                }
            }
            await fsp.appendFile(this.file, JSON.stringify(candidate) + "\n", "utf8");
            return candidate;
        } finally {
            await this.lock.release();
        }
    }

    /** Search streaming; collects matches then sorts by createdAt desc, title asc. */
    async search(q: SearchQuery): Promise<Snippet[]> {
        const limit = q.limit ?? 20;
        const needle = q.text?.toLowerCase();
        const tagSet = q.tags?.map(t => t.toLowerCase());
        const lang = q.lang;

        const matches: Snippet[] = [];
        for await (const s of this.iterAll()) {
            if (lang && s.lang !== lang) continue;
            if (tagSet && tagSet.length && !tagSet.every(t => s.tags.some(st => st.toLowerCase() === t))) continue;
            if (needle) {
                const hay = [s.title, s.notes ?? "", s.code].join("\n").toLowerCase();
                if (!hay.includes(needle)) continue;
            }
            matches.push(s);
        }

        matches.sort((a, b) => (b.createdAt.localeCompare(a.createdAt)) || a.title.localeCompare(b.title));
        return matches.slice(0, limit);
    }

    /** Dedupe by content hash (keep first occurrence). */
    async dedupe(): Promise<{ before: number; after: number }> {
        await this.lock.acquire();
        try {
            const all = await this.readAll();
            const seen = new Map<string, Snippet>();
            for (const s of all) if (!seen.has(s.hash)) seen.set(s.hash, s);
            const kept = Array.from(seen.values());
            await this.writeAllAtomic(kept);
            return { before: all.length, after: kept.length };
        } finally {
            await this.lock.release();
        }
    }

    async listAll(limit = 100): Promise<Snippet[]> {
        return this.search({ limit });
    }
}
