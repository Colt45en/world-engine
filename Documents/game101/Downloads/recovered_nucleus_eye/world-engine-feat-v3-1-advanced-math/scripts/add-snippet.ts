#!/usr/bin/env tsx
import * as fs from "fs";
import * as path from "path";
import { SnipStore } from "../src/snips/store.js";

type Argv = Record<string, string | boolean | undefined>;

function parseArgv(argv: string[]): Argv {
    const out: Argv = {};
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a.startsWith("--")) {
            const k = a.slice(2);
            const v = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
            out[k] = v;
        }
    }
    return out;
}

async function readStdin(): Promise<string> {
    return await new Promise(res => {
        let data = "";
        process.stdin.setEncoding("utf8");
        process.stdin.on("data", c => (data += c));
        process.stdin.on("end", () => res(data));
        if (process.stdin.isTTY) res(""); // no stdin
    });
}

(async () => {
    const args = parseArgv(process.argv);
    const root = (args.store as string) || process.cwd();
    const store = new SnipStore(root);

    let code = "";
    if (args.file) code = fs.readFileSync(path.resolve(String(args.file)), "utf8");
    else code = await readStdin();

    if (!code.trim()) {
        console.error("No code provided. Pipe code via stdin or pass --file path/to/code.");
        console.error("");
        console.error("Examples:");
        console.error("  # Add from file:");
        console.error("  npm run snip:add -- --file src/math.js --title 'Math utilities' --tags math,utils --lang js");
        console.error("");
        console.error("  # Add from stdin:");
        console.error("  cat src/glyph.ts | npm run snip:add -- --title 'Glyph collation' --tags glyph,morpheme --lang ts");
        process.exit(1);
    }

    const snip = await store.add({
        title: (args.title as string) || "snippet",
        tags: (args.tags as string | undefined)?.split(",").map(s => s.trim()).filter(Boolean),
        lang: (args.lang as string) || undefined,
        source: (args.source as string) || undefined,
        notes: (args.notes as string) || undefined,
        code
    });

    console.log(`✔ saved: ${snip.title} [${snip.lang ?? "n/a"}] #${snip.id}`);
    console.log(`   tags: ${snip.tags.join(", ") || "(none)"}`);
    if (snip.notes) console.log(`  notes: ${snip.notes}`);
})();
