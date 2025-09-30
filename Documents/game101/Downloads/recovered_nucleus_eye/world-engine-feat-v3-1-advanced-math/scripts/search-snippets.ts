#!/usr/bin/env tsx
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

function fmt(s: string, n = 80) {
    s = s.replace(/\r\n/g, "\n").trimEnd();
    if (s.length <= n) return s;
    return s.slice(0, n - 1) + "…";
}

(async () => {
    const args = parseArgv(process.argv);
    const store = new SnipStore((args.store as string) || process.cwd());

    if (args.help) {
        console.log("Usage: npm run snip:find [options]");
        console.log("");
        console.log("Options:");
        console.log("  --text <query>    Search in title, notes, and code");
        console.log("  --lang <lang>     Filter by language");
        console.log("  --tags <t1,t2>    Filter by tags (comma-separated, all must match)");
        console.log("  --limit <n>       Limit results (default: 20)");
        console.log("  --store <path>    Snippet store directory (default: cwd)");
        console.log("");
        console.log("Examples:");
        console.log("  npm run snip:find -- --text 'glyph'");
        console.log("  npm run snip:find -- --lang js --tags math,utils");
        console.log("  npm run snip:find -- --text 'matrix' --limit 5");
        process.exit(0);
    }

    const results = await store.search({
        text: (args.text as string) || undefined,
        lang: (args.lang as string) || undefined,
        tags: (args.tags as string | undefined)?.split(",").map(s => s.trim()).filter(Boolean),
        limit: args.limit ? Number(args.limit) : undefined
    });

    if (!results.length) {
        console.log("No matches found.");
        console.log("");
        console.log("Try:");
        console.log("  npm run snip:find -- --help");
        process.exit(0);
    }

    console.log(`Found ${results.length} snippet${results.length === 1 ? '' : 's'}:\n`);

    for (const s of results) {
        console.log(`• ${s.title}  [${s.lang ?? "n/a"}]  #${s.id}`);
        if (s.tags.length) console.log(`  tags: ${s.tags.join(", ")}`);
        if (s.source) console.log(`  src:  ${s.source}`);
        if (s.notes) console.log(`  note: ${fmt(s.notes, 100)}`);
        console.log(`  code: ${fmt(s.code, 120).replace(/\n/g, "\\n")}`);
        console.log("");
    }
})();
