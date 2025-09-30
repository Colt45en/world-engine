// synthetic/reader.ts
import fs from "fs";
import path from "path";

export type SynKind = "example" | "property" | "contract" | "doc" | "fixture";
export type SynBlock = { kind: SynKind; file: string; line: number; raw: string; data: Record<string, any>; };

const START = /^\/\/\s*@syn:(example|property|contract|doc|fixture)\s*$/;
const END = /^\/\/\s*---\s*$/;

export function parseSynBlocks(code: string, file: string): SynBlock[] {
    const lines = code.split(/\r?\n/);
    const blocks: SynBlock[] = [];
    let i = 0;
    while (i < lines.length) {
        const m = lines[i].match(START);
        if (!m) { i++; continue; }
        const kind = m[1] as SynKind;
        const startLine = i;
        i++;
        const buf: string[] = [];
        while (i < lines.length && !END.test(lines[i])) {
            const l = lines[i].replace(/^\/\/\s?/, ""); // strip //
            buf.push(l);
            i++;
        }
        // skip end line
        if (i < lines.length && END.test(lines[i])) i++;
        const raw = buf.join("\n").trim();
        const data = parseLooseYAML(raw);
        blocks.push({ kind, file, line: startLine + 1, raw, data });
    }
    return blocks;
}

// tiny YAML-ish parser (strings, numbers, booleans, arrays, objects)
function parseLooseYAML(src: string): any {
    // allow JSON as a fast path
    try { return JSON.parse(src); } catch { }
    const out: any = {};
    let cur: any = out;
    const stack: any[] = [];
    let curIndent = 0;

    const lines = src.split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
        const m = line.match(/^(\s*)([^:]+):\s*(.*)$/);
        if (!m) continue;
        const [, ind, kRaw, vRaw] = m;
        const key = kRaw.trim();
        const indent = ind.length;

        while (indent < curIndent && stack.length) {
            cur = stack.pop();
            curIndent -= 2;
        }
        if (vRaw === "") {
            // nested object
            const obj: any = {};
            cur[key] = obj;
            stack.push(cur);
            cur = obj;
            curIndent = indent + 2;
            continue;
        }
        cur[key] = parseScalarOrArray(vRaw.trim());
    }
    return out;
}

function parseScalarOrArray(v: string): any {
    if (/^\[.*\]$/.test(v)) {
        // simple array of scalars
        const inner = v.slice(1, -1).trim();
        if (!inner) return [];
        return inner.split(",").map(s => parseScalar(s.trim()));
    }
    return parseScalar(v);
}
function parseScalar(v: string): any {
    if (v === "true") return true;
    if (v === "false") return false;
    if (!isNaN(Number(v))) return Number(v);
    if (/^".*"$/.test(v) || /^'.*'$/.test(v)) return v.slice(1, -1);
    return v; // bare string
}

export function readProjectFiles(root: string, exts = [".ts", ".tsx", ".js", ".jsx"]) {
    const files: string[] = [];
    const walk = (dir: string) => {
        for (const f of fs.readdirSync(dir)) {
            const p = path.join(dir, f);
            const st = fs.statSync(p);
            if (st.isDirectory() && !f.startsWith('.') && f !== 'node_modules') walk(p);
            else if (exts.includes(path.extname(f))) files.push(p);
        }
    };
    walk(root);
    return files;
}
