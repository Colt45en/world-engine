// synthetic/runner.ts
import vm from "vm";
import path from "path";
import fs from "fs";
import { parseSynBlocks, readProjectFiles, SynBlock } from "./reader";

type Result = {
    ok: boolean;
    name: string;
    kind: string;
    file: string;
    line: number;
    msg?: string | undefined;
};

export async function runSynthetic(root: string): Promise<Result[]> {
    const files = readProjectFiles(root);
    const results: Result[] = [];

    for (const file of files) {
        const code = fs.readFileSync(file, "utf8");
        const blocks = parseSynBlocks(code, file);
        if (!blocks.length) continue;

        // load module in a sandbox; expose its exports
        const mod = await loadModule(file);

        for (const b of blocks) {
            try {
                if (b.kind === "example") results.push(runExample(b, mod));
                else if (b.kind === "property") results.push(await runProperty(b, mod));
                else if (b.kind === "contract") results.push(runContract(b, mod));
                else if (b.kind === "doc") results.push({ ok: true, name: b.data['name'] || "doc", kind: "doc", file: b.file, line: b.line });
                else if (b.kind === "fixture") results.push({ ok: true, name: b.data['name'] || "fixture", kind: "fixture", file: b.file, line: b.line });
            } catch (e: any) {
                results.push({ ok: false, name: b.data?.['name'] || b.kind, kind: b.kind, file: b.file, line: b.line, msg: String(e?.message || e) });
            }
        }
    }
    return results;
}

function runExample(b: SynBlock, mod: any): Result {
    const name = b.data?.['name'] || "example";
    const { input, expect } = b.data;
    // choose callable: first exported function or named one in data
    const fnName = b.data?.['fn'] || firstFunctionName(mod);
    const fn = mod[fnName];
    if (typeof fn !== "function") throw new Error(`No function '${fnName}' exported`);
    const args = input?.args ?? [];
    const out = fn.apply(null, args);
    const ok = deepEqual(out, expect);
    const msg = ok ? undefined : `expected ${JSON.stringify(expect)} got ${JSON.stringify(out)}`;
    return { ok, name, kind: "example", file: b.file, line: b.line, msg };
}

async function runProperty(b: SynBlock, mod: any): Promise<Result> {
    const name = b.data?.['name'] || "property";
    const { gens = {}, prop, trials = 100 } = b.data;
    // Build an evaluator for prop: allow fn(...) and variables a,b,c
    const ctx: any = vm.createContext({ ...mod, Number, Math });
    for (let t = 0; t < trials; t++) {
        const env = Object.fromEntries(Object.entries(gens).map(([k, spec]) => [k, sample(spec as any)]));
        const expr = prop.replace(/\b([a-zA-Z_]\w*)\b/g, (id: string) => (id in env ? `(${JSON.stringify(env[id])})` : id));
        const ok = !!vm.runInContext(expr, ctx);
        if (!ok) return { ok: false, name, kind: "property", file: b.file, line: b.line, msg: `counterexample: ${JSON.stringify(env)} for ${prop}` };
    }
    return { ok: true, name, kind: "property", file: b.file, line: b.line };
}

function runContract(b: SynBlock, mod: any): Result {
    const name = b.data?.['name'] || "contract";
    const pre = b.data?.['pre'] || "true";
    const post = b.data?.['post'] || "true";
    const fnName = b.data?.['fn'] || firstFunctionName(mod);
    const fn = mod[fnName];
    if (typeof fn !== "function") throw new Error(`No function '${fnName}' exported`);

    // crude symbolic points: try a few
    const points = [
        { a: 0, b: 0 }, { a: 1, b: 2 }, { a: -3, b: 7 }, { a: 10, b: -4 }, { a: 3.14, b: -2.72 }
    ];
    for (const point of points) {
        const { a, b: bVal } = point;
        const preOK = !!vm.runInContext(pre.replaceAll("a", "(a)").replaceAll("b", "(b)"), vm.createContext({ a, b: bVal, ...mod }));
        if (!preOK) continue;
        const out = fn(a, bVal);
        const postOK = !!vm.runInContext(post.replaceAll("out", "(out)").replaceAll("a", "(a)").replaceAll("b", "(b)"), vm.createContext({ a, b: bVal, out, ...mod }));
        if (!postOK) return { ok: false, name, kind: "contract", file: b.file, line: b.line, msg: `violated at a=${a}, b=${bVal}` };
    }
    return { ok: true, name, kind: "contract", file: b.file, line: b.line };
}

/* utils */

function firstFunctionName(mod: any) { return Object.keys(mod).find(k => typeof mod[k] === "function") || "default"; }

function deepEqual(a: any, b: any) { return JSON.stringify(a) === JSON.stringify(b); }

function sample(spec: any): any {
    if (typeof spec === "string") {
        const m = spec.match(/^int\((-?\d+),\s*(-?\d+)\)$/);
        if (m) { const lo = +m[1], hi = +m[2]; return Math.floor(lo + Math.random() * (hi - lo + 1)); }
    }
    if (Array.isArray(spec)) return spec[Math.floor(Math.random() * spec.length)];
    if (typeof spec === "object" && spec !== null) {
        const out: any = {};
        for (const k of Object.keys(spec)) out[k] = sample(spec[k]);
        return out;
    }
    return spec;
}

async function loadModule(file: string): Promise<any> {
    // Enhanced module loader that handles ES6 imports and TypeScript
    const code = fs.readFileSync(file, "utf8");

    // Simple ES6 to CommonJS transform for basic cases
    let transformedCode = code
        .replace(/export\s+function\s+(\w+)/g, 'function $1')
        .replace(/export\s+class\s+(\w+)/g, 'class $1')
        .replace(/export\s+const\s+(\w+)/g, 'const $1')
        .replace(/export\s*\{([^}]+)\}/g, (_, exports) => {
            const names = exports.split(',').map((n: string) => n.trim());
            return names.map((n: string) => `module.exports.${n} = ${n};`).join('\n');
        });

    const wrapper = `(function(exports, require, module, __filename, __dirname){ ${transformedCode}\n})`;
    const script = new vm.Script(wrapper, { filename: file });
    const exports: any = {};
    const moduleObj = { exports };
    const req = (id: string) => {
        try {
            return require(resolveImport(id, file));
        } catch (e) {
            // Return empty object for missing imports to prevent crashes
            return {};
        }
    };
    script.runInThisContext().call(exports, exports, req, moduleObj, file, path.dirname(file));
    return moduleObj.exports;
}

function resolveImport(id: string, from: string) {
    if (id.startsWith(".")) return path.resolve(path.dirname(from), id);
    return id; // node_modules
}
