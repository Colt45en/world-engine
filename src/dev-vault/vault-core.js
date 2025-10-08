/**
 * RNES VectorLab Dev Vault - Secure sandboxed development environment
 * - Signed bundle verification with ECDSA P-256
 * - Per-file SHA-256 integrity checking
 * - Sandboxed blob imports with allow-list
 * - Stealth UI with long-press and Shift×5 reveal
 * - Cross-browser file picker fallback
 */

/*** Utilities ***/
const u8 = s => new TextEncoder().encode(s);
const toHex = buf => [...new Uint8Array(buf)].map(v => v.toString(16).padStart(2, '0')).join('');
const b64uToBuf = (b64u) => {
    const b64 = b64u.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(b64u.length / 4) * 4, '=');
    const str = atob(b64); const a = new Uint8Array(str.length);
    for (let i = 0; i < str.length; i++) a[i] = str.charCodeAt(i);
    return a.buffer;
};
const sha256 = async (buf) => toHex(await crypto.subtle.digest('SHA-256', buf));

/*** Public key — pin alg & curve ***/
const PUB_JWK = {
    kty: "EC", crv: "P-256", alg: "ES256", ext: true,
    // Replace with your real 32-byte coords (base64url, no padding)
    x: "fQvKXQy5a9cfm5fN4o9gXb9r71o0a0kZ0Zq3sYx2N1M",
    y: "6G8r2Yq2qfahfDGg0X5Y2WmOeN3n5kI3x3y3Jp1r7sQ"
};
const pubKeyPromise = crypto.subtle.importKey(
    "jwk", PUB_JWK, { name: "ECDSA", namedCurve: "P-256" }, true, ["verify"]
);

/*** Canonical JSON (stable key sort) ***/
const sortObj = (o) => {
    if (Array.isArray(o)) return o.map(sortObj);
    if (o && typeof o === 'object') {
        return Object.keys(o).sort().reduce((acc, k) => { acc[k] = sortObj(o[k]); return acc; }, {});
    }
    return o;
};
const canonicalJSONString = (o) => JSON.stringify(sortObj(o));

/*** IndexedDB wrapper ***/
const DB = (() => {
    let db;
    const open = () => new Promise((res, rej) => {
        const r = indexedDB.open("RNES_VAULT", 1);
        r.onupgradeneeded = () => {
            const store = r.result.createObjectStore("files");
            // Create additional stores for Math Pro integration
            r.result.createObjectStore("snippets");
            r.result.createObjectStore("glyphs");
        };
        r.onsuccess = () => { db = r.result; res(); };
        r.onerror = () => rej(r.error);
    });
    const put = (k, v, store = "files") => new Promise((res, rej) => {
        const tx = db.transaction(store, "readwrite"); tx.objectStore(store).put(v, k);
        tx.oncomplete = res; tx.onerror = () => rej(tx.error);
    });
    const get = (k, store = "files") => new Promise((res, rej) => {
        const tx = db.transaction(store, "readonly"); const rq = tx.objectStore(store).get(k);
        rq.onsuccess = () => res(rq.result); rq.onerror = () => rej(rq.error);
    });
    const list = (store = "files") => new Promise((res, rej) => {
        const tx = db.transaction(store, "readonly");
        const rq = tx.objectStore(store).getAllKeys();
        rq.onsuccess = () => res(rq.result); rq.onerror = () => rej(rq.error);
    });
    return { open, put, get, list };
})();

/*** Pak ingest with signature & allow-list ***/
const ALLOW = new Set([
    "script.mjs", "shader.frag", "notes.md",
    "math-pro.js", "glyph-map.json", "morpheme-config.js",
    "scene-config.json", "vector-lab.js"
]);

async function importPak(jsonText) {
    const statusEl = document.getElementById('vaultStatus');
    const status = s => statusEl.textContent = s;

    try {
        status("verifying…");
        const pak = JSON.parse(jsonText);

        // Verify signature (DER ECDSA over canonical(entries))
        if (!pak || typeof pak !== 'object' || !pak.entries || !pak.signature) {
            throw new Error("bad pak structure");
        }

        const canonical = canonicalJSONString(pak.entries);
        const sigBuf = b64uToBuf(pak.signature); // DER-encoded ECDSA
        const key = await pubKeyPromise;
        const ok = await crypto.subtle.verify(
            { name: "ECDSA", hash: "SHA-256" }, key, sigBuf, u8(canonical)
        );
        if (!ok) throw new Error("signature invalid");

        // Per-file hashes + allow-list
        const importedFiles = [];
        for (const [name, meta] of Object.entries(pak.entries)) {
            if (!ALLOW.has(name)) throw new Error(`blocked file: ${name}`);
            if (!meta || typeof meta !== 'object' || !meta.sha256 || !meta.data) {
                throw new Error(`bad entry: ${name}`);
            }

            const buf = b64uToBuf(meta.data);
            const hash = await sha256(buf);
            if (hash.toLowerCase() !== String(meta.sha256).toLowerCase()) {
                throw new Error(`hash mismatch: ${name}`);
            }

            await DB.put(name, buf);
            importedFiles.push(name);

            // Special handling for Math Pro files
            if (name === 'glyph-map.json') {
                await DB.put('glyph-map', buf, 'glyphs');
            }
        }

        status(`imported ${importedFiles.length} files`);

        // Preview notes
        const notesBuf = await DB.get("notes.md");
        if (notesBuf) {
            const notesEl = document.getElementById('notes');
            if (notesEl) notesEl.value = new TextDecoder().decode(notesBuf);
        }

        return { success: true, files: importedFiles };
    } catch (err) {
        status(`error: ${err.message}`);
        throw err;
    }
}

/*** Load & run hidden script with Math Pro integration ***/
let __devLoop = { id: 0, last: 0 };

async function runHiddenScript() {
    const status = s => document.getElementById('vaultStatus').textContent = s;
    status("loading…");

    const buf = await DB.get("script.mjs");
    if (!buf) {
        status("no script");
        return;
    }

    const blobUrl = URL.createObjectURL(new Blob([buf], { type: "text/javascript" }));

    try {
        // Teardown previous
        window.__hiddenDev = window.__hiddenDev || {};
        cancelAnimationFrame(__devLoop.id);
        window.__hiddenDev.mod?.teardown?.();

        const mod = await import(/* @vite-ignore */ blobUrl);
        window.__hiddenDev.mod = mod;

        // Initialize with Math Pro context
        const context = {
            THREE: window.THREE,
            scene: window.scene,
            renderer: window.renderer,
            // Math Pro integration
            LLEMath: window.LLEMath,
            GlyphCollationMap: window.GlyphCollationMap,
            MathPro: window.MathPro,
            // Vault access
            vault: {
                getFile: (name) => DB.get(name),
                putFile: (name, data) => DB.put(name, data),
                listFiles: () => DB.list(),
                getGlyph: (name) => DB.get(name, 'glyphs'),
                putGlyph: (name, data) => DB.put(name, data, 'glyphs')
            }
        };

        mod.init?.(context);

        // Start update loop if exposed
        __devLoop.last = performance.now();
        const tick = (t) => {
            const dt = (t - __devLoop.last) / 1000;
            __devLoop.last = t;
            window.__hiddenDev.mod?.update?.(dt, t / 1000);
            __devLoop.id = requestAnimationFrame(tick);
        };
        __devLoop.id = requestAnimationFrame(tick);

        status("running");
    } finally {
        URL.revokeObjectURL(blobUrl);
    }
}

/*** Stealth UI (long-press; Shift×5; Escape) ***/
function initStealthUI() {
    const panel = document.getElementById('vaultPanel');
    const btnArea = document.getElementById('vaultBtn');
    let pressTimer, shiftCount = 0, lastShift = 0;

    const togglePanel = (show) => {
        const isShowing = show ?? !panel.classList.contains('show');
        panel.classList.toggle('show', isShowing);
        panel.setAttribute('aria-hidden', isShowing ? 'false' : 'true');
    };

    // Long press detection
    btnArea.addEventListener('pointerdown', () => {
        pressTimer = setTimeout(() => togglePanel(), 800);
    });

    const cancelPress = () => clearTimeout(pressTimer);
    btnArea.addEventListener('pointerup', cancelPress);
    btnArea.addEventListener('pointerleave', cancelPress);
    btnArea.addEventListener('pointercancel', cancelPress);

    // Shift×5 and Escape hotkeys
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Shift') {
            const now = performance.now();
            shiftCount = (now - lastShift < 600) ? (shiftCount + 1) : 1;
            lastShift = now;
            if (shiftCount >= 5) {
                togglePanel(true);
                shiftCount = 0;
            }
        }
        if (e.key === 'Escape') togglePanel(false);
    });

    return { togglePanel };
}

/*** Cross-browser file picker ***/
const pickFile = async () => {
    if ('showOpenFilePicker' in window) {
        const handles = await window.showOpenFilePicker({
            types: [{
                description: 'Vault Paks',
                accept: { 'application/json': ['.pak', '.json'] }
            }]
        });
        return handles[0].getFile();
    } else {
        return new Promise((res) => {
            const inp = document.createElement('input');
            inp.type = 'file';
            inp.accept = '.pak,.json,application/json';
            inp.onchange = () => res(inp.files?.[0] ?? null);
            inp.click();
        });
    }
};

/*** Math Pro Integration Utilities ***/
const MathProVault = {
    async saveGlyphMapping(glyphMap) {
        const data = JSON.stringify(glyphMap.exportMappings());
        await DB.put('glyph-mappings', u8(data), 'glyphs');
        return { success: true };
    },

    async loadGlyphMapping() {
        const buf = await DB.get('glyph-mappings', 'glyphs');
        if (!buf) return null;
        return JSON.parse(new TextDecoder().decode(buf));
    },

    async saveSnippet(snippet) {
        const key = `snippet_${snippet.id}`;
        await DB.put(key, u8(JSON.stringify(snippet)), 'snippets');
        return { success: true };
    },

    async exportMathProPak() {
        const files = await DB.list();
        const glyphs = await DB.list('glyphs');
        const snippets = await DB.list('snippets');

        const manifest = {
            version: "1.0.0",
            type: "math-pro-vault",
            timestamp: new Date().toISOString(),
            files: files.length,
            glyphs: glyphs.length,
            snippets: snippets.length
        };

        return {
            manifest,
            files: await Promise.all(files.map(async f => ({
                name: f,
                data: await DB.get(f)
            }))),
            glyphs: await Promise.all(glyphs.map(async g => ({
                name: g,
                data: await DB.get(g, 'glyphs')
            }))),
            snippets: await Promise.all(snippets.map(async s => ({
                name: s,
                data: await DB.get(s, 'snippets')
            })))
        };
    }
};

// Export the vault system
export {
    DB,
    importPak,
    runHiddenScript,
    initStealthUI,
    pickFile,
    MathProVault,
    canonicalJSONString,
    sha256
};
