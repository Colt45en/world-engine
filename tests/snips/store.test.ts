import { describe, it, expect } from "vitest";
import * as fs from "fs";
import * as fsp from "fs/promises";
import * as path from "path";
import { SnipStore } from "../../src/snips/store.js";

const tmp = path.join(process.cwd(), ".tmp-snips-test");

async function resetTmp() {
    await fsp.rm(tmp, { recursive: true, force: true });
    await fsp.mkdir(tmp, { recursive: true });
}

describe("snips", () => {
    it("adds, dedupes, searches", async () => {
        await resetTmp();
        const store = new SnipStore(tmp);

        // Add first snippet
        const a = await store.add({
            title: "hello",
            tags: ["ts"],
            lang: "ts",
            code: "const a=1;\n"
        });

        // Add duplicate - should return same snippet
        const b = await store.add({
            title: "hello",
            tags: ["ts"],
            lang: "ts",
            code: "const a=1;\n"
        });
        expect(b.id).toBe(a.id); // deduped

        // Add different snippet
        const c = await store.add({
            title: "variant",
            tags: ["ts", "util"],
            lang: "ts",
            code: "const a=2;\n"
        });
        expect(a.hash === c.hash).toBe(false);

        // Search by text
        const byText = await store.search({ text: "const a=" });
        expect(byText.length).toBe(2);

        // Search by tag
        const byTag = await store.search({ tags: ["util"] });
        expect(byTag.length).toBe(1);
        expect(byTag[0]?.title).toBe("variant");

        // Test dedupe
        const d = await store.dedupe();
        const file = path.join(tmp, "snips", "snippets.ndjson");
        const lines = fs.readFileSync(file, "utf8").trim().split("\n");
        expect(lines.length).toBe(d.after);
    });

    it("handles code normalization", async () => {
        await resetTmp();
        const store = new SnipStore(tmp);

        // Different line endings and whitespace should normalize to same hash
        const code1 = "const x = 1;  \r\nconst y = 2;\r\n";
        const code2 = "const x = 1;\nconst y = 2;\n";

        const a = await store.add({ title: "test1", code: code1 });
        const b = await store.add({ title: "test2", code: code2 });

        // Should have different IDs but same hash due to normalization
        expect(a.id).not.toBe(b.id);
        expect(a.hash).toBe(b.hash);
    });

    it("searches with multiple criteria", async () => {
        await resetTmp();
        const store = new SnipStore(tmp);

        await store.add({
            title: "matrix math",
            tags: ["math", "linear"],
            lang: "js",
            code: "function multiply(a, b) { return a * b; }",
            notes: "Basic multiplication"
        });

        await store.add({
            title: "vector ops",
            tags: ["math", "vector"],
            lang: "ts",
            code: "const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);",
            notes: "Vector dot product"
        });

        await store.add({
            title: "glyph mapping",
            tags: ["glyph", "morpheme"],
            lang: "ts",
            code: "const glyphMap = new Map();",
            notes: "Symbol to morpheme mapping"
        });

        // Search by language
        const jsResults = await store.search({ lang: "js" });
        expect(jsResults.length).toBe(1);
        expect(jsResults[0]?.title).toBe("matrix math");

        // Search by multiple tags
        const mathResults = await store.search({ tags: ["math"] });
        expect(mathResults.length).toBe(2);

        // Search by text in notes
        const vectorResults = await store.search({ text: "dot product" });
        expect(vectorResults.length).toBe(1);
        expect(vectorResults[0]?.title).toBe("vector ops");

        // Combined search
        const combinedResults = await store.search({
            text: "multiply",
            lang: "js",
            tags: ["math"]
        });
        expect(combinedResults.length).toBe(1);
    });
});
