#!/usr/bin/env tsx
/**
 * Demo: Integration between Math Pro Glyph System and Snippet Storage
 * Shows how to store and retrieve mathematical expressions as snippets
 */

import { GlyphCollationMap, LLEMath, StationaryUnit, ButtonFactory } from "../src/types/Math Pro.js";
import { SnipStore } from "../src/snips/store.js";

async function demoGlyphSnippetIntegration() {
    console.log("🧮 Math Pro + Snippet Integration Demo");
    console.log("=".repeat(50));

    const glyphMap = new GlyphCollationMap();
    const snippetStore = new SnipStore();

    // 1. Store some mathematical expression snippets
    console.log("\n📝 Adding mathematical expression snippets...");

    const mathExpressions = [
        {
            title: "Matrix Multiplication",
            code: `// Basic matrix multiplication using Math Pro
const A = [[1, 2], [3, 4]];
const B = [[5, 6], [7, 8]];
const result = LLEMath.multiply(A, B);
console.log(result); // [[19, 22], [43, 50]]`,
            tags: ["matrix", "multiplication", "linear-algebra"],
            lang: "js",
            notes: "Safe matrix multiplication with shape validation"
        },
        {
            title: "Glyph to Button Conversion",
            code: `// Convert mathematical glyphs to executable buttons
const glyphMap = new GlyphCollationMap();
const button = glyphMap.createButtonFromGlyph('∑', 'Summation');
const su = new StationaryUnit(3, [1, 2, 3]);
const result = button.apply(su);`,
            tags: ["glyph", "morpheme", "transformation"],
            lang: "js",
            notes: "Transform mathematical symbols into morpheme operations"
        },
        {
            title: "Cholesky Decomposition",
            code: `// Numerically stable Cholesky with SPD enforcement
const S = [[4, 2], [2, 3]];
const safeSPD = LLEMath.ensureSPD(S);
const L = LLEMath.cholesky(safeSPD);
const solution = LLEMath.solveCholesky(S, [6, 7]);`,
            tags: ["cholesky", "linear-solver", "spd"],
            lang: "js",
            notes: "Safe positive definite matrix decomposition"
        },
        {
            title: "Greek Letter Glyph Sequence",
            code: `// Parse Greek mathematical symbols
const glyphSequence = "αβγδε"; // alpha, beta, gamma, delta, epsilon
const tokens = glyphMap.parseGlyphSequence(glyphSequence);
const buttons = glyphMap.createButtonSequence(glyphSequence);
console.log(\`Created \${buttons.length} transformation buttons\`);`,
            tags: ["greek", "sequence", "parsing"],
            lang: "js",
            notes: "Convert Greek mathematical notation to morpheme sequences"
        }
    ];

    const addedSnippets = [];
    for (const expr of mathExpressions) {
        const snippet = await snippetStore.add(expr);
        addedSnippets.push(snippet);
        console.log(`  ✔ ${snippet.title} [${snippet.lang}] #${snippet.id.slice(0, 8)}...`);
    }

    // 2. Search for snippets by mathematical concepts
    console.log("\n🔍 Searching for mathematical snippets...");

    const searches = [
        { query: { tags: ["matrix"] }, desc: "Matrix operations" },
        { query: { text: "glyph" }, desc: "Glyph-related code" },
        { query: { tags: ["linear-solver"] }, desc: "Linear algebra solvers" },
        { query: { text: "LLEMath" }, desc: "Math Pro library usage" }
    ];

    for (const { query, desc } of searches) {
        const results = await snippetStore.search(query);
        console.log(`  ${desc}: ${results.length} result${results.length === 1 ? '' : 's'}`);
        for (const result of results.slice(0, 2)) {
            console.log(`    • ${result.title} [${result.tags.join(', ')}]`);
        }
    }

    // 3. Demonstrate glyph-driven snippet generation
    console.log("\n🔮 Glyph-driven snippet generation...");

    const glyphExpressions = [
        { glyph: "∑", desc: "Summation operator" },
        { glyph: "∫", desc: "Integration/construction" },
        { glyph: "∇", desc: "Gradient/movement" },
        { glyph: "α", desc: "Alpha scaling factor" },
        { glyph: "Δ", desc: "Delta difference operator" }
    ];

    for (const { glyph, desc } of glyphExpressions) {
        const morpheme = glyphMap.getMorphemeFromGlyph(glyph);
        const cluster = glyphMap.getSemanticCluster(glyph);

        if (morpheme) {
            const codeExample = generateCodeExample(glyph, morpheme, cluster);
            const snippet = await snippetStore.add({
                title: `${glyph} (${desc})`,
                code: codeExample,
                tags: ["glyph-generated", morpheme, cluster?.name || "misc"],
                lang: "js",
                notes: `Auto-generated example for glyph ${glyph} → morpheme ${morpheme}`
            });
            console.log(`  ✔ Generated snippet for ${glyph} → ${morpheme} #${snippet.id.slice(0, 8)}...`);
        }
    }

    // 4. Show snippet statistics
    console.log("\n📊 Snippet storage statistics...");
    const allSnippets = await snippetStore.listAll(100);
    const langStats = allSnippets.reduce((acc, s) => {
        const lang = s.lang || 'unknown';
        acc[lang] = (acc[lang] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

    const tagStats = allSnippets.flatMap(s => s.tags)
        .reduce((acc, tag) => {
            acc[tag] = (acc[tag] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);

    console.log(`  Total snippets: ${allSnippets.length}`);
    console.log(`  Languages: ${Object.entries(langStats).map(([k, v]) => `${k}(${v})`).join(', ')}`);
    console.log(`  Top tags: ${Object.entries(tagStats)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([k, v]) => `${k}(${v})`)
        .join(', ')}`);

    // 5. Demonstrate search and mathematical execution
    console.log("\n⚡ Live mathematical execution from snippets...");

    const matrixResults = await snippetStore.search({ tags: ["matrix"], limit: 1 });
    if (matrixResults.length > 0) {
        const snippet = matrixResults[0];
        console.log(`  Executing: ${snippet?.title}`);
        console.log(`  Code: ${snippet?.code.split('\n')[0]}...`);

        // This would normally execute the code in a sandbox
        console.log(`  ✔ Matrix operation completed (simulated)`);
    }

    console.log("\n🎯 Demo completed!");
    console.log("Your Math Pro library now has a robust snippet management system!");
    console.log("\nNext steps:");
    console.log("  • npm run snip:add -- --file src/types/Math\\ Pro.js --title 'Math Pro Core' --tags math,core");
    console.log("  • npm run snip:find -- --tags glyph");
    console.log("  • npm run test:snips");
}

function generateCodeExample(glyph: string, morpheme: string, cluster: any): string {
    const examples = {
        multi: `// ${glyph} - Multiplication/accumulation operation
const values = [1, 2, 3, 4, 5];
const sum = values.reduce((acc, val) => acc + val, 0);
console.log('Sum:', sum);`,

        build: `// ${glyph} - Construction/building operation
const components = [[1, 0], [0, 1]];
const matrix = LLEMath.multiply(components[0], components[1]);
console.log('Built matrix:', matrix);`,

        move: `// ${glyph} - Movement/transformation operation
const position = [0, 0, 0];
const velocity = [1, 1, 0];
const newPos = LLEMath.vectorAdd(position, velocity);
console.log('New position:', newPos);`,

        scale: `// ${glyph} - Scaling operation
const vector = [2, 3, 4];
const scaleFactor = 0.5;
const scaled = LLEMath.vectorScale(vector, scaleFactor);
console.log('Scaled vector:', scaled);`,

        counter: `// ${glyph} - Opposition/difference operation
const a = [5, 3, 2];
const b = [2, 1, 1];
const diff = LLEMath.vectorSub(a, b);
console.log('Difference:', diff);`
    };

    return examples[morpheme as keyof typeof examples] || `// ${glyph} - ${morpheme} operation
const result = performOperation();
console.log('Result:', result);`;
}

// Run the demo
if (import.meta.url === `file://${process.argv[1]}`) {
    demoGlyphSnippetIntegration().catch(console.error);
}
