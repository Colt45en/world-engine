// synthetic/cli.ts
import http from "http";
import { runSynthetic } from "./runner";

function renderHTML(results: any[]) {
    const ok = results.filter(r => r.ok).length, total = results.length;
    const rows = results.map(r => `
    <div class="row ${r.ok ? "ok" : "fail"}">
      <b>[${r.kind}]</b> ${escape(r.name)} — <span>${escape(r.file)}:${r.line}</span>
      ${r.msg ? `<div class="msg">${escape(r.msg)}</div>` : ""}
    </div>`).join("");
    return `<!doctype html><meta charset="utf-8"><style>
    body{margin:0;background:#0b1418;color:#cfe;font:12px ui-monospace}
    .head{padding:8px;border-bottom:1px solid #244;background:#0f1523}
    .row{padding:8px;border-bottom:1px solid #223;transition:background 0.2s}
    .row:hover{background:#0f1523}
    .ok{color:#54f0b8} .fail{color:#ff6b6b}
    .msg{opacity:.85;margin-top:4px;font-style:italic}
    b{color:#9fd6ff}
    span{opacity:0.7;font-size:11px}
  </style>
  <div class="head">🧮 Synthetic Results: ${ok}/${total} passing</div>
  ${rows}`;
}

function escape(s: string) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] as string)); }

async function main() {
    const root = process.argv[2] || process.cwd();
    console.log(`🔍 Scanning ${root} for synthetic directives...`);

    const results = await runSynthetic(root);
    const html = renderHTML(results);

    // 1) print to stdout (CI)
    const passing = results.filter(r => r.ok).length;
    console.log(`\n🧮 Synthetic: ${passing}/${results.length} passing`);

    if (results.length === 0) {
        console.log('💡 No synthetic directives found. Add some //@syn: blocks to your code!');
        return;
    }

    // Show failures
    const failures = results.filter(r => !r.ok);
    if (failures.length > 0) {
        console.log('\n❌ Failures:');
        failures.forEach(f => {
            console.log(`   [${f.kind}] ${f.name} at ${f.file}:${f.line}`);
            if (f.msg) console.log(`      ${f.msg}`);
        });
    }

    // 2) start ephemeral server for IDE integration
    const srv = http.createServer((req, res) => {
        if (req.url === "/synthetic") {
            res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
            res.end(html);
            return;
        }
        res.writeHead(404); res.end("not found");
    });

    srv.listen(7077, () => {
        console.log('\n🌐 Synthetic UI available at: http://localhost:7077/synthetic');
        console.log('💡 Add this to your World Engine Studio for live results!');
    });

    // Keep alive for 5 minutes
    setTimeout(() => {
        console.log('\n⏰ Synthetic server timeout. Run again to refresh.');
        srv.close();
        process.exit(0);
    }, 300000);
}

main().catch(e => { console.error('💥 Synthetic error:', e); process.exit(1); });
