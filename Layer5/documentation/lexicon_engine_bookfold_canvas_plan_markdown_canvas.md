# CANVAS-0 • FOUNDATION (Bookfold Frame)

> Compact frame shell only. Everything else nests inside. **Keep font small; no swrick effect—just size.**

```html
<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Lexicon Engine • Bookfold</title>
<style>html,body{margin:0;height:100%;background:#0b0e14;color:#d8e5ff;font:12px/1.25 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}*{box-sizing:border-box} :root{--gut:8px;--rail:320px;--dock:32px;--top:36px;--sm:11px;--xs:10px} .top{height:var(--top);display:flex;gap:6px;align-items:center;padding:6px 8px;border-bottom:1px solid #1a2740;background:#0e1726} .brand{font-weight:700;color:#7cfccb} .i{opacity:.7;font-size:var(--xs)} .btn{border:1px solid #27324a;background:#111927;color:#e6f2ff;border-radius:8px;padding:4px 8px;height:26px;cursor:pointer;font-size:var(--xs)} .wrap{position:absolute;inset:var(--top) 0 var(--dock) 0;padding:var(--gut)} .book{height:100%;display:grid;grid-template-columns:minmax(260px,var(--rail)) var(--gut) 1fr;grid-template-rows:1fr;grid-template-areas:"left gap right"} .left{grid-area:left;border:1px solid #22314a;border-radius:10px;background:#0f1523;overflow:auto} .right{grid-area:right;border:1px solid #22314a;border-radius:10px;background:#07080a;overflow:auto} .gap{grid-area:gap} .dock{position:absolute;left:0;right:0;bottom:0;height:var(--dock);display:flex;gap:6px;align-items:center;padding:4px 8px;border-top:1px solid #1a2740;background:#0e1726} .small{font-size:var(--sm)} .mono{font:11px/1.2 ui-monospace,SFMono-Regular,Consolas,Menlo,monospace} .row{display:flex;gap:6px;align-items:center} .pill{flex:1;display:flex;gap:6px;align-items:center;border:1px solid #22314a;border-radius:10px;padding:4px 8px;background:#0f1523} .pill input{flex:1;background:transparent;border:0;outline:0;color:#d8e5ff;font-size:var(--xs)}
/* anchors (ordered weak→strong) */
.anchor{display:block;padding:6px 8px;border-top:1px dashed #1a2740;color:#8fb6ff;font-size:var(--xs)}
</style>
<body>
  <div class=top>
    <span class=brand>NEXUS·LEX</span>
    <div class=pill><span class=i>⌘K</span><input placeholder="Command…"/></div>
    <button class=btn id=btnSave>Save</button>
    <button class=btn id=btnLoad>Load</button>
    <button class=btn id=btnFold>Fold</button>
  </div>
  <div class=wrap>
    <div class=book>
      <aside class=left id=left><!-- Canvas-L • controls --></aside>
      <div class=gap></div>
      <main class=right id=right><!-- Canvas-R • output --></main>
    </div>
  </div>
  <div class=dock>
    <button class=btn id=btnInfo>ℹ︎</button>
    <div class=i>anchors: a1…a5 • no overlap; sowing appends below</div>
  </div>
<script>/* minimal fold + storage (no innerHTML writes) */
(()=>{const $=s=>document.querySelector(s);const put=(el,html)=>{const n=document.createElement('div');n.className='small mono';n.textContent=html;el.appendChild(n)};const K='lex-bookfold-v1';btnSave.onclick=()=>localStorage.setItem(K,JSON.stringify({L:$('#left').innerHTML,R:$('#right').innerHTML}));btnLoad.onclick=()=>{try{const d=JSON.parse(localStorage.getItem(K)||'{}');if(d.L!==undefined)$('#left').innerHTML=d.L;if(d.R!==undefined)$('#right').innerHTML=d.R}catch{}};btnFold.onclick=()=>{document.documentElement.style.setProperty('--rail', getComputedStyle(document.documentElement).getPropertyValue('--rail')==='320px'?'420px':'320px')};btnInfo.onclick=()=>alert('Info tabs live in anchors; hover to expand.');window.lexAppend=(side,title,block)=>{const host=side==='L'?$('#left'):$('#right');const a=document.createElement('a');a.className='anchor';a.textContent=title;host.appendChild(a);const pre=document.createElement('pre');pre.className='mono small';pre.textContent=block;host.appendChild(pre)};window.addEventListener('error',e=>console.warn(e.message));})();
</script>
</body></html>
```

> **How to use foundation:** `lexAppend('L','Section·Title',"…content…")` appends compact blocks without HTML parsing. Add more sections by sowing new anchors (bottom-only append).

---

# CANVAS-1 • LEFT PAGE (Controls) — Foundation Layer

```text
[Left Controls]
- A1 (weak): Session • Save/Load • Fold • Compact-font
- A2: Input • Word/Corpus dropzone • Language • Tokenization
- A3: Affix Rules • Prefix/Suffix catalogs • Exceptions • Heuristics
- A4: Morphology • Stemmer select • Lemmatizer • POS tag hints
- A5 (strong): Pipelines • Run → Extract → Classify → Export
```

```js
// seed compact control sets (append-only)
lexAppend('L','A1 · Session',`save,load,fold,info`)
lexAppend('L','A2 · Input',`source: textarea|file|url\nlanguage: en|es|…\ntokenize: whitespace|wordpiece`)
lexAppend('L','A3 · Affixes',`prefixes: pre-, re-, un-, inter-, trans- …\nsuffixes: -ing,-ed,-tion,-able,-ness …\nexceptions: geo-, bio-, -ious rules`)
lexAppend('L','A4 · Morphology',`stemmer: porter|snowball\nlemma: wordnet?\npos-hints: noun|verb|adj`)
lexAppend('L','A5 · Pipeline',`run: extract→segment→score→render\nexport: json|csv|minimal`)
```

---

# CANVAS-2 • RIGHT PAGE (Output) — Foundation Layer

```text
[Right Output]
- A1 (weak): Log (compact)
- A2: Summary (counts)
- A3: Table: token • root • prefix • suffix • role • behavior
- A4: Details by token (accordion)
- A5 (strong): Exports preview
```

```js
lexAppend('R','A1 · Log',`ready.`)
lexAppend('R','A2 · Summary',`tokens:0 roots:0 unique-prefix:0 unique-suffix:0`)
lexAppend('R','A3 · Table',`token,root,prefix,suffix,role,behavior`)
lexAppend('R','A4 · Details',`(select a token)`)
lexAppend('R','A5 · Export',`(awaiting run)`)
```

---

# DATA MODEL (Compact)

```json
{
  "token":"state-of-the-art",
  "segments":["state","of","the","art"],
  "affixes":{ "prefix":[], "suffix":[], "compound":true },
  "root":"state",
  "role":"compound-noun",
  "behavior":{
    "morph":"fixed hyphenated multiword",
    "syntax":"attributive modifier",
    "semantics":"superlative-quality/modern",
    "notes":"treat as lexicalized unit; do not split during affix mining"
  }
}
```

---

# PIPELINE (Compact Pseudocode)

```pseudo
extract(text)->tokens
for t in tokens:
  if hyphenated_multiword(t) and in_lexicon(t):
    yield unit(t); continue
  (pre,stem,suf)=affix_split(t,catalog)
  pos=pos_hint(t,ctx)
  behavior=classify(stem,pre,suf,pos)
  emit{token:t,root:stem,prefix:pre,suffix:suf,role:pos,behavior}
```

---

# TESTS (Additive; do not change once added)

```csv
# token,expected.root,expected.prefix,expected.suffix,expected.role,notes
state-of-the-art,state,,,-,compound-unit
unbelievable,believe,un-, -able,adj,
reprocessing,process,re-, -ing,verb-nominal,
transnational,nation,trans-, -al,adj,
mask,mask,,,-,bare-root
```

> If expected behavior differs, tell me desired outputs; I’ll extend the rules.

---

# ANCHORS (Ordered Weak→Strong; No Overlap)

```text
A1: Session/Log (weak)
A2: Corpus/Summary
A3: Tables
A4: Details
A5: Exports (strong)
Rule: sow new info by appending to the *lowest applicable anchor*. Never rewrite prior blocks. Each block is a self-contained snapshot.
```

---

# SPACE-SAVING RULES

```text
• Font sizes: 12px base, 11px mono small, 10px hints.
• Buttons ≤ 26px height; grid/rails fixed; overflow:auto.
• No verbose prose in canvas blocks; prefer CSV/JSON.
• Use `lexAppend(side,title,block)` with newline-compressed content.
```

---

# NEXT STEPS

```text
1) Wire real tokenizer + affix catalog.
2) Implement behavior classifier (roles: compound, derivational, inflectional).
3) Render compact tables on A3; accordion on A4.
4) Local export (CSV/JSON) in A5.
5) Optional: left-rail pull-over micro-help via btnInfo.
```

