
# World Engine Studio — Bottom Sheet (Editable Plan)

> This canvas is the **planning sheet**. We’ll mark every section “back by back,” edit until it’s right, then lock Layer 1. Nothing here auto-runs. It’s for alignment and review.

---

## 0) Working Agreements

- **Two-pass process**: (A) Plan on this sheet → (B) Ship Layer 1 (locked, idempotent) → (C) Only then Layer 2.
- **No duplication**: single installer symbol; re-entry is a no-op when same/newer version is present.
- **Boundary guards**: freeze/seal exports; read-only globals; no `document.write` or eval; text-only events.
- **Persistence**: namespaced keys; JSON-safe; external store pass-through.
- **Accessibility**: keyboard focus rings, controls keyboardable (applies in UI components later).

---

## 1) Version & Guard (Header)

**Objective**: One global symbol `__STUDIO_BRIDGE__` holds `{version, installedAt}`. If already present with same or newer semver, skip init.

**Notes**:

- Semver: `0.1.0-base` for Layer 1; bump minor for non-breaking, major for breaking.
- Banner log at init: `[Studio:init] vX.Y.Z transport=...` (once).

**Editable fields**:

- Package name: **studio-bridge**
- Initial version: **0.1.0-base**

---

## 2) Immutable Config & Constants

**Objective**: Centralize literals; freeze them. Examples:

- `BUS_NAME = 'studio-bus'`
- `NAMESPACE = 'studio:'`
- Message types: `'eng.run'|'eng.test'|'eng.result'|'eng.status'|'rec.start'|'rec.stop'|'rec.clip'|'rec.transcript'|'rec.mark'|'chat.cmd'|'chat.announce'`

**Open questions**:

- Any additional command verbs? (e.g., `/export`, `/reset`)

---

## 3) Event Bus (BroadcastChannel + fallback)

**Objective**: Reliable intra-tab/inter-tab routing.

**API**:

- `onBus(fn) -> off()` — multi-listener, safe fan-out, returns unsubscribe.
- `sendBus(msg)` — delivers to channel or window event fallback.

**Guards**:

- Validate `msg.type` is string.
- Try/catch listener calls; isolated failures don’t crash the bus.

---

## 4) Store (Namespaced, JSON-safe)

**Objective**: Simple async facade with external-store passthrough.

**API**:

- `Store.save(key, value)` → `externalStore.upsert` or `localStorage.setItem(NAMESPACE+key, JSON.stringify(value))`
- `Store.load(key)` → `externalStore.get` or parsed `localStorage` value.

**Guards**:

- JSON parse errors handled; return `null` on failure.
- Size-awareness (log a warning when value > \~1MB).

---

## 5) Utils

**Objective**: small helpers, no heavy deps.

- `generateId()` → prefer `crypto.randomUUID()`, fallback to time+rand.
- `parseCommand(line)` → supports `/run`, `/test`, `/rec start|screen|stop`, `/mark`, `/status`, `/history`, `/clear`, `/help`; else treat as `/run`.
- `log(msg, level='info')` → ISO timestamp; respects `debug=studio` for verbose levels.

---

## 6) Engine Transport

**Objective**: Safely operate same-origin iframe; warn on cross-origin.

**API**:

- `setupEngineTransport(iframe) -> { isSameOrigin(), withEngine(fn) }`
- `withEngine(fn)` calls fn(doc) or warns (no-op) if cross-origin.

**Guards**:

- Try/catch DOM access.
- No DOM mutation outside allowed selectors (`#input`, `#run`, `#out`, `#tests`).

---

## 7) Public APIs (Recorder / Engine / Chat)

**Objective**: Small surface, bus-only communication. Methods are non-writable, non-configurable.

- `RecorderAPI`: `startMic(meta)`, `startScreen(meta)`, `startBoth(meta)`, `stop()`, `mark(tag, runId)`
- `EngineAPI`: `run(text)`, `test(name)`, `getStatus()`
- `ChatAPI`: `command(line)`, `announce(message, level)`

**Guards**:

- Coerce strings; validate enums for `mode`.

---

## 8) Bridge Export

**Objective**: Expose a single object, and CommonJS mirror for bundlers.

- `window.StudioBridge = { onBus, sendBus, RecorderAPI, EngineAPI, ChatAPI, Store, Utils, setupEngineTransport }`
- Freeze the bridge object and each member function.
- CJS: `module.exports = ...` if present.

---

## 9) Security / Boundary Guards

- Install-once guard using version compare.
- `Object.freeze` and `Object.defineProperty` (non-configurable) on exports.
- No `document.write`, no `eval`, no Function constructor.
- Input sanitation: ensure `.type` is string; drop messages otherwise.
- Defensive timers (no unbounded `setInterval`).

---

## 10) Diagnostics & Telemetry (non-PHI)

- One-time banner log with version and channel type.
- Optional `debug=studio` query flips verbose logging.

---

## 11) Message Contracts (quick reference)

- `eng.run`: `{type, text}`
- `eng.test`: `{type, name}`
- `eng.result`: `{type, runId, outcome, input?}`
- `rec.start`: `{type, mode:'mic'|'screen'|'both', meta?}`
- `rec.stop`: `{type}`
- `rec.clip`: `{type, clipId, url?, meta?, size?}`
- `rec.transcript`: `{type, clipId?, text, ts}`
- `rec.mark`: `{type, tag, runId?}`
- `chat.cmd`: `{type, line}`
- `chat.announce`: `{type, message, level}`

---

## 12) Storage Keys (namespaced)

- `runs.<runId>` — run record
- `clips.<clipId>` — clip record
- `marks.<markId>` — marker record
- `wordEngine.lastRun` — last parsed output

All prefixed under `studio:` in `localStorage`.

---

## 13) Test Plan (Layer 1)

- **Idempotency**: load script twice → second load logs skip.
- **Bus fallback**: disable `BroadcastChannel` to force `window` fallback.
- **Store**: save/load round-trip, JSON error handling.
- **Transport**: same-origin vs cross-origin path exercised.
- **Freeze**: attempting to overwrite `StudioBridge` fails silently with warning.

---

## 14) Rollback Plan

- Layer 1 is additive, non-breaking. To rollback, bump patch and re-ship with guards; previous instances ignore lower versions.

---

## 15) Layer 1 Scaffold (Draft — to be finalized before locking)

```js
(function(){
  'use strict';

  // ===== 1) Version & Install Guard =====
  var NAME = 'studio-bridge';
  var VERSION = '0.1.0-base';

  if (typeof window.__STUDIO_BRIDGE__ === 'object') {
    try {
      var existing = window.__STUDIO_BRIDGE__;
      if (existing && existing.version && compareSemver(existing.version, VERSION) >= 0) {
        // Already installed with same/newer version → no-op
        return; // idempotent
      }
    } catch(_) {}
  }

  Object.defineProperty(window, '__STUDIO_BRIDGE__', { value: { version: VERSION, installedAt: Date.now() }, writable: false, configurable: false, enumerable: true });

  // ===== 2) Config & Constants =====
  var CONST = Object.freeze({ BUS_NAME: 'studio-bus', NAMESPACE: 'studio:' });

  // ===== 3) Bus =====
  var bc = ('BroadcastChannel' in self) ? new BroadcastChannel(CONST.BUS_NAME) : null;
  var listeners = new Set();
  function _fan(msg){
    if (!msg || typeof msg.type !== 'string') return; // boundary
    listeners.forEach(function(fn){ try{ fn(msg); }catch(e){ console.warn('[Studio:bus-listener]', e); } });
  }
  if (bc) bc.onmessage = function(e){ _fan(e.data); };
  else window.addEventListener('studio:msg', function(e){ _fan(e.detail); });

  function onBus(fn){ listeners.add(fn); return function off(){ listeners.delete(fn); }; }
  function sendBus(msg){ if (!msg || typeof msg.type !== 'string') return; if (bc) bc.postMessage(msg); else window.dispatchEvent(new CustomEvent('studio:msg', { detail: msg })); }

  // ===== 4) Store =====
  var Store = Object.freeze({
    save: function(key, value){
      try{
        if (window.externalStore && typeof window.externalStore.upsert === 'function') return Promise.resolve(window.externalStore.upsert(key, value));
        if (window.localStorage){ localStorage.setItem(CONST.NAMESPACE+key, JSON.stringify(value)); return Promise.resolve(value); }
      }catch(e){ return Promise.reject(e); }
      return Promise.reject(new Error('No storage available'));
    },
    load: function(key){
      try{
        if (window.externalStore && typeof window.externalStore.get === 'function') return Promise.resolve(window.externalStore.get(key));
        if (window.localStorage){ var raw = localStorage.getItem(CONST.NAMESPACE+key); return Promise.resolve(raw ? JSON.parse(raw) : null); }
      }catch(e){ return Promise.resolve(null); }
      return Promise.resolve(null);
    }
  });

  // ===== 5) Utils =====
  function compareSemver(a,b){
    var pa=a.split('-')[0].split('.').map(Number), pb=b.split('-')[0].split('.').map(Number);
    for(var i=0;i<3;i++){ var d=(pa[i]||0)-(pb[i]||0); if(d) return d; }
    return 0;
  }
  var Utils = Object.freeze({
    generateId: function(){ if (crypto && crypto.randomUUID) return crypto.randomUUID(); return String(Date.now())+Math.random().toString(36).slice(2,10); },
    parseCommand: function(line){ var t=(line||'').trim(); if(t.startsWith('/run ')) return {type:'run', args:t.slice(5)}; if(t.startsWith('/test ')) return {type:'test', args:t.slice(6)}; if(t==='/rec start') return {type:'rec', args:'start'}; if(t==='/rec screen') return {type:'rec', args:'screen'}; if(t==='/rec stop') return {type:'rec', args:'stop'}; if(t.startsWith('/mark ')) return {type:'mark', args:t.slice(6)}; if(t==='/status') return {type:'status', args:''}; if(t==='/history') return {type:'history', args:''}; if(t==='/clear') return {type:'clear', args:''}; if(t==='/help') return {type:'help', args:''}; return {type:'run', args:t}; },
    log: function(message, level){ var ts=new Date().toISOString(); var lv=level||'info'; (console[lv]||console.log)("[Studio:"+lv+"] "+ts+" - "+message); }
  });

  // ===== 6) Engine Transport =====
  function setupEngineTransport(engineFrame){
    function isSameOrigin(){ try{ return !!engineFrame && !!engineFrame.contentWindow && !!engineFrame.contentWindow.document; }catch(e){ return false; } }
    function withEngine(fn){ if (!isSameOrigin()) { console.warn('Cross-origin detected; use postMessage path'); return; } try{ fn(engineFrame.contentWindow.document); }catch(e){ console.warn('withEngine error', e); } }
    return { isSameOrigin: isSameOrigin, withEngine: withEngine };
  }

  // ===== 7) Public APIs =====
  var RecorderAPI = Object.freeze({
    startMic: function(meta){ sendBus({type:'rec.start', mode:'mic', meta: meta||{}}); },
    startScreen: function(meta){ sendBus({type:'rec.start', mode:'screen', meta: meta||{}}); },
    startBoth: function(meta){ sendBus({type:'rec.start', mode:'both', meta: meta||{}}); },
    stop: function(){ sendBus({type:'rec.stop'}); },
    mark: function(tag, runId){ sendBus({type:'rec.mark', tag: (tag||'mark'), runId: (runId==null?null:String(runId))}); }
  });

  var EngineAPI = Object.freeze({
    run: function(text){ sendBus({type:'eng.run', text: String(text||'')}); },
    test: function(name){ sendBus({type:'eng.test', name: String(name||'')}); },
    getStatus: function(){ sendBus({type:'eng.status'}); }
  });

  var ChatAPI = Object.freeze({
    command: function(line){ sendBus({type:'chat.cmd', line: String(line||'')}); },
    announce: function(message, level){ sendBus({type:'chat.announce', message: String(message||''), level: String(level||'info')}); }
  });

  // ===== 8) Bridge Export =====
  var Bridge = Object.freeze({ onBus: onBus, sendBus: sendBus, RecorderAPI: RecorderAPI, EngineAPI: EngineAPI, ChatAPI: ChatAPI, Store: Store, Utils: Utils, setupEngineTransport: setupEngineTransport });

  Object.defineProperty(window, 'StudioBridge', { value: Bridge, writable: false, configurable: false, enumerable: true });
  if (typeof module !== 'undefined' && module.exports){ module.exports = Bridge; }

  // ===== 10) Diagnostics =====
  Utils.log('init '+NAME+' v'+VERSION + (bc?' [BroadcastChannel]':' [window-event]'));
})();
```

> **Review checklist before we lock Layer 1:**
>
> -



---

## 16) Universal Space‑Saving (C‑Shrink Modes)

> Headings and titles here are canonical. Use these exact labels when wiring UI toggles and code comments so we can embed/turtle reliably without duplication.

### [C1] Shrink Mode — Compact‑1 (safe default)

- **Purpose**: Mild tightening without harming readability.
- **Rules**:
  - Reduce base font to **13 → 12px** (UI) and **12 → 11px** (mono), never below 11 for body text.
  - Tighten vertical rhythm: margins/paddings −10% to −15%.
  - Collapse consecutive blank lines to **max 1** (outside code fences only).
  - Keep headings at least **+2px** above body.
- **Turtling**: Wrap large note blocks in triple‑backtick fences with a one‑line title right above. The title must match the section heading.
- **Selector tokens**: `.cshrink-1`, `.mono-shrink-1`.

### [C2] Shrink Mode — Compact‑2 (aggressive)

- **Purpose**: Maximize on-screen density for review.
- **Rules**:
  - Base font **12 → 11px** (UI) and **11 → 10px** (mono); headings scale down proportionally but remain ≥ body+1px.
  - Reduce line-height to **1.20–1.25** where readable (never below 1.15).
  - Trim leading/trailing spaces; normalize bullet indents to **2 spaces**.
  - Collapse multi-blank regions to **0** outside code fences and tables.
- **Turtling**: Prefer fenced blocks with **language tag** (`text or `md) to discourage syntax highlighters from expanding.
- **Selector tokens**: `.cshrink-2`, `.mono-shrink-2`.

### [C3] Shrink Mode — Compact‑3 (review only)

- **Purpose**: Extreme density for snapshot/fit-to-screen reviews; not for long reading.
- **Rules**:
  - Body **10px** hard minimum; monospace **10px**; headings may equal body size.
  - Line-height **1.10–1.15**; gutters narrowed to the minimum visually distinct.
  - All nonessential callouts/tips collapsed behind a turtle fence.
  - Ellipsize mid‑paragraph parentheticals with `[…]` markers; original preserved in fenced note immediately below.
- **Turtling**: Every optional block must have a matching **title line** immediately above the fence, e.g., `C3: Diagnostics (collapsed)` and then `text … `.
- **Selector tokens**: `.cshrink-3`, `.mono-shrink-3`.

### Turtle / Embedding Protocol (No‑duplication)

1. **Title match**: The single line above any fenced block must **exactly** match the canonical heading for that block.
2. **Single source**: If a section is turtled, the fence is **the** source of truth; do not repeat the content elsewhere.
3. **Re‑entry safe**: All compressors operate **outside** fenced code blocks and tables.
4. **Markers**: Begin and end optional regions with HTML comments for tooling: `<!-- C2:Start:Options --> … <!-- C2:End:Options -->`.

### Whitespace/Gaps Compression (idempotent)

- Collapse 3+ spaces between words to **1** (outside code/fences/tables).
- Normalize list bullets to `-` and indent by **2 spaces** per level.
- Trim trailing spaces at EOL.
- Convert CRLF → LF.
- Guarantee single blank line **between** top-level sections (`##`).

### Font Policies (global, space‑saving aware)

- Font stacks: UI = system‑ui; code = `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`.
- Minimums: body 10px, mono 10px; default targets per mode above.
- Use `.mono-shrink-*` classes to scale code blocks independently from UI text.

### Boundary Guards for Space‑Saving

- **Lock/Unlock**: Only Layer 1 can set or clear `.cshrink-*` classes on the **document root**.
- **Idempotency**: Applying a mode twice produces no change; switching modes first removes previous class.
- **Safe zones**: No compressors run inside fenced code, blockquotes, or tables.

### Operator Checklist (pre‑lock)

-

<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>World Engine Studio — Chat ↔ Engine ↔ Recorder</title>
<style>
  :root{
    --bg:#0b0e14; --panel:#0f1523; --ink:#e6f0ff; --mut:#9ab0d6; --line:#1e2b46; --acc:#54f0b8;
    --chip:#182741;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.35 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;overflow:hidden}
  .shell{display:grid;grid-template-columns:320px 1fr;grid-template-rows:auto 1fr;gap:8px;height:100%;padding:8px}
  .top{grid-column:1/-1;display:flex;align-items:center;gap:10px;padding:8px;border:1px solid var(--line);border-radius:12px;background:#0d1322}
  .brand{font-weight:800;color:var(--acc)}
  .sp{flex:1}
  .col{border:1px solid var(--line);border-radius:12px;background:linear-gradient(180deg,#0f1523,#0b1020);overflow:hidden;min-height:0}
  .L{display:grid;grid-template-rows:auto 1fr auto;gap:8px;padding:8px;min-width:0}
  .R{display:grid;grid-template-rows:auto 1fr;gap:8px;padding:8px;min-width:0}
  .title{font-weight:700;color:#bfeaff;border-bottom:1px dashed var(--line);padding-bottom:6px;margin-bottom:6px}
  .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
  .mini{font-size:12px;color:var(--mut)}
  .btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:6px 10px;cursor:pointer}
  .chip{border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:999px;padding:6px 10px}
  textarea{width:100%;min-height:90px;background:#0b1120;border:1px solid var(--line);border-radius:8px;padding:9px;color:var(--ink);font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  .chat{display:grid;grid-template-rows:auto 1fr auto;gap:8px;min-height:0}
  .msgs{border:1px solid var(--line);border-radius:10px;background:#0a0f1c;padding:8px;overflow:auto;min-height:0}
  .msg{margin:6px 0;padding:8px;border-radius:10px}
  .me{background:#11203a}
  .ai{background:#0f1a2f}
  .sys{background:#122222;color:#9eead3}
  input[type=text]{width:100%;background:#0b1120;border:1px solid var(--line);border-radius:8px;padding:9px;color:var(--ink)}
  .space{display:grid;grid-template-columns:1fr 1fr;gap:8px;min-height:0}
  .frame{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#06101c;min-height:0}
  .frame>iframe{display:block;width:100%;height:100%;border:0}
  .label{position:absolute;top:6px;left:10px;font-size:12px;background:#0008;padding:3px 8px;border-radius:999px;border:1px solid #ffffff20}
  .hint{font-size:12px;color:#96a8cc}
  .footer{padding:6px;text-align:center;color:#6eaeb0}
  .kbd{font:11px/1.1 ui-monospace;background:#0a1322;border:1px solid var(--line);border-radius:6px;padding:2px 6px}
</style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div class="brand">World Engine Studio</div>
      <div class="mini">Chat controls engine • Recording UI hosts visuals</div>
      <div class="sp"></div>
      <span class="chip">/run</span><span class="chip">/test</span><span class="chip">/script</span>
    </div>

    <section class="col L chat">
      <div class="title">AI Chatbot — Engine Control</div>

      <div id="msgs" class="msgs" aria-live="polite"></div>

      <div class="row">
        <input id="chat" type="text" placeholder="Type: /run state-of-the-art, unbelievable  or  Analyze these terms for me..."/>
        <button id="send" class="btn">Send</button>
      </div>

      <div class="mini">
        Commands: <span class="kbd">/run</span> terms… • <span class="kbd">/test</span> name • <span class="kbd">/script</span> JS
      </div>
    </section>

    <section class="col R">
      <div class="title row">
        <div>Surfaces</div>
        <div class="sp"></div>
        <div class="hint">The Recording Toolkit preview is replaced with the World Engine visual.</div>
      </div>

      <div class="space">
        <div class="frame" id="fWorld">
          <div class="label">World Engine</div>
          <iframe id="we" src="worldengine.html" title="World Engine"></iframe>
        </div>

        <div class="frame" id="fRecording">
          <div class="label">Recording Toolkit (UI host)</div>
          <iframe id="rt" src="recording-toolkit.html" title="Recording Toolkit"></iframe>
        </div>
      </div>
    </section>

    <div class="footer mini">Prototype: same-origin iframes + DOM bridge. Replace the stub AI with your backend when ready.</div>
  </div>

<script>
(function(){
  'use strict';
  const $ = (id)=>document.getElementById(id);
  const msgs = $('msgs');
  function say(text, cls='ai'){ const div=document.createElement('div'); div.className='msg '+cls; div.textContent=text; msgs.appendChild(div); msgs.scrollTop=msgs.scrollHeight; }
  function sayJSON(obj, cls='ai'){ const div=document.createElement('div'); div.className='msg '+cls; div.textContent = JSON.stringify(obj, null, 2); msgs.appendChild(div); msgs.scrollTop=msgs.scrollHeight; }

  const we = $('we'); const rt = $('rt');

  let weReady = false;
  let rtReady = false;

  we.addEventListener('load', ()=>{
    weReady = true;
    say('World Engine loaded.', 'sys');
  });

  rt.addEventListener('load', ()=>{
    rtReady = true;
    say('Recording Toolkit loaded.', 'sys');
    try {
      // Replace Recording Toolkit's <video id="live"> with an embedded World Engine view
      const doc = rt.contentWindow.document;
      const live = doc.getElementById('live');
      if(live){
        const slot = document.createElement('iframe');
        slot.src = 'worldengine.html'; // second view inside recording UI
        slot.style.width='100%'; slot.style.height='400px'; slot.style.border='0'; slot.title='World Engine Visual';
        live.parentNode.replaceChild(slot, live);
      }
      // Minimize the heavy recording controls into a corner widget (non-destructive)
      const controlsSection = doc.querySelector('.controls');
      if(controlsSection){
        controlsSection.style.position = 'fixed';
        controlsSection.style.right = '12px';
        controlsSection.style.bottom = '12px';
        controlsSection.style.width = '360px';
        controlsSection.style.maxWidth = '38vw';
        controlsSection.style.transform = 'scale(0.9)';
        controlsSection.style.transformOrigin = 'bottom right';
        controlsSection.style.zIndex = '9999';
        controlsSection.style.opacity = '0.95';
      }
    } catch(e){
      say('Note: could not tweak Recording Toolkit layout (likely cross-origin). Using default UI.', 'sys');
    }
  });

  // Helpers to drive World Engine DOM directly (same-origin)
  function withWE(fn){
    if(!weReady){ say('World Engine not ready yet.', 'sys'); return; }
    try{
      const d = we.contentWindow.document;
      fn(d);
    }catch(e){
      say('Cannot access World Engine frame (cross-origin?).', 'sys');
    }
  }

  function runAnalysisFromText(text){
    withWE((d)=>{
      const input = d.getElementById('input');
      const btn = d.getElementById('run');
      if(!input || !btn){ say('World Engine controls not found.', 'sys'); return; }
      input.value = text.trim();
      // trigger click to analyze
      btn.click();
      // harvest JSON summary
      setTimeout(()=>{
        const out = d.getElementById('out');
        if(out) {
          try { sayJSON(JSON.parse(out.textContent), 'ai'); }
          catch{ say(out.textContent.slice(0,2000), 'ai'); }
        }
      }, 120);
    });
  }

  function clickTestByName(name){
    withWE((d)=>{
      const host = d.getElementById('tests');
      if(!host){ say('No tests panel found.', 'sys'); return; }
      const btns = host.querySelectorAll('button');
      let ok=false;
      btns.forEach(b=>{
        if(b.textContent.trim().toLowerCase()===name.trim().toLowerCase()){
          b.click(); ok=true;
        }
      });
      say(ok ? `Loaded test "${name}".` : `Test "${name}" not found.`, 'ai');
    });
  }

  function runScript(jsCode){
    try{
      const res = we.contentWindow.eval(jsCode);
      say('Script executed. Result: '+res, 'ai');
    }catch(e){
      say('Script error: '+e, 'ai');
    }
  }

  // Super-light "AI" stub that recognizes intents. Replace with backend later.
  function interpretAndDispatch(text){
    const t = text.trim();
    if(t.startsWith('/run')){
      const payload = t.replace(/^\/run\s*/,'').trim();
      runAnalysisFromText(payload);
    }else if(t.startsWith('/test')){
      const name = t.replace(/^\/test\s*/,'').trim();
      clickTestByName(name);
    }else if(t.startsWith('/script')){
      const js = t.replace(/^\/script\s*/,'').trim();
      runScript(js);
    }else{
      // Default: treat as "analyze these lines"
      runAnalysisFromText(t);
    }
  }

  $('send').addEventListener('click', ()=>{
    const i = $('chat'); const v = i.value;
    if(!v) return;
    say(v, 'me'); i.value='';
    interpretAndDispatch(v);
  });

  $('chat').addEventListener('keydown', (e)=>{
    if(e.key==='Enter'){ e.preventDefault(); $('send').click(); }
  });

  // Boot tips
  say('Type /run followed by terms (comma or newline separated). Example: /run state-of-the-art, restate, unbelievable', 'sys');
})();
</script>
</body>
</html>







<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>World Engine Studio — Corner Chat Widget</title>
<style>
  :root{
    --bg:#0b0e14; --panel:#0f1523; --ink:#e6f0ff; --mut:#9ab0d6; --line:#1e2b46; --acc:#54f0b8;
    --chip:#182741;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.35 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
  .wrap{display:grid;grid-template-rows:auto 1fr auto;gap:8px;height:100%;padding:10px}
  .top{display:flex;align-items:center;gap:10px;padding:8px 12px;border:1px solid var(--line);border-radius:12px;background:#0d1322}
  .brand{font-weight:800;color:var(--acc)}
  .mini{font-size:12px;color:var(--mut)}
  .sp{flex:1}
  .content{display:grid;grid-template-columns:1fr 1fr;gap:10px;min-height:0}
  .frame{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#06101c;min-height:0}
  .frame>iframe{display:block;width:100%;height:100%;border:0}
  .label{position:absolute;top:6px;left:10px;font-size:12px;background:#0008;padding:3px 8px;border-radius:999px;border:1px solid #ffffff20}
  .footer{padding:6px;text-align:center;color:#6eaeb0}
  /* Corner Chat Widget */
  .dock{position:fixed;right:18px;bottom:18px;z-index:999999}
  .fab{display:flex;align-items:center;gap:8px;background:linear-gradient(135deg,#1a2440,#0f1523);
       color:var(--ink);border:1px solid var(--line);border-radius:999px;padding:10px 12px;cursor:pointer;
       box-shadow:0 8px 30px #0008}
  .fab .dot{width:8px;height:8px;background:var(--acc);border-radius:999px;box-shadow:0 0 10px var(--acc)}
  .drawer{position:fixed;right:18px;bottom:74px;width:360px;max-width:85vw;height:60vh;max-height:78vh;
          background:linear-gradient(180deg,#0f1523,#0b1020);border:1px solid var(--line);border-radius:16px;
          overflow:hidden;box-shadow:0 16px 60px #0009;transform:translateY(18px) scale(0.98);opacity:0;
          pointer-events:none;transition:all .18s ease}
  .drawer.open{transform:none;opacity:1;pointer-events:auto}
  .drawer .head{display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid var(--line)}
  .drawer .title{font-weight:700}
  .drawer .msgs{height:calc(60vh - 130px);overflow:auto;padding:8px}
  .msg{margin:6px 8px;padding:8px;border-radius:10px;background:#0a0f1c;white-space:pre-wrap}
  .msg.me{background:#11203a}
  .msg.sys{background:#122222;color:#9eead3}
  .drawer .ctrls{display:grid;gap:6px;padding:8px;border-top:1px solid var(--line)}
  .drawer input[type=text]{width:100%;background:#0b1120;border:1px solid var(--line);border-radius:8px;padding:9px;color:var(--ink)}
  .drawer .row{display:flex;gap:6px;align-items:center}
  .btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:6px 10px;cursor:pointer}
  .kbd{font:11px/1.1 ui-monospace;background:#0a1322;border:1px solid var(--line);border-radius:6px;padding:2px 6px}
</style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="brand">World Engine Studio</div>
      <div class="mini">Recording app hosts visuals & tests • Chat lives as a corner widget</div>
      <div class="sp"></div>
      <span class="kbd">/run</span>
      <span class="kbd">/test</span>
      <span class="kbd">/script</span>
    </div>

    <div class="content">
      <div class="frame">
        <div class="label">World Engine (primary)</div>
        <iframe id="we" src="worldengine.html" title="World Engine"></iframe>
      </div>
      <div class="frame">
        <div class="label">Recording Toolkit (UI host)</div>
        <iframe id="rt" src="recording-toolkit.html" title="Recording Toolkit"></iframe>
      </div>
    </div>

    <div class="footer mini">Prototype: open locally or serve from one origin for full DOM bridging.</div>
  </div>

  <!-- Corner Chat Widget -->
  <div class="dock">
    <div id="fab" class="fab" title="Chat with the World Engine">
      <div class="dot"></div><div>Chat</div>
    </div>
    <div id="drawer" class="drawer" role="dialog" aria-modal="true" aria-label="AI Chatbot Engine Control">
      <div class="head">
        <div class="title">AI Chatbot — Engine Control</div>
        <div class="sp"></div>
        <button id="close" class="btn" title="Close">×</button>
      </div>
      <div id="msgs" class="msgs" aria-live="polite"></div>
      <div class="ctrls">
        <div class="row">
          <input id="chat" type="text" placeholder="Type /run …  /test …  /script … then Enter"/>
          <button id="send" class="btn">Send</button>
        </div>
        <div class="mini">Commands: <span class="kbd">/run</span> terms… • <span class="kbd">/test</span> name • <span class="kbd">/script</span> JS</div>
      </div>
    </div>
  </div>

<script>
(function(){
  'use strict';
  const $ = (id)=>document.getElementById(id);
  const we = $('we'); const rt = $('rt');
  const drawer = $('drawer'); const fab = $('fab'); const closeBtn = $('close');
  const msgs = $('msgs'); const input = $('chat'); const send = $('send');

  function toggleDrawer(open){
    drawer.classList[open ? 'add' : 'remove']('open');
  }
  fab.addEventListener('click', ()=>toggleDrawer(true));
  closeBtn.addEventListener('click', ()=>toggleDrawer(false));
  document.addEventListener('keydown', (e)=>{
    if(e.key === 'Escape') toggleDrawer(false);
    if(e.ctrlKey && e.key.toLowerCase()==='k'){ e.preventDefault(); toggleDrawer(true); input.focus(); }
  });

  function say(text, cls='ai'){ const div=document.createElement('div'); div.className='msg '+cls; div.textContent=text; msgs.appendChild(div); msgs.scrollTop=msgs.scrollHeight; }
  function sayJSON(obj, cls='ai'){ const div=document.createElement('div'); div.className='msg '+cls; div.textContent = JSON.stringify(obj, null, 2); msgs.appendChild(div); msgs.scrollTop=msgs.scrollHeight; }

  let weReady=false, rtReady=false;
  we.addEventListener('load', ()=>{ weReady=true; say('World Engine loaded.', 'sys'); });
  rt.addEventListener('load', ()=>{
    rtReady=true; say('Recording Toolkit loaded.', 'sys');
    // Try to embed engine visuals + tests inside the Recording Toolkit as its UI panels
    try {
      const rtDoc = rt.contentWindow.document;

      // Create a left sidebar for Tests/Visuals if not present
      let left = rtDoc.getElementById('we-panel');
      if(!left){
        left = rtDoc.createElement('div');
        left.id = 'we-panel';
        left.innerHTML = `
          <style>
            #we-panel{position:fixed;left:12px;top:12px;bottom:12px;width:360px;max-width:42vw;
                      background:linear-gradient(180deg,#0f1523,#0b1020);border:1px solid #1e2b46;border-radius:12px;
                      box-shadow:0 16px 40px #0008; color:#e6f0ff; font:12px/1.35 system-ui; overflow:hidden; z-index: 9998;}
            #we-panel .tabs{display:flex;gap:4px;padding:8px;border-bottom:1px solid #1e2b46}
            #we-panel .tabs .t{border:1px solid #1e2b46;background:#182741;padding:6px 8px;border-radius:8px;cursor:pointer}
            #we-panel .tabs .t.active{background:#223355}
            #we-panel .body{position:absolute;top:40px;bottom:0;left:0;right:0;overflow:auto;padding:8px}
            #we-panel iframe{width:100%;height:220px;border:0;border-radius:8px}
            #we-panel .tests{display:grid;gap:6px;margin-top:8px}
            #we-panel .tests button{padding:6px 8px;border-radius:8px;border:1px solid #1e2b46;background:#0a1222;color:#e6f0ff;cursor:pointer;text-align:left}
            #we-panel .log{white-space:pre-wrap;background:#0a0f1c;border:1px solid #1e2b46;border-radius:8px;padding:8px;margin-top:8px;min-height:80px}
            #we-panel .hint{color:#9ab0d6;margin-top:6px}
            #we-rec{position:fixed;right:12px;bottom:12px;width:360px;max-width:40vw;background:#0f1523;
                    border:1px solid #1e2b46;border-radius:12px;padding:8px;box-shadow:0 16px 40px #0008; z-index: 9999;}
          </style>
          <div class="tabs">
            <div class="t active" data-id="visual">Visual</div>
            <div class="t" data-id="tests">Tests</div>
            <div class="t" data-id="logs">Logs</div>
          </div>
          <div class="body">
            <section id="panel-visual">
              <iframe src="worldengine.html" title="World Visual"></iframe>
              <div class="hint">This is the engine visual embedded inside the Recording app.</div>
            </section>
            <section id="panel-tests" style="display:none">
              <div class="tests" id="tests-host"><em>Loading tests…</em></div>
            </section>
            <section id="panel-logs" style="display:none">
              <div class="log" id="rt-log"></div>
            </section>
          </div>
          <div id="we-rec"></div>
        `;
        rtDoc.body.appendChild(left);

        // Tabs
        const tabs = Array.from(left.querySelectorAll('.t'));
        function setTab(id){
          tabs.forEach(t=>t.classList.toggle('active', t.dataset.id===id));
          left.querySelector('#panel-visual').style.display = id==='visual' ? '' : 'none';
          left.querySelector('#panel-tests').style.display  = id==='tests'  ? '' : 'none';
          left.querySelector('#panel-logs').style.display   = id==='logs'   ? '' : 'none';
        }
        tabs.forEach(t=>t.addEventListener('click', ()=>setTab(t.dataset.id)));

        // Move/clone recording controls into a compact widget on the right
        let widget = left.querySelector('#we-rec');
        const bigControls = rtDoc.querySelector('.controls') || rtDoc.querySelector('#controls') || null;
        if(bigControls){
          widget.innerHTML = '<div class="hint">Recording controls (compact)</div>';
          widget.appendChild(bigControls.cloneNode(true));
        } else {
          widget.innerHTML = '<div class="hint">Recording controls not detected; using defaults.</div>';
        }

        // Fill Tests panel by cloning the engine tests list and wiring click-through
        function injectTests(){
          try{
            const host = left.querySelector('#tests-host');
            const engineDoc = we.contentWindow.document;
            const tests = engineDoc.getElementById('tests');
            if(tests){
              host.innerHTML = '';
              const buttons = tests.querySelectorAll('button');
              buttons.forEach((b,i)=>{
                const clone = rtDoc.createElement('button');
                clone.textContent = b.textContent || ('Test '+(i+1));
                clone.addEventListener('click', ()=>{
                  // click through into the engine
                  b.click();
                  // tiny log
                  const log = left.querySelector('#rt-log');
                  if(log) log.textContent += `\nLoaded test: ${clone.textContent}`;
                });
                host.appendChild(clone);
              });
              if(!buttons.length){ host.innerHTML = '<em>No tests found in engine.</em>'; }
            }else{
              host.innerHTML = '<em>Engine tests panel (#tests) not found.</em>';
            }
          }catch(e){
            // Cross-origin or timing issue
            const host = left.querySelector('#tests-host');
            host.innerHTML = '<em>Cannot access engine tests (cross-origin). Serve from same origin.</em>';
          }
        }

        // Try now, and also after engine load
        injectTests();
        we.addEventListener('load', injectTests);
      }

    } catch(e){
      say('Could not integrate panels inside Recording Toolkit (likely cross-origin).', 'sys');
    }
  });

  // Chat → Engine bridge (same as previous prototype)
  function withWE(fn){
    if(!weReady){ say('World Engine not ready.', 'sys'); return; }
    try{
      const d = we.contentWindow.document;
      fn(d);
    }catch(e){
      say('Cannot access World Engine (cross-origin?).', 'sys');
    }
  }

  function runAnalysisFromText(text){
    withWE((d)=>{
      const input = d.getElementById('input');
      const btn = d.getElementById('run');
      if(!input || !btn){ say('World Engine controls not found.', 'sys'); return; }
      input.value = text.trim();
      btn.click();
      setTimeout(()=>{
        const out = d.getElementById('out');
        if(out) {
          try { sayJSON(JSON.parse(out.textContent), 'ai'); }
          catch{ say(out.textContent.slice(0,2000), 'ai'); }
        }
      }, 120);
    });
  }
  function clickTestByName(name){
    withWE((d)=>{
      const host = d.getElementById('tests');
      if(!host){ say('No tests panel found.', 'sys'); return; }
      const btns = host.querySelectorAll('button');
      let ok=false;
      btns.forEach(b=>{
        if((b.textContent||'').trim().toLowerCase()===name.trim().toLowerCase()){
          b.click(); ok=true;
        }
      });
      say(ok ? `Loaded test "${name}".` : `Test "${name}" not found.`, 'ai');
    });
  }
  function runScript(jsCode){
    try{
      const res = we.contentWindow.eval(jsCode);
      say('Script executed. Result: '+res, 'ai');
    }catch(e){
      say('Script error: '+e, 'ai');
    }
  }
  function interpretAndDispatch(text){
    const t = text.trim();
    if(t.startsWith('/run')){
      runAnalysisFromText(t.replace(/^\/run\s*/,'').trim());
    }else if(t.startsWith('/test')){
      clickTestByName(t.replace(/^\/test\s*/,'').trim());
    }else if(t.startsWith('/script')){
      runScript(t.replace(/^\/script\s*/,'').trim());
    }else{
      runAnalysisFromText(t);
    }
  }

  send.addEventListener('click', ()=>{ const v=input.value; if(!v) return; say(v,'me'); input.value=''; interpretAndDispatch(v); });
  input.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ e.preventDefault(); send.click(); }});

  // Boot tip
  toggleDrawer(false);
  say('Press the Chat button (or Ctrl+K) to open the chatbot. Use /run, /test, or /script.', 'sys');

})();
</script>
</body>
</html>
