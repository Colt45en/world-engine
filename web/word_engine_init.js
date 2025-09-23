// word_engine_init.js
// Minimal bootstrap to load toggle schema, apply WOW→GGL aliasing on ingest, and wire 'T' to flip views.

export async function initWordEngine({
  schemaUrl = 'toggle_schema.json',
  aliasUrl = 'alias_wow_ggl.json',
  onToggle = (view)=>console.log('[WordEngine] view ->', view),
  getDataFrame = async (path)=>{ const r=await fetch(path); return await r.text(); } // replace with your CSV/MD loader
} = {}){
  const [schema, alias, bookmarks] = await Promise.all([
    fetch(schemaUrl).then(r=>r.json()),
    fetch(aliasUrl).then(r=>r.json()),
    fetch((await fetch(schemaUrl).then(r=>r.json())).ui?.bookmarks_url || 'bookmarks.json').then(r=>r.json()).catch(()=>({}))
  ]);

  // Simple alias applier
  const rules = (alias.apply_rules||[]).map(r=>({
    targets: new Set(r.where||[]),
    re: new RegExp(r.pattern, r.flags?.includes('ignore_case') ? 'gi' : 'g'),
    replacement: r.replacement
  }));

  const applyAliases = (text, targetLabel) => {
    let out = text;
    for (const rule of rules){
      if (rule.targets.has(targetLabel)) out = out.replace(rule.re, rule.replacement);
    }
    return out;
  };

  // Load both datasets
  const lexPath = schema.views.lexicon.source;
  const runesPath = schema.views.runes.source;

  let lexRaw = await getDataFrame(lexPath);
  let runesRaw = await getDataFrame(runesPath);

  // Apply WOW→GGL to headers/tags/code-ish places; here we blanket-transform the whole text.
  lexRaw  = applyAliases(lexRaw, 'headers');
  runesRaw = applyAliases(runesRaw, 'code');

  // Keep state
  const state = {
    schema,
    alias,
    data: { lexRaw, runesRaw },
    view: (schema.default_view || 'lexicon')
  };

  // Hotkey: T to toggle
  const handler = (e)=>{
    if (e.key === 't' || e.key === 'T'){
      state.view = state.view === 'lexicon' ? 'runes' : 'lexicon';
      try { onToggle(state.view, state); } catch{}
    }
  };
  window.addEventListener('keydown', handler);

  // Restore last-selected (best-effort)
  const bm = bookmarks?.views?.[state.view]?.last_selected;
  state.bookmarks = bookmarks || {};
  state.last_selected = bm || { rows: [], cols: [] };

  // Initial notify
  onToggle(state.view, state);

  return {
    state,
    destroy(){ window.removeEventListener('keydown', handler); },
    toggle(){ state.view = state.view === 'lexicon' ? 'runes' : 'lexicon'; onToggle(state.view, state); },
    applyAliases
  };
}
