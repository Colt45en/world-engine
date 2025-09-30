/**
 * Morphology Integration for Upflow Automation - V2
 * Enhanced longest-match, multi-affix, positions, & hooks system
 *
 * MATHEMATICAL UPGRADES AVAILABLE:
 * - Complete mathematical safety system in './lle-stable-math.js'
 * - Dimension-agnostic engine in './lexical-logic-engine-enhanced.js'
 * - Full V3 integration in './world-engine-v3-mathematical.js'
 * - Live demo in './world-engine-math-demo.js'
 *
 * Key mathematical improvements:
 * ✅ Shape validation with strict checks
 * ✅ Moore-Penrose pseudo-inverse for stable reconstruction
 * ✅ Morpheme-driven button composition
 * ✅ Dimension-agnostic operations (2D, 3D, 4D, NxN)
 * ✅ Undo/redo with mathematical validation
 * ✅ Preview composition without side effects
 * ✅ NaN/Infinity guards throughout
 * ✅ Type safety and composition validation
 * ✅ Self-testing capabilities
 *
 * Usage: import { WorldEngineV3Factory } from './world-engine-v3-mathematical.js'
 */

/** LLE Morphology v2 */
export function createLLEMorphologyV2() {
  const morphemePatterns = {
    prefixes: [
      'counter','inter','trans','pseudo','proto','super','multi','under','over',
      'anti','auto','micro','macro','semi','sub','pre','re','un','dis','out','up','down'
    ],
    suffixes: [
      'ization','ational','acious','ically','fulness','lessly','ations',
      'ization','ability','ically','ation','sion','ness','ment',
      'able','ible','ward','wise','ship','hood','dom','ism','ist',
      'ize','ise','fy','ate','ent','ant','ive','ous','eous','ious','al','ic','ical','ar','ary',
      'ing','ed','er','est','ly','s'
    ]
  };

  // sort by length desc for longest match
  morphemePatterns.prefixes.sort((a,b)=>b.length-a.length);
  morphemePatterns.suffixes.sort((a,b)=>b.length-a.length);

  function minimalStemFix(stem, suffix) {
    // very small sample to avoid "over-stemming"
    // e.g., make+ing → making ; stop+ing → stopping (double p is ok to keep)
    if (suffix === 'ing' && /e$/.test(stem)) return stem.replace(/e$/,'');
    return stem;
  }

  return function morpho(input) {
    const original = input ?? '';
    const word = original.toLowerCase().trim();

    const morphemes = [];
    const prefixes = [];
    const suffixes = [];

    // collect multiple prefixes (greedy, left→right)
    let consumedStart = 0;
    let search = true;
    while (search) {
      search = false;
      for (const pre of morphemePatterns.prefixes) {
        if (word.startsWith(pre, consumedStart)) {
          prefixes.push(pre);
          morphemes.push({ type:'prefix', text:pre, start:consumedStart, end:consumedStart+pre.length });
          consumedStart += pre.length;
          search = true;
          break;
        }
      }
    }

    // collect multiple suffixes (greedy, right→left)
    let consumedEnd = word.length;
    search = true;
    while (search) {
      search = false;
      for (const suf of morphemePatterns.suffixes) {
        const startIdx = consumedEnd - suf.length;
        if (startIdx > consumedStart + 1 && word.slice(startIdx, consumedEnd) === suf) {
          suffixes.unshift(suf); // keep order from inner→outer
          morphemes.push({ type:'suffix', text:suf, start:startIdx, end:consumedEnd });
          consumedEnd = startIdx;
          search = true;
          break;
        }
      }
    }

    let root = word.slice(consumedStart, consumedEnd);
    // tiny stem correction if we attached an -ing/-ed etc.
    const lastSuf = suffixes[suffixes.length-1];
    root = minimalStemFix(root, lastSuf ?? '');

    // rebuild morpheme order (prefixes, root, suffixes) with updated root bounds
    const rebuilt = [];
    let cursor = 0;
    for (const p of prefixes) {
      rebuilt.push({ type:'prefix', text:p, start:cursor, end:cursor+p.length });
      cursor += p.length;
    }
    const rootStart = cursor, rootEnd = rootStart + root.length;
    rebuilt.push({ type:'root', text:root, start:rootStart, end:rootEnd });
    cursor = rootEnd;
    for (const s of suffixes) {
      rebuilt.push({ type:'suffix', text:s, start:cursor, end:cursor+s.length });
      cursor += s.length;
    }

    return {
      original,
      word,
      prefixes,
      root,
      suffixes,
      morphemes: rebuilt,
      complexity: prefixes.length + (suffixes.length ? 1 : 0)
    };
  };
}

// Word Engine integration: safer parsing, better tags, semantic class via morphology
export function createWordEngineIntegrationV2() {
  function safeJSON(x) {
    if (typeof x !== 'string') return x;
    try { return JSON.parse(x); } catch { return null; }
  }

  function coalesceWordRecord(anyData) {
    if (!anyData) return null;
    const d = anyData;
    if (typeof d.word === 'string') return { word:d.word, english:d.english||d.gloss||d.meaning||'', context:d.context||'', metadata:d };
    if (d.result && typeof d.result.word === 'string') {
      const r = d.result;
      return { word:r.word, english:r.english||r.gloss||'', context:r.context||'', metadata:d };
    }
    if (Array.isArray(d) && d.length) {
      const f = d[0];
      return { word:f.word||f.string||f.text||'', english:f.english||f.gloss||f.meaning||'', context:f.context||'', metadata:f };
    }
    return null;
  }

  function extractTags(meta) {
    if (!meta || typeof meta !== 'object') return [];
    const tags = new Set();
    const maybe = ['class','type','category','priority','status','domain','lang','source','project','module','topic','tags'];
    for (const k of maybe) {
      const v = meta[k];
      if (v == null) continue;
      if (Array.isArray(v)) v.forEach(x => tags.add(`${k}:${String(x)}`));
      else tags.add(`${k}:${String(v)}`);
    }
    return Array.from(tags);
  }

  function classifyWord(word, english, morph) {
    const e = (english||'').toLowerCase();
    const w = (word||'').toLowerCase();
    const suf = new Set((morph?.suffixes)||[]);

    const looksVerb = suf.has('ize') || suf.has('ise') || suf.has('fy') || suf.has('ate') || suf.has('ing') || suf.has('ed');
    const looksNoun = suf.has('ness') || suf.has('tion') || suf.has('sion') || suf.has('ment') || suf.has('ism') || suf.has('ship') || suf.has('hood') || suf.has('dom');
    const looksAdj  = suf.has('ive') || suf.has('al') || suf.has('ous') || suf.has('ical') || suf.has('ary') || suf.has('ic');

    if (/\b(verb|action|perform|do|execute|transform|convert)\b/.test(e) || looksVerb) return 'Action';
    if (/\b(state|condition|status|being)\b/.test(e)) return 'State';
    if (/\b(component|module|part|structure|system)\b/.test(e)) return 'Structure';
    if (/\b(property|quality|attribute|trait)\b/.test(e) || looksAdj) return 'Property';
    if (looksNoun) return 'Entity';
    return 'General';
  }

  return {
    extractWordData(lastRun) {
      const data = typeof lastRun === 'string' ? safeJSON(lastRun) : lastRun;
      return coalesceWordRecord(data);
    },

    buildWordRecord(wordData, morpho) {
      if (!wordData || !wordData.word) return null;
      const morphology = morpho(wordData.word);
      return {
        ...wordData,
        ...morphology,
        indexed: Date.now(),
        source: 'wordEngine',
        tags: extractTags(wordData.metadata),
        semanticClass: classifyWord(wordData.word, wordData.english, morphology)
      };
    }
  };
}

// Integration script: no top-level await, clean init, richer queries
export async function initEnhancedUpflow({
  idbFactory
}) {
  const morpho = createLLEMorphologyV2();
  const wordEngine = createWordEngineIntegrationV2();

  const { createIDBIndexStorage, buildShardKeys, createUpflowAutomation } = await idbFactory();

  const storage = await createIDBIndexStorage({
    dbName: 'WorldEngineUpflow',
    storeName: 'lexicon',
    preloadKeys: buildShardKeys('lexi.index', 16)
  });

  const upflow = createUpflowAutomation({
    storage,
    indexKey: 'lexi.index',
    shards: 16,
    morph: morpho
  });

  async function ingestEnhancedRun(storageKey = 'wordEngine.lastRun') {
    try {
      const lastRunData = localStorage.getItem(storageKey);
      if (!lastRunData) throw new Error('No lastRun data found');

      const wordData = wordEngine.extractWordData(lastRunData);
      if (!wordData) throw new Error('Could not extract word data');

      const record = wordEngine.buildWordRecord(wordData, morpho);
      if (!record) throw new Error('Could not build word record');

      const result = upflow.addWord(record.word, record.english, {
        ...record,
        tags: record.tags,
        semanticClass: record.semanticClass
      });

      return { ok: true, ...result, record };
    } catch (error) {
      console.warn('Enhanced ingest failed:', error);
      return { ok: false, error: String(error?.message || error) };
    }
  }

  // richer semantic queries using stored morphology
  const semantic = {
    actions: () => upflow.query.byPrefix('').filter(word => {
      const rec = upflow.query.getWord(word);
      return rec?.metadata?.semanticClass === 'Action';
    }),
    states: () => upflow.query.byPrefix('').filter(word => {
      const rec = upflow.query.getWord(word);
      return rec?.metadata?.semanticClass === 'State';
    }),
    structures: () => upflow.query.byPrefix('').filter(word => {
      const rec = upflow.query.getWord(word);
      return rec?.metadata?.semanticClass === 'Structure';
    }),
    byRoot: (root) => upflow.query.byPrefix('').filter(w => upflow.query.getWord(w)?.root === root),
    byPrefix: (pre) => upflow.query.byPrefix(pre),
    bySuffix: (suf) => upflow.query.byPrefix('').filter(w => (upflow.query.getWord(w)?.suffixes||[]).includes(suf))
  };

  return {
    ...upflow,
    ingestEnhanced: ingestEnhancedRun,
    morpho,
    wordEngine,
    storage,
    semantic
  };
}

// Legacy compatibility wrapper
export function createLLEMorphology() {
  return createLLEMorphologyV2();
}

export function createWordEngineIntegration() {
  return createWordEngineIntegrationV2();
}
