// ESM service replacement for services/pain/service.js
import { clusterTexts } from './cluster.js';

function painScore(text = '', eng = 0) {
  const keywords = ['error', 'stuck', 'fail', 'broken', 'bug', 'issue', 'problem', 'difficult'];
  let score = 0;
  const lower = String(text).toLowerCase();
  for (const k of keywords) if (lower.includes(k)) score += 1;
  const normalizedEng = Math.min(eng / 10, 1);
  return Math.min(score * (1 + normalizedEng), 1);
}

function burst7d(dates = []) {
  const now = Date.now();
  const last7 = dates.filter(d => (now - Date.parse(d)) <= 7 * 24 * 60 * 60 * 1000);
  const prev7 = dates.filter(d => {
    const age = now - Date.parse(d);
    return age > 7 * 24 * 60 * 60 * 1000 && age <= 14 * 24 * 60 * 60 * 1000;
  });
  return last7.length / Math.max(prev7.length, 1);
}

const store = { events: [], clusters: new Map() };

function ingestPain(ev = {}) {
  const pain = painScore(ev.text, ev.eng);
  const severity = pain >= 0.8 ? 3 : pain >= 0.5 ? 2 : pain >= 0.25 ? 1 : 0;
  const rec = Object.assign({}, ev, { pain, severity });
  store.events.push(rec);
  return rec;
}

function recomputeClusters() {
  const ids = store.events.map(e => e.id);
  const texts = store.events.map(e => e.text || '');
  const { assignments = [], labels = [] } = (typeof clusterTexts === 'function') ? clusterTexts(ids, texts) : { assignments: [], labels: [] };
  const byC = {};
  store.events.forEach((e, i) => {
    const c = assignments[i] || 0;
    e.clusterId = String(c);
    (byC[c] = byC[c] || []).push(e);
  });
  store.clusters.clear();
  Object.entries(byC).forEach(([k, rows]) => {
    const id = k;
    const avg = rows.reduce((s, r) => s + (r.pain || 0), 0) / rows.length;
    const last = rows.reduce((m, r) => (m > r.time ? m : r.time), rows[0].time);
    store.clusters.set(id, { id, label: labels[+k] || 'cluster', members: rows.map(r => r.id), avgPain: avg, count: rows.length, lastSeen: last });
  });
}

function summary(now = Date.now()) {
  const dates = store.events.map(e => e.time);
  const burst = burst7d(dates);
  const toDays = (t) => Math.floor((now - Date.parse(t)) / 864e5);
  const last7 = store.events.filter(e => toDays(e.time) <= 7).length;
  const prev21 = store.events.filter(e => { const d = toDays(e.time); return d > 7 && d <= 28; }).length;
  const byOrg = {};
  store.events.forEach(e => { if (!e.org) return; byOrg[e.org] = (byOrg[e.org] || 0) + 1; });
  const topOrgs = Object.entries(byOrg).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([org,count])=>({org,count}));
  return { last7, prev21, burst, topOrgs };
}

function topClusters(limit = 10) {
  return Array.from(store.clusters.values()).sort((a,b) => (b.avgPain*b.count) - (a.avgPain*a.count)).slice(0, limit);
}

export { ingestPain, recomputeClusters, summary, topClusters };
export const _store = store;
