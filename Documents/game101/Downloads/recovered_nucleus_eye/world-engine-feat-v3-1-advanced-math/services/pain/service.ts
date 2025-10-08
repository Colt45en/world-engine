import { PainEvent, PainRecord, PainCluster, PainSummary } from './types.js';
import { clusterTexts } from './cluster.js';

// Simple pain scoring function (you can replace with your existing painScore)
function painScore(text: string, eng: number): number {
  const keywords = ['error', 'stuck', 'fail', 'broken', 'bug', 'issue', 'problem', 'difficult'];
  let score = 0;
  
  keywords.forEach(keyword => {
    if (text.toLowerCase().includes(keyword)) score += 1;
  });
  
  // Boost by engagement
  const normalizedEng = Math.min(eng / 10, 1);
  return Math.min(score * (1 + normalizedEng), 1);
}

// Simple burst detection (you can replace with your existing burst7d)
function burst7d(dates: string[]): number {
  const now = Date.now();
  const last7Days = dates.filter(d => (now - Date.parse(d)) <= 7 * 24 * 60 * 60 * 1000);
  const prev7Days = dates.filter(d => {
    const age = now - Date.parse(d);
    return age > 7 * 24 * 60 * 60 * 1000 && age <= 14 * 24 * 60 * 60 * 1000;
  });
  
  return last7Days.length / Math.max(prev7Days.length, 1);
}

const store = { 
  events: [] as PainRecord[], 
  clusters: new Map<string, PainCluster>() 
};

export function ingestPain(ev: PainEvent): PainRecord {
  const pain = painScore(ev.text, ev.eng);
  const severity = pain >= 0.8 ? 3 : pain >= 0.5 ? 2 : pain >= 0.25 ? 1 : 0;
  const rec: PainRecord = { ...ev, pain, severity };
  store.events.push(rec);
  return rec;
}

export function recomputeClusters(): void {
  const ids = store.events.map(e => e.id);
  const texts = store.events.map(e => e.text);
  const { assignments, labels } = clusterTexts(ids, texts);
  
  const byC: Record<number, PainRecord[]> = {};
  store.events.forEach((e, i) => {
    const c = assignments[i];
    if (c !== undefined) {
      e.clusterId = String(c);
      if (!byC[c]) byC[c] = [];
      byC[c].push(e);
    }
  });
  
  store.clusters.clear();
  Object.entries(byC).forEach(([k, rows]) => {
    const id = k;
    const avg = rows.reduce((s, r) => s + r.pain, 0) / rows.length;
    const firstTime = rows[0]?.time ?? new Date().toISOString();
    const last = rows.reduce((m, r) => m > r.time ? m : r.time, firstTime);
    const label = labels[+k] ?? 'Unknown';
    
    store.clusters.set(id, {
      id,
      label,
      members: rows.map(r => r.id),
      avgPain: avg,
      count: rows.length,
      lastSeen: last
    });
  });
}

export function summary(now = Date.now()): PainSummary {
  const dates = store.events.map(e => e.time);
  const burst = burst7d(dates);
  const toDays = (t: string) => Math.floor((now - Date.parse(t)) / 864e5);
  
  const last7 = store.events.filter(e => toDays(e.time) <= 7).length;
  const prev21 = store.events.filter(e => {
    const d = toDays(e.time);
    return d > 7 && d <= 28;
  }).length;
  
  const byOrg: Record<string, number> = {};
  store.events.forEach(e => {
    if (!e.org) return;
    byOrg[e.org] = (byOrg[e.org] || 0) + 1;
  });
  
  const topOrgs = Object.entries(byOrg)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([org, count]) => ({ org, count }));
  
  return { last7, prev21, burst, topOrgs };
}

export function topClusters(limit = 10) {
  return Array.from(store.clusters.values())
    .sort((a, b) => (b.avgPain * b.count) - (a.avgPain * a.count))
    .slice(0, limit);
}

export const _store = store; // for tests