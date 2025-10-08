// Clustering using tri-gram Jaccard similarity
const grams = (s: string) => {
  const t = s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const g = new Set<string>();
  for (let i = 0; i < t.length - 2; i++) g.add(t.slice(i, i + 3));
  return g;
};

const jaccard = (a: Set<string>, b: Set<string>) => {
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter || 1);
};

export function clusterTexts(ids: string[], texts: string[], thr = 0.35) {
  const G = texts.map(grams);
  const C: number[] = Array(texts.length).fill(-1);
  let cid = 0;
  
  for (let i = 0; i < texts.length; i++) {
    if (C[i] !== -1) continue;
    C[i] = cid;
    
    for (let j = i + 1; j < texts.length; j++) {
      const gi = G[i];
      const gj = G[j];
      if (gi && gj && C[j] === -1 && jaccard(gi, gj) >= thr) {
        C[j] = cid;
      }
    }
    cid++;
  }
  
  const groups: Record<number, string[]> = {};
  ids.forEach((id, i) => {
    const k = C[i];
    if (k !== undefined) {
      if (!groups[k]) groups[k] = [];
      groups[k].push(id);
    }
  });
  
  const labels: Record<number, string> = {};
  Object.entries(groups).forEach(([k, groupIds]) => {
    const i = parseInt(k, 10);
    const firstId = groupIds[0];
    if (firstId) {
      const idx = ids.indexOf(firstId);
      const rep = texts[idx] ?? '';
      labels[i] = rep.slice(0, 80);
    }
  });
  
  return { assignments: C, labels };
}