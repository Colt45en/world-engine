// Minimal clustering stub for services/pain/cluster.js (ESM)
export function clusterTexts(ids = [], texts = []) {
  const assignments = texts.map(() => 0);
  const labels = ['general'];
  return { assignments, labels };
}
