// ESM router replacement for services/pain
import express from 'express';
import { ingestPain, recomputeClusters, summary, topClusters } from './service.js';

const router = express.Router();

router.post('/ingest', (req, res) => {
  const body = req.body || {};
  if (!body.id || !body.time || !body.text) {
    return res.status(400).json({ error: 'id, time, text required' });
  }
  const rec = ingestPain(body);
  res.json(rec);
});

router.post('/recluster', (_req, res) => {
  recomputeClusters();
  res.json({ ok: true });
});

router.get('/summary', (_req, res) => res.json(summary()));

router.get('/clusters', (_req, res) => res.json(topClusters(20)));

export default router;
