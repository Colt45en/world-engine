import { Router, Request, Response } from 'express';
import { ingestPain, recomputeClusters, summary, topClusters } from './service.js';
import { PainEvent } from './types.js';

export const painRouter = Router();

painRouter.post('/ingest', (req: Request, res: Response) => {
  const body = req.body as PainEvent;
  if (!body?.id || !body?.time || !body?.text) {
    return res.status(400).json({ error: 'id, time, text required' });
  }
  const rec = ingestPain(body);
  res.json(rec);
});

painRouter.post('/recluster', (_req: Request, res: Response) => {
  recomputeClusters();
  res.json({ ok: true });
});

painRouter.get('/summary', (_req: Request, res: Response) => res.json(summary()));

painRouter.get('/clusters', (_req: Request, res: Response) => res.json(topClusters(20)));