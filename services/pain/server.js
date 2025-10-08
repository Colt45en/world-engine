import express from 'express';
import router from './router.js';

const app = express();
app.use(express.json());

// CORS
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  next();
});

app.use('/api/pain', router);

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/', (req, res) => {
  res.json({ status: 'pain service (stub)', uptime: process.uptime() });
});

const PORT = process.env.PAIN_PORT || process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Pain service (stub) listening on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
});