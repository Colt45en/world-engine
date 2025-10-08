# Pain Detection System Setup

## Quick Start

1. **Install dependencies:**
```bash
npm install express body-parser
pip install scikit-learn joblib fastapi uvicorn
```

2. **Start the Pain API server:**
```bash
cd services/pain
npx ts-node server.ts
```

3. **Test the API:**
```bash
# Ingest a pain event
curl -X POST http://localhost:3001/api/pain/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "id": "1",
    "time": "2025-10-06T12:00:00Z",
    "source": "issues",
    "text": "webhook retries keep failing, stuck on manual steps",
    "eng": 12,
    "org": "Contoso"
  }'

# Get summary
curl http://localhost:3001/api/pain/summary

# Recluster events
curl -X POST http://localhost:3001/api/pain/recluster

# Get top clusters
curl http://localhost:3001/api/pain/clusters
```

## System Components

### 1. Core Services
- `services/pain/types.ts` - Type definitions
- `services/pain/service.ts` - Main business logic
- `services/pain/router.ts` - Express API routes
- `services/pain/server.ts` - Standalone server

### 2. ML Pipeline
- `services/finder/weak_labeler.py` - Heuristic labeling
- `datasets/opportunity_pain.schema.json` - Data schema

### 3. API Endpoints
- `POST /api/pain/ingest` - Add new pain event
- `GET /api/pain/summary` - Get metrics summary
- `GET /api/pain/clusters` - Get pain clusters
- `POST /api/pain/recluster` - Recompute clusters

## Example Usage

```javascript
// Ingest pain from various sources
const painEvent = {
  id: crypto.randomUUID(),
  time: new Date().toISOString(),
  source: 'issues',
  text: 'API integration fails with timeout errors',
  eng: 15,
  org: 'CustomerCorp'
};

fetch('http://localhost:3001/api/pain/ingest', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(painEvent)
});
```

## Next Steps

1. **Connect to your existing data sources** (RSS feeds, GitHub issues, etc.)
2. **Train the ML classifier** using the weak labeler
3. **Add dashboard integration** to your existing UI
4. **Set up alerts** for high-burst periods

The system is now ready to start collecting and analyzing user pain points!