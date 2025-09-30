const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

// Basic World Engine API with Meta-Librarian Pipeline integration
const worldEngineAPI = {
  // Mock seed data
  seeds: {
    'excellent': 0.8,
    'good': 0.6,
    'okay': 0.2,
    'neutral': 0.0,
    'bad': -0.6,
    'terrible': -0.8
  },

  // AI Bot system
  aiBot: {
    interactions: [],
    knowledge: new Map(),
    stats: {
      total_interactions: 0,
      successful_responses: 0,
      learning_events: 0
    }
  },

  // Semantic Vector Converter
  vectorConverter: {
    state: [0.0, 0.5, 0.3, 0.6],
    operators: {
      'REBUILD': { scaling: [1.0, 1.2, 1.3, 0.8], bias: [0.0, 0.2, 0.2, -0.1] },
      'UPDATE': { scaling: [1.0, 1.1, 1.0, 1.2], bias: [0.0, 0.1, 0.0, 0.15] },
      'OPTIMIZE': { scaling: [1.0, 0.95, 1.1, 1.3], bias: [0.0, 0.0, 0.05, 0.2] },
      'ENHANCE': { scaling: [1.1, 1.2, 1.1, 1.1], bias: [0.05, 0.1, 0.05, 0.1] },
      'DEBUG': { scaling: [1.0, 1.2, 1.4, 0.9], bias: [0.0, 0.15, 0.25, -0.05] },
      'SIMPLIFY': { scaling: [1.0, 0.8, 0.7, 1.2], bias: [0.0, -0.1, -0.2, 0.1] },
      'AMPLIFY': { scaling: [1.3, 1.4, 1.2, 1.0], bias: [0.1, 0.2, 0.1, 0.0] },
      'STABILIZE': { scaling: [0.9, 0.9, 0.9, 1.1], bias: [0.0, 0.0, 0.0, 0.1] }
    },

    transform(operator, metadata = {}) {
      const oldState = [...this.state];
      const op = this.operators[operator];
      if (op) {
        for (let i = 0; i < 4; i++) {
          this.state[i] = Math.tanh(this.state[i] * op.scaling[i] + op.bias[i]);
        }
      }
      return { oldState, newState: [...this.state], operator, metadata };
    },

    getCurrentState() { return [...this.state]; },

    getStateDescription() {
      const [p, i, g, c] = this.state;
      return {
        polarity: { value: p, description: p > 0.2 ? 'positive' : p < -0.2 ? 'negative' : 'neutral' },
        intensity: { value: i, description: i > 0.7 ? 'high' : i < 0.3 ? 'low' : 'moderate' },
        granularity: { value: g, description: g > 0.7 ? 'detailed' : g < 0.3 ? 'coarse' : 'balanced' },
        confidence: { value: c, description: c > 0.7 ? 'confident' : c < 0.3 ? 'uncertain' : 'moderate' },
        magnitude: Math.sqrt(p*p + i*i + g*g + c*c)
      };
    }
  },

  analyzeWord(word) {
    return { word, score: this.seeds[word] || 0.0 };
  },

  analyzeText(text) {
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    const results = words.map(word => this.analyzeWord(word));
    const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length || 0;
    return {
      text: text,
      words: results,
      overall_score: avgScore,
      word_count: words.length,
      timestamp: new Date().toISOString()
    };
  },

  // Meta-Librarian Pipeline simulation
  processThroughPipeline(input, metadata = {}) {
    return {
      content: input,
      metadata: {
        pipeline_zones_processed: ['ROOT_EXTRACTION', 'CONTEXT_STRIPPING', 'AXIOMATIC_MAPPING', 'RELATIONSHIP_WEAVING', 'PATTERN_RECOGNITION', 'SYNTHETIC_REBUILD', 'ACTIONABLE_OUTPUT'],
        core_concepts: input.toLowerCase().match(/\b\w{4,}\b/g)?.slice(0, 5) || [],
        emergent_properties: input.length > 100 ? ['high_complexity_emergence'] : ['simple_structure'],
        actionable_output: {
          recommendations: input.includes('?') ? ['clarify_question_intent'] : ['continue_analysis'],
          action_plan: ['analyze_deeper', 'apply_transformations', 'generate_insights']
        }
      },
      zone_history: ['HEAD', 'ROOT_EXTRACTION', 'CONTEXT_STRIPPING', 'SYNTHETIC_REBUILD', 'ACTIONABLE_OUTPUT'],
      transformations: [
        {operator: 'REBUILD', zone: 'ROOT_EXTRACTION'},
        {operator: 'ENHANCE', zone: 'SYNTHETIC_REBUILD'}
      ],
      pipeline_active: true,
      timestamp: new Date().toISOString()
    };
  },

  // Enhanced chat with pipeline integration
  chatWithBot(message, context = {}) {
    const interactionId = Date.now().toString();

    // Process through pipeline
    const pipelineResult = this.processThroughPipeline(message, context);

    // Detect operators
    const detectedOperators = this.detectSemanticOperators(message);

    // Apply transformations
    let vectorResult = null;
    if (detectedOperators.length > 0) {
      vectorResult = this.vectorConverter.transform(detectedOperators[0], {
        source: 'chat_message',
        interaction_id: interactionId
      });
    }

    // Generate response
    let response = "I'm still learning about that topic. Can you tell me more?";
    let confidence = 0.3;
    let knowledgeSources = [];

    const lowerMessage = message.toLowerCase();
    const currentState = this.vectorConverter.getStateDescription();

    if (lowerMessage.includes('pipeline') || lowerMessage.includes('meta')) {
      response = `🌱 Meta-Librarian Pipeline Analysis:\n\nProcessed through zones: ${pipelineResult.zone_history.join(' → ')}\n\nCore concepts: ${pipelineResult.metadata.core_concepts.join(', ')}\n\nThe recursive pipeline analyzes your input through 7 cognitive zones with turtle-like progression through each analytical layer.`;
      confidence = 0.9;
      knowledgeSources = ['meta_librarian_pipeline'];
    } else if (lowerMessage.includes('transform') || detectedOperators.length > 0) {
      response = `Applied transformations: ${detectedOperators.join(', ')}\n\nSemantic state: ${currentState.polarity.description} polarity (${currentState.polarity.value.toFixed(2)}), ${currentState.intensity.description} intensity (${currentState.intensity.value.toFixed(2)})`;
      confidence = 0.85;
      knowledgeSources = ['vector_converter'];
    } else if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
      response = "Hello! I'm your AI assistant with Meta-Librarian Pipeline capabilities. I can process your input through 7 analytical zones and apply semantic transformations. Try asking about the pipeline or requesting transformations!";
      confidence = 0.95;
      knowledgeSources = ['general'];
    }

    // Record interaction
    this.aiBot.interactions.push({
      interaction_id: interactionId,
      input_text: message,
      bot_response: response,
      detected_operators: detectedOperators,
      vector_transformation: vectorResult,
      pipeline_result: pipelineResult,
      timestamp: new Date().toISOString()
    });

    this.aiBot.stats.total_interactions++;

    return {
      response,
      interaction_id: interactionId,
      confidence,
      knowledge_sources: knowledgeSources,
      detected_operators: detectedOperators,
      vector_transformation: vectorResult,
      pipeline_result: pipelineResult,
      world_engine_analysis: this.analyzeText(message)
    };
  },

  detectSemanticOperators(text) {
    const operators = [];
    const lowerText = text.toLowerCase();

    if (/rebuild|restructure/.test(lowerText)) operators.push('REBUILD');
    if (/enhance|boost|amplify/.test(lowerText)) operators.push('ENHANCE');
    if (/optimize|improve/.test(lowerText)) operators.push('OPTIMIZE');
    if (/debug|analyze/.test(lowerText)) operators.push('DEBUG');
    if (/simplify|reduce/.test(lowerText)) operators.push('SIMPLIFY');

    return operators;
  }
};

// HTTP Server
const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // API Routes
  if (pathname.startsWith('/api/')) {
    handleAPIRequest(req, res, pathname);
    return;
  }

  // Static file serving
  if (pathname.startsWith('/web/')) {
    serveStaticFile(req, res, pathname.substring(5));
    return;
  }

  // Redirect root to studio
  if (pathname === '/') {
    res.writeHead(302, { 'Location': '/web/studio.html' });
    res.end();
    return;
  }

  res.writeHead(404);
  res.end('Not Found');
});

function handleAPIRequest(req, res, pathname) {
  // AI Bot chat
  if (req.method === 'POST' && pathname === '/api/chat') {
    let body = '';
    req.on('data', chunk => body += chunk.toString());
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const result = worldEngineAPI.chatWithBot(data.message, data.context);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON' }));
      }
    });
    return;
  }

  // Pipeline processing
  if (req.method === 'POST' && pathname === '/api/pipeline/process') {
    let body = '';
    req.on('data', chunk => body += chunk.toString());
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const result = worldEngineAPI.processThroughPipeline(data.input, data.metadata);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Processing error' }));
      }
    });
    return;
  }

  // Pipeline canvas
  if (req.method === 'GET' && pathname === '/api/pipeline/canvas') {
    fs.readFile(path.join(__dirname, 'web', 'meta-librarian-canvas.html'), 'utf8', (err, data) => {
      if (err) {
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('<h1>Canvas not available</h1>');
      } else {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(data);
      }
    });
    return;
  }

  // Vector transform
  if (req.method === 'POST' && pathname === '/api/vector/transform') {
    let body = '';
    req.on('data', chunk => body += chunk.toString());
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const result = worldEngineAPI.vectorConverter.transform(data.operator, data.metadata);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Transform error' }));
      }
    });
    return;
  }

  // Analysis endpoint
  if (req.method === 'POST' && pathname === '/api/analyze') {
    let body = '';
    req.on('data', chunk => body += chunk.toString());
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const result = worldEngineAPI.analyzeText(data.text);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Analysis error' }));
      }
    });
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'API endpoint not found' }));
}

function serveStaticFile(req, res, filePath) {
  const fullPath = path.join(__dirname, 'web', filePath);

  fs.readFile(fullPath, (error, content) => {
    if (error) {
      res.writeHead(404, { 'Content-Type': 'text/html' });
      res.end(`
        <!DOCTYPE html>
        <html>
        <head><title>404 - Not Found</title></head>
        <body>
          <h1>404 - File Not Found</h1>
          <p>The file "${filePath}" was not found.</p>
          <p><a href="/web/studio.html">Go to World Engine Studio</a></p>
        </body>
        </html>
      `);
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentTypes = {
      '.html': 'text/html',
      '.js': 'application/javascript',
      '.css': 'text/css',
      '.json': 'application/json',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.gif': 'image/gif',
      '.svg': 'image/svg+xml'
    };

    res.writeHead(200, { 'Content-Type': contentTypes[ext] || 'text/plain' });
    res.end(content, 'utf-8');
  });
}

const PORT = process.env.PORT || 8001;
server.listen(PORT, () => {
  console.log(`
🌍 World Engine Studio Server Running
🚀 Server: http://localhost:${PORT}
🌱 Meta-Librarian Pipeline: http://localhost:${PORT}/api/pipeline/canvas
💬 AI Bot Chat Interface: http://localhost:${PORT}/web/studio.html
🔀 Vector Transformations: Available via API
📊 Real-time Analysis: Integrated
  `);
});
