const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

// Import and setup Meta-Librarian Pipeline integration
let MetaLibrarianPipeline = null;
let pipelineInstance = null;

// Try to import Python pipeline via child_process for integration
const { spawn } = require('child_process');

// Meta-Librarian Pipeline integration adapter
const createPipelineAdapter = () => {
  return {
    processThroughPipeline: async (inputData, metadata = {}) => {
      return new Promise((resolve, reject) => {
        // Call Python pipeline via subprocess
        const pythonProcess = spawn('python', [
          '-c', `
import sys
sys.path.append('${__dirname.replace(/\\/g, '/')}')
try:
    from meta_librarian_pipeline import MetaLibrarianPipeline
    from semantic_vector_converter import SemanticVectorConverter
    import json

    # Create vector converter
    vector_converter = SemanticVectorConverter()

    # Create pipeline instance
    pipeline = MetaLibrarianPipeline(vector_converter)

    # Process input data
    input_data = """${inputData.replace(/"/g, '\\"')}"""
    metadata = ${JSON.stringify(metadata)}

    result = pipeline.process_through_pipeline(input_data, metadata)

    # Convert result to JSON
    output = {
        'content': result.content,
        'metadata': result.metadata,
        'zone_history': result.zone_history,
        'transformations': result.transformations,
        'vector_state': result.vector_state,
        'timestamp': result.timestamp
    }

    print(json.dumps(output))

except Exception as e:
    print(json.dumps({'error': str(e)}))
`
        ]);

        let output = '';
        pythonProcess.stdout.on('data', (data) => {
          output += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
          console.error('Pipeline stderr:', data.toString());
        });

        pythonProcess.on('close', (code) => {
          try {
            const result = JSON.parse(output.trim());
            resolve(result);
          } catch (e) {
            reject(new Error(`Pipeline failed: ${e.message}`));
          }
        });
      });
    },

    getGrowthMetrics: () => {
      return {
        total_processes: 0,
        feedback_cycles: 0,
        active_branches: 0,
        complexity_evolution: []
      };
    },

    harvestInsights: () => {
      return {
        insights: [],
        frameworks_created: 0,
        total_cycles_completed: 0
      };
    }
  };
};

// Initialize pipeline adapter
const pipelineAdapter = createPipelineAdapter();

// Basic World Engine API mock for development
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

  // Semantic Vector Converter System
  vectorConverter: {
    state: [0.0, 0.5, 0.3, 0.6], // [polarity, intensity, granularity, confidence]
    snapshots: [],
    transformationHistory: [],

    // Operator library for transformations
    operators: {
      'REBUILD': { scaling: [1.0, 1.2, 1.3, 0.8], bias: [0.0, 0.2, 0.2, -0.1] },
      'UPDATE': { scaling: [1.0, 1.1, 1.0, 1.2], bias: [0.0, 0.1, 0.0, 0.15] },
      'OPTIMIZE': { scaling: [1.0, 0.95, 1.1, 1.3], bias: [0.0, 0.0, 0.05, 0.2] },
      'ENHANCE': { scaling: [1.1, 1.2, 1.1, 1.1], bias: [0.05, 0.1, 0.05, 0.1] },
      'DEBUG': { scaling: [1.0, 1.2, 1.4, 0.9], bias: [0.0, 0.15, 0.25, -0.05] },
      'SIMPLIFY': { scaling: [1.0, 0.8, 0.7, 1.2], bias: [0.0, -0.1, -0.2, 0.1] },
      'AMPLIFY': { scaling: [1.3, 1.4, 1.2, 1.0], bias: [0.1, 0.2, 0.1, 0.0] },
      'STABILIZE': { scaling: [0.9, 0.9, 0.9, 1.1], bias: [0.0, 0.0, 0.0, 0.1] },
      'POSITIVE': { scaling: [1.2, 1.0, 1.0, 1.0], bias: [0.3, 0.0, 0.0, 0.0] },
      'NEGATIVE': { scaling: [1.2, 1.0, 1.0, 1.0], bias: [-0.3, 0.0, 0.0, 0.0] },
      'NEUTRAL': { scaling: [0.5, 1.0, 1.0, 1.0], bias: [0.0, 0.0, 0.0, 0.0] },
      'LEARN': { scaling: [1.0, 1.1, 1.0, 1.2], bias: [0.0, 0.1, 0.0, 0.15] },
      'STATUS': { scaling: [1.0, 1.0, 1.0, 1.0], bias: [0.0, 0.0, 0.0, 0.0] },
      'RESTORE': { scaling: [1.0, 1.0, 1.0, 1.0], bias: [0.0, 0.0, 0.0, 0.0] },
      'RESET': { scaling: [0.5, 0.5, 0.5, 0.5], bias: [0.0, 0.25, 0.15, 0.3] }
    },

    applyTransformation(operator, metadata = {}) {
      const oldState = [...this.state];

      // Handle special operators
      if (operator === 'STATUS') {
        this.snapshots.push([...this.state]);
        return {
          oldState,
          newState: [...this.state],
          operator,
          timestamp: new Date().toISOString(),
          metadata: { action: 'snapshot_saved', count: this.snapshots.length }
        };
      }

      if (operator === 'RESTORE' && this.snapshots.length > 0) {
        this.state = [...this.snapshots[this.snapshots.length - 1]];
        return {
          oldState,
          newState: [...this.state],
          operator,
          timestamp: new Date().toISOString(),
          metadata: { action: 'restored_from_snapshot' }
        };
      }

      // Apply standard transformation
      const op = this.operators[operator];
      if (op) {
        for (let i = 0; i < 4; i++) {
          this.state[i] = (this.state[i] * op.scaling[i]) + op.bias[i];
          // Clamp to reasonable bounds
          this.state[i] = Math.max(-2.0, Math.min(2.0, this.state[i]));
        }
      }

      const result = {
        oldState,
        newState: [...this.state],
        operator,
        timestamp: new Date().toISOString(),
        metadata
      };

      this.transformationHistory.push(result);
      if (this.transformationHistory.length > 100) {
        this.transformationHistory = this.transformationHistory.slice(-50);
      }

      return result;
    },

    runSequence(operators, metadata = {}) {
      const results = [];
      for (const op of operators) {
        results.push(this.applyTransformation(op, { ...metadata, sequenceStep: results.length }));
      }
      return {
        finalState: [...this.state],
        transformations: results,
        sequence: operators
      };
    },

    getStateDescription() {
      const [p, i, g, c] = this.state;
      return {
        polarity: {
          value: p,
          description: p > 0.2 ? 'positive' : p < -0.2 ? 'negative' : 'neutral'
        },
        intensity: {
          value: i,
          description: i > 0.7 ? 'high' : i < 0.3 ? 'low' : 'moderate'
        },
        granularity: {
          value: g,
          description: g > 0.7 ? 'detailed' : g < 0.3 ? 'coarse' : 'balanced'
        },
        confidence: {
          value: c,
          description: c > 0.7 ? 'confident' : c < 0.3 ? 'uncertain' : 'moderate'
        },
        vector: this.state,
        magnitude: Math.sqrt(this.state.reduce((sum, x) => sum + x * x, 0))
      };
    },

    suggestNextOperators(targetState = [0.0, 0.6, 0.5, 0.8]) {
      const suggestions = [];
      const currentState = [...this.state];

      for (const [opName, op] of Object.entries(this.operators)) {
        if (opName === 'STATUS' || opName === 'RESTORE') continue;

        // Simulate transformation
        const predicted = currentState.map((val, i) =>
          Math.max(-2.0, Math.min(2.0, (val * op.scaling[i]) + op.bias[i]))
        );

        // Calculate distance improvement
        const currentDistance = Math.sqrt(
          currentState.reduce((sum, val, i) => sum + Math.pow(val - targetState[i], 2), 0)
        );
        const predictedDistance = Math.sqrt(
          predicted.reduce((sum, val, i) => sum + Math.pow(val - targetState[i], 2), 0)
        );

        const improvement = Math.max(0, currentDistance - predictedDistance);
        suggestions.push({ operator: opName, effectiveness: improvement });
      }

      return suggestions
        .sort((a, b) => b.effectiveness - a.effectiveness)
        .slice(0, 5);
    }
  },

  // Mock AI bot knowledge system
  aiBot: {
    knowledge: [
      { id: '1', content: 'World Engine is a lexicon processing and semantic analysis system', category: 'system', confidence: 0.95 },
      { id: '2', content: 'Use /run <text> to analyze content, /rec start to begin recording', category: 'commands', confidence: 0.9 },
      { id: '3', content: 'The system uses hand-labeled seeds for sentiment analysis', category: 'semantic', confidence: 0.85 }
    ],
    interactions: [],
    stats: {
      knowledge_entries: 3,
      total_interactions: 0,
      average_confidence: 0.9,
      success_rate: 1.0,
      world_engine_connected: true
    }
  },

  // Mock analysis function
  analyzeWord(word, context = '') {
    const baseScore = this.seeds[word.toLowerCase()] || 0;
    return {
      word: word,
      score: baseScore,
      confidence: 0.8,
      context: context,
      timestamp: new Date().toISOString()
    };
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

  // AI Bot chat function
  // AI Bot chat function with Meta-Librarian Pipeline and vector converter integration
  chatWithBot(message, context = {}) {
    const interactionId = Date.now().toString();

    // For now, create a simple pipeline simulation until Python integration is ready
    const simulatePipelineProcessing = (input) => {
      return {
        content: input,
        metadata: {
          pipeline_zones_processed: ['ROOT_EXTRACTION', 'CONTEXT_STRIPPING', 'AXIOMATIC_MAPPING', 'RELATIONSHIP_WEAVING', 'PATTERN_RECOGNITION', 'SYNTHETIC_REBUILD', 'ACTIONABLE_OUTPUT'],
          core_concepts: input.toLowerCase().match(/\b\w{4,}\b/g)?.slice(0, 3) || [],
          emergent_properties: input.length > 100 ? ['high_complexity_emergence'] : ['simple_structure'],
          actionable_output: {
            recommendations: input.includes('?') ? ['clarify_question_intent'] : ['continue_analysis'],
            action_plan: ['analyze_deeper', 'apply_transformations', 'generate_insights']
          }
        },
        zone_history: ['HEAD', 'ROOT_EXTRACTION', 'SYNTHETIC_REBUILD', 'ACTIONABLE_OUTPUT'],
        transformations: [],
        pipeline_active: true
      };
    };

    // Process through Meta-Librarian Pipeline simulation
    const pipelineResult = simulatePipelineProcessing(message);

    // Analyze message for semantic operators
    const detectedOperators = this.detectSemanticOperators(message);

    // Apply vector transformations based on message content
    let vectorResult = null;
    if (detectedOperators.length > 0) {
      vectorResult = this.vectorConverter.runSequence(detectedOperators, {
        source: 'chat_message',
        message: message,
        interaction_id: interactionId,
        pipeline_processing: true
      });
    }

    // Generate response based on pipeline analysis and vector state
    let response = "I'm still learning about that topic. Can you tell me more?";
    let confidence = 0.3;
    let knowledgeSources = [];

    const lowerMessage = message.toLowerCase();
    const currentState = this.vectorConverter.getStateDescription();

    // Enhanced response generation with pipeline integration
    if (pipelineResult && pipelineResult.metadata && pipelineResult.metadata.actionable_output) {
      const actionableOutput = pipelineResult.metadata.actionable_output;
      const recommendations = actionableOutput.recommendations || [];
      const actionPlan = actionableOutput.action_plan || [];
      const coreConcepts = pipelineResult.metadata.core_concepts || [];

      response = `🌱 Meta-Librarian Pipeline Analysis:\n\n`;
      response += `**Zones processed:** ${pipelineResult.zone_history.join(' → ')}\n\n`;

      if (coreConcepts.length > 0) {
        response += `**Core concepts extracted:** ${coreConcepts.join(', ')}\n\n`;
      }

      if (recommendations.length > 0) {
        response += `**Recommendations:** ${recommendations.join(', ')}\n\n`;
      }

      if (pipelineResult.metadata.emergent_properties) {
        response += `**Emergent properties:** ${pipelineResult.metadata.emergent_properties.join(', ')}\n\n`;
      }

      response += `The recursive pipeline has analyzed your input through multiple cognitive zones, extracting patterns and generating actionable insights.`;

      confidence = 0.9;
      knowledgeSources = ['meta_librarian_pipeline', 'deep_analysis'];

    } else if (lowerMessage.includes('pipeline') || lowerMessage.includes('meta') || lowerMessage.includes('recursive') || lowerMessage.includes('librarian')) {
      response = `🌱 The Meta-Librarian Recursive Pipeline processes your input through 7 zones:\n\n🐢 ROOT EXTRACTION → 🐢 CONTEXT STRIPPING → 🐢 AXIOMATIC MAPPING → 🐢 RELATIONSHIP WEAVING → 🐢 PATTERN RECOGNITION → 🐢 SYNTHETIC REBUILD → 🚀 ACTIONABLE OUTPUT\n\nEach zone can branch or nest for parallel analysis. The pipeline features recursive feedback loops where outputs flow back to the HEAD zone for continuous learning. Try asking complex questions to see the full analysis!`;
      confidence = 0.95;
      knowledgeSources = ['pipeline_info'];

    } else if (lowerMessage.includes('zone') || lowerMessage.includes('camera') || lowerMessage.includes('zoom')) {
      response = `🎥 Camera Zones allow multi-scale analysis:\n• **Zoom in** for micro-analysis (see inside the turtle shells 🐢)\n• **Zoom out** for macro-patterns (see fields/branches as a whole)\n• **Pan across** for cross-domain links\n\nThe agricultural metaphor tracks how the pipeline grows: roots, vines, branches showing adaptation over time with irrigation (data flows) and harvest (actionable outputs).`;
      confidence = 0.88;
      knowledgeSources = ['camera_zones', 'agriculture_metaphor'];

    } else if (lowerMessage.includes('transform') || lowerMessage.includes('convert') || detectedOperators.length > 0) {
      const stateDesc = currentState;
      response = `Applied transformations: ${detectedOperators.join(', ')}.\n\nCurrent semantic state: ${stateDesc.polarity.description} polarity (${stateDesc.polarity.value.toFixed(2)}), ${stateDesc.intensity.description} intensity (${stateDesc.intensity.value.toFixed(2)}), ${stateDesc.granularity.description} granularity, ${stateDesc.confidence.description} confidence.`;
      confidence = 0.9;
      knowledgeSources = ['vector_converter', 'transformation'];

      // Add suggestions for next steps
      const suggestions = this.vectorConverter.suggestNextOperators();
      if (suggestions.length > 0) {
        response += `\n\nSuggested next operators: ${suggestions.slice(0, 3).map(s => s.operator).join(', ')}.`;
      }

    } else if (lowerMessage.includes('state') || lowerMessage.includes('status')) {
      const stateDesc = currentState;
      response = `Current semantic vector state:\n• Polarity: ${stateDesc.polarity.description} (${stateDesc.polarity.value.toFixed(2)})\n• Intensity: ${stateDesc.intensity.description} (${stateDesc.intensity.value.toFixed(2)})\n• Granularity: ${stateDesc.granularity.description} (${stateDesc.granularity.value.toFixed(2)})\n• Confidence: ${stateDesc.confidence.description} (${stateDesc.confidence.value.toFixed(2)})\n• Vector magnitude: ${stateDesc.magnitude.toFixed(2)}`;
      confidence = 0.95;
      knowledgeSources = ['vector_state'];

    } else if (lowerMessage.includes('world engine') || lowerMessage.includes('system')) {
      response = "World Engine is a lexicon processing and semantic analysis system with integrated recording, chat interface, and real-time analysis capabilities. It now includes the Meta-Librarian Recursive Pipeline for deep cognitive processing through 7 analytical zones, plus an advanced vector transformation system where each semantic operation evolves a 4D state space (polarity, intensity, granularity, confidence).";
      confidence = 0.9;
      knowledgeSources = ['system', 'vector_converter', 'meta_librarian'];

    } else if (lowerMessage.includes('command') || lowerMessage.includes('how') || lowerMessage.includes('use')) {
      response = `Commands available:\n• \`/run <text>\` - analyze content through World Engine\n• \`/rec start\` - begin recording\n• \`/help\` - full command list\n• Transform with operators: REBUILD, ENHANCE, OPTIMIZE, AMPLIFY, etc.\n• For deep analysis, ask complex questions to trigger Meta-Librarian Pipeline processing through its 7 recursive zones!`;
      confidence = 0.85;
      knowledgeSources = ['commands', 'vector_operations', 'pipeline_usage'];

    } else if (lowerMessage.includes('semantic') || lowerMessage.includes('analysis')) {
      response = "The World Engine uses semantic scaling with hand-labeled seeds (-1.0 to 1.0). The vector transformation system extends this with a 4D state space evolving through operator sequences. The Meta-Librarian Pipeline adds recursive processing: ROOT EXTRACTION 🐢 → CONTEXT STRIPPING 🐢 → AXIOMATIC MAPPING 🐢 → RELATIONSHIP WEAVING 🐢 → PATTERN RECOGNITION 🐢 → SYNTHETIC REBUILD 🐢 → ACTIONABLE OUTPUT 🚀 with recursive feedback loops.";
      confidence = 0.8;
      knowledgeSources = ['semantic', 'vector_converter', 'pipeline_analysis'];

    } else if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('help')) {
      response = "Hello! I'm your AI assistant for World Engine Studio with advanced capabilities:\n\n🌱 **Meta-Librarian Recursive Pipeline** - Deep analysis through 7 cognitive zones\n🔀 **Vector Transformation System** - 4D semantic state evolution\n🧠 **Learning & Memory** - Persistent knowledge from conversations\n\nTry asking complex questions, request transformations like 'enhance with OPTIMIZE', or explore the pipeline zones!";
      confidence = 0.95;
      knowledgeSources = ['general', 'vector_intro', 'pipeline_intro'];
    }

    // Record interaction with pipeline and vector state
    this.aiBot.interactions.push({
      interaction_id: interactionId,
      input_text: message,
      bot_response: response,
      detected_operators: detectedOperators,
      vector_transformation: vectorResult,
      semantic_state: currentState,
      pipeline_result: pipelineResult,
      timestamp: new Date().toISOString(),
      world_engine_analysis: this.analyzeText(message)
    });

    // Update stats
    this.aiBot.stats.total_interactions++;

    return {
      response: response,
      interaction_id: interactionId,
      confidence: confidence,
      knowledge_sources: knowledgeSources,
      detected_operators: detectedOperators,
      vector_transformation: vectorResult,
      semantic_state: currentState,
      pipeline_result: pipelineResult,
      world_engine_analysis: this.analyzeText(message)
    };
  },

    // Analyze message for semantic operators
    const detectedOperators = this.detectSemanticOperators(message);

    // Apply vector transformations based on message content
    let vectorResult = null;
    if (detectedOperators.length > 0) {
      vectorResult = this.vectorConverter.runSequence(detectedOperators, {
        source: 'chat_message',
        message: message,
        interaction_id: interactionId
      });
    }

    // Generate response based on message content and vector state
    let response = "I'm still learning about that topic. Can you tell me more?";
    let confidence = 0.3;
    let knowledgeSources = [];

    const lowerMessage = message.toLowerCase();
    const currentState = this.vectorConverter.getStateDescription();

    // Enhanced response generation with vector state awareness
    if (lowerMessage.includes('transform') || lowerMessage.includes('convert') || detectedOperators.length > 0) {
      const stateDesc = currentState;
      response = `I've applied the transformations: ${detectedOperators.join(', ')}. Current semantic state: ${stateDesc.polarity.description} polarity (${stateDesc.polarity.value.toFixed(2)}), ${stateDesc.intensity.description} intensity (${stateDesc.intensity.value.toFixed(2)}), ${stateDesc.granularity.description} granularity, ${stateDesc.confidence.description} confidence.`;
      confidence = 0.9;
      knowledgeSources = ['vector_converter', 'transformation'];

      // Add suggestions for next steps
      const suggestions = this.vectorConverter.suggestNextOperators();
      if (suggestions.length > 0) {
        response += ` Suggested next operators: ${suggestions.slice(0, 3).map(s => s.operator).join(', ')}.`;
      }
    } else if (lowerMessage.includes('state') || lowerMessage.includes('status')) {
      const stateDesc = currentState;
      response = `Current semantic vector state: Polarity is ${stateDesc.polarity.description} (${stateDesc.polarity.value.toFixed(2)}), Intensity is ${stateDesc.intensity.description} (${stateDesc.intensity.value.toFixed(2)}), Granularity is ${stateDesc.granularity.description} (${stateDesc.granularity.value.toFixed(2)}), Confidence is ${stateDesc.confidence.description} (${stateDesc.confidence.value.toFixed(2)}). Vector magnitude: ${stateDesc.magnitude.toFixed(2)}.`;
      confidence = 0.95;
      knowledgeSources = ['vector_state'];
    } else if (lowerMessage.includes('world engine') || lowerMessage.includes('system')) {
      response = "World Engine is a lexicon processing and semantic analysis system with integrated recording, chat interface, and real-time analysis capabilities. It now includes an advanced vector transformation system where each semantic operation evolves a 4D state space representing polarity, intensity, granularity, and confidence.";
      confidence = 0.9;
      knowledgeSources = ['system', 'vector_converter'];
    } else if (lowerMessage.includes('command') || lowerMessage.includes('how') || lowerMessage.includes('use')) {
      response = `You can use commands like \`/run <text>\` to analyze content, \`/rec start\` to begin recording, and \`/help\` for a full command list. The studio also supports semantic vector operations like REBUILD, UPDATE, OPTIMIZE, ENHANCE, AMPLIFY, and more. Try saying 'transform with REBUILD' or 'apply ENHANCE OPTIMIZE sequence' to modify the semantic state.`;
      confidence = 0.85;
      knowledgeSources = ['commands', 'vector_operations'];
    } else if (lowerMessage.includes('semantic') || lowerMessage.includes('analysis')) {
      response = "The World Engine uses semantic scaling with hand-labeled seeds ranging from -1.0 to 1.0. The new vector transformation system extends this with a 4D state space that evolves through operator sequences. Each transformation applies mathematical operations to shift the semantic understanding along dimensions of polarity, intensity, granularity, and confidence.";
      confidence = 0.8;
      knowledgeSources = ['semantic', 'vector_converter'];
    } else if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('help')) {
      response = "Hello! I'm your AI assistant for World Engine Studio with advanced vector transformation capabilities. I can help you understand semantic operations, apply vector transformations to evolve meaning spaces, and learn from our conversations. Try commands like 'show state', 'transform with ENHANCE', or ask about available operators!";
      confidence = 0.95;
      knowledgeSources = ['general', 'vector_intro'];
    }

    // Record interaction with vector state
    this.aiBot.interactions.push({
      interaction_id: interactionId,
      input_text: message,
      bot_response: response,
      detected_operators: detectedOperators,
      vector_transformation: vectorResult,
      semantic_state: currentState,
      timestamp: new Date().toISOString(),
      world_engine_analysis: this.analyzeText(message)
    });

    // Update stats
    this.aiBot.stats.total_interactions++;

    return {
      response: response,
      interaction_id: interactionId,
      confidence: confidence,
      knowledge_sources: knowledgeSources,
      detected_operators: detectedOperators,
      vector_transformation: vectorResult,
      semantic_state: currentState,
      world_engine_analysis: this.analyzeText(message)
    };
  },

  // Detect semantic operators in natural language
  detectSemanticOperators(text) {
    const operators = [];
    const lowerText = text.toLowerCase();

    // Map natural language to operators
    const operatorPatterns = {
      'rebuild': /rebuild|restructure|recreate/,
      'update': /update|refresh|modify/,
      'optimize': /optimize|improve|tune/,
      'enhance': /enhance|boost|amplify|strengthen/,
      'debug': /debug|analyze|examine|investigate/,
      'simplify': /simplify|reduce|streamline/,
      'stabilize': /stabilize|balance|steady/,
      'positive': /positive|good|better|improve/,
      'negative': /negative|bad|worse|problem/,
      'neutral': /neutral|balanced|center/,
      'learn': /learn|study|remember|absorb/,
      'reset': /reset|clear|start over|beginning/,
      'status': /status|state|current|show/,
      'restore': /restore|rollback|undo|previous/
    };

    // Check for explicit operator mentions
    for (const [op, pattern] of Object.entries(operatorPatterns)) {
      if (pattern.test(lowerText)) {
        operators.push(op.toUpperCase());
      }
    }

    // Look for transformation sequences
    const sequenceMatch = lowerText.match(/(?:transform|apply|run)\s+(?:with\s+)?([a-z\s]+)(?:sequence)?/);
    if (sequenceMatch) {
      const words = sequenceMatch[1].split(/\s+/);
      for (const word of words) {
        const upperWord = word.toUpperCase();
        if (this.vectorConverter.operators[upperWord]) {
          if (!operators.includes(upperWord)) {
            operators.push(upperWord);
          }
        }
      }
    }

    return [...new Set(operators)]; // Remove duplicates
  }
};

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
  let filePath = '.' + pathname;
  if (filePath === './') filePath = './web/studio.html';

  const extname = String(path.extname(filePath)).toLowerCase();
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.wav': 'audio/wav',
    '.mp4': 'video/mp4',
    '.woff': 'application/font-woff',
    '.ttf': 'application/font-ttf',
    '.eot': 'application/vnd.ms-fontobject',
    '.otf': 'application/font-otf',
    '.wasm': 'application/wasm'
  };

  const contentType = mimeTypes[extname] || 'application/octet-stream';

  fs.readFile(filePath, (error, content) => {
    if (error) {
      if (error.code == 'ENOENT') {
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end(`
          <!DOCTYPE html>
          <html>
          <head><title>404 - Not Found</title></head>
          <body>
            <h1>404 - File Not Found</h1>
            <p>The requested file "${pathname}" was not found.</p>
            <p><a href="/web/studio.html">Go to World Engine Studio</a></p>
          </body>
          </html>
        `);
      } else {
        res.writeHead(500);
        res.end('Server error: ' + error.code);
      }
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    }
  });
});

function handleAPIRequest(req, res, pathname) {
  if (req.method === 'GET' && pathname === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: 'ok',
      service: 'World Engine (Node.js Mock)',
      timestamp: new Date().toISOString()
    }));
    return;
  }

  if (req.method === 'POST' && pathname === '/api/analyze') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const result = worldEngineAPI.analyzeText(data.text || '');
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON' }));
      }
    });
    return;
  }

  if (req.method === 'GET' && pathname === '/api/seeds') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(worldEngineAPI.seeds));
    return;
  }

  // AI Bot endpoints
  if (req.method === 'POST' && pathname === '/api/ai-bot/chat') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
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

  if (req.method === 'POST' && pathname === '/api/ai-bot/feedback') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        // Mock feedback processing
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'success', message: 'Feedback recorded' }));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON' }));
      }
    });
    return;
  }

  if (req.method === 'GET' && pathname === '/api/ai-bot/stats') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(worldEngineAPI.aiBot.stats));
    return;
  }

  if (req.method === 'GET' && pathname === '/api/ai-bot/knowledge') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(worldEngineAPI.aiBot.knowledge));
    return;
  }

  // Meta-Librarian Pipeline endpoints
  if (req.method === 'POST' && pathname === '/api/pipeline/process') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);

        // Use the simplified pipeline simulation for now
        const simulatePipelineProcessing = (input, metadata = {}) => {
          return {
            content: input,
            metadata: {
              pipeline_zones_processed: ['ROOT_EXTRACTION', 'CONTEXT_STRIPPING', 'AXIOMATIC_MAPPING', 'RELATIONSHIP_WEAVING', 'PATTERN_RECOGNITION', 'SYNTHETIC_REBUILD', 'ACTIONABLE_OUTPUT'],
              core_concepts: input.toLowerCase().match(/\\b\\w{4,}\\b/g)?.slice(0, 5) || [],
              emergent_properties: input.length > 100 ? ['high_complexity_emergence', 'deep_analysis_required'] : ['simple_structure', 'direct_processing'],
              axioms: input.includes('?') ? ['interrogative_assumption'] : input.includes('!') ? ['declarative_certainty'] : ['neutral_stance'],
              relationships: input.split(' ').length > 10 ? [{type: 'complex_discourse', strength: 'high'}] : [{type: 'simple_statement', strength: 'medium'}],
              patterns: {
                micro: input.includes('.') ? ['sentence_structure'] : ['fragment_pattern'],
                macro: input.length > 200 ? ['essay_form'] : ['brief_communication']
              },
              actionable_output: {
                recommendations: input.includes('?') ? ['clarify_question_intent', 'provide_context'] : ['continue_analysis', 'apply_transformations'],
                action_plan: ['analyze_deeper', 'apply_transformations', 'generate_insights', 'validate_results'],
                implementation_steps: ['extract_core', 'build_connections', 'synthesize_framework']
              },
              processing_summary: {
                zones_traversed: 7,
                transformations_applied: Math.floor(Math.random() * 5) + 3,
                recursive_depth: Math.floor(input.length / 100) + 1
              },
              ...metadata
            },
            zone_history: ['HEAD', 'ROOT_EXTRACTION', 'CONTEXT_STRIPPING', 'AXIOMATIC_MAPPING', 'RELATIONSHIP_WEAVING', 'PATTERN_RECOGNITION', 'SYNTHETIC_REBUILD', 'ACTIONABLE_OUTPUT'],
            transformations: [
              {operator: 'REBUILD', zone: 'ROOT_EXTRACTION', timestamp: new Date().toISOString()},
              {operator: 'SIMPLIFY', zone: 'CONTEXT_STRIPPING', timestamp: new Date().toISOString()},
              {operator: 'DEBUG', zone: 'AXIOMATIC_MAPPING', timestamp: new Date().toISOString()},
              {operator: 'ENHANCE', zone: 'RELATIONSHIP_WEAVING', timestamp: new Date().toISOString()},
              {operator: 'OPTIMIZE', zone: 'PATTERN_RECOGNITION', timestamp: new Date().toISOString()},
              {operator: 'AMPLIFY', zone: 'SYNTHETIC_REBUILD', timestamp: new Date().toISOString()},
              {operator: 'STABILIZE', zone: 'ACTIONABLE_OUTPUT', timestamp: new Date().toISOString()}
            ],
            vector_state: worldEngineAPI.vectorConverter.getCurrentState(),
            pipeline_active: true,
            timestamp: new Date().toISOString()
          };
        };

        const result = simulatePipelineProcessing(data.input, data.metadata);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON or processing error' }));
      }
    });
    return;
  }

  if (req.method === 'GET' && pathname === '/api/pipeline/metrics') {
    const metrics = {
      total_processes: Math.floor(Math.random() * 100) + 10,
      feedback_cycles: Math.floor(Math.random() * 20) + 2,
      active_branches: Math.floor(Math.random() * 5),
      complexity_evolution: [0.2, 0.4, 0.6, 0.8, 0.9, 1.1, 1.3],
      zones_performance: {
        'ROOT_EXTRACTION': { avg_time: 120, success_rate: 0.95 },
        'CONTEXT_STRIPPING': { avg_time: 80, success_rate: 0.92 },
        'AXIOMATIC_MAPPING': { avg_time: 200, success_rate: 0.88 },
        'RELATIONSHIP_WEAVING': { avg_time: 150, success_rate: 0.91 },
        'PATTERN_RECOGNITION': { avg_time: 180, success_rate: 0.89 },
        'SYNTHETIC_REBUILD': { avg_time: 250, success_rate: 0.85 },
        'ACTIONABLE_OUTPUT': { avg_time: 100, success_rate: 0.98 }
      },
      recursive_feedback_efficiency: 0.76,
      harvest_insights: {
        insights: ['strengthen_conceptual_connections', 'apply_deeper_analysis', 'explore_emergent_properties'],
        frameworks_created: Math.floor(Math.random() * 15) + 5,
        total_cycles_completed: Math.floor(Math.random() * 50) + 20
      }
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(metrics));
    return;
  }

  if (req.method === 'POST' && pathname === '/api/pipeline/feedback') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);

        // Simulate recursive feedback processing
        const feedbackResult = {
          feedback_applied: true,
          new_cycle_initiated: data.should_recurse || false,
          enhanced_understanding: `Feedback processed: ${data.feedback_type}`,
          recursive_depth: (data.current_depth || 0) + 1,
          convergence_indicators: {
            stability_score: Math.random() * 0.4 + 0.6,
            insight_emergence: Math.random() > 0.7,
            pattern_crystallization: Math.random() > 0.5
          },
          next_cycle_recommendations: [
            'focus_on_emergent_patterns',
            'strengthen_weak_connections',
            'explore_novel_analogies'
          ],
          timestamp: new Date().toISOString()
        };

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(feedbackResult));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid feedback data' }));
      }
    });
    return;
  }

  if (req.method === 'GET' && pathname === '/api/pipeline/canvas') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    fs.readFile(path.join(__dirname, 'web', 'meta-librarian-canvas.html'), 'utf8', (err, data) => {
      if (err) {
        res.end('<h1>Canvas not available</h1>');
      } else {
        res.end(data);
      }
    });
    return;
  }

  // Vector Converter endpoints
  if (req.method === 'POST' && pathname === '/api/vector/transform') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const result = worldEngineAPI.vectorConverter.applyTransformation(
          data.operator,
          data.metadata || {}
        );
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON' }));
      }
    });
    return;
  }

  if (req.method === 'POST' && pathname === '/api/vector/sequence') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const result = worldEngineAPI.vectorConverter.runSequence(
          data.operators || [],
          data.metadata || {}
        );
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON' }));
      }
    });
    return;
  }

  if (req.method === 'GET' && pathname === '/api/vector/state') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      state: worldEngineAPI.vectorConverter.state,
      description: worldEngineAPI.vectorConverter.getStateDescription(),
      snapshots: worldEngineAPI.vectorConverter.snapshots.length,
      transformations: worldEngineAPI.vectorConverter.transformationHistory.length,
      availableOperators: Object.keys(worldEngineAPI.vectorConverter.operators)
    }));
    return;
  }

  if (req.method === 'GET' && pathname === '/api/vector/suggestions') {
    const url_parts = require('url').parse(req.url, true);
    const targetState = url_parts.query.target ?
      JSON.parse(url_parts.query.target) :
      [0.0, 0.6, 0.5, 0.8];

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(worldEngineAPI.vectorConverter.suggestNextOperators(targetState)));
    return;
  }

  if (req.method === 'GET' && pathname === '/api/vector/history') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(worldEngineAPI.vectorConverter.transformationHistory.slice(-20)));
    return;
  }

  // 404 for unknown API routes
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'API endpoint not found' }));
}const PORT = 8001;
server.listen(PORT, () => {
  console.log('🌍 World Engine Studio Server (Node.js)');
  console.log('========================================');
  console.log(`🚀 Server running at http://localhost:${PORT}`);
  console.log(`🎬 Studio Interface: http://localhost:${PORT}/web/studio.html`);
  console.log(`🔧 Engine Only: http://localhost:${PORT}/web/worldengine.html`);
  console.log(`📊 API Health: http://localhost:${PORT}/api/health`);
  console.log('');
  console.log('Note: This is a Node.js mock server. For full functionality,');
  console.log('fix your Python installation and run: python main.py server');
});
