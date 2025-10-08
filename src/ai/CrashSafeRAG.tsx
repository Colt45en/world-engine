// Enhanced RAG Chat with Crash Prevention and Recovery
import * as React from 'react';

interface RAGResult {
  content: string;
  metadata: {
    source: string;
    category: string;
    priority: string;
  };
  source: string;
  category: string;
  priority: string;
}

interface RAGResponse {
  success: boolean;
  query?: string;
  results: RAGResult[];
  sources: string[];
  error?: string;
}

interface TrainingState {
  isTraining: boolean;
  dataPoints: number;
  errors: number;
  lastError?: string;
  memoryUsage: number;
  maxMemoryMB: number;
}

export class CrashSafeWorldEngineRAG {
  private bridgeUrl = process.env.REACT_APP_API_URL || 'http://localhost:8888';
  private initialized = false;
  private fallbackMode = false;
  private docs: Array<{ name: string; content: string }> = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private trainingState: TrainingState = {
    isTraining: false,
    dataPoints: 0,
    errors: 0,
    memoryUsage: 0,
    maxMemoryMB: 512 // 512MB limit
  };

  // Circuit breaker pattern
  private circuitBreaker = {
    failures: 0,
    maxFailures: 3,
    timeout: 30000, // 30 seconds
    nextAttempt: 0,
    isOpen: false
  };

  async initialize(bridgeUrl?: string) {
    if (this.initialized) return;

    if (bridgeUrl) {
      this.bridgeUrl = bridgeUrl;
    }

    try {
      // Test connection to Python RAG bridge with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${this.bridgeUrl}/health`, {
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'NexusRAG/1.0'
        }
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const health = await response.json() as { status: string; system: string };
        console.log('✅ Connected to RAG bridge:', health.system);
        this.initialized = true;
        this.fallbackMode = false;
        this.circuitBreaker.failures = 0;
        this.circuitBreaker.isOpen = false;
        return;
      }
    } catch (error) {
      console.warn('RAG bridge not available, using fallback mode:', error);
      this.circuitBreaker.failures++;
    }

    // Fallback to built-in docs if bridge is not available
    this.docs.push(...this.getBuiltInDocs());
    this.initialized = true;
    this.fallbackMode = true;
    console.log(`RAG initialized in fallback mode with ${this.docs.length} documents`);
  }

  async retrieve(query: string): Promise<RAGResult | null> {
    if (!this.initialized) {
      await this.initialize();
    }

    // Check circuit breaker
    if (this.circuitBreaker.isOpen) {
      if (Date.now() < this.circuitBreaker.nextAttempt) {
        console.log('Circuit breaker open, using fallback');
        return this.fallbackRetrieve(query);
      } else {
        // Reset circuit breaker
        this.circuitBreaker.isOpen = false;
        this.circuitBreaker.failures = 0;
      }
    }

    // Try Python RAG bridge first with enhanced error handling
    if (!this.fallbackMode && !this.circuitBreaker.isOpen) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

        const response = await fetch(`${this.bridgeUrl}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: query,
            top_k: 1,
            training_mode: this.trainingState.isTraining
          }),
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (response.ok) {
          const data: RAGResponse = await response.json();
          if (data.success && data.results.length > 0) {
            this.circuitBreaker.failures = 0; // Reset on success
            return data.results[0];
          }
        } else {
          throw new Error(`Bridge returned ${response.status}: ${response.statusText}`);
        }
      } catch (error) {
        console.warn('RAG bridge query failed:', error);
        this.handleBridgeFailure();
      }
    }

    // Fallback to local search
    return this.fallbackRetrieve(query);
  }

  private handleBridgeFailure() {
    this.circuitBreaker.failures++;
    this.trainingState.errors++;

    if (this.circuitBreaker.failures >= this.circuitBreaker.maxFailures) {
      this.circuitBreaker.isOpen = true;
      this.circuitBreaker.nextAttempt = Date.now() + this.circuitBreaker.timeout;
      console.warn(`Circuit breaker activated. Next attempt in ${this.circuitBreaker.timeout/1000}s`);

      // Switch to fallback mode temporarily
      this.fallbackMode = true;
    }
  }

  // Enhanced training management
  startTraining(): boolean {
    if (this.trainingState.isTraining) {
      console.warn('Training already in progress');
      return false;
    }

    try {
      this.trainingState = {
        isTraining: true,
        dataPoints: 0,
        errors: 0,
        memoryUsage: this.estimateMemoryUsage(),
        maxMemoryMB: 512
      };

      console.log('🤖 Training session started with crash protection');
      return true;
    } catch (error) {
      console.error('Failed to start training:', error);
      return false;
    }
  }

  stopTraining(): void {
    this.trainingState.isTraining = false;
    console.log('🤖 Training session stopped');

    // Memory cleanup
    this.performMemoryCleanup();
  }

  private estimateMemoryUsage(): number {
    // Rough estimation based on data structures
    const docMemory = this.docs.length * 1000; // ~1KB per doc
    const stateMemory = JSON.stringify(this.trainingState).length;
    return (docMemory + stateMemory) / (1024 * 1024); // Convert to MB
  }

  private performMemoryCleanup(): void {
    // Clear large data structures if memory usage is high
    if (this.trainingState.memoryUsage > this.trainingState.maxMemoryMB * 0.8) {
      console.log('🧹 Performing memory cleanup');

      // Keep only essential docs
      if (this.docs.length > 10) {
        this.docs = this.docs.slice(0, 10);
      }

      // Force garbage collection if available
      if ((globalThis as any).gc) {
        (globalThis as any).gc();
      }
    }
  }

  getTrainingState(): TrainingState {
    return { ...this.trainingState };
  }

  // Enhanced error handling for training data collection
  collectTrainingData(data: any): boolean {
    if (!this.trainingState.isTraining) {
      return false;
    }

    try {
      // Check memory limits
      this.trainingState.memoryUsage = this.estimateMemoryUsage();
      if (this.trainingState.memoryUsage > this.trainingState.maxMemoryMB) {
        console.warn('Memory limit reached, performing cleanup');
        this.performMemoryCleanup();
        return false;
      }

      // Process data safely
      this.trainingState.dataPoints++;

      // Limit data points to prevent memory overflow
      if (this.trainingState.dataPoints > 10000) {
        console.warn('Training data limit reached, stopping collection');
        this.stopTraining();
        return false;
      }

      return true;
    } catch (error) {
      this.trainingState.errors++;
      this.trainingState.lastError = error instanceof Error ? error.message : String(error);
      console.error('Training data collection error:', error);

      // Auto-stop if too many errors
      if (this.trainingState.errors > 50) {
        console.error('Too many training errors, stopping session');
        this.stopTraining();
      }

      return false;
    }
  }

  async generateContextualResponse(query: string): Promise<string> {
    try {
      const result = await this.retrieve(query);

      if (result) {
        return result.content;
      }

      return this.getDefaultResponse(query);
    } catch (error) {
      console.error('Error generating response:', error);
      return `I apologize, but I encountered an error processing your question. The system is in ${this.fallbackMode ? 'fallback' : 'bridge'} mode. Please try again or contact support if the issue persists.`;
    }
  }

  private fallbackRetrieve(query: string): RAGResult | null {
    if (this.docs.length === 0) {
      return null;
    }

    const queryWords = this.extractWords(query.toLowerCase());
    const scores: Array<{ name: string; content: string; score: number }> = [];

    for (const doc of this.docs) {
      const docWords = this.extractWords(doc.content.toLowerCase());
      const intersection = [...queryWords].filter(word => docWords.has(word));
      const score = intersection.length / Math.max(queryWords.size, 1);

      if (score > 0) {
        scores.push({ name: doc.name, content: doc.content, score });
      }
    }

    if (scores.length === 0) {
      return null;
    }

    scores.sort((a, b) => b.score - a.score);
    const bestDoc = scores[0];
    const relevantSections = this.extractRelevantSections(bestDoc.content, query);

    return {
      content: this.formatResponse(query, bestDoc.name, relevantSections),
      metadata: {
        source: bestDoc.name,
        category: 'documentation',
        priority: 'high'
      },
      source: bestDoc.name,
      category: 'documentation',
      priority: 'high'
    };
  }

  private extractWords(text: string): Set<string> {
    const words = text.match(/[a-z0-9]+/g) || [];
    return new Set(words.filter(word => word.length > 2));
  }

  private extractRelevantSections(content: string, query: string): string[] {
    const sections = content.split('\n').filter(line => line.trim().length > 0);
    const queryWords = this.extractWords(query.toLowerCase());
    const queryWordsArray = Array.from(queryWords);

    const relevantSections = sections.filter(section => {
      const sectionWords = this.extractWords(section.toLowerCase());
      return queryWordsArray.some(word => sectionWords.has(word));
    });

    return relevantSections.slice(0, 5);
  }

  private formatResponse(query: string, docName: string, sections: string[]): string {
    const responses = [
      `Based on the ${docName} documentation:`,
      '',
      ...sections.map(section => `• ${section.replace(/^#+\s*/, '').trim()}`),
      '',
      'Would you like more specific information about any of these topics?'
    ];

    return responses.join('\n');
  }

  private getDefaultResponse(query: string): string {
    const lowerQuery = query.toLowerCase();

    // Add training-specific responses
    if (lowerQuery.includes('training') || lowerQuery.includes('crash')) {
      return `Training System Status:
• Current State: ${this.trainingState.isTraining ? 'Active' : 'Stopped'}
• Data Points: ${this.trainingState.dataPoints}
• Errors: ${this.trainingState.errors}
• Memory Usage: ${this.trainingState.memoryUsage.toFixed(2)}MB
• Bridge Status: ${this.fallbackMode ? 'Fallback Mode' : 'Connected'}

If you're experiencing crashes, try reducing the training data size or restarting the session.`;
    }

    if (lowerQuery.includes('visual') || lowerQuery.includes('bleedway')) {
      return `Visual Bleedway converts PNG silhouettes into 3D meshes with sensory overlay integration...`;
    }

    if (lowerQuery.includes('nexus') || lowerQuery.includes('room')) {
      return `NEXUS Room provides an immersive 3D environment with multiple iframe panels...`;
    }

    return `World Engine is an integrated development and visualization platform featuring crash-safe training, memory management, and robust error handling.`;
  }

  private getBuiltInDocs(): Array<{ name: string; content: string }> {
    return [
      {
        name: 'crash-prevention.md',
        content: `# Crash Prevention Guide

## Training Safety Features
- Memory limit monitoring (512MB default)
- Circuit breaker pattern for bridge failures
- Automatic cleanup on memory overflow
- Error threshold auto-stop (50 errors max)
- Data point limits (10,000 max)

## Recovery Procedures
1. Check training state with status command
2. Clear memory with cleanup command
3. Restart bridge connection if needed
4. Switch to fallback mode if bridge unavailable

## Error Handling
- Timeout protection (10s for queries, 5s for health checks)
- Graceful degradation to local mode
- Automatic reconnection attempts
- Memory usage monitoring`
      },
      {
        name: 'training-troubleshooting.md',
        content: `# Training System Troubleshooting

## Common Issues
### Bridge Connection Failed
- Check if Python backend is running on localhost:8888
- Verify network connectivity
- System will automatically switch to fallback mode

### Memory Overflow
- Reduce training batch size
- Clear accumulated data
- Restart training session

### Training Crashes
- Check error logs in training state
- Verify data format compatibility
- Ensure sufficient system resources

## Prevention
- Monitor memory usage regularly
- Set appropriate data limits
- Use incremental training approaches`
      }
    ];
  }
}

// Enhanced React hook with crash protection
export function useCrashSafeRAG(): {
  query: (question: string) => Promise<string>;
  isReady: boolean;
  trainingState: TrainingState;
  startTraining: () => boolean;
  stopTraining: () => void;
} {
  const [rag] = React.useState(() => new CrashSafeWorldEngineRAG());
  const [isReady, setIsReady] = React.useState(false);
  const [trainingState, setTrainingState] = React.useState<TrainingState>({
    isTraining: false,
    dataPoints: 0,
    errors: 0,
    memoryUsage: 0,
    maxMemoryMB: 512
  });

  React.useEffect(() => {
    rag.initialize().then(() => {
      setIsReady(true);
    });
  }, [rag]);

  // Update training state periodically
  React.useEffect(() => {
    const interval = setInterval(() => {
      setTrainingState(rag.getTrainingState());
    }, 1000);

    return () => clearInterval(interval);
  }, [rag]);

  const query = React.useCallback(async (question: string): Promise<string> => {
    if (!isReady) {
      return 'RAG system is still initializing. Please try again in a moment.';
    }

    return rag.generateContextualResponse(question);
  }, [rag, isReady]);

  const startTraining = React.useCallback(() => {
    return rag.startTraining();
  }, [rag]);

  const stopTraining = React.useCallback(() => {
    rag.stopTraining();
  }, [rag]);

  return { query, isReady, trainingState, startTraining, stopTraining };
}

// Enhanced RAG Chat Interface with crash protection
export function CrashSafeRAGChat() {
  const [isOpen, setIsOpen] = React.useState(false);
  const [messages, setMessages] = React.useState<Array<{
    type: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: number;
  }>>([]);
  const [input, setInput] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);
  const { query, isReady, trainingState, startTraining, stopTraining } = useCrashSafeRAG();

  const sendMessage = async (message: string) => {
    if (!message.trim() || isLoading) return;

    const userMessage = { type: 'user' as const, content: message, timestamp: Date.now() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await query(message);
      const assistantMessage = { type: 'assistant' as const, content: response, timestamp: Date.now() };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'assistant' as const,
        content: 'Sorry, I encountered an error processing your question. The system has crash protection enabled and will recover automatically.',
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const toggleTraining = () => {
    if (trainingState.isTraining) {
      stopTraining();
      const message = {
        type: 'system' as const,
        content: 'Training stopped. Data points collected: ' + trainingState.dataPoints,
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, message]);
    } else {
      if (startTraining()) {
        const message = {
          type: 'system' as const,
          content: 'Training started with crash protection enabled.',
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, message]);
      }
    }
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 z-50 w-12 h-12 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full shadow-lg flex items-center justify-center text-white hover:shadow-xl transition-shadow"
        title="Ask World Engine AI (Crash-Safe)"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80 h-96 bg-gray-900 border border-gray-700 rounded-lg shadow-2xl flex flex-col overflow-hidden">
      {/* Header with Training Status */}
      <div className="p-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <span className="font-medium">Nexus AI (Safe)</span>
          <div className={`w-2 h-2 rounded-full ${isReady ? 'bg-green-400' : 'bg-yellow-400'}`} title={isReady ? 'Ready' : 'Loading...'} />
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleTraining}
            className={`px-2 py-1 text-xs rounded ${trainingState.isTraining ? 'bg-red-500' : 'bg-green-500'}`}
            title={`Training: ${trainingState.isTraining ? 'Active' : 'Stopped'}`}
          >
            {trainingState.isTraining ? 'Stop' : 'Train'}
          </button>
          <button
            onClick={() => setIsOpen(false)}
            className="text-white/80 hover:text-white"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Status Bar */}
      {trainingState.isTraining && (
        <div className="px-3 py-1 bg-gray-800 text-xs text-gray-300 border-b border-gray-700">
          Training: {trainingState.dataPoints} points, {trainingState.errors} errors, {trainingState.memoryUsage.toFixed(1)}MB
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 && (
          <div className="text-gray-400 text-sm text-center py-8">
            Ask me about World Engine, training issues, or system status!
            <div className="mt-2 text-xs">
              Try: "training status" or "help with crashes"
            </div>
          </div>
        )}

        {messages.map((message, i) => (
          <div key={i} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xs px-3 py-2 rounded-lg text-sm ${
              message.type === 'user'
                ? 'bg-blue-600 text-white'
                : message.type === 'system'
                ? 'bg-orange-600 text-white'
                : 'bg-gray-700 text-gray-100'
            }`}>
              <div className="whitespace-pre-wrap">{message.content}</div>
              <div className={`text-xs mt-1 ${
                message.type === 'user' ? 'text-blue-200' :
                message.type === 'system' ? 'text-orange-200' : 'text-gray-400'
              }`}>
                {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-700 text-gray-100 px-3 py-2 rounded-lg text-sm">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                <span>Thinking...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isReady ? "Ask me anything..." : "Loading..."}
            disabled={!isReady || isLoading}
            className="flex-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 text-sm focus:outline-none focus:border-blue-500"
          />
          <button
            type="submit"
            disabled={!isReady || isLoading || !input.trim()}
            className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
}
