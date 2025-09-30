// RAG (Retrieval-Augmented Generation) Integration for World Engine
// Modern implementation with Python backend bridge
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

export class WorldEngineRAG {
  private bridgeUrl = 'http://localhost:8888';
  private initialized = false;
  private fallbackMode = false;
  private docs: Array<{ name: string; content: string }> = [];

  async initialize(bridgeUrl?: string) {
    if (this.initialized) return;

    if (bridgeUrl) {
      this.bridgeUrl = bridgeUrl;
    }

    try {
      // Test connection to Python RAG bridge
      const response = await fetch(`${this.bridgeUrl}/health`);
      if (response.ok) {
        const health = await response.json() as { status: string; system: string };
        console.log('✅ Connected to RAG bridge:', health.system);
        this.initialized = true;
        this.fallbackMode = false;
        return;
      }
    } catch (error) {
      console.warn('RAG bridge not available, using fallback mode:', error);
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

    // Try Python RAG bridge first
    if (!this.fallbackMode) {
      try {
        const response = await fetch(`${this.bridgeUrl}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: query,
            top_k: 1
          })
        });

        if (response.ok) {
          const data: RAGResponse = await response.json();
          if (data.success && data.results.length > 0) {
            return data.results[0];
          }
        }
      } catch (error) {
        console.warn('RAG bridge query failed, falling back to local search:', error);
        this.fallbackMode = true;
      }
    }

    // Fallback to local search
    return this.fallbackRetrieve(query);
  }

  private fallbackRetrieve(query: string): RAGResult | null {
    if (this.docs.length === 0) {
      return null;
    }

    const queryWords = this.extractWords(query.toLowerCase());
    const scores: Array<{ name: string; content: string; score: number }> = [];

    for (const doc of this.docs) {
      const docWords = this.extractWords(doc.content.toLowerCase());
      const queryWordsArray = Array.from(queryWords);
      const overlap = queryWordsArray.filter(word => docWords.has(word)).length;
      const score = overlap / Math.max(queryWordsArray.length, 1);

      if (score > 0) {
        scores.push({ ...doc, score });
      }
    }

    scores.sort((a, b) => b.score - a.score);
    const best = scores[0];

    if (best) {
      return {
        content: best.content,
        metadata: {
          source: best.name,
          category: 'fallback',
          priority: 'medium'
        },
        source: best.name,
        category: 'fallback',
        priority: 'medium'
      };
    }

    return null;
  }

  async generateContextualResponse(query: string, context?: string): Promise<string> {
    if (!this.initialized) {
      await this.initialize();
    }

    // Try Python RAG bridge first
    if (!this.fallbackMode) {
      try {
        const response = await fetch(`${this.bridgeUrl}/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: query,
            context: context || '',
            top_k: 3
          })
        });

        if (response.ok) {
          const data: RAGResponse = await response.json();
          if (data.success && data.results.length > 0) {
            return this.formatAdvancedResponse(query, data.results);
          }
        }
      } catch (error) {
        console.warn('RAG bridge query failed, falling back to local search:', error);
        this.fallbackMode = true;
      }
    }

    // Fallback to local processing
    const retrieved = await this.retrieve(query);

    if (!retrieved) {
      return this.getDefaultResponse(query);
    }

    // Extract relevant sections from the document
    const sections = this.extractRelevantSections(retrieved.content, query);

    return this.formatResponse(query, retrieved.source, sections);
  }

  private formatAdvancedResponse(query: string, results: RAGResult[]): string {
    const responses = [
      `Based on World Engine documentation:`,
      ''
    ];

    for (const result of results) {
      responses.push(`**${result.source}** (${result.category}):`);
      const sections = result.content.split('\n').filter(line => line.trim().length > 0);
      const relevantSections = sections.slice(0, 3); // Top 3 sections

      for (const section of relevantSections) {
        responses.push(`• ${section.replace(/^#+\s*/, '').trim()}`);
      }
      responses.push('');
    }

    responses.push('Need more specific information? Ask about any component!');

    return responses.join('\n');
  }

  private extractWords(text: string): Set<string> {
    const words = text.match(/[a-z0-9]+/g) || [];
    return new Set(words.filter(word => word.length > 2)); // Filter short words
  }

  private extractRelevantSections(content: string, query: string): string[] {
    const sections = content.split('\n').filter(line => line.trim().length > 0);
    const queryWords = this.extractWords(query.toLowerCase());
    const queryWordsArray = Array.from(queryWords);

    const relevantSections = sections.filter(section => {
      const sectionWords = this.extractWords(section.toLowerCase());
      return queryWordsArray.some(word => sectionWords.has(word));
    });

    return relevantSections.slice(0, 5); // Return top 5 relevant sections
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

    if (lowerQuery.includes('visual') || lowerQuery.includes('bleedway')) {
      return `Visual Bleedway converts PNG silhouettes into 3D meshes with sensory overlay integration. Key features:

• Drag & drop PNG silhouettes (front/side views)
• Advanced algorithms: Otsu thresholding, morphological operations
• Real-time 3D mesh generation
• Sensory framework integration (6-channel system)
• GLB export for external use

Navigate to #visual-bleedway to try it out!`;
    }

    if (lowerQuery.includes('nexus') || lowerQuery.includes('room')) {
      return `NEXUS Room provides an immersive 3D environment with multiple iframe panels:

• Grounded first-person camera with axis-locked controls
• 12 configurable panels across 4 walls
• Direct integration with World Engine routes
• Real-time content switching and configuration
• WebGL + CSS3D hybrid rendering

Navigate to #nexus-room to explore!`;
    }

    if (lowerQuery.includes('crypto') || lowerQuery.includes('dashboard')) {
      return `Crypto Dashboard offers advanced trading analytics in 3D:

• Real-time cryptocurrency data visualization
• Technical indicators: RSI, MACD, OBV
• 3D spatial chart rendering
• Multiple API provider support
• Interactive camera controls and chart panels

Navigate to #crypto-dashboard to start trading analysis!`;
    }

    return `World Engine is an integrated development and visualization platform featuring:

• Visual Bleedway: PNG → 3D mesh conversion
• NEXUS Room: 3D iframe environment
• Crypto Dashboard: Advanced trading analytics
• Sensory Framework: 6-channel contextual system
• Bridge System: Cross-component data sharing

Use the dashboard to explore different components or ask more specific questions!`;
  }

  private getBuiltInDocs(): Array<{ name: string; content: string }> {
    return [
      {
        name: 'world-engine-overview.md',
        content: `# World Engine Overview

World Engine is a comprehensive development platform integrating multiple specialized components:

## Core Components

### Visual Bleedway
- Converts PNG silhouettes to 3D meshes
- Advanced image processing algorithms
- Sensory framework integration
- GLB export capability

### NEXUS Room
- 3D iframe environment with 12 panels
- WebGL + CSS3D rendering
- Dynamic content management
- Immersive navigation controls

### Crypto Dashboard
- Real-time cryptocurrency analytics
- 3D spatial data visualization
- Technical indicator support
- Multi-provider API integration

### Sensory Framework
- 6-channel sensory data system
- Procedural moment generation
- R3F overlay rendering
- Environmental context integration

## Navigation
- Hash-based routing system
- Dashboard with proximity volumes
- Cross-component data bridge
- Real-time storage management`
      },
      {
        name: 'api-reference.md',
        content: `# World Engine API Reference

## Bridge System
\`\`\`typescript
import { bridge, useBridgeData } from './bridge/WorldEngineBridge';

// Subscribe to shared data
const [meshData, setMeshData] = useBridgeData<THREE.Mesh>('shared-mesh');

// Publish data
bridge.publish('shared-mesh', generatedMesh);
\`\`\`

## Visual Bleedway API
\`\`\`typescript
import { buildMeshFromSilhouette } from './visual/SilhouetteProcessing';

const mesh = await buildMeshFromSilhouette({
  frontTexture: frontImage,
  sideTexture: sideImage,
  preset: 'Auto (Otsu)',
  resolution: 32
});
\`\`\`

## Storage API
\`\`\`typescript
import { worldEngineStorage } from './storage/WorldEngineStorage';

await worldEngineStorage.initialize();
const stats = worldEngineStorage.getStorageStats();
\`\`\``
      },
      {
        name: 'troubleshooting.md',
        content: `# Troubleshooting Guide

## Visual Bleedway Issues

### Empty Mesh Generation
- Ensure silhouette has sufficient white pixels
- Try Auto (Otsu) preset for automatic thresholding
- Check that input image is a valid PNG

### Performance Problems
- Reduce resolution parameter (16-32)
- Use Performance preset
- Resize large input images

## NEXUS Room Issues

### Panel Loading Problems
- Check console for iframe security errors
- Ensure URLs are accessible
- Try relative paths for local content

### Camera Controls Not Working
- Verify WebGL context is available
- Check browser hardware acceleration
- Clear browser cache and reload

## Crypto Dashboard Issues

### API Connection Failed
- Check API endpoint configuration
- Verify API key validity
- Try switching to demo data mode

### Chart Rendering Problems
- Ensure WebGL is supported
- Check canvas element initialization
- Verify data format compatibility`
      }
    ];
  }
}

// React hook for RAG functionality
export function useRAG(): {
  query: (question: string) => Promise<string>;
  isReady: boolean;
} {
  const [rag] = React.useState(() => new WorldEngineRAG());
  const [isReady, setIsReady] = React.useState(false);

  React.useEffect(() => {
    rag.initialize().then(() => {
      setIsReady(true);
    });
  }, [rag]);

  const query = React.useCallback(async (question: string): Promise<string> => {
    if (!isReady) {
      return 'RAG system is still initializing. Please try again in a moment.';
    }

    return rag.generateContextualResponse(question);
  }, [rag, isReady]);

  return { query, isReady };
}

// RAG Chat Interface Component
export function RAGChat() {
  const [isOpen, setIsOpen] = React.useState(false);
  const [messages, setMessages] = React.useState<Array<{
    type: 'user' | 'assistant';
    content: string;
    timestamp: number;
  }>>([]);
  const [input, setInput] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);
  const { query, isReady } = useRAG();

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
        content: 'Sorry, I encountered an error processing your question. Please try again.',
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

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 z-50 w-12 h-12 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full shadow-lg flex items-center justify-center text-white hover:shadow-xl transition-shadow"
        title="Ask World Engine AI"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80 h-96 bg-gray-900 border border-gray-700 rounded-lg shadow-2xl flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <span className="font-medium">World Engine AI</span>
          <div className={`w-2 h-2 rounded-full ${isReady ? 'bg-green-400' : 'bg-yellow-400'}`} title={isReady ? 'Ready' : 'Loading...'} />
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="text-white/80 hover:text-white"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 && (
          <div className="text-gray-400 text-sm text-center py-8">
            Ask me about World Engine components, features, or troubleshooting!
            <div className="mt-2 text-xs">
              Try: "How does Visual Bleedway work?" or "Show me NEXUS room features"
            </div>
          </div>
        )}

        {messages.map((message, i) => (
          <div key={i} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xs px-3 py-2 rounded-lg text-sm ${
              message.type === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-100'
            }`}>
              <div className="whitespace-pre-wrap">{message.content}</div>
              <div className={`text-xs mt-1 ${
                message.type === 'user' ? 'text-blue-200' : 'text-gray-400'
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
