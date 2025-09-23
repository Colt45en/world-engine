/**
 * Type definitions for World Engine Studio
 * Provides TypeScript support for the bridge system and components.
 */

// ===== Studio Bridge Types =====

export interface StudioMessage {
  type: string;
  [key: string]: any;
}

export interface EngineRunMessage extends StudioMessage {
  type: 'eng.run';
  text: string;
}

export interface EngineTestMessage extends StudioMessage {
  type: 'eng.test';
  name: string;
}

export interface EngineResultMessage extends StudioMessage {
  type: 'eng.result';
  runId: string;
  outcome: AnalysisOutcome;
  input?: string;
}

export interface RecorderStartMessage extends StudioMessage {
  type: 'rec.start';
  mode: 'mic' | 'screen' | 'both';
  meta?: Record<string, any>;
}

export interface RecorderClipMessage extends StudioMessage {
  type: 'rec.clip';
  clipId: string;
  url?: string;
  meta?: Record<string, any>;
  size?: number;
}

export interface RecorderTranscriptMessage extends StudioMessage {
  type: 'rec.transcript';
  clipId?: string;
  text: string;
  ts: number;
}

export interface ChatCommandMessage extends StudioMessage {
  type: 'chat.cmd';
  line: string;
}

// ===== Analysis Types =====

export interface AnalysisOutcome {
  count?: number;
  items?: AnalysisItem[];
  opts?: Record<string, any>;
  type?: string;
  result?: string;
  input?: string;
  timestamp?: number;
}

export interface AnalysisItem {
  lemma?: string;
  word?: string;
  text?: string;
  root?: string;
  prefix?: string;
  suffix?: string;
  pos?: string;
  score?: number;
  [key: string]: any;
}

// ===== Component APIs =====

export interface RecorderAPI {
  startMic(meta?: Record<string, any>): Promise<void>;
  startScreen(meta?: Record<string, any>): Promise<void>;
  startBoth(meta?: Record<string, any>): Promise<void>;
  stop(): Promise<void>;
  mark(tag?: string, runId?: string | null): Promise<void>;
}

export interface EngineAPI {
  run(text: string): Promise<void>;
  test(name: string): Promise<void>;
  getStatus(): Promise<void>;
}

export interface ChatAPI {
  command(line: string): Promise<void>;
  announce(message: string, level?: string): Promise<void>;
}

// ===== Storage Types =====

export interface StorageInterface {
  save(key: string, value: any): Promise<any>;
  load(key: string): Promise<any>;
}

export interface RunData {
  runId: string;
  ts: number;
  input: string;
  outcome: AnalysisOutcome;
  clipId?: string | null;
}

export interface ClipData {
  clipId: string;
  url: string;
  blob?: Blob;
  duration?: number;
  size: number;
  timestamp: number;
  meta: Record<string, any>;
}

export interface MarkerData {
  id: string;
  tag: string;
  runId?: string | null;
  timestamp: number;
  clipId?: string | null;
}

// ===== Controller Types =====

export interface EngineControllerOptions {
  timeout?: number;
}

export interface RecorderControllerOptions {
  transcription?: boolean;
  autoMarkRuns?: boolean;
  chunkSize?: number;
}

export interface ChatControllerOptions {
  autoLinkClips?: boolean;
  transcriptCommands?: boolean;
  commandPrefix?: string;
}

// ===== Global Window Extensions =====

declare global {
  interface Window {
    StudioBridge: {
      onBus: (fn: (msg: StudioMessage) => void) => void;
      sendBus: (msg: StudioMessage) => void;
      RecorderAPI: RecorderAPI;
      EngineAPI: EngineAPI;
      ChatAPI: ChatAPI;
      Store: StorageInterface;
      Utils: {
        generateId(): string;
        parseCommand(line: string): { type: string; args: string };
        log(message: string, level?: string): void;
      };
      setupEngineTransport: (frame: HTMLIFrameElement) => any;
    };
    
    EngineController: new (frame: HTMLIFrameElement, options?: EngineControllerOptions) => any;
    RecorderController: new (options?: RecorderControllerOptions) => any;
    ChatController: new (options?: ChatControllerOptions) => any;
    
    externalStore?: {
      upsert(key: string, value: any): Promise<any>;
      get(key: string): Promise<any>;
    };
    
    Studio?: {
      showHelp(): void;
      exportData(): Promise<void>;
      reset(): void;
      getStatus(): any;
    };
  }
  
  // Speech Recognition API types
  interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    grammars: SpeechGrammarList;
    interimResults: boolean;
    lang: string;
    maxAlternatives: number;
    serviceURI: string;
    start(): void;
    stop(): void;
    abort(): void;
    onaudiostart: ((this: SpeechRecognition, ev: Event) => any) | null;
    onaudioend: ((this: SpeechRecognition, ev: Event) => any) | null;
    onend: ((this: SpeechRecognition, ev: Event) => any) | null;
    onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null;
    onnomatch: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
    onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
    onsoundstart: ((this: SpeechRecognition, ev: Event) => any) | null;
    onsoundend: ((this: SpeechRecognition, ev: Event) => any) | null;
    onspeechstart: ((this: SpeechRecognition, ev: Event) => any) | null;
    onspeechend: ((this: SpeechRecognition, ev: Event) => any) | null;
    onstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  }
  
  const webkitSpeechRecognition: {
    prototype: SpeechRecognition;
    new(): SpeechRecognition;
  };
  
  const SpeechRecognition: {
    prototype: SpeechRecognition;
    new(): SpeechRecognition;
  };
}

export {};