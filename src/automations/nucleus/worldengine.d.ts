// Types for WorldEngine Tier4 Integration

export interface WorldEngineTier4 {
    createTier4RoomBridge(iframe: HTMLIFrameElement): Tier4Bridge;
    StudioBridge: StudioBridge;
}

export interface Tier4Bridge {
    on(event: 'connectionStatus', callback: (isConnected: boolean) => void): void;
    on(event: 'operatorApplied', callback: (operator: string, previousState: State, newState: State) => void): void;
    applyOperator(operator: string, options?: { source?: string }): void;
    triggerNucleusEvent(role: string): void;
    processAIBotMessage(message: string, messageType: string): void;
    processLibrarianData(librarian: string, dataType: string, data: any): void;
}

export interface StudioBridge {
    onBus(event: string, callback: (message: any) => void): void;
}

export interface State {
    x: number[];
    kappa: number;
    level: number;
    [key: string]: any;
}

export interface EventLogEntry {
    type: 'operator' | 'nucleus' | 'ai_bot' | 'librarian';
    operator?: string;
    role?: string;
    messageType?: string;
    nucleusRole?: string;
    librarian?: string;
    dataType?: string;
    timestamp: number;
    confidence?: number;
}

// Extend Window interface
declare global {
    interface Window {
        WorldEngineTier4?: WorldEngineTier4;
    }
}
