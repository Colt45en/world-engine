// Integration hook for World Engine master system
import * as React from 'react';
import { worldEngineStorage } from '../storage/WorldEngineStorage';
import { SensoryTokenStream, type SceneMoment, type SensoryDetail } from '../sensory/SensoryTokenStream';

export interface WorldEngineState {
    storage: {
        initialized: boolean;
        stats: ReturnType<typeof worldEngineStorage.getStorageStats>;
    };
    sensory: {
        currentMoment: SceneMoment | null;
        isStreaming: boolean;
    };
    routing: {
        currentRoute: string;
        availableRoutes: string[];
    };
}

const defaultSensoryMoment: SceneMoment = {
    timestamp: Date.now(),
    details: [
        { channel: 'sight', strength: 0.5 },
        { channel: 'sound', strength: 0.3 },
        { channel: 'touch', strength: 0.2 },
        { channel: 'scent', strength: 0.1 },
        { channel: 'taste', strength: 0.1 },
        { channel: 'inner', strength: 0.4 },
    ]
};

export function useWorldEngine(): {
    state: WorldEngineState;
    actions: {
        initializeStorage: () => Promise<void>;
        storeData: (path: string, data: string | Blob, tags?: string[]) => Promise<string>;
        setSensoryText: (text: string) => void;
        navigateTo: (route: string) => void;
        exportData: () => Promise<void>;
    };
} {
    const [state, setState] = React.useState<WorldEngineState>({
        storage: {
            initialized: false,
            stats: worldEngineStorage.getStorageStats()
        },
        sensory: {
            currentMoment: null,
            isStreaming: false
        },
        routing: {
            currentRoute: '',
            availableRoutes: ['free-mode', 'sandbox-360']
        }
    });

    const [sensoryText, setSensoryText] = React.useState('');

    // Initialize storage on mount
    React.useEffect(() => {
        worldEngineStorage.initialize()
            .then(() => {
                setState(prev => ({
                    ...prev,
                    storage: {
                        initialized: true,
                        stats: worldEngineStorage.getStorageStats()
                    }
                }));
            })
            .catch(error => {
                console.error('World Engine initialization failed:', error);
            });
    }, []);

    // Handle route changes
    React.useEffect(() => {
        const handleHashChange = () => {
            const route = window.location.hash.slice(1);
            setState(prev => ({
                ...prev,
                routing: {
                    ...prev.routing,
                    currentRoute: route
                }
            }));
        };

        handleHashChange(); // Initial load
        window.addEventListener('hashchange', handleHashChange);
        return () => window.removeEventListener('hashchange', handleHashChange);
    }, []);

    const handleSensoryMoment = React.useCallback((moment: SceneMoment) => {
        setState(prev => ({
            ...prev,
            sensory: {
                ...prev.sensory,
                currentMoment: moment
            }
        }));
    }, []);

    const actions = React.useMemo(() => ({
        initializeStorage: async () => {
            await worldEngineStorage.initialize();
            setState(prev => ({
                ...prev,
                storage: {
                    initialized: true,
                    stats: worldEngineStorage.getStorageStats()
                }
            }));
        },

        storeData: async (path: string, data: string | Blob, tags: string[] = []) => {
            const id = await worldEngineStorage.storeFile(path, data, tags);
            setState(prev => ({
                ...prev,
                storage: {
                    ...prev.storage,
                    stats: worldEngineStorage.getStorageStats()
                }
            }));
            return id;
        },

        setSensoryText: (text: string) => {
            setSensoryText(text);
            setState(prev => ({
                ...prev,
                sensory: {
                    ...prev.sensory,
                    isStreaming: text.length > 0
                }
            }));
        },

        navigateTo: (route: string) => {
            window.location.hash = `#${route}`;
        },

        exportData: async () => {
            await worldEngineStorage.exportToFSA();
        }
    }), []);

    return {
        state,
        actions,
        // Include the sensory stream component as a render prop
        SensoryStream: sensoryText ? (
            <SensoryTokenStream
        text= { sensoryText }
        base={ defaultSensoryMoment }
        onMoment={ handleSensoryMoment }
        tps={ 6}
        />
    ) : null
  };
}
