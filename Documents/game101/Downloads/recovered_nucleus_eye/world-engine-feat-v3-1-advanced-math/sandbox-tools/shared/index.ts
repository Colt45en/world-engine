// =============================================================================
// R3F Sandbox Shared Utilities - Main Export
// =============================================================================

// Export all hooks and utilities
export * from './utilities';

// Export all shared components
export * from './components';

// =============================================================================
// Quick Access Exports for Common Use Cases
// =============================================================================

// Audio/Video
export {
    useAudioAnalysis,
    useVideoStream
} from './utilities';

// Performance
export {
    usePerformanceStats,
    PerformanceMonitor
} from './utilities';

// Math & Animation
export {
    MathUtils,
    useSpring,
    useTween,
    useGameTime
} from './utilities';

// UI Components
export {
    FloatingPanel,
    InfoDisplay,
    AudioVisualizer,
    ParticleSystem
} from './components';

// Interactive Components
export {
    Draggable,
    Slider3D
} from './components';

// Effect Components
export {
    Grid3D,
    Trail3D,
    AnimatedGroup,
    MorphingGeometry,
    BoundingBox
} from './components';

// Storage & Export
export {
    useLocalStorage,
    exportToJson,
    copyToClipboard
} from './utilities';

// Validation
export {
    validateWebGL,
    validateWebGL2,
    getWebGLCapabilities
} from './utilities';

// Geometry Creation
export {
    createParametricGeometry,
    createNoiseGeometry,
    createDataTexture,
    createNoiseTexture
} from './utilities';

// =============================================================================
// Preset Configurations
// =============================================================================

export const COMMON_COLORS = {
    // Neon colors for sandbox UI
    neonGreen: '#00ff88',
    neonBlue: '#00ccff',
    neonPink: '#ff0099',
    neonYellow: '#ffff00',
    neonPurple: '#9900ff',
    neonOrange: '#ff6600',

    // Dark theme colors
    darkBg: 'rgba(0, 0, 0, 0.9)',
    darkBorder: '#333',
    darkText: '#ccc',

    // Accent colors
    success: '#00ff00',
    warning: '#ffaa00',
    error: '#ff0000',
    info: '#0099ff'
};

export const ANIMATION_PRESETS = {
    smooth: { stiffness: 0.1, damping: 0.8 },
    bouncy: { stiffness: 0.3, damping: 0.6 },
    snappy: { stiffness: 0.5, damping: 0.9 },
    wobbly: { stiffness: 0.2, damping: 0.4 }
};

export const CAMERA_PRESETS = {
    default: { position: [0, 0, 5], fov: 75 },
    wide: { position: [0, 0, 8], fov: 90 },
    close: { position: [0, 0, 3], fov: 60 },
    bird: { position: [0, 10, 0], fov: 75 },
    side: { position: [5, 0, 0], fov: 75 }
};

export const LIGHTING_PRESETS = {
    studio: {
        ambient: { intensity: 0.3 },
        directional: { position: [10, 10, 5], intensity: 1 },
        point: { position: [10, 10, 10], intensity: 0.5 }
    },
    dramatic: {
        ambient: { intensity: 0.1 },
        directional: { position: [5, 15, 5], intensity: 1.5 },
        point: { position: [-10, 5, 10], intensity: 0.8 }
    },
    soft: {
        ambient: { intensity: 0.5 },
        directional: { position: [0, 10, 0], intensity: 0.8 },
        point: { position: [0, 0, 10], intensity: 0.3 }
    }
};

// =============================================================================
// Tool Configuration Types
// =============================================================================

export interface ToolConfig {
    id: string;
    name: string;
    category: string;
    complexity: 1 | 2 | 3 | 4 | 5;
    requiresAudio?: boolean;
    requiresWebcam?: boolean;
    mathematical?: boolean;
    interactive?: boolean;
    description: string;
}

export interface SandboxEnvironment {
    camera: typeof CAMERA_PRESETS[keyof typeof CAMERA_PRESETS];
    lighting: typeof LIGHTING_PRESETS[keyof typeof LIGHTING_PRESETS];
    colors: Partial<typeof COMMON_COLORS>;
    showGrid?: boolean;
    showStats?: boolean;
    showControls?: boolean;
    background?: string;
    fog?: { near: number; far: number; color: string };
}

// =============================================================================
// Default Sandbox Environment
// =============================================================================

export const DEFAULT_SANDBOX_ENV: SandboxEnvironment = {
    camera: CAMERA_PRESETS.default,
    lighting: LIGHTING_PRESETS.studio,
    colors: COMMON_COLORS,
    showGrid: false,
    showStats: false,
    showControls: true,
    background: '#000000'
};

// =============================================================================
// Helper Functions for Tool Integration
// =============================================================================

/**
 * Creates a standardized error boundary for tools
 */
export const createToolErrorBoundary = (toolName: string) => {
    return class extends React.Component<
        React.PropsWithChildren<{}>,
        { hasError: boolean; error?: Error }
    > {
        constructor(props: React.PropsWithChildren<{}>) {
            super(props);
            this.state = { hasError: false };
        }

        static getDerivedStateFromError(error: Error) {
            return { hasError: true, error };
        }

        componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
            console.error(`${toolName} Error:`, error, errorInfo);
        }

        render() {
            if (this.state.hasError) {
                return (
                    <FloatingPanel
            title= {`${toolName} Error`
            }
            background = "rgba(255, 0, 0, 0.1)"
            borderColor = "#ff4444"
                >
                <div style={ { color: '#ff4444', textAlign: 'center' } }>
                    <p>Tool encountered an error and had to be disabled.</p>
                        < p style = {{ fontSize: '12px', opacity: 0.7 }
        }>
        { this.state.error?.message }
        </p>
        < button
    onClick = {() => this.setState({ hasError: false })}
style = {{
    marginTop: '10px',
        padding: '8px 16px',
            background: 'rgba(255, 68, 68, 0.2)',
                border: '1px solid #ff4444',
                    color: '#ff4444',
                        borderRadius: '4px',
                            cursor: 'pointer'
}}
              >
    Retry
    </button>
    </div>
    </FloatingPanel>
        );
      }

return this.props.children;
    }
  };
};

/**
 * Standard loading component for tools
 */
export const ToolLoadingFallback: React.FC<{ toolName: string }> = ({ toolName }) => (
    <FloatingPanel title= {`Loading ${toolName}...`}>
        <div style={ { textAlign: 'center', padding: '20px' } }>
            <div style={ { marginBottom: '15px' } }> Initializing R3F Environment...</div>
                < div style = {{
    width: '100%',
        height: '4px',
            background: 'rgba(0, 255, 136, 0.2)',
                borderRadius: '2px',
                    overflow: 'hidden'
}}>
    <div style={
        {
            width: '30%',
                height: '100%',
                    background: '#00ff88',
                        animation: 'loading 2s infinite',
                            borderRadius: '2px'
        }
} />
    </div>
    <style>
{
    `
          @keyframes loading {
            0% { width: 0%; margin-left: 0%; }
            50% { width: 75%; margin-left: 25%; }
            100% { width: 0%; margin-left: 100%; }
          }
        `}
</style>
    </div>
    </FloatingPanel>
);

/**
 * Standard permission request handler
 */
export const requestPermissions = async (requirements: {
    audio?: boolean;
    video?: boolean;
}): Promise<boolean> => {
    try {
        if (requirements.audio && requirements.video) {
            await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
        } else if (requirements.audio) {
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } else if (requirements.video) {
            await navigator.mediaDevices.getUserMedia({ video: true });
        }
        return true;
    } catch (error) {
        console.warn('Permission request failed:', error);
        return false;
    }
};

/**
 * Create tool-specific controls configuration
 */
export const createToolControls = (toolName: string, config: Record<string, any>) => {
    return {
        [toolName]: config
    };
};

// Re-export React for convenience
import React from 'react';
export { React };
