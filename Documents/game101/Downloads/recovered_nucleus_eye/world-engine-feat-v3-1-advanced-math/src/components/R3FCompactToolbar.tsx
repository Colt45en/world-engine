import React, { useState, useRef, useEffect } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';

interface R3FToolbarSettings {
  // Display Settings
  resolution: number;
  brightness: number;
  contrast: number;
  saturation: number;

  // Vector Forge Tools
  vectorForgeEnabled: boolean;
  spectralMode: boolean;
  heartPulse: number;

  // General
  isExpanded: boolean;
  position: 'top' | 'bottom';
}

interface VectorForgeTools {
  addVector: () => void;
  addNote: () => void;
  clearScene: () => void;
  exportScene: () => void;
  toggleSpectral: () => void;
  heartPulse: (intensity: number) => void;
}

export function R3FCompactToolbar({
  vectorForgeTools,
  onSettingsChange
}: {
  vectorForgeTools?: VectorForgeTools;
  onSettingsChange?: (settings: Partial<R3FToolbarSettings>) => void;
}) {
  const [settings, setSettings] = useState<R3FToolbarSettings>({
    resolution: 1.0,
    brightness: 1.0,
    contrast: 1.0,
    saturation: 1.0,
    vectorForgeEnabled: false,
    spectralMode: false,
    heartPulse: 0.0,
    isExpanded: false,
    position: 'top'
  });

  const [activeTab, setActiveTab] = useState<'display' | 'vector' | 'tools'>('display');
  const toolbarRef = useRef<HTMLDivElement>(null);

  const updateSetting = <K extends keyof R3FToolbarSettings>(
    key: K,
    value: R3FToolbarSettings[K]
  ) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    onSettingsChange?.(newSettings);
  };

  const toggleToolbar = () => {
    updateSetting('isExpanded', !settings.isExpanded);
  };

  const resetDisplaySettings = () => {
    updateSetting('resolution', 1.0);
    updateSetting('brightness', 1.0);
    updateSetting('contrast', 1.0);
    updateSetting('saturation', 1.0);
  };

  // Apply display settings to canvas
  useEffect(() => {
    const canvas = document.querySelector('canvas');
    if (canvas) {
      canvas.style.filter = `
        brightness(${settings.brightness})
        contrast(${settings.contrast})
        saturate(${settings.saturation})
      `;

      // Apply resolution scaling
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * settings.resolution;
      canvas.height = rect.height * settings.resolution;
    }
  }, [settings.resolution, settings.brightness, settings.contrast, settings.saturation]);

  return (
    <div
      ref={toolbarRef}
      className={`r3f-compact-toolbar ${settings.position} ${settings.isExpanded ? 'expanded' : 'collapsed'}`}
    >
      {/* Toggle Button */}
      <button
        className="toolbar-toggle"
        onClick={toggleToolbar}
        title={settings.isExpanded ? 'Collapse Toolbar' : 'Expand Toolbar'}
      >
        {settings.isExpanded ? '▲' : '🔧'}
      </button>

      {/* Position Toggle */}
      <button
        className="position-toggle"
        onClick={() => updateSetting('position', settings.position === 'top' ? 'bottom' : 'top')}
        title="Toggle Position"
      >
        {settings.position === 'top' ? '⬇️' : '⬆️'}
      </button>

      {/* Expanded Content */}
      {settings.isExpanded && (
        <div className="toolbar-content">
          {/* Tab Navigation */}
          <div className="tab-nav">
            <button
              className={`tab-btn ${activeTab === 'display' ? 'active' : ''}`}
              onClick={() => setActiveTab('display')}
            >
              📺
            </button>
            <button
              className={`tab-btn ${activeTab === 'vector' ? 'active' : ''}`}
              onClick={() => setActiveTab('vector')}
              disabled={!vectorForgeTools}
            >
              🔥
            </button>
            <button
              className={`tab-btn ${activeTab === 'tools' ? 'active' : ''}`}
              onClick={() => setActiveTab('tools')}
            >
              🛠️
            </button>
          </div>

          {/* Display Settings Tab */}
          {activeTab === 'display' && (
            <div className="tab-content display-tab">
              <div className="setting-group">
                <label>Resolution</label>
                <input
                  type="range"
                  min="0.25"
                  max="2.0"
                  step="0.25"
                  value={settings.resolution}
                  onChange={(e) => updateSetting('resolution', parseFloat(e.target.value))}
                />
                <span>{settings.resolution}x</span>
              </div>

              <div className="setting-group">
                <label>Brightness</label>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  value={settings.brightness}
                  onChange={(e) => updateSetting('brightness', parseFloat(e.target.value))}
                />
                <span>{Math.round(settings.brightness * 100)}%</span>
              </div>

              <div className="setting-group">
                <label>Contrast</label>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  value={settings.contrast}
                  onChange={(e) => updateSetting('contrast', parseFloat(e.target.value))}
                />
                <span>{Math.round(settings.contrast * 100)}%</span>
              </div>

              <div className="setting-group">
                <label>Saturation</label>
                <input
                  type="range"
                  min="0.0"
                  max="2.0"
                  step="0.1"
                  value={settings.saturation}
                  onChange={(e) => updateSetting('saturation', parseFloat(e.target.value))}
                />
                <span>{Math.round(settings.saturation * 100)}%</span>
              </div>

              <button className="reset-btn" onClick={resetDisplaySettings}>
                🔄 Reset
              </button>
            </div>
          )}

          {/* Vector Forge Tab */}
          {activeTab === 'vector' && vectorForgeTools && (
            <div className="tab-content vector-tab">
              <div className="tool-row">
                <button
                  className="tool-btn"
                  onClick={vectorForgeTools.addVector}
                  title="Add Vector"
                >
                  ➕
                </button>
                <button
                  className="tool-btn"
                  onClick={vectorForgeTools.addNote}
                  title="Add Note"
                >
                  📝
                </button>
                <button
                  className="tool-btn"
                  onClick={vectorForgeTools.clearScene}
                  title="Clear Scene"
                >
                  🗑️
                </button>
                <button
                  className="tool-btn"
                  onClick={vectorForgeTools.exportScene}
                  title="Export Scene"
                >
                  💾
                </button>
              </div>

              <div className="tool-row">
                <button
                  className={`tool-btn ${settings.spectralMode ? 'active' : ''}`}
                  onClick={() => {
                    vectorForgeTools.toggleSpectral();
                    updateSetting('spectralMode', !settings.spectralMode);
                  }}
                  title="Toggle Spectral Mode"
                >
                  👁️
                </button>
                <button
                  className="tool-btn"
                  onClick={() => vectorForgeTools.heartPulse(0.3)}
                  title="Heart Pulse"
                >
                  💓
                </button>
              </div>

              <div className="setting-group">
                <label>Heart Intensity</label>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.1"
                  value={settings.heartPulse}
                  onChange={(e) => {
                    const value = parseFloat(e.target.value);
                    updateSetting('heartPulse', value);
                    vectorForgeTools.heartPulse(value);
                  }}
                />
                <span>{Math.round(settings.heartPulse * 100)}%</span>
              </div>
            </div>
          )}

          {/* Tools Tab */}
          {activeTab === 'tools' && (
            <div className="tab-content tools-tab">
              <div className="tool-row">
                <button
                  className="tool-btn"
                  onClick={() => window.open('./public/vector-forge-unified-engine.html', '_blank')}
                  title="Open Vector Forge"
                >
                  🚀
                </button>
                <button
                  className="tool-btn"
                  onClick={() => window.open('./public/nexus-3d-sculptor.html', '_blank')}
                  title="Open 3D Sculptor"
                >
                  🎯
                </button>
                <button
                  className="tool-btn"
                  onClick={() => window.open('./public/nexus-math-academy-connected.html', '_blank')}
                  title="Open Math Academy"
                >
                  🧮
                </button>
              </div>

              <div className="setting-group">
                <label>
                  <input
                    type="checkbox"
                    checked={settings.vectorForgeEnabled}
                    onChange={(e) => updateSetting('vectorForgeEnabled', e.target.checked)}
                  />
                  Vector Forge Integration
                </label>
              </div>
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        .r3f-compact-toolbar {
          position: fixed;
          z-index: 1000;
          background: rgba(0, 0, 0, 0.9);
          border: 1px solid #333;
          border-radius: 8px;
          backdrop-filter: blur(10px);
          font-family: monospace;
          font-size: 12px;
          color: white;
          transition: all 0.3s ease;
          max-width: 400px;
        }

        .r3f-compact-toolbar.top {
          top: 10px;
          right: 10px;
        }

        .r3f-compact-toolbar.bottom {
          bottom: 10px;
          right: 10px;
        }

        .r3f-compact-toolbar.collapsed {
          width: 60px;
          height: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 5px;
        }

        .r3f-compact-toolbar.expanded {
          width: 300px;
          min-height: 200px;
          padding: 10px;
        }

        .toolbar-toggle {
          background: none;
          border: none;
          color: white;
          cursor: pointer;
          font-size: 14px;
          padding: 5px;
          border-radius: 4px;
          transition: background 0.2s;
        }

        .toolbar-toggle:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .position-toggle {
          background: none;
          border: none;
          color: white;
          cursor: pointer;
          font-size: 12px;
          padding: 2px;
          border-radius: 4px;
          transition: background 0.2s;
        }

        .position-toggle:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .toolbar-content {
          margin-top: 10px;
        }

        .tab-nav {
          display: flex;
          gap: 2px;
          margin-bottom: 10px;
          border-bottom: 1px solid #333;
          padding-bottom: 5px;
        }

        .tab-btn {
          background: rgba(255, 255, 255, 0.1);
          border: none;
          color: white;
          cursor: pointer;
          padding: 6px 10px;
          border-radius: 4px;
          font-size: 14px;
          transition: background 0.2s;
          flex: 1;
        }

        .tab-btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .tab-btn.active {
          background: #667eea;
        }

        .tab-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .tab-content {
          min-height: 120px;
        }

        .setting-group {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
          font-size: 11px;
        }

        .setting-group label {
          min-width: 60px;
          color: #ccc;
        }

        .setting-group input[type="range"] {
          flex: 1;
          height: 20px;
          background: #333;
          border-radius: 10px;
          outline: none;
          -webkit-appearance: none;
        }

        .setting-group input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 15px;
          height: 15px;
          border-radius: 50%;
          background: #667eea;
          cursor: pointer;
        }

        .setting-group span {
          min-width: 35px;
          text-align: right;
          color: #999;
          font-size: 10px;
        }

        .tool-row {
          display: flex;
          gap: 5px;
          margin-bottom: 8px;
        }

        .tool-btn {
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid #333;
          color: white;
          cursor: pointer;
          padding: 8px;
          border-radius: 4px;
          font-size: 14px;
          transition: all 0.2s;
          flex: 1;
          min-width: 35px;
        }

        .tool-btn:hover {
          background: rgba(255, 255, 255, 0.2);
          transform: translateY(-1px);
        }

        .tool-btn.active {
          background: #667eea;
          border-color: #667eea;
        }

        .reset-btn {
          background: #ff6b6b;
          border: none;
          color: white;
          cursor: pointer;
          padding: 6px 12px;
          border-radius: 4px;
          font-size: 11px;
          width: 100%;
          margin-top: 5px;
          transition: background 0.2s;
        }

        .reset-btn:hover {
          background: #ff5252;
        }

        .setting-group input[type="checkbox"] {
          margin-right: 5px;
        }

        /* Mobile responsiveness */
        @media (max-width: 768px) {
          .r3f-compact-toolbar.expanded {
            width: 250px;
            right: 5px;
          }

          .r3f-compact-toolbar.top {
            top: 5px;
          }

          .r3f-compact-toolbar.bottom {
            bottom: 5px;
          }
        }
      `}</style>
    </div>
  );
}

// Hook to provide Vector Forge integration
export function useVectorForgeIntegration(): VectorForgeTools {
  const addVector = () => {
    // Send message to Vector Forge iframe if present
    const iframe = document.querySelector('#vector-forge-iframe') as HTMLIFrameElement;
    if (iframe?.contentWindow) {
      iframe.contentWindow.postMessage({ type: 'ADD_VECTOR' }, '*');
    } else {
      console.log('Vector Forge: Add Vector');
    }
  };

  const addNote = () => {
    const iframe = document.querySelector('#vector-forge-iframe') as HTMLIFrameElement;
    if (iframe?.contentWindow) {
      iframe.contentWindow.postMessage({ type: 'ADD_NOTE' }, '*');
    } else {
      console.log('Vector Forge: Add Note');
    }
  };

  const clearScene = () => {
    const iframe = document.querySelector('#vector-forge-iframe') as HTMLIFrameElement;
    if (iframe?.contentWindow) {
      iframe.contentWindow.postMessage({ type: 'CLEAR_SCENE' }, '*');
    } else {
      console.log('Vector Forge: Clear Scene');
    }
  };

  const exportScene = () => {
    const iframe = document.querySelector('#vector-forge-iframe') as HTMLIFrameElement;
    if (iframe?.contentWindow) {
      iframe.contentWindow.postMessage({ type: 'EXPORT_SCENE' }, '*');
    } else {
      console.log('Vector Forge: Export Scene');
    }
  };

  const toggleSpectral = () => {
    const iframe = document.querySelector('#vector-forge-iframe') as HTMLIFrameFrame;
    if (iframe?.contentWindow) {
      iframe.contentWindow.postMessage({ type: 'TOGGLE_SPECTRAL' }, '*');
    } else {
      console.log('Vector Forge: Toggle Spectral');
    }
  };

  const heartPulse = (intensity: number) => {
    const iframe = document.querySelector('#vector-forge-iframe') as HTMLIFrameElement;
    if (iframe?.contentWindow) {
      iframe.contentWindow.postMessage({ type: 'HEART_PULSE', intensity }, '*');
    } else {
      console.log(`Vector Forge: Heart Pulse ${intensity}`);
    }
  };

  return {
    addVector,
    addNote,
    clearScene,
    exportScene,
    toggleSpectral,
    heartPulse
  };
}

// Integration component for R3F scenes
export function R3FToolbarProvider({ children }: { children: React.ReactNode }) {
  const vectorForgeTools = useVectorForgeIntegration();
  const [settings, setSettings] = useState<Partial<R3FToolbarSettings>>({});

  const handleSettingsChange = (newSettings: Partial<R3FToolbarSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));

    // Apply settings to scene
    if (newSettings.spectralMode !== undefined) {
      document.body.classList.toggle('spectral-mode', newSettings.spectralMode);
    }
  };

  return (
    <>
      {children}
      <R3FCompactToolbar
        vectorForgeTools={vectorForgeTools}
        onSettingsChange={handleSettingsChange}
      />
    </>
  );
}
