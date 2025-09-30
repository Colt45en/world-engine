import * as React from 'react';
import { ProximityVolume } from './spatial/ProximityVolume';
import { worldEngineStorage } from './storage/WorldEngineStorage';

export function Dashboard() {
  const [currentRoute, setCurrentRoute] = React.useState('');
  const [storageStats, setStorageStats] = React.useState(worldEngineStorage.getStorageStats());

  React.useEffect(() => {
    const handleHashChange = () => {
      setCurrentRoute(window.location.hash.slice(1));
    };

    handleHashChange(); // Initial load
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  React.useEffect(() => {
    // Initialize storage and update stats
    worldEngineStorage.initialize().then(() => {
      setStorageStats(worldEngineStorage.getStorageStats());
    }).catch(error => {
      console.error('Storage initialization failed:', error);
    });
  }, []);

  const navigateTo = (route: string) => {
    window.location.hash = `#${route}`;
  };

  const renderCurrentView = () => {
    switch (currentRoute) {
      case 'free-mode':
        return <FreeMode storageStats={storageStats} />;
      case 'sandbox-360':
        return <Camera360Sandbox />;
      default:
        return <MainDashboard onNavigate={navigateTo} storageStats={storageStats} />;
    }
  };

  return (
    <div className="h-full bg-gradient-to-b from-gray-900 to-black text-white overflow-hidden">
      {renderCurrentView()}
    </div>
  );
}

function MainDashboard({ onNavigate, storageStats }: {
  onNavigate: (route: string) => void;
  storageStats: ReturnType<typeof worldEngineStorage.getStorageStats>;
}) {
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-700">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
          World Engine Dashboard
        </h1>
        <p className="text-gray-400 mt-1">Master System Integration</p>
      </div>

      {/* Navigation Grid */}
      <div className="flex-1 p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

        {/* Free Mode Card */}
        <ProximityVolume
          radius={100}
          thickness={12}
          onClick={() => onNavigate('free-mode')}
          className="group"
        >
          <div className="w-full h-40 bg-gradient-to-br from-green-900 to-green-700 rounded-lg p-6 border border-green-600/30 hover:border-green-500/60 transition-all duration-300 cursor-pointer">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-green-300">Free Mode</h3>
                <p className="text-green-400/70 text-sm mt-2">
                  Open canvas with complete toolset
                </p>
              </div>
              <div className="text-green-400 opacity-60 group-hover:opacity-100 transition-opacity">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </div>
            </div>
          </div>
        </ProximityVolume>

        {/* Visual Bleedway Card */}
        <ProximityVolume
          radius={100}
          thickness={12}
          onClick={() => onNavigate('visual-bleedway')}
          className="group"
        >
          <div className="w-full h-40 bg-gradient-to-br from-cyan-900 to-emerald-700 rounded-lg p-6 border border-cyan-600/30 hover:border-cyan-500/60 transition-all duration-300 cursor-pointer">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-cyan-300">Visual Bleedway</h3>
                <p className="text-cyan-400/70 text-sm mt-2">
                  Mask → Mesh → Sensory Pipeline
                </p>
                <span className="inline-block mt-2 px-2 py-1 bg-cyan-800/50 rounded text-xs text-cyan-300">
                  NEW
                </span>
              </div>
              <div className="text-cyan-400 opacity-60 group-hover:opacity-100 transition-opacity">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
            </div>
          </div>
        </ProximityVolume>

        {/* 360 Camera Sandbox */}
        <ProximityVolume
          radius={100}
          thickness={12}
          onClick={() => onNavigate('sandbox-360')}
          className="group"
        >
          <div className="w-full h-40 bg-gradient-to-br from-purple-900 to-blue-700 rounded-lg p-6 border border-purple-600/30 hover:border-purple-500/60 transition-all duration-300 cursor-pointer">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-purple-300">360° Camera</h3>
                <p className="text-purple-400/70 text-sm mt-2">
                  Holographic capture prototype
                </p>
                <span className="inline-block mt-2 px-2 py-1 bg-purple-800/50 rounded text-xs text-purple-300">
                  DEV ONLY
                </span>
              </div>
              <div className="text-purple-400 opacity-60 group-hover:opacity-100 transition-opacity">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
            </div>
          </div>
        </ProximityVolume>

        {/* NEXUS Room */}
        <ProximityVolume
          radius={100}
          thickness={12}
          onClick={() => onNavigate('nexus-room')}
          className="group"
        >
          <div className="w-full h-40 bg-gradient-to-br from-indigo-900 to-violet-700 rounded-lg p-6 border border-indigo-600/30 hover:border-indigo-500/60 transition-all duration-300 cursor-pointer">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-indigo-300">NEXUS Room</h3>
                <p className="text-indigo-400/70 text-sm mt-2">
                  3D Iframe Environment
                </p>
                <span className="inline-block mt-2 px-2 py-1 bg-indigo-800/50 rounded text-xs text-indigo-300">
                  NEW
                </span>
              </div>
              <div className="text-indigo-400 opacity-60 group-hover:opacity-100 transition-opacity">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
            </div>
          </div>
        </ProximityVolume>

        {/* Crypto Dashboard */}
        <ProximityVolume
          radius={100}
          thickness={12}
          onClick={() => onNavigate('crypto-dashboard')}
          className="group"
        >
          <div className="w-full h-40 bg-gradient-to-br from-yellow-900 to-amber-700 rounded-lg p-6 border border-yellow-600/30 hover:border-yellow-500/60 transition-all duration-300 cursor-pointer">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-yellow-300">Crypto Dashboard</h3>
                <p className="text-yellow-400/70 text-sm mt-2">
                  3D Trading Analytics
                </p>
                <span className="inline-block mt-2 px-2 py-1 bg-yellow-800/50 rounded text-xs text-yellow-300">
                  NEW
                </span>
              </div>
              <div className="text-yellow-400 opacity-60 group-hover:opacity-100 transition-opacity">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
          </div>
        </ProximityVolume>

        {/* Nexus Widget Dashboard */}
        <ProximityVolume
          radius={100}
          thickness={12}
          onClick={() => onNavigate('nexus-widgets')}
          className="group"
        >
          <div className="w-full h-40 bg-gradient-to-br from-purple-900 to-pink-700 rounded-lg p-6 border border-purple-600/30 hover:border-purple-500/60 transition-all duration-300 cursor-pointer">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-purple-300">Nexus Widgets</h3>
                <p className="text-purple-400/70 text-sm mt-2">
                  Connect to All Widgets
                </p>
                <span className="inline-block mt-2 px-2 py-1 bg-purple-800/50 rounded text-xs text-purple-300">
                  🚀 INTEGRATED
                </span>
              </div>
              <div className="text-purple-400 opacity-60 group-hover:opacity-100 transition-opacity">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
            </div>
          </div>
        </ProximityVolume>

        {/* Codex Automations */}
        <div className="w-full h-40 bg-gradient-to-br from-orange-900 to-red-700 rounded-lg p-6 border border-orange-600/30">
          <div>
            <h3 className="text-lg font-semibold text-orange-300">Codex Automations</h3>
            <p className="text-orange-400/70 text-sm mt-2">
              Cards → Actions → Outputs
            </p>
            <div className="mt-3 text-xs text-orange-400/60">
              Available in Free Mode
            </div>
          </div>
        </div>
      </div>

      {/* Additional Tools Section */}
      <div className="px-6 pb-6">
        <h3 className="text-lg font-semibold text-gray-300 mb-4">Storage & Analytics</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Storage Management */}
          <div className="h-32 bg-gradient-to-br from-gray-800 to-gray-600 rounded-lg p-4 border border-gray-600/30">
            <div>
              <h4 className="font-medium text-gray-300">Storage</h4>
              <p className="text-gray-400/70 text-sm mt-2">
                OPFS → ZIP → Manifest
              </p>
              <div className="mt-3 flex space-x-2 text-xs">
                <span className="px-2 py-1 bg-green-800/40 rounded text-green-300">OPFS</span>
                <span className="px-2 py-1 bg-blue-800/40 rounded text-blue-300">FSA</span>
                <span className="px-2 py-1 bg-purple-800/40 rounded text-purple-300">ZIP</span>
              </div>
            </div>
          </div>

          {/* Performance Monitor */}
          <div className="h-32 bg-gradient-to-br from-blue-800 to-indigo-600 rounded-lg p-4 border border-blue-600/30">
            <div>
              <h4 className="font-medium text-blue-300">Analytics</h4>
              <p className="text-blue-400/70 text-sm mt-2">
                Performance & Usage Metrics
              </p>
              <div className="mt-3 text-xs text-blue-400/60">
                Real-time monitoring
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="p-4 border-t border-gray-700 bg-black/50">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <span>System: Operational</span>
          <span>Files: {storageStats.totalFiles} ({(storageStats.totalSize / 1024 / 1024).toFixed(1)}MB)</span>
          <span>Graphics: {storageStats.opfsSupported ? 'OPFS Ready' : 'Limited'}</span>
        </div>
      </div>
    </div>
  );
}

function FreeMode({ storageStats }: {
  storageStats: ReturnType<typeof worldEngineStorage.getStorageStats>;
}) {
  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-green-700 bg-green-900/20">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-green-300">Free Mode</h2>
          <button
            onClick={() => window.location.hash = '#'}
            className="px-3 py-1 bg-green-800/50 rounded text-green-300 hover:bg-green-700/50 transition-colors"
          >
            ← Dashboard
          </button>
        </div>
      </div>

      <div className="flex-1 p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
          {/* Codex Automation Panel */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h3 className="text-lg font-medium text-green-400 mb-4">Codex Automations</h3>
            <div className="space-y-3">
              <div className="p-3 bg-green-900/30 rounded border border-green-600/30">
                <div className="text-sm font-medium text-green-300">Card → Action → Output</div>
                <div className="text-xs text-green-400/70 mt-1">Ready for integration</div>
              </div>
            </div>
          </div>

          {/* Storage Monitor */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h3 className="text-lg font-medium text-blue-400 mb-4">Storage Status</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">OPFS:</span>
                <span className={storageStats.opfsSupported ? "text-green-400" : "text-red-400"}>
                  {storageStats.opfsSupported ? "Active" : "Unsupported"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">FSA Support:</span>
                <span className={storageStats.fsaSupported ? "text-blue-400" : "text-yellow-400"}>
                  {storageStats.fsaSupported ? "Available" : "Limited"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Files:</span>
                <span className="text-purple-400">{storageStats.totalFiles}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Bundles:</span>
                <span className="text-purple-400">{storageStats.bundles}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Camera360Sandbox() {
  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-purple-700 bg-purple-900/20">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-purple-300">360° Camera Sandbox</h2>
            <span className="inline-block mt-1 px-2 py-1 bg-red-800/50 rounded text-xs text-red-300">
              DEV ONLY - NOT FOR PRODUCTION
            </span>
          </div>
          <button
            onClick={() => window.location.hash = '#'}
            className="px-3 py-1 bg-purple-800/50 rounded text-purple-300 hover:bg-purple-700/50 transition-colors"
          >
            ← Dashboard
          </button>
        </div>
      </div>

      <div className="flex-1 p-6">
        <div className="bg-gray-800/50 rounded-lg p-6 h-full">
          <h3 className="text-lg font-medium text-purple-400 mb-4">Holographic Capture System</h3>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
            {/* Camera Ring */}
            <div className="bg-purple-900/30 rounded-lg p-4">
              <h4 className="font-medium text-purple-300 mb-3">Camera Ring</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-300">Cameras:</span>
                  <span className="text-purple-400">12-ring setup</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Resolution:</span>
                  <span className="text-purple-400">4K per cam</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Sync:</span>
                  <span className="text-green-400">Locked</span>
                </div>
              </div>
            </div>

            {/* Hologram Processing */}
            <div className="bg-blue-900/30 rounded-lg p-4">
              <h4 className="font-medium text-blue-300 mb-3">Hologram Engine</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-300">Raymarch:</span>
                  <span className="text-blue-400">Active</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Particles:</span>
                  <span className="text-blue-400">GPU-based</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Quality:</span>
                  <span className="text-yellow-400">Prototype</span>
                </div>
              </div>
            </div>

            {/* Capture Presets */}
            <div className="bg-green-900/30 rounded-lg p-4">
              <h4 className="font-medium text-green-300 mb-3">Capture Presets</h4>
              <div className="space-y-2">
                <button className="w-full p-2 bg-green-800/40 rounded text-green-300 hover:bg-green-700/40 transition-colors">
                  Standard 360°
                </button>
                <button className="w-full p-2 bg-green-800/40 rounded text-green-300 hover:bg-green-700/40 transition-colors">
                  High Quality
                </button>
                <button className="w-full p-2 bg-green-800/40 rounded text-green-300 hover:bg-green-700/40 transition-colors">
                  Performance Mode
                </button>
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-red-900/20 border border-red-600/30 rounded">
            <div className="text-red-300 text-sm">
              <strong>Development Notice:</strong> This is a prototype feature for testing holographic capture systems.
              Not intended for production use. Camera hardware integration required.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
