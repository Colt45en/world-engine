import React from 'react';
import AppCanvas from './AppCanvas';
import { ToolManager } from './tools/ToolManager';
import { StoreProvider } from './state/StoreProvider';

export default function R3FEditorFramework() {
  return (
    <StoreProvider>
      <div className="grid grid-cols-4 h-screen bg-black">
        {/* Main 3D Viewport */}
        <div className="col-span-3 relative">
          <AppCanvas />
        </div>

        {/* Tool Panel Sidebar */}
        <div className="p-4 bg-black/80 overflow-y-auto border-l border-gray-700">
          <ToolManager />
        </div>
      </div>
    </StoreProvider>
  );
}

// Export store and utilities for external use
export {
  useScene,
  setSpotlightPosition,
  nudgeSpotlight
} from './state/store';

export {
  StoreProvider,
  StoreUtils
} from './state/StoreProvider';

export type {
  Entity,
  CubeEntity,
  EntityId,
  Vec3
} from './state/store';

export type {
  Tool,
  CommandItem,
  TransformData
} from './tools/types';
