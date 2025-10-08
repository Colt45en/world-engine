import React from 'react';
import { useScene } from './store';

export const StoreProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return <>{children}</>;
};

// Hook for accessing store with TypeScript safety
export const useSceneStore = () => {
  const store = useScene();
  return store;
};

// Utility functions for common operations
export const StoreUtils = {
  // Get all cubes
  getCubes: () => {
    const entities = useScene.getState().entities;
    return Object.values(entities).filter(e => e.kind === 'cube');
  },

  // Get selected cube
  getSelectedCube: () => {
    const state = useScene.getState();
    const selectedId = state.selected[0];
    return selectedId ? state.entities[selectedId] : null;
  },

  // Check if entity is selected
  isSelected: (entityId: string) => {
    const selected = useScene.getState().selected;
    return selected.includes(entityId);
  },

  // Get entity by id
  getEntity: (id: string) => {
    const entities = useScene.getState().entities;
    return entities[id] || null;
  },

  // Get all entities of specific kind
  getEntitiesByKind: (kind: 'cube' | 'light' | 'camera' | 'empty') => {
    const entities = useScene.getState().entities;
    return Object.values(entities).filter(e => e.kind === kind);
  },

  // Check if undo is available
  canUndo: () => {
    const history = useScene.getState().history;
    return history.length > 0;
  },

  // Check if redo is available
  canRedo: () => {
    const future = useScene.getState().future;
    return future.length > 0;
  },

  // Get entity count
  getEntityCount: () => {
    const entities = useScene.getState().entities;
    return Object.keys(entities).length;
  },

  // Get cube count
  getCubeCount: () => {
    const entities = useScene.getState().entities;
    return Object.values(entities).filter(e => e.kind === 'cube').length;
  }
};
