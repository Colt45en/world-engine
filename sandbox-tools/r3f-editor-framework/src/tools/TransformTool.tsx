import React from 'react';
import { useScene } from '../state/store';

// Simple transform tool placeholder
// In production, this would integrate with @react-three/drei's TransformControls

export default function TransformTool() {
  const { selected, entities, updateEntity } = useScene();

  const selectedEntity = selected.length > 0 ? entities[selected[0]] : null;

  if (!selectedEntity) {
    return (
      <div className="p-4 text-gray-400 text-center">
        Select an entity to transform
      </div>
    );
  }

  const handlePositionChange = (axis: number, value: number) => {
    const newPosition = [...selectedEntity.position] as [number, number, number];
    newPosition[axis] = value;
    updateEntity(selectedEntity.id, { position: newPosition });
  };

  const handleRotationChange = (axis: number, value: number) => {
    const newRotation = [...selectedEntity.rotation] as [number, number, number];
    newRotation[axis] = value * Math.PI / 180; // Convert degrees to radians
    updateEntity(selectedEntity.id, { rotation: newRotation });
  };

  const handleScaleChange = (axis: number, value: number) => {
    const newScale = [...selectedEntity.scale] as [number, number, number];
    newScale[axis] = value;
    updateEntity(selectedEntity.id, { scale: newScale });
  };

  return (
    <div className="p-4 space-y-4 bg-gray-800 text-white">
      <h3 className="text-sm font-semibold">Transform: {selectedEntity.name || selectedEntity.id}</h3>

      {/* Position */}
      <div>
        <label className="text-xs font-medium text-gray-400">Position</label>
        <div className="grid grid-cols-3 gap-2 mt-1">
          {(['X', 'Y', 'Z'] as const).map((axis, index) => (
            <div key={axis}>
              <label className="text-xs text-gray-500">{axis}</label>
              <input
                type="number"
                value={selectedEntity.position[index].toFixed(2)}
                onChange={(e) => handlePositionChange(index, parseFloat(e.target.value) || 0)}
                className="w-full px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded"
                step="0.1"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Rotation */}
      <div>
        <label className="text-xs font-medium text-gray-400">Rotation (degrees)</label>
        <div className="grid grid-cols-3 gap-2 mt-1">
          {(['X', 'Y', 'Z'] as const).map((axis, index) => (
            <div key={axis}>
              <label className="text-xs text-gray-500">{axis}</label>
              <input
                type="number"
                value={(selectedEntity.rotation[index] * 180 / Math.PI).toFixed(1)}
                onChange={(e) => handleRotationChange(index, parseFloat(e.target.value) || 0)}
                className="w-full px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded"
                step="1"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Scale */}
      <div>
        <label className="text-xs font-medium text-gray-400">Scale</label>
        <div className="grid grid-cols-3 gap-2 mt-1">
          {(['X', 'Y', 'Z'] as const).map((axis, index) => (
            <div key={axis}>
              <label className="text-xs text-gray-500">{axis}</label>
              <input
                type="number"
                value={selectedEntity.scale[index].toFixed(2)}
                onChange={(e) => handleScaleChange(index, parseFloat(e.target.value) || 0)}
                className="w-full px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded"
                step="0.1"
                min="0.01"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Entity Properties */}
      {selectedEntity.kind === 'cube' && (
        <div>
          <label className="text-xs font-medium text-gray-400">Properties</label>
          <div className="space-y-2 mt-1">
            <div>
              <label className="text-xs text-gray-500">Color</label>
              <input
                type="color"
                value={selectedEntity.color}
                onChange={(e) => updateEntity(selectedEntity.id, { color: e.target.value })}
                className="w-full h-8 bg-gray-700 border border-gray-600 rounded"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500">Name</label>
              <input
                type="text"
                value={selectedEntity.name || ''}
                onChange={(e) => updateEntity(selectedEntity.id, { name: e.target.value })}
                className="w-full px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded"
                placeholder="Entity name"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
