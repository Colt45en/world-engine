import React from 'react';

// Fallback grid helper component
function GridHelper({
  size = 20,
  divisions = 20,
  colorCenterLine = "#444",
  colorGrid = "#222"
}) {
  return (
    <gridHelper
      args={[size, divisions, colorCenterLine, colorGrid]}
      position={[0, -0.001, 0]}
    />
  );
}

// Fallback axis indicator
function AxisIndicator({
  axisScale = 2,
  labelColor = "white"
}) {
  return (
    <group>
      {/* X Axis - Red */}
      <mesh position={[axisScale / 2, 0, 0]}>
        <cylinderGeometry args={[0.02, 0.02, axisScale, 8]} />
        <meshBasicMaterial color="#ff0000" />
      </mesh>
      {/* Y Axis - Green */}
      <mesh position={[0, axisScale / 2, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.02, 0.02, axisScale, 8]} />
        <meshBasicMaterial color="#00ff00" />
      </mesh>
      {/* Z Axis - Blue */}
      <mesh position={[0, 0, axisScale / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.02, 0.02, axisScale, 8]} />
        <meshBasicMaterial color="#0000ff" />
      </mesh>
    </group>
  );
}

export default function HelpersLayer() {
  return (
    <group name="helpers-layer">
      <GridHelper
        size={20}
        divisions={20}
        colorCenterLine="#444"
        colorGrid="#222"
      />
      <AxisIndicator
        axisScale={2}
        labelColor="white"
      />
    </group>
  );
}
