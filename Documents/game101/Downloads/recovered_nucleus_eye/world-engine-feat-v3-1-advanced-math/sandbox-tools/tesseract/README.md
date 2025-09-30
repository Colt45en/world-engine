# Tesseract (4D Hypercube) Visualization

A standalone interactive visualization of a 4-dimensional hypercube (tesseract) with real-time rotation and stereographic projection to 3D space.

## Features

- **True 4D Mathematics**: Genuine 4D rotations in 6 possible planes
- **Real-time Animation**: Smooth rotation with customizable speeds
- **Stereographic Projection**: 4D → 3D projection for visualization
- **Interactive Controls**: Adjust all rotation planes independently
- **Vertex & Edge Display**: Toggle visualization components
- **Educational Interface**: Learn about 4D geometry concepts

## Mathematical Background

### 4D Hypercube Structure
- **16 Vertices**: All combinations of (±1, ±1, ±1, ±1)
- **32 Edges**: Connect vertices differing in exactly one coordinate
- **Connectivity**: Each vertex connects to 4 neighbors (one per dimension)

### 4D Rotations
Unlike 3D rotation (3 axes), 4D has **6 rotation planes**:
- **XY Plane**: Rotation in X-Y, preserving Z-W
- **XZ Plane**: Rotation in X-Z, preserving Y-W
- **XW Plane**: Rotation in X-W, preserving Y-Z
- **YZ Plane**: Rotation in Y-Z, preserving X-W
- **YW Plane**: Rotation in Y-W, preserving X-Z
- **ZW Plane**: Rotation in Z-W, preserving X-Y

### Projection Method
Uses **stereographic projection** from 4D to 3D:
```
(x, y, z, w) → (x, y, z) × scale
where scale = distance / (distance - w)
```

## Controls

### Visualization
- **scale**: Overall size of tesseract (0.5-3)
- **rotationSpeed**: Global animation speed multiplier
- **projectionDistance**: 4D→3D projection distance (affects perspective)
- **showVertices**: Toggle vertex spheres
- **showEdges**: Toggle edge lines
- **vertexSize**: Size of vertex spheres
- **edgeColor**: Color of edges
- **vertexColor**: Color of vertices

### 4D Rotations
Independent rotation speeds for each plane:
- **rotateXY**: XY-plane rotation speed
- **rotateXZ**: XZ-plane rotation speed
- **rotateXW**: XW-plane rotation speed
- **rotateYZ**: YZ-plane rotation speed
- **rotateYW**: YW-plane rotation speed
- **rotateZW**: ZW-plane rotation speed

## Understanding 4D Rotation

### Single Plane Rotations
- **XY only**: Looks like normal 3D rotation around Z-axis
- **ZW only**: Creates "inside-out" turning effect
- **XW only**: Vertices appear/disappear as they rotate through W

### Combined Rotations
- **XY + ZW**: Classic tesseract animation (most recognizable)
- **XW + YW**: Creates complex folding patterns
- **All planes**: Chaotic but mesmerizing 4D tumbling

## Educational Value

### Key Insights
1. **Dimensionality**: Experience how 4D objects behave
2. **Projection**: Understand limitations of viewing higher dimensions
3. **Symmetry**: Observe 4D rotational symmetry groups
4. **Connectivity**: See how 4D adjacency differs from 3D

### Recommended Experiments
1. **Start Simple**: Enable only XY rotation (familiar 3D-like motion)
2. **Add ZW**: Enable ZW rotation to see "hyperrotation"
3. **Pure 4D**: Disable XY/YZ/XZ, enable only XW/YW/ZW
4. **Full Chaos**: Enable all rotations with different speeds

## Technical Implementation

### 4D Vector Operations
```typescript
class Vector4 {
  constructor(x, y, z, w);

  // 4D rotation in XY plane
  rotateXY(angle) {
    return new Vector4(
      cos(angle) * x - sin(angle) * y,
      sin(angle) * x + cos(angle) * y,
      z, w
    );
  }
}
```

### Edge Generation Algorithm
```typescript
// Two vertices are connected if they differ in exactly one bit
for (let i = 0; i < 16; i++) {
  for (let bit = 0; bit < 4; bit++) {
    const j = i ^ (1 << bit); // Flip one bit
    if (i < j) edges.push([i, j]);
  }
}
```

## Mathematical Properties

### Tesseract Numbers
- **Vertices**: 2⁴ = 16
- **Edges**: 4 × 2³ = 32
- **Faces**: 6 × 2² = 24 (squares)
- **Cells**: 4 × 2¹ = 8 (cubes)
- **4-faces**: 1 (the tesseract itself)

### Euler Characteristic
In 4D: V - E + F - C + H = 0
Where: 16 - 32 + 24 - 8 + 1 = 1 ✗

Actually: χ = 0 for 4D sphere, but tesseract is boundary of 4D cube.

## Usage

```tsx
import Tesseract from './Tesseract';

function MyApp() {
  return <Tesseract />;
}
```

## Integration

The TesseractVisualization component can be used in other 3D scenes:

```tsx
import { TesseractVisualization } from './Tesseract';

<Canvas>
  <TesseractVisualization />
</Canvas>
```
