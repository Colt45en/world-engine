# Graphics Mini-Labs Collection

An interactive educational tool for learning fundamental computer graphics concepts through hands-on React Three.js implementations.

## 🎯 Overview

This collection contains three interactive mini-labs that demonstrate core graphics programming concepts:

- **Mini-lab A**: Barycentric Vertex Paint
- **Mini-lab B**: Catmull-Rom Curves with Parameterization
- **Mini-lab C**: Texture Paint via Barycentrics-as-UVs

## 📚 Educational Concepts

### Barycentric Coordinates
- **Definition**: A coordinate system for triangles using area weights
- **Properties**: Always sum to 1.0 inside the triangle, ≥0 inside, can be negative outside
- **Applications**: Smooth interpolation, collision detection, texture mapping

### Catmull-Rom Splines
- **Purpose**: Smooth curves through control points with different parameterization strategies
- **Types**:
  - Uniform (α=0): Equal spacing, may overshoot
  - Chordal (α=1): Distance-based spacing, higher tension
  - Centripetal (α=0.5): Optimal balance, prevents loops and cusps

### Texture Coordinate Systems
- **UV Mapping**: 2D coordinate system for 3D surfaces
- **Barycentric-to-UV**: Using triangle weights as texture coordinates
- **Canvas Texture**: Real-time painting directly onto 3D surfaces

## 🚀 Usage

```tsx
import GraphicsMiniLabs from './graphics-mini-labs';

function App() {
  return <GraphicsMiniLabs />;
}
```

## 🎮 Interactive Features

### Mini-lab A: Barycentric Vertex Paint
- **Visual**: Triangle with smooth RGB color interpolation
- **Controls**:
  - Toggle vertex labels and weight displays
  - Wireframe mode toggle
  - Auto-rotation option
- **Learning Points**:
  - Vertex A (1,0,0) → Pure red
  - Vertex B (0,1,0) → Pure green
  - Vertex C (0,0,1) → Pure blue
  - Midpoint AB (0.5,0.5,0) → Red + Green blend
  - Centroid (⅓,⅓,⅓) → Average of all three colors

### Mini-lab B: Catmull-Rom Curves
- **Visual**: Interactive curve with moveable control points
- **Controls**:
  - Parameterization type selector (Uniform/Chordal/Centripetal)
  - Curve resolution adjustment
  - Control point visibility
  - Tangent visualization option
- **Interactive**: Click to add control points
- **Learning Points**:
  - Compare different parameterization effects
  - Understand overshoot vs. tension trade-offs
  - See how centripetal prevents loops

### Mini-lab C: Texture Paint
- **Visual**: Triangle surface with paintable texture
- **Controls**:
  - Brush size adjustment
  - Color picker for paint
  - UV grid overlay toggle
  - Clear canvas button
- **Interactive**: Click on triangle to paint
- **Learning Points**:
  - Barycentric coordinates become UV coordinates
  - Triangle vertices map to A(0,0), B(1,0), C(0,1)
  - Direct 3D surface painting

## 🔧 Implementation Details

### Shader Programming
```glsl
// Barycentric vertex shader
attribute vec3 barycentric;
varying vec3 vBary;

void main() {
  vBary = barycentric;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

// Fragment shader using barycentrics as colors
varying vec3 vBary;
void main() {
  vec3 color = vec3(vBary.x, vBary.y, vBary.z);
  gl_FragColor = vec4(color, 1.0);
}
```

### Mathematical Functions
```typescript
// Barycentric weight calculation using areas
function baryWeights(A: V3, B: V3, C: V3, P: V3): [number, number, number] {
  const areaABC = 0.5 * len(cross(sub(B, A), sub(C, A)));
  const w0 = 0.5 * len(cross(sub(B, P), sub(C, P))) / areaABC;
  const w1 = 0.5 * len(cross(sub(A, P), sub(C, P))) / areaABC;
  const w2 = 0.5 * len(cross(sub(A, P), sub(B, P))) / areaABC;
  return [w0, w1, w2];
}

// Catmull-Rom spline evaluation
function catmullRom(Pm1: V3, P0: V3, P1: V3, P2: V3, u: number, t: number[], i: number): V3 {
  // Tangent calculation with non-uniform parameterization
  const m1 = mul(sub(P1, Pm1), (t2 - t1) / (t1 - t0));
  const m2 = mul(sub(P2, P0), (t2 - t1) / (t2 - t0));

  // Hermite basis functions
  const s = (u01 - t1) / (t2 - t1);
  const h00 = 2 * s**3 - 3 * s**2 + 1;
  const h10 = s**3 - 2 * s**2 + s;
  const h01 = -2 * s**3 + 3 * s**2;
  const h11 = s**3 - s**2;

  // Combine using Hermite interpolation
  return add(add(mul(P0, h00), mul(m1, h10 * (t2 - t1))),
             add(mul(P1, h01), mul(m2, h11 * (t2 - t1))));
}
```

### Canvas Texture Integration
```typescript
// Real-time texture painting
const handleTriangleClick = (event: any) => {
  const point = event.point;
  const [w0, w1, w2] = baryWeights(A, B, C, point);

  if (w0 >= 0 && w1 >= 0 && w2 >= 0) {  // Inside triangle
    const u = w1;  // B vertex contributes to U
    const v = w2;  // C vertex contributes to V

    // Paint on canvas at UV coordinates
    ctx.arc(u * canvas.width, (1-v) * canvas.height, brushSize, 0, Math.PI * 2);
    ctx.fill();
    texture.needsUpdate = true;
  }
};
```

## 🎓 Learning Outcomes

After completing these mini-labs, students will understand:

1. **Barycentric Coordinates**:
   - How to calculate area-based weights for any point relative to a triangle
   - Why they're useful for interpolation and inside/outside tests
   - The relationship between barycentrics and color blending

2. **Parametric Curves**:
   - The difference between uniform, chordal, and centripetal parameterization
   - Why centripetal is often the best choice for smooth curves
   - How to implement Catmull-Rom splines with proper tangent calculation

3. **Texture Mapping**:
   - The connection between 3D geometry and 2D texture space
   - How barycentric coordinates can serve as UV coordinates
   - Real-time texture generation and updates

## 🔍 Self-Check Exercises

### Barycentric Coordinates
- [ ] Verify that at vertex A, barycentric coordinates are (1,0,0) → pure red
- [ ] Check that at the midpoint of edge AB, coordinates are approximately (0.5,0.5,0)
- [ ] Confirm that at the triangle centroid, all three weights are approximately ⅓

### Catmull-Rom Curves
- [ ] Compare uniform vs. centripetal parameterization with the same control points
- [ ] Notice how chordal parameterization affects curve tension
- [ ] Observe the absence of loops in centripetal mode

### Texture Paint
- [ ] Paint near vertex A and verify it appears in the (0,0) region
- [ ] Paint near vertex B and check the (1,0) region
- [ ] Paint near vertex C and confirm the (0,1) region
- [ ] Verify that painting inside the triangle always produces valid UV coordinates

## 🚀 Extensions

Advanced students can extend these labs by:
- Adding more complex geometries (quads, arbitrary meshes)
- Implementing Bézier curves alongside Catmull-Rom
- Creating multi-triangle texture painting systems
- Adding real-time deformation based on barycentric coordinates
- Implementing more sophisticated brush effects

## 📊 Performance Notes

- Uses efficient BufferGeometry for triangle representation
- Canvas texture updates only when painting occurs
- Curve generation uses configurable resolution for performance tuning
- All calculations use optimized vector operations

## 🎨 Visual Feedback

Each mini-lab provides immediate visual feedback:
- **Color changes** reflect mathematical relationships
- **Interactive elements** respond to user input
- **Real-time updates** show parameter effects
- **Visual guides** help understand coordinate systems

This collection serves as an excellent introduction to computer graphics programming while providing hands-on experience with fundamental mathematical concepts.
