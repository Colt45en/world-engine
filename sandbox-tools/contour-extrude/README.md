# Contour Extrude Tool

An advanced **image-to-3D contour extrusion system** that detects edges in images, traces contours using marching squares algorithm, simplifies polygons with Ramer-Douglas-Peucker, and extrudes them into detailed 3D geometry with configurable beveling.

## 🎯 Core Technology

### Image Processing Pipeline
- **Luminance Conversion**: RGB to grayscale using standard luminance formula (0.2126*R + 0.7152*G + 0.0722*B)
- **Binary Thresholding**: Configurable threshold for foreground/background separation
- **Mask Generation**: Creates binary mask for contour detection algorithms
- **Multi-Source Support**: Text rendering, geometric shapes, and custom image loading

### Marching Squares Algorithm
- **Boundary Following**: 8-directional neighbor analysis for edge detection
- **Contour Tracing**: Clockwise boundary following with sub-pixel precision
- **Multiple Contour Support**: Detects all contours and selects the longest primary shape
- **Safety Mechanisms**: Infinite loop prevention and robust termination conditions

## 📊 Features

### Image Source Options
- **Text Mode**: Custom text with configurable font and rotation
- **Geometric Shapes**: Circle, star, heart, rectangle primitives
- **Custom Images**: Load and process external image files
- **Real-Time Preview**: Instant visualization of contour detection results

### Contour Processing
- **Adaptive Threshold**: 0.1-0.9 range with real-time adjustment
- **Polygon Simplification**: Ramer-Douglas-Peucker algorithm with epsilon control
- **Contour Statistics**: Point count, perimeter length, and complexity metrics
- **Edge Quality Analysis**: Boundary coherence and smoothness evaluation

### 3D Extrusion Controls
- **Extrude Depth**: 0.05-1.0 unit depth with precision control
- **Bevel System**: Enable/disable with thickness and size parameters
- **Bevel Segments**: 1-12 segments for smooth curved edges
- **Extrusion Steps**: Multi-step extrusion for complex depth profiles

## 🏗️ Geometric Processing

### Polygon Simplification (RDP Algorithm)
```typescript
// Distance from point to line segment calculation
const distanceToSegment = (p: MSPoint, a: MSPoint, b: MSPoint): number => {
  const vx = b[0] - a[0], vy = b[1] - a[1];
  const wx = p[0] - a[0], wy = p[1] - a[1];
  const c1 = vx * wx + vy * wy, c2 = vx * vx + vy * vy;
  let t = c2 ? c1 / c2 : 0;
  t = Math.max(0, Math.min(1, t));
  const dx = a[0] + t * vx - p[0], dy = a[1] + t * vy - p[1];
  return Math.hypot(dx, dy);
};

// Recursive simplification with epsilon tolerance
const simplify = (i: number, j: number) => {
  let maxDistance = -1, maxIndex = -1;
  for (let k = i + 1; k < j; k++) {
    const distance = distanceToSegment(pts[k], pts[i], pts[j]);
    if (distance > maxDistance) { maxDistance = distance; maxIndex = k; }
  }
  if (maxDistance > epsilon) {
    keep[maxIndex] = 1;
    simplify(i, maxIndex); simplify(maxIndex, j);
  }
};
```

### Boundary Following Algorithm
- **8-Connected Analysis**: Comprehensive neighbor checking for boundary detection
- **Directional Tracing**: Maintains consistent clockwise boundary traversal
- **Sub-Pixel Precision**: 0.5-pixel offset for smoother contour positioning
- **Closure Detection**: Automatic contour loop completion and validation

## 🎮 Usage Guide

### Basic Operation
1. **Select Source**: Choose text, geometric shape, or load custom image
2. **Adjust Threshold**: Fine-tune foreground/background separation
3. **Configure Simplification**: Balance detail vs smoothness with epsilon parameter
4. **Set Extrusion**: Define depth, bevel settings, and geometry complexity

### Advanced Techniques
- **Text Optimization**: Use bold fonts and adequate size for clean contours
- **Shape Processing**: Experiment with different primitives for various geometric forms
- **Custom Images**: Load logos, icons, or artwork for specialized 3D conversion
- **Quality Tuning**: Balance contour points vs polygon complexity for optimal results

## 🔧 Technical Implementation

### Marching Squares Implementation
```typescript
// Boundary detection with 8-directional neighbors
const isBoundary = (x: number, y: number) => {
  if (!at(x, y)) return false;
  for (let k = 0; k < 8; k++) {
    const nx = x + dirs[k][0], ny = y + dirs[k][1];
    if (!inside(nx, ny) || at(nx, ny) === 0) return true;
  }
  return false;
};

// Contour following with direction tracking
do {
  visited[cy * W + cx] = 1;
  contour.push([cx + 0.5, cy + 0.5]);
  let found = false;
  for (let turn = 0; turn < 8; turn++) {
    const ndir = (dir + 7 + turn) % 8;
    const nx = cx + dirs[ndir][0], ny = cy + dirs[ndir][1];
    if (inside(nx, ny) && isBoundary(nx, ny)) {
      cx = nx; cy = ny; dir = ndir; found = true; break;
    }
  }
  if (!found) break;
} while (!(cx === sx && cy === sy));
```

### Performance Characteristics
- **Image Processing**: Linear O(W*H) complexity for threshold conversion
- **Contour Detection**: O(P) where P is perimeter length of shape
- **Polygon Simplification**: O(N log N) average case for RDP algorithm
- **3D Extrusion**: Depends on contour complexity and bevel segments

## 🚀 Applications

### Logo and Branding
- **3D Logos**: Convert company logos to dimensional representations
- **Signage Design**: Create extruded text for architectural visualization
- **Product Mockups**: Transform 2D designs into 3D prototype models

### Artistic Creation
- **Typography Art**: Extrude custom fonts and lettering into sculptural forms
- **Icon Processing**: Convert 2D icons into dimensional interface elements
- **Pattern Generation**: Create repeating 3D elements from 2D motifs

### Technical Applications
- **CAD Integration**: Generate 3D geometry from engineering drawings
- **Rapid Prototyping**: Convert sketches to 3D printable models
- **Game Assets**: Create 3D models from concept art and sprites

## 📈 Quality Metrics

### Contour Analysis
- **Point Density**: Optimal balance between detail and performance
- **Smoothness Factor**: Measured via angular deviation along contour
- **Closure Quality**: Validation of contour loop completion
- **Simplification Ratio**: Reduction percentage from original to simplified

### Geometric Validation
- **Triangle Quality**: Aspect ratio and area distribution analysis
- **Vertex Distribution**: Even spacing and density validation
- **Normal Consistency**: Surface orientation and lighting compatibility
- **Topology Verification**: Manifold surface and edge connectivity checks

---

*Advanced contour detection and 3D extrusion system with real-time processing, configurable quality parameters, and comprehensive geometric analysis.*
