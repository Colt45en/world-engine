# Sandbox Tools Collection

A comprehensive collection of standalone mathematical and 3D visualization tools extracted and optimized from the multi-view avatar impostor system. Each tool is self-contained, optimized, and ready for integration.

## 🛠️ Available Tools

### 1. **Sonic FX** (`/sonic-fx/`)
**Cinematic speed overlay system**
- Real-time motion streaks and radial effects
- Customizable parameters (speed, intensity, color)
- Non-destructive overlay rendering
- Perfect for speed perception enhancement

### 2. **Mask Mesher** (`/mask-mesher/`)
**Volume mesh generator from silhouettes**
- Front/side mask to 3D volume conversion
- Voxel-based surface extraction
- Real-time preview and export
- Ideal for avatar generation from photos

### 3. **RBF Solver** (`/rbf-solver/`)
**Radial Basis Function PDE solver**
- Multiple kernel types (IMQ, Gaussian, Multiquadric, Thin Plate)
- Meshless Poisson equation solver
- Real-time parameter adjustment
- Educational and research applications

### 4. **Tesseract** (`/tesseract/`)
**4D hypercube visualization**
- True 4D mathematics with 6 rotation planes
- Stereographic projection to 3D
- Interactive controls for all rotation planes
- Educational 4D geometry tool

### 5. **Avatar Impostor** (`/avatar-impostor/`) *[Planned]*
**Multi-view avatar rendering system**
- 9-camera ring capture system
- Real-time depth-aware fusion
- Quality diagnostics and debugging
- Production-ready avatar rendering

### 6. **Contour Extrude** (`/contour-extrude/`) *[Planned]*
**Image to 3D extrusion pipeline**
- Marching squares contour extraction
- Automatic shape simplification
- 3D extrusion with beveling
- Text and logo to 3D conversion

### 7. **HyperBowl** (`/hyperbowl/`) *[Planned]*
**Analytic surface visualization**
- Mathematical surface equations
- Real-time deformation
- Educational mathematical visualization
- Research and presentation tool

### 8. **Mushroom Codex** (`/mushroom-codex/`) *[Planned]*
**Parametric mushroom generator**
- Profile-based generation system
- Toon shading and outline effects
- Batch export capabilities
- Procedural natural object creation

## 🚀 Quick Start

Each tool is self-contained with its own dependencies and documentation:

```bash
# Navigate to any tool
cd sandbox-tools/sonic-fx

# Install dependencies (if needed)
npm install three @react-three/fiber @react-three/drei leva

# Use in your project
import SonicFX from './sonic-fx/SonicFX';
```

## 🎯 Design Principles

### **Standalone Architecture**
- Each tool is completely independent
- No shared dependencies between tools
- Self-contained with own documentation
- Ready for individual integration

### **Optimization Focus**
- Performance-optimized algorithms
- Minimal render overhead
- Efficient memory usage
- Real-time parameter adjustment

### **Educational Value**
- Comprehensive documentation
- Mathematical background explanations
- Interactive learning experiences
- Algorithm implementation details

### **Production Ready**
- Export capabilities (GLB/OBJ)
- Integration-friendly APIs
- Error handling and validation
- Professional UI/UX design

## 📊 Complexity Matrix

| Tool | Math Complexity | Render Complexity | Use Case |
|------|----------------|-------------------|----------|
| Sonic FX | Low | Low | Effects, Gaming |
| Mask Mesher | Medium | Medium | Avatar Creation |
| RBF Solver | High | Medium | Scientific Computing |
| Tesseract | High | Low | Education, Visualization |
| Avatar Impostor | High | High | Production Rendering |

## 🔧 Technical Requirements

### **Minimum Dependencies**
```json
{
  "three": "^0.150.0",
  "@react-three/fiber": "^8.0.0",
  "@react-three/drei": "^9.0.0",
  "leva": "^0.9.0",
  "react": "^18.0.0"
}
```

### **Optional Enhancements**
```json
{
  "@react-three/postprocessing": "^2.0.0",
  "postprocessing": "^6.0.0"
}
```

## 🎨 Integration Patterns

### **Component Integration**
```tsx
import { SonicOverlay } from './sandbox-tools/sonic-fx/SonicFX';
import { TesseractVisualization } from './sandbox-tools/tesseract/Tesseract';

function MyApp() {
  return (
    <Canvas>
      <TesseractVisualization />
      <SonicOverlay />
    </Canvas>
  );
}
```

### **Utility Functions**
```tsx
import { solveRBFProblem } from './sandbox-tools/rbf-solver/RBFSolver';
import { buildVolumeFromMasks } from './sandbox-tools/mask-mesher/MaskMesher';

// Use mathematical functions directly
const solution = solveRBFProblem('IMQ', {c: 0.25}, 196, 24, 1.0, 0.002, 42);
```

## 📚 Learning Path

### **Beginner** (Start Here)
1. **Sonic FX** - Simple shader effects
2. **Tesseract** - 4D visualization basics
3. **HyperBowl** - Mathematical surfaces

### **Intermediate**
1. **Mask Mesher** - Voxel algorithms
2. **Contour Extrude** - Image processing
3. **Mushroom Codex** - Parametric modeling

### **Advanced**
1. **RBF Solver** - Numerical methods
2. **Avatar Impostor** - Multi-view rendering

## 🔬 Research Applications

- **Computer Graphics**: Avatar rendering, procedural generation
- **Mathematical Visualization**: 4D geometry, PDE solving
- **Educational Technology**: Interactive learning tools
- **Game Development**: Visual effects, procedural content

## 🤝 Contributing

Each tool follows the same structure:
```
tool-name/
├── ToolName.tsx      # Main component
├── README.md         # Documentation
├── utils.ts          # Utility functions
└── types.ts          # Type definitions
```

## 📄 License

MIT License - Use freely in personal and commercial projects.

## 🎓 Educational Resources

- **Mathematical Background**: Each tool includes theory explanations
- **Algorithm Details**: Step-by-step implementation guides
- **Interactive Examples**: Learn by experimentation
- **Research References**: Academic papers and resources

---

*Built for the World Engine v3.1 Advanced Math project*
*Optimized for learning, research, and production use*
