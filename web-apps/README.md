# Web Applications

This folder contains interactive web applications built for the World Engine project.

## Applications

### 3D Canvas Physics (`3d-canvas-physics/`)
- **File**: `3d_canvas_editor.html`
- **Description**: 3D visualization system with cube physics and radial graph generation
- **Features**:
  - Dual-layer canvas system (back grid at z=0, front grid at z=0.5)
  - Interactive transparency controls
  - Radial mathematical pattern generator
  - Real-time 3D cube physics simulation
  - PNG export functionality

## Usage

Open any HTML file directly in a web browser. No external dependencies required - all applications are self-contained.

## Development Notes

- Canvas rendering issues: Check browser console for initialization logs
- Black screen with corner cube: Likely canvas layering/z-index conflicts
- All applications use vanilla JavaScript for maximum compatibility
