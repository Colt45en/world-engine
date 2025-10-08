# Cubes Spotlight - Interactive 3D Scene Controller

## Overview
A sophisticated R3F conversion of a C++ GLFW/GLEW/ImGui cube scene, featuring interactive cube management, dynamic spotlight control, and programmatic scene manipulation capabilities.

## Key Features

### 🎯 Interactive Cube Management
- **Dynamic Addition/Deletion**: Add new cubes or remove selected ones
- **Selection System**: Click-to-select cubes with visual feedback
- **Animated Bobbing**: Each cube has unique bobbing animation with configurable speed
- **Color Variation**: Random HSL color generation for visual distinction

### 💡 Advanced Spotlight System
- **Toggle Control**: Enable/disable spotlight with instant visual feedback
- **Mouse Dragging**: Interactive ground-plane spotlight positioning
- **Programmatic Control**: Direct coordinate input and nudge controls
- **Visual Marker**: Glowing sphere indicator for spotlight position

### 🎮 Sophisticated Controls
- **Animation Speed Slider**: Real-time animation speed adjustment (0-2x)
- **Spotlight Positioning**: Direct X,Z coordinate input with set button
- **Nudge System**: Directional controls with adjustable step size
- **Cube Selection Dropdown**: Easy selection from numbered cube list

### 🎨 Visual Excellence
- **Dynamic Shadows**: Real-time shadow casting from all light sources
- **Grid Helper**: Visual ground reference with customizable spacing
- **Hemisphere Lighting**: Natural ambient lighting simulation
- **Material System**: Standard materials with proper light interaction

## Technical Implementation

### State Management (Zustand)
```typescript
interface CubeData {
  id: number;
  position: [number, number, number];
  offset: number; // Animation phase offset
  color: string;  // HSL color string
}

interface SpotlightState {
  enabled: boolean;
  position: [number, number, number];
}

interface StoreState {
  cubes: CubeData[];
  selected: number; // Index of selected cube (-1 = none)
  animationSpeed: number;
  spotlight: SpotlightState;
  // Action methods...
}
```

### Ground Plane Interaction
- **Ray Casting**: Accurate mouse-to-world coordinate conversion
- **Plane Intersection**: Y=0 ground plane intersection calculation
- **Drag Detection**: Mouse down/move/up state management
- **Coordinate Validation**: Finite number checking for robustness

### Animation System
- **Per-Cube Offsets**: Unique animation phases prevent synchronization
- **Speed Scaling**: Global animation speed multiplier affects all cubes
- **Sinusoidal Bobbing**: Smooth sine wave vertical movement
- **Rotation Animation**: Continuous X/Y axis rotation for visual interest

### Programmatic API
```typescript
// External control functions
setSpotlightPosition(x: number, y: number, z: number): void
nudgeSpotlight(dx: number, dz: number): void

// Store access
useStore.getState().addCube()
useStore.getState().setSelected(index: number)
useStore.getState().setAnimationSpeed(speed: number)
```

## User Interface

### Main Controls Panel
- **Scene Management**: Add/delete cube buttons with disabled states
- **Animation Control**: Speed slider with real-time value display
- **Spotlight Toggle**: Clear on/off status with toggle button

### Advanced Spotlight Controls
- **Direct Positioning**: Numeric X,Z input fields with set button
- **Nudge Grid**: 3x3 directional control grid with center reset
- **Step Size**: Adjustable nudge increment (0.05-1.0 units)
- **Live Updates**: Position values update in real-time during drag

### Cube Selection
- **Dropdown Menu**: All cubes listed by index with "None" option
- **Visual Feedback**: Selected cube highlighted in cyan (#66ccff)
- **State Synchronization**: UI selection matches 3D scene selection

## Performance Optimizations

### Efficient Rendering
- **Suspense Boundaries**: Lazy loading prevention of render blocking
- **Memoized Calculations**: Ray caster and plane objects cached
- **Minimal Re-renders**: Zustand state slicing reduces unnecessary updates

### Memory Management
- **Geometry Reuse**: Shared box/sphere geometries across instances
- **Event Cleanup**: Proper event listener management
- **State Cleanup**: Automatic cleanup on cube deletion

### Animation Performance
- **useFrame Integration**: React Three Fiber's optimized animation loop
- **Direct Transform**: Bypassing React reconciliation for position updates
- **Conditional Updates**: Animation only when speed > 0

## Integration Examples

### Standalone Usage
```tsx
import CubesSpotlight from './cubes-spotlight/CubesSpotlight';

function App() {
  return <CubesSpotlight />;
}
```

### Programmatic Control
```tsx
import { setSpotlightPosition, nudgeSpotlight } from './cubes-spotlight/CubesSpotlight';

// Position spotlight at coordinates
setSpotlightPosition(2.5, 1, -1.8);

// Move spotlight relatively
nudgeSpotlight(0.5, 0); // Move right
nudgeSpotlight(0, -0.3); // Move forward
```

### Store Integration
```tsx
import { useStore } from './cubes-spotlight/CubesSpotlight';

function ExternalController() {
  const cubes = useStore(state => state.cubes);
  const addCube = useStore(state => state.addCube);

  return (
    <button onClick={addCube}>
      Add Cube (Total: {cubes.length})
    </button>
  );
}
```

## Dependencies
- **React Three.js**: Core 3D rendering and interaction
- **Zustand**: Lightweight state management
- **Three.js**: 3D mathematics and geometry utilities
- **Tailwind CSS**: Styling system for UI components

## Educational Value
- **State Management Patterns**: Zustand store design and usage
- **3D Interaction**: Mouse-to-world coordinate conversion
- **Animation Techniques**: Frame-based animation with React Three.js
- **UI/3D Integration**: Bridging 2D controls with 3D scene state

This tool demonstrates professional-grade 3D scene management with sophisticated user interaction, making it ideal for learning advanced React Three.js techniques and state management patterns.
