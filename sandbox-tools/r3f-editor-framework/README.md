# R3F Editor Framework - Professional 3D Scene Editor

## Overview
A comprehensive, production-grade 3D editor framework built with React Three.js, featuring advanced state management, modular tooling system, and professional editing capabilities. This framework transforms the basic cube scene concept into a sophisticated development platform.

## 🚀 Key Features

### 🎯 **Advanced State Management**
- **Zustand Store**: Professional state management with reactive updates
- **Immer Integration**: Immutable state updates with patch-based history
- **Undo/Redo System**: Full history management with operation tracking
- **Entity Management**: Type-safe entity system with extensible architecture

### 🛠️ **Professional Tooling System**
- **Modular Tool Architecture**: Extensible plugin-based tool system
- **Command Palette**: VS Code-style command interface (Ctrl+K)
- **Scene Graph Inspector**: Hierarchical entity management
- **Transform Tools**: Direct manipulation of entity properties
- **Keyboard Shortcuts**: Professional workflow acceleration

### 🎨 **Layered Scene Architecture**
- **EntityLayer**: Modular entity rendering with separation of concerns
- **LightLayer**: Advanced lighting system with interactive controls
- **HelpersLayer**: Development aids (grids, axis indicators)
- **PostFX Pipeline**: Extensible post-processing system

### 🎮 **Interactive Controls**
- **Mouse-to-World Mapping**: Precise ground plane interaction
- **Click Selection**: Professional entity selection workflow
- **Drag Controls**: Interactive spotlight positioning
- **Animation System**: Per-entity animation with global speed control

## 📁 Architecture Overview

```
src/
├── App.tsx                    # Main application component
├── AppCanvas.tsx             # 3D Canvas wrapper with Suspense
├── state/
│   ├── store.ts              # Enhanced Zustand store with Immer
│   └── StoreProvider.tsx     # State management utilities
├── scene/
│   ├── SceneRoot.tsx         # Main scene composition
│   ├── layers/
│   │   ├── EntityLayer.tsx   # Entity rendering system
│   │   ├── LightLayer.tsx    # Lighting and shadows
│   │   └── HelpersLayer.tsx  # Development helpers
│   ├── hooks/
│   │   └── useGroundPointer.ts # Ground plane interaction
│   └── postfx/
│       └── PostFXPipeline.tsx # Post-processing effects
└── tools/
    ├── ToolManager.tsx       # Tool system management
    ├── types.ts              # TypeScript interfaces
    ├── TransformTool.tsx     # Entity transformation
    ├── SceneGraphInspector.tsx # Hierarchy management
    ├── CommandPalette.tsx    # Command interface
    └── CubeScenePanel.tsx    # Original cube scene controls
```

## 🎯 State Management System

### Entity System
```typescript
interface Entity {
  id: EntityId;
  kind: 'cube' | 'light' | 'camera' | 'empty';
  position: Vec3;
  rotation: Vec3;
  scale: Vec3;
  name?: string;
  visible?: boolean;
  // Entity-specific properties...
}
```

### Store Actions
```typescript
// Entity management
addEntity(entity: Entity): void
updateEntity(id: EntityId, partial: Partial<Entity>): void
removeEntity(id: EntityId): void
selectOnly(id?: EntityId): void

// Cube scene specific
addCube(): void
deleteSelected(): void
setAnimationSpeed(speed: number): void
toggleSpotlight(): void

// History management
undo(): void
redo(): void
```

## 🛠️ Tool System

### Built-in Tools

#### 1. **Transform Tool** (`Alt+T`)
- Direct manipulation of position, rotation, scale
- Real-time property editing with numeric inputs
- Color and name property management
- Visual feedback for selected entities

#### 2. **Scene Graph Inspector** (`Alt+G`)
- Hierarchical view of all entities
- Click-to-select functionality
- Entity type indicators
- Quick delete operations
- Property inspection panel

#### 3. **Cube Scene Panel** (`Alt+C`)
- Original cube scene controls
- Animation speed slider
- Spotlight toggle and positioning
- Undo/redo history display
- Programmatic spotlight control

### Command Palette (`Ctrl+K`)
```typescript
interface CommandItem {
  id: string;
  label: string;
  description?: string;
  shortcut?: string;
  category: string;
  action: () => void;
  isEnabled?: boolean;
}
```

Available commands:
- **Add Cube** (`Ctrl+N`): Create new cube entity
- **Delete Selected** (`Delete`): Remove selected entities
- **Toggle Spotlight** (`L`): Control scene lighting
- **Undo** (`Ctrl+Z`) / **Redo** (`Ctrl+Shift+Z`): History navigation

## 🎨 Advanced Features

### History System with Immer
```typescript
const commit = (fn: (draft: SceneState) => void) => {
  const [next, patches, inverse] = produceWithPatches(state, fn);
  set({
    ...next,
    history: [...state.history, { patches, inverse }],
    future: []
  });
};
```

### Ground Plane Interaction
```typescript
export function useGroundPointer(
  onPoint: (position: [number, number, number]) => void
) {
  // Accurate ray casting for mouse-to-world coordinate conversion
  // Handles both R3F events and fallback manual ray calculation
}
```

### Layered Scene Composition
- **Modular Architecture**: Each layer handles specific responsibilities
- **Performance Optimization**: Efficient rendering with selective updates
- **Development Workflow**: Clear separation of rendering concerns

## 💻 Usage Examples

### Basic Usage
```tsx
import R3FEditorFramework from './r3f-editor-framework/src/App';

function App() {
  return <R3FEditorFramework />;
}
```

### External State Access
```tsx
import { useScene, StoreUtils } from './r3f-editor-framework/src/App';

function ExternalController() {
  const cubes = useScene(state => StoreUtils.getCubes());
  const addCube = useScene(state => state.addCube);

  return (
    <button onClick={addCube}>
      Add Cube (Total: {cubes.length})
    </button>
  );
}
```

### Programmatic Control
```tsx
import { setSpotlightPosition, nudgeSpotlight } from './r3f-editor-framework/src/App';

// Direct spotlight positioning
setSpotlightPosition(2.5, 1, -1.8);

// Relative movement
nudgeSpotlight(0.5, 0);  // Move right
nudgeSpotlight(0, -0.3); // Move forward
```

## 🔧 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Open command palette |
| `Alt+T` | Toggle transform tool |
| `Alt+G` | Toggle scene graph |
| `Alt+C` | Toggle cube scene panel |
| `Ctrl+N` | Add new cube |
| `Delete` | Delete selected |
| `L` | Toggle spotlight |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Esc` | Close dialogs |

## 🚀 Extension Points

### Custom Tools
```typescript
const customTool: Tool = {
  id: 'my-tool',
  label: 'My Tool',
  category: 'custom',
  enable: () => console.log('Tool enabled'),
  disable: () => console.log('Tool disabled'),
  Panel: MyCustomPanel
};
```

### Custom Entities
```typescript
interface MyEntity extends Entity {
  kind: 'my-type';
  customProperty: string;
}
```

### Custom Commands
```typescript
const customCommands: CommandItem[] = [
  {
    id: 'custom-action',
    label: 'Custom Action',
    category: 'Custom',
    action: () => { /* custom logic */ }
  }
];
```

## 📊 Performance Features

- **Lazy Loading**: Components loaded on-demand
- **Selective Re-renders**: Zustand state slicing
- **Memory Management**: Automatic cleanup
- **Efficient Animation**: Direct Three.js manipulation
- **Optimized Ray Casting**: Cached raycaster instances

## 🎯 Production Considerations

### Dependencies
```json
{
  "@react-three/fiber": "^8.x",
  "@react-three/drei": "^9.x",
  "three": "^0.157.x",
  "zustand": "^4.x",
  "immer": "^10.x"
}
```

### Optional Enhancements
- **@react-three/postprocessing**: Advanced post-processing effects
- **@react-three/rapier**: Physics integration
- **@react-three/xr**: VR/AR support
- **React Developer Tools**: State inspection

## 🎓 Educational Value

This framework demonstrates:
- **Professional React Architecture**: Component composition patterns
- **Advanced State Management**: Zustand + Immer integration
- **3D Development Workflows**: Professional editor patterns
- **TypeScript Best Practices**: Type-safe development
- **Performance Optimization**: React Three.js best practices

The R3F Editor Framework represents a complete evolution from basic 3D scenes to professional-grade development tools, suitable for education, prototyping, and production applications.
