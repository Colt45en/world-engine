# Nexus Forge v2.1 - Visual Reality Constructor

An advanced React Three.js application for creating interactive 3D scenes with integrated media capabilities, thematic environments, and real-time controls.

## 🎯 Overview

Nexus Forge is a sophisticated visual reality constructor that combines:
- **3D Scene Management** - Interactive cubes with customizable properties
- **Media Integration** - Image slideshow and video playback in 3D space
- **Thematic Environments** - Six distinct atmospheric themes
- **Advanced Lighting** - Spotlight and ambient light controls
- **Particle Systems** - Dynamic particle effects
- **State Management** - Zustand-powered reactive state

## ✨ Key Features

### **Interactive 3D Objects**
- Click-to-select cube system
- Real-time property editing (scale, color, tag)
- Animated rotation and pulsing effects
- Customizable positioning and themes

### **Media Systems**
- **Image Slideshow**: Upload multiple images displayed in 3D strip
- **Video Playback**: Video files displayed on 3D screen
- **Auto-advancing**: Configurable slideshow timing
- **Real-time Updates**: Live media switching and controls

### **Thematic Environments**
- **Neutral Void**: Pure potential in perfect balance
- **Karma Echoes**: Search through echoes of past actions
- **Threading the Dao**: Patterns interlace through reality
- **Blank Origin**: Clean slate of pure potential
- **Celestial Ascension**: Rising into realms of pure thought
- **Ethereal Twilight**: Liminal space of mysteries

### **Advanced Lighting**
- **Spotlight System**: Positioned light with intensity and color controls
- **Ambient Lighting**: Global illumination adjustment
- **Dynamic Shadows**: Real-time shadow casting
- **Theme-based Ambience**: Environment-specific lighting

### **Particle Effects**
- **500 Particles**: Spherically distributed around scene
- **Theme Colors**: Particles match environment ambience
- **Smooth Animation**: Gentle rotation and color variation
- **Performance Optimized**: Efficient buffer geometry

## 🎮 Controls & Interface

### **Scene Controls**
- Animation Speed: Adjust cube rotation speed (0-2x)
- Fog Density: Control atmospheric fog (0.05-1.0)
- View Mode: Switch between orbit and fixed camera
- Particle Toggle: Enable/disable particle effects
- Reset Scene: Return to default state
- Add Cube: Generate new random cube

### **Light Controls**
- Spotlight Toggle: Enable/disable directional light
- Intensity: Adjust spotlight brightness (0.1-3.0)
- Color Picker: Six preset spotlight colors
- Ambient: Global lighting intensity (0-1.0)

### **Cube Editing**
- Tag Selection: Assign thematic identities
- Scale Adjustment: Size scaling (0.5-2.0x)
- Color Palette: Six preset colors per cube
- Remove: Delete selected cubes

### **Media Controls**
- Image Upload: Multiple file selection
- Slideshow: Previous/Next navigation
- Auto-play: Configurable timing (1-10 seconds)
- Video Upload: Single video file support
- Live Preview: Real-time 3D display

## 🏗 Architecture

### **State Management (Zustand)**
```typescript
interface Store {
  // 3D Objects
  cubes: Cube[];
  selectedCube: number | null;

  // Lighting
  spotlight: SpotlightConfig;
  ambientLight: AmbientConfig;

  // Environment
  theme: string;
  animationSpeed: number;
  fogDensity: number;
  viewMode: 'orbit' | 'fixed';
  particles: boolean;

  // Media
  imageTextures: THREE.Texture[];
  currentSlide: number;
  slideshowSpeed: number;
  slideshowActive: boolean;
  videoURL: string;
}
```

### **Component Structure**
```
NexusForge/
├── 3D Components
│   ├── Cube - Interactive 3D objects
│   ├── EnhancedSpotlight - Dynamic lighting
│   ├── ParticleSystem - Atmospheric effects
│   ├── VideoScreen - 3D video display
│   └── SlideshowStrip - Image carousel
├── UI Panels
│   ├── ThemeSelector - Environment themes
│   ├── SceneControls - Animation & fog
│   ├── LightControls - Spotlight system
│   ├── CubeControls - Object editing
│   ├── MediaPanel - File uploads
│   └── MythicCodex - Lore accordion
└── Store Integration - Zustand state
```

### **Shader Integration**
- **Standard Materials**: PBR rendering with metalness/roughness
- **Emissive Effects**: Selection highlighting
- **Video Textures**: Real-time video display
- **Particle Materials**: Vertex colors with transparency

## 🎨 Usage Examples

### **Basic Setup**
```tsx
import NexusForge from './nexus-forge';

function App() {
  return <NexusForge />;
}
```

### **Standalone Canvas Integration**
```tsx
import { Scene, MediaBridges } from './nexus-forge/NexusForge';
import { Canvas } from '@react-three/fiber';

function CustomForge() {
  return (
    <Canvas>
      <Scene />
      <MediaBridges />
    </Canvas>
  );
}
```

### **State Access**
```tsx
import { useStore } from './nexus-forge/NexusForge';

function ExternalComponent() {
  const cubes = useStore(s => s.cubes);
  const addCube = useStore(s => s.addCube);

  return (
    <div>
      <p>Cubes: {cubes.length}</p>
      <button onClick={addCube}>Add Cube</button>
    </div>
  );
}
```

## 🎵 Media Integration

### **Supported Formats**
- **Images**: JPG, PNG, GIF, WebP
- **Videos**: MP4, WebM, OGV

### **3D Display System**
- **Slideshow Strip**: Up to 8 images in horizontal layout
- **Video Screen**: Large display with aspect ratio preservation
- **Real-time Updates**: Immediate texture application
- **Auto-advance**: Configurable slideshow timing

### **File Handling**
```typescript
// Image processing
const handleImages = (files: FileList) => {
  const textures = Array.from(files).map(file => {
    const url = URL.createObjectURL(file);
    return new THREE.TextureLoader().load(url);
  });
  addImages(textures);
};

// Video processing
const handleVideo = (file: File) => {
  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.src = url;
  video.play();
  // Creates THREE.VideoTexture automatically
};
```

## 🌟 Thematic System

Each theme provides:
- **Background Color**: Scene backdrop
- **Fog Color**: Atmospheric tinting
- **Ambient Color**: Particle system base
- **Narrative**: Philosophical description
- **Title**: Mystical designation

### **Theme Implementation**
```typescript
const themeMap = {
  Seeker: {
    bg: "#001122",
    fog: "#223344",
    title: "Karma Echoes",
    ambience: "#113355",
    description: "Those who search through echoes..."
  }
  // ... other themes
};
```

## 🔧 Performance Optimizations

### **Particle System**
- Uses `BufferGeometry` for efficiency
- Spherical distribution algorithm
- Color variation with HSL manipulation
- Conditional rendering based on user toggle

### **Media Handling**
- Lazy texture loading
- Video element recycling
- Memory cleanup on component unmount
- Optimized texture filtering

### **State Management**
- Zustand for minimal re-renders
- Selective subscriptions
- Immutable state updates
- Efficient store selectors

## 🎭 Philosophical Integration

Nexus Forge includes a "Mythic Symbol Codex" featuring:
- **The Oars of Karma**: Fate determination through action
- **The Drowned Monastery**: Hidden wisdom in the depths
- **The Whirlpool of Forgotten Names**: Erasure of the past
- **The Void of Echoes**: Thoughts becoming reality
- **The Crystal Nexus**: Timeline intersection point

## 🚀 Advanced Features

### **Real-time Interactions**
- Click-to-select cube system
- Drag-responsive controls
- Live parameter adjustment
- Immediate visual feedback

### **Media Synchronization**
- Slideshow auto-advance
- Video loop controls
- Texture update optimization
- Cross-origin media support

### **Environment Management**
- Dynamic fog distance calculation
- Theme-based particle colors
- Lighting intensity scaling
- Camera mode switching

## 📊 Technical Specifications

- **Rendering**: WebGL via React Three Fiber
- **State**: Zustand reactive store
- **UI**: Custom shadcn/ui-inspired components
- **Media**: HTML5 video/canvas integration
- **Performance**: 60fps targeting with optimized renders
- **Memory**: Automatic cleanup and garbage collection

Nexus Forge represents the culmination of interactive 3D web experiences, combining artistic vision with technical sophistication to create a truly immersive visual reality constructor.
