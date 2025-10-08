# Mushroom Codex - Advanced Parametric Mushroom Generator

## Overview

The Mushroom Codex is a sophisticated 3D mushroom generation system that creates parametric mushroom models with toon-shaded materials, procedural spot generation, outline shells, and profile management. Built with React Three.js, it offers both preset profiles and random generation capabilities.

## Features

### 🎨 Visual Features
- **Toon Shading Pipeline**: Custom gradient mapping with configurable shade levels
- **Outline Shell System**: Back-face rendered outlines with adjustable thickness
- **Procedural Spot Generation**: Stochastic white spots using normal distribution
- **Multi-Profile System**: 10+ built-in mushroom varieties
- **Real-time Animation**: Gentle rotation and floating effects

### 🔧 Technical Capabilities
- **Parametric Generation**: Full control over cap/stem dimensions and colors
- **Material System**: Advanced toon materials with texture support
- **Profile Management**: JSON-based configuration system
- **Export Functionality**: Download mushroom configurations as JSON
- **Interactive Selection**: Click-to-select with visual feedback
- **Garden Layout**: Grid-based arrangement with automatic positioning

### 🎯 Mathematical Features
- **Hemisphere Mapping**: Proper UV projection for surface spots
- **Normal Distribution**: Gaussian randomization for natural spot placement
- **Quaternion Orientation**: Accurate surface normal alignment
- **Parametric Control**: Linear interpolation for smooth variations

## Usage

### Basic Implementation

```tsx
import MushroomCodex from './MushroomCodex';

function App() {
  return <MushroomCodex />;
}
```

### Custom Mushroom Parameters

```tsx
interface MushroomParams {
  capRadius?: number;           // Cap radius (0.4-1.2)
  capHeightScale?: number;      // Cap height scaling (0.6-1.4)
  stemHeight?: number;          // Stem height (0.6-1.5)
  stemRadiusTop?: number;       // Top stem radius (0.1-0.3)
  stemRadiusBottom?: number;    // Bottom stem radius (0.15-0.35)
  capColor?: string;            // Cap color (CSS/hex)
  stemColor?: string;           // Stem color (CSS/hex)
  spotDensity?: number;         // Spots per unit area (0-15)
  spotScaleMin?: number;        // Minimum spot size (0.02-0.1)
  spotScaleMax?: number;        // Maximum spot size (0.05-0.15)
  outline?: boolean;            // Enable outline shell
  outlineScale?: number;        // Outline thickness (1.01-1.1)
  outlineColor?: string;        // Outline color
  shades?: number;              // Toon shading levels (2-10)
}
```

### Profile System

```tsx
const customProfile: MushroomProfile = {
  name: "custom_variety",
  seed: 123,
  capRadius: 0.8,
  capHeightScale: 0.9,
  stemHeight: 1.1,
  spotDensity: 6,
  capColor: "#ff6b35",
  stemColor: "#f4e4bc",
  outlineScale: 1.03
};
```

## Built-in Profiles

### Standard Varieties
- **Scarlet Tiny**: Classic red-capped mushroom (small)
- **Plump Fairy**: Round, fairy-tale style mushroom
- **Tall Cap**: Elongated cap with increased height
- **Ghost White**: Pure white variety without spots
- **Speckled Amber**: Orange-tinted with heavy spotting

### Specialty Varieties
- **Forest King**: Large, dark red forest mushroom
- **Violet Glow**: Purple-hued mystical variety
- **Inky Blue**: Deep blue cap with minimal spots
- **Sunset Peach**: Warm peach coloration
- **Midnight Teal**: Dark teal forest variety

## API Reference

### Core Functions

#### `buildMushroom(params: MushroomParams): BuiltMushroom`
Creates a complete mushroom with all components:
- Cap geometry with hemisphere mapping
- Stem with tapered cylinder geometry
- Procedural spot generation
- Optional outline shell
- Toon-shaded materials

#### `createToonMaterial(color, shades, texture?): THREE.MeshToonMaterial`
Generates toon-shaded materials with custom gradient mapping:
- Configurable shade levels
- Optional texture mapping
- Proper color space handling

#### `generateSpots(cap: THREE.Mesh, params): THREE.Mesh[]`
Creates procedural spots on mushroom cap:
- Normal distribution positioning
- Hemisphere surface projection
- Size variation with min/max bounds
- Surface normal alignment

#### `createOutlineGroup(root, scale, color): THREE.Group`
Builds outline shell system:
- Back-face material rendering
- Configurable thickness scaling
- Depth-write disabled for proper layering

### Component Props

#### `MushroomPrimitive`
```tsx
interface MushroomPrimitiveProps {
  params: MushroomParams;       // Mushroom configuration
  selected?: boolean;           // Selection state
  onPointerDown?: (event) => void; // Click handler
  animated?: boolean;           // Enable animation
}
```

## Interactive Controls

### Garden Management
- **Profile Spawning**: Click any profile to add to garden
- **Random Generation**: Create randomized parameters
- **Selection System**: Click mushrooms to select
- **Export Function**: Download selected mushroom as JSON
- **Clear Garden**: Remove all mushrooms

### Visual Feedback
- **Selection Indicator**: Floating "Selected" label
- **Profile Preview**: Color swatches and parameter display
- **Stats Panel**: Count and selection information
- **Parameter Display**: Real-time values for radius/height

## Mathematical Foundations

### Spot Generation Algorithm
```
1. Calculate hemisphere surface area: 2πr²
2. Determine spot count: density × area
3. Generate random direction vectors using Box-Muller transform
4. Project to hemisphere surface (y ≥ 0)
5. Apply cap height scaling to y-coordinate
6. Orient spots to surface normals using quaternions
```

### Toon Shading Implementation
```
1. Create gradient texture with specified shade levels
2. Use nearest-neighbor filtering for hard edges
3. Apply gradient map to MeshToonMaterial
4. Configure proper UV wrapping and color space
```

### Outline Shell Mathematics
```
1. Traverse mesh hierarchy to find geometry
2. Clone meshes with scaled vertices
3. Apply back-face culling for interior rendering
4. Disable depth-write for proper layering
```

## Performance Considerations

- **Geometry Caching**: Reuse base geometries when possible
- **Material Pooling**: Share toon materials between similar mushrooms
- **LOD System**: Consider distance-based detail reduction
- **Spot Optimization**: Limit maximum spot count for performance
- **Memory Management**: Proper cleanup of geometries and materials

## Export System

### JSON Structure
```json
{
  "name": "mushroom_name",
  "parameters": {
    "capRadius": 0.8,
    "capHeightScale": 0.9,
    "stemHeight": 1.1,
    "capColor": "#ff6b35",
    "stemColor": "#f4e4bc",
    "spotDensity": 6,
    "outline": true,
    "outlineScale": 1.03
  },
  "timestamp": 1640995200000
}
```

### GLB Export (Advanced)
The system supports geometry export to GLB format:
1. Collect all mesh geometries
2. Merge materials and textures
3. Generate proper UV coordinates
4. Export using THREE.js GLTFExporter

## Integration Examples

### With Leva Controls
```tsx
const controls = useControls({
  profile: { options: MUSHROOM_PROFILES.map(p => p.name) },
  animated: true,
  spotDensity: { value: 5, min: 0, max: 15 },
  outlineScale: { value: 1.02, min: 1.0, max: 1.1 }
});
```

### With Animation System
```tsx
useFrame(({ clock }) => {
  if (animated) {
    mushroomRef.current.rotation.y = clock.elapsedTime * 0.5;
    mushroomRef.current.position.y = Math.sin(clock.elapsedTime * 2) * 0.02;
  }
});
```

### Custom Profile Loading
```tsx
const loadCustomProfile = async (url: string) => {
  const response = await fetch(url);
  const profile: MushroomProfile = await response.json();
  return buildMushroom(profile);
};
```

## Advanced Features

### Texture Mapping
- Support for custom cap textures
- Proper UV coordinate generation
- Color space management
- Texture filtering and wrapping

### Shader Customization
- Custom toon gradient creation
- Outline thickness variation
- Spot material properties
- Lighting model adjustments

### Procedural Variations
- Seed-based reproducible generation
- Parameter interpolation
- Mutation algorithms
- Genetic breeding system

## Best Practices

1. **Parameter Validation**: Clamp values to reasonable ranges
2. **Memory Management**: Dispose geometries and materials properly
3. **Performance Monitoring**: Track polygon count and draw calls
4. **Profile Organization**: Group related varieties together
5. **Export Compatibility**: Ensure cross-platform JSON compatibility

## Troubleshooting

### Common Issues
- **Invisible Spots**: Check normal vector calculations
- **Outline Bleeding**: Adjust outline scale and depth settings
- **Performance Drops**: Limit concurrent mushroom count
- **Material Conflicts**: Ensure unique material instances

### Debug Tools
- Geometry wireframe visualization
- Normal vector display
- Spot position debugging
- Material property inspection

---

The Mushroom Codex represents a complete parametric generation system combining mathematical precision with artistic flexibility, suitable for games, simulations, and interactive experiences requiring organic procedural content.
