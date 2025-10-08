# Sonic FX Tool

A standalone cinematic speed overlay system that creates motion streaks and radial visual effects to enhance speed perception without modifying underlying materials.

## Features

- **Real-time Speed Overlay**: Full-screen shader overlay for motion effects
- **Customizable Parameters**: Control speed, intensity, bands, rotation, falloff
- **Color Customization**: RGB color controls for visual theming
- **Performance Optimized**: Minimal render overhead with efficient shaders
- **Non-destructive**: Overlays existing scenes without material changes

## Controls

- **enabled**: Toggle effect on/off
- **speed**: Animation speed multiplier (0-10)
- **intensity**: Overall effect intensity (0-2)
- **bands**: Number of radial stripes (50-500)
- **rotation**: Stripe rotation speed (0-20)
- **falloff**: Radial falloff distance (0.5-2.0)
- **core**: Central core size (0.05-0.5)
- **colorR/G/B**: RGB color components (0-1)

## Usage

```tsx
import SonicFX from './SonicFX';

function MyApp() {
  return <SonicFX />;
}
```

## Integration

The SonicOverlay component can be imported and used in any Three.js/React scene:

```tsx
import { SonicOverlay } from './SonicFX';

// Add to your scene
<SonicOverlay />
```

## Technical Details

- Uses custom shader material with time-based animations
- Renders at high render order (1000) to appear above other objects
- Transparent blending with depth testing disabled
- Optimized fragment shader with smooth interpolation functions
