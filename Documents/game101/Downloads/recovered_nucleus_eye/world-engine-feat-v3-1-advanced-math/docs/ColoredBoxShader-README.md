# 🎨 Colored Box Shader System

A powerful and flexible shader system for creating colored boxes with various effects in Three.js applications.

## 📁 Files Created

- `src/shaders/ColoredBoxShader.ts` - Core shader implementation
- `src/demos/ColoredBoxDemo.tsx` - React component demo
- `src/web/colored_box_shader_demo.html` - Standalone HTML demo
- `src/utils/ShaderUtils.ts` - Utility functions and helpers

## 🚀 Quick Start

### Basic Usage

```typescript
import { ColoredBoxShader } from '../shaders/ColoredBoxShader';

// Create a simple red box
const redBox = ColoredBoxShader.createColoredBox(
  [1, 1, 1],                    // size [width, height, depth]
  new THREE.Color(0xff0000)     // color
);

// Add to scene
scene.add(redBox);
```

### Using Presets

```typescript
import { ColoredBoxPresets } from '../shaders/ColoredBoxShader';

// Gradient box
const gradientBox = ColoredBoxShader.createColoredBox(
  [1, 1, 1],
  new THREE.Color(0x00ff00),
  ColoredBoxPresets.gradient(
    new THREE.Color(0x00ff00),  // start color
    new THREE.Color(0x0000ff)   // end color
  )
);

// Animated box
const animatedBox = ColoredBoxShader.createColoredBox(
  [1, 1, 1],
  new THREE.Color(0xff00ff),
  ColoredBoxPresets.animated(new THREE.Color(0xff00ff))
);
```

### Using Utility Functions

```typescript
import ShaderUtils from '../utils/ShaderUtils';

// Create simple boxes with minimal code
const simpleBox = ShaderUtils.createSimpleBox(0xff0000, 1, [0, 0, 0]);
const gradientBox = ShaderUtils.createGradientBox(0x00ff00, 0x0000ff, 1, [2, 0, 0]);
const animatedBox = ShaderUtils.createAnimatedBox(0xff00ff, 1, [-2, 0, 0]);

// Create a grid of boxes
const boxes = ShaderUtils.createBoxGrid(scene, 3, 2, true);
```

## 🎛️ Shader Parameters

### Uniforms

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `boxColor` | `vec3` | Base color of the box | `(1, 0, 0)` |
| `time` | `float` | Animation time | `0.0` |
| `intensity` | `float` | Color intensity multiplier | `1.0` |
| `gradient` | `bool` | Enable gradient effect | `false` |
| `gradientColor` | `vec3` | Secondary color for gradient | `(0, 0, 1)` |
| `animate` | `bool` | Enable color animation | `false` |
| `pattern` | `float` | Pattern type (0-3) | `0.0` |

### Pattern Types

- **0**: Solid color
- **1**: Gradient (vertical)
- **2**: Animated colors
- **3**: Noise pattern

## 🎨 Animation

### Basic Animation Loop

```typescript
const clock = new THREE.Clock();

function animate() {
  const deltaTime = clock.getDelta();

  // Update shader time
  ColoredBoxShader.updateTime(material, deltaTime);

  // Render
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
```

### Advanced Animation Effects

```typescript
import { ShaderUtils, ColorPalettes } from '../utils/ShaderUtils';

// Rainbow effect
ShaderUtils.createRainbowEffect(material, time, 0.5);

// Pulsing intensity
ShaderUtils.applyPulseEffect(material, time, 0.5, 1.5, 2);

// Color palette cycling
ShaderUtils.cyclePalette(material, ColorPalettes.sunset, time, 1);

// Wave motion for multiple boxes
ShaderUtils.applyWaveMotion(boxes, time, 1, 1);
```

## 🎯 Interactive Features

### Dynamic Color Changes

```typescript
// Change color instantly
ColoredBoxShader.setColor(material, new THREE.Color(0x00ff00));

// Smooth color transition
ShaderUtils.changeColorSmooth(material, targetColor, 0.02);

// Randomize color
ShaderUtils.randomizeColor(material);
```

### Pattern Switching

```typescript
// Change pattern type
ColoredBoxShader.setPattern(material, 2); // Switch to animated pattern

// Enable/disable effects
material.uniforms.gradient.value = true;
material.uniforms.animate.value = false;
```

## 📱 HTML Demo Usage

Open `src/web/colored_box_shader_demo.html` in a web browser to see:

- **5 different box types** with various effects
- **Interactive controls** for real-time manipulation
- **Click anywhere** to randomize colors
- **Control buttons** for pattern changes and animation toggles

### Controls

- 🎲 **Random Colors**: Randomize all box colors
- 🔄 **Change Pattern**: Switch to random patterns
- ⏯️ **Toggle Animation**: Start/stop rotation animation
- 🔄 **Reset View**: Reset camera and box positions

## ⚛️ React Integration

Use the `ColoredBoxDemo` component:

```tsx
import ColoredBoxDemo from '../demos/ColoredBoxDemo';

function App() {
  return (
    <div>
      <ColoredBoxDemo />
    </div>
  );
}
```

## 🎨 Color Palettes

Pre-defined color palettes available in `ShaderUtils`:

- **Sunset**: Warm orange/red tones
- **Ocean**: Deep blue/purple tones
- **Forest**: Natural green tones
- **Neon**: Bright electric colors
- **Fire**: Hot red/orange/yellow tones

```typescript
// Use a palette
ShaderUtils.cyclePalette(material, ColorPalettes.neon, time, 1);
```

## 🛠️ Customization

### Custom Shader Material

```typescript
const customMaterial = ColoredBoxShader.createMaterial({
  boxColor: { value: new THREE.Color(0x00ffff) },
  gradientColor: { value: new THREE.Color(0xff8800) },
  gradient: { value: true },
  animate: { value: true },
  pattern: { value: 1.5 },
  intensity: { value: 1.3 }
});
```

### Extending the Shader

You can modify the fragment shader to add new effects:

```glsl
// Add custom patterns in the fragmentShader
} else if (pattern < 3.5) {
  // Your custom pattern here
  float custom = sin(vUv.x * 20.0) * cos(vUv.y * 20.0);
  finalColor = mix(boxColor, gradientColor, custom);
}
```

## 🧹 Cleanup

Always dispose of resources when done:

```typescript
// Dispose individual materials
material.dispose();

// Dispose multiple materials
ShaderUtils.disposeShaderMaterials(materials);

// Dispose geometries
geometry.dispose();
```

## 📊 Performance Tips

1. **Reuse materials** when possible for boxes with the same appearance
2. **Limit the number of animated boxes** for better performance
3. **Use instanced rendering** for large numbers of similar boxes
4. **Dispose resources** properly to prevent memory leaks

## 🎮 Example Applications

This shader system is perfect for:

- **Game environments** with interactive objects
- **Data visualization** with color-coded elements
- **Educational demos** showing shader concepts
- **Art installations** with dynamic visuals
- **UI elements** with animated feedback

## 🐛 Troubleshooting

### Common Issues

1. **Boxes appear black**: Check that lighting is properly set up
2. **Animation not working**: Ensure time uniform is being updated
3. **Colors not changing**: Verify uniform values are being set correctly
4. **Performance issues**: Reduce number of animated boxes or lower animation frequency

### Debug Tips

```typescript
// Log uniform values
console.log('Uniforms:', material.uniforms);

// Check material compilation
console.log('Material:', material);

// Monitor performance
console.time('render');
renderer.render(scene, camera);
console.timeEnd('render');
```

---

**Happy Shading! 🎨✨**
