# Shared Utilities

Common utilities, types, and helper functions used across all sandbox tools.

## Files

### `types.ts`
- **Common Type Definitions**: Vector3D, Vector4D, MaterialConfig, etc.
- **Tool Interfaces**: BaseTool, RenderableTool
- **Event System**: ToolEvent, ProgressEvent, ErrorEvent
- **Geometry Types**: MeshData, VolumeData, ImageMask

### `utils.ts`
- **Mathematical Functions**: clamp, lerp, smoothstep, random generators
- **Vector Operations**: Vec3, Vec4 utility classes
- **Image Processing**: fileToImageData, imageDataToMask
- **Geometry Processing**: bounding box calculation, normal computation
- **Performance Utilities**: debounce, throttle, object pooling
- **File Operations**: download, clipboard access

## Usage

```typescript
import { Vec3, clamp, debounce } from '../shared/utils';
import { Vector3D, MaterialConfig } from '../shared/types';

// Vector math
const point1: Vector3D = Vec3.create(1, 2, 3);
const point2: Vector3D = Vec3.create(4, 5, 6);
const distance = Vec3.distance(point1, point2);

// Utility functions
const clamped = clamp(value, 0, 1);
const debouncedUpdate = debounce(updateFunction, 300);
```

## Key Features

- **Type Safety**: Comprehensive TypeScript definitions
- **Performance**: Optimized algorithms and object pooling
- **Reusability**: Common patterns abstracted into utilities
- **Documentation**: Well-documented functions with examples
