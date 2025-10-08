# Mask Mesher Tool

A standalone tool for creating 3D volumetric meshes from 2D silhouette images. Converts front/side mask images into solid 3D geometry using voxel-based intersection.

## Features

- **Dual-Mask Input**: Front mask (required) + optional side mask for depth constraint
- **Volumetric Intersection**: Creates volume by intersecting 2D silhouettes in 3D space
- **Voxel Surface Generation**: Generates mesh faces for exposed voxel surfaces
- **Real-time Preview**: Interactive 3D preview with rotation
- **Export Ready**: Optimized geometry suitable for GLB/OBJ export

## How It Works

1. **Front Mask**: Gates the X-Y plane (defines the front silhouette)
2. **Side Mask** (optional): Gates the Z-Y plane (defines depth profile)
3. **Volume Building**: For each Y-level, intersects front and side constraints
4. **Surface Extraction**: Generates faces for voxels with exposed neighbors
5. **Mesh Generation**: Creates optimized BufferGeometry with normals

## Controls

### File Input
- **Front Silhouette**: Primary shape definition (required)
- **Side Silhouette**: Depth profile constraint (optional)

### Mesh Settings
- **resolution**: Voxel grid resolution (16-128)
- **threshold**: Luminance threshold for mask detection (0.1-0.9)
- **meshColor**: Material color
- **wireframe**: Toggle wireframe view
- **metalness/roughness**: PBR material properties

## Usage Tips

1. **Image Preparation**: Use high-contrast silhouettes (white shape on black background)
2. **Resolution**: Higher resolution = more detail but slower processing
3. **Threshold**: Adjust based on image contrast (0.5 works for most cases)
4. **Side Mask**: Use for more realistic depth constraints (e.g., avatar profiles)

## Technical Details

- **Algorithm**: 3D voxel rasterization with 6-neighbor face culling
- **Performance**: O(n³) complexity where n = resolution
- **Memory**: Temporary volume buffer of size resolution³
- **Output**: Indexed BufferGeometry with computed vertex normals

## Integration

```tsx
import MaskMesher from './MaskMesher';

function MyApp() {
  return <MaskMesher />;
}
```

## File Format Support

- Input: Any image format supported by HTML5 (PNG, JPG, WebP, etc.)
- Output: THREE.BufferGeometry (ready for GLB/OBJ export)
