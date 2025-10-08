# GeoHex Flower Observatory

This scene replaces the previous HSV workspace demo with a React Three Fiber
experience that highlights sacred-geometry overlays inside a glass Goldberg
dome.

## File Layout

- `src/App.tsx` boots the canvas, applies a minimal loading overlay, and points
   the camera toward the observatory.
- `src/GeoHexFlowerScene.jsx` contains the plain-JavaScript scene logic.
- `src/GeoHexFlowerScene.tsx` re-exports the JSX module for any TypeScript-aware
   imports.

## Visual Elements

- **Glass Dome:** `THREE.IcosahedronGeometry` with transparent physical
   material and animated thickness to simulate refractive glass.
- **Flower Platform:** Hex-packed circles on the ground plane, animated subtly
   to give the flower-of-life motion. Sparkles add depth.
- **Energy Streams:** Pulsing line segments connecting the dome to the core
   platform, modulated in the render loop.
- **Bloom:** Post-processing bloom driven by `@react-three/postprocessing` to
   emphasize highlights.

## Running the Scene

1. Install dependencies if you have not already:
   ```powershell
   npm install three @react-three/fiber @react-three/drei @react-three/postprocessing
   ```
2. Start your preferred React dev server (for example, Vite or CRA) from the
   repository root.
3. Open the served page in a browser to explore the GeoHex Flower Observatory.

> **Note:** The scene uses plain JavaScript modules with `// @ts-nocheck`; no
> TypeScript annotations are required.
