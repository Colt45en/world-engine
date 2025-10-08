# Avatar Impostor Tool

A sophisticated **multi-view avatar impostor rendering system** that creates high-quality real-time avatars using a ring of 9 cameras capturing synchronized views, depth-aware occlusion testing, and advanced shader-based view blending.

## 🎯 Core Technology

### Multi-View Camera Ring System
- **9-Camera Setup**: Configurable ring of cameras orbiting the avatar at parametric radius and height
- **Real-Time Capture**: Synchronized Frame Buffer Object (FBO) rendering with depth textures
- **Dynamic Positioning**: Continuous rotation with configurable speed and look-at targeting
- **Depth-Aware Processing**: Per-pixel depth testing with occlusion epsilon for surface accuracy

### Advanced Impostor Shader
- **Multi-View Blending**: Weighted combination of up to 9 simultaneous view contributions
- **Surface-Normal Weighting**: Angular bias compensation using surface normal dot product with view direction
- **Gradient-Based Confidence**: Luminance gradient analysis for texture detail quality assessment
- **Occlusion Detection**: Depth buffer comparison with epsilon tolerance for hidden surface removal

## 📊 Features

### Camera System Controls
- **Camera Count**: 3-9 cameras (default: 9)
- **Texture Resolution**: 256/512/1024 per view (default: 512)
- **Rotation Speed**: 0-2 rad/s orbital motion (default: 0.35)
- **Ring Geometry**: Configurable radius (1-4m) and height (1-4m)
- **Capture Rate**: 1-10fps texture update frequency

### Rendering Parameters
- **Surface Bias**: Normal-based weighting threshold (0-0.1, default: 0.02)
- **Gradient Filtering**: Enable/disable texture detail confidence analysis
- **Gradient Thresholds**: Low/high gradient cutoffs for quality assessment
- **Occlusion Epsilon**: Depth comparison tolerance (0-0.1, default: 0.02)

### Debug Visualization Modes
1. **View ID**: Color-coded contribution from each camera
2. **Quality**: Blend quality heatmap (red = low, green = high)
3. **Occlusion**: Blue intensity showing occluded pixel ratio
4. **Confidence**: Green intensity from gradient-based confidence
5. **Cos Theta**: Surface normal alignment visualization

## 🏗️ Platform Environment

### Hexagonal Platform Generation
- **Procedural Geometry**: Algorithmic hexagonal grid with configurable ring count
- **Height Function**: Combined bowl curvature and hexagonal ridge patterns
- **Physical Materials**: Glass-like transmission with clearcoat and subsurface effects
- **Grid Visualization**: Circular line overlays showing hexagonal cell boundaries

### Interactive Controls
- **Orbit Camera**: Full 3DOF navigation around the impostor scene
- **Real-Time Updates**: Live parameter adjustment with immediate visual feedback
- **Performance Monitoring**: Memory usage, capture rates, and system statistics
- **Data Export**: JSON export of camera positions and multi-view configuration

## 🎮 Usage Guide

### Basic Operation
1. **Camera Ring**: Observe the 9 cameras orbiting the central avatar mesh
2. **View Blending**: Watch real-time impostor updates as cameras capture different angles
3. **Debug Modes**: Use diagnostic visualizations to understand rendering quality
4. **Parameter Tuning**: Adjust bias, gradients, and epsilon values for optimal results

### Advanced Techniques
- **Quality Optimization**: Use gradient confidence to filter high-detail regions
- **Occlusion Tuning**: Adjust epsilon for complex geometry depth handling
- **Performance Scaling**: Balance camera count vs texture resolution for target framerate
- **Memory Management**: Monitor texture memory usage in status panel

## 🔬 Technical Implementation

### Shader Pipeline
```glsl
// Multi-view texture sampling with depth testing
for (int i = 0; i < viewCount; i++) {
    // Project world position to view space
    vec4 clipSpacePos = camVP[i] * vec4(vWorldPos, 1.0);
    vec2 screenUV = (clipSpacePos.xy / clipSpacePos.w) * 0.5 + 0.5;

    // Depth occlusion test
    float expectedDepth = linearizeDepth(clipSpacePos.z, near[i], far[i]);
    float sampledDepth = linearizeDepth(getDepth(i, screenUV), near[i], far[i]);

    if (sampledDepth + occEps < expectedDepth) continue; // Occluded

    // Surface normal weighting
    float cosTheta = max(0.0, dot(normal, viewDir) - bias);

    // Gradient confidence analysis
    float confidence = getGradientConfidence(i, screenUV, gradLo, gradHi);

    // Weighted color accumulation
    accumulator += getColor(i, screenUV) * cosTheta * confidence;
    weightSum += cosTheta * confidence;
}
```

### Performance Characteristics
- **Memory Usage**: ~18MB for 9 cameras at 512² resolution (color + depth)
- **Render Calls**: 9 FBO renders per frame + final impostor pass
- **Shader Complexity**: 9-way texture sampling with depth testing per pixel
- **Update Frequency**: Configurable 1-10fps for texture capture optimization

## 🚀 Applications

### Real-Time Avatars
- **VR/AR Avatars**: High-quality representation with minimal geometry
- **Game Characters**: LOD system for distant characters with impostor fallback
- **Video Conferencing**: Depth-aware avatar representation for virtual meetings

### Research Applications
- **Multi-View Stereo**: Algorithm validation for view synthesis techniques
- **Depth Estimation**: Real-time depth fusion from multiple viewpoints
- **Impostor Rendering**: Advanced techniques for complex geometry approximation

---

*Advanced multi-view avatar impostor system with real-time depth-aware rendering, gradient-based confidence analysis, and comprehensive diagnostic visualization modes.*
