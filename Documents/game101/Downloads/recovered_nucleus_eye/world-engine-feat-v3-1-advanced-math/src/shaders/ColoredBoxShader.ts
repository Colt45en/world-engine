import * as THREE from 'three';

/**
 * Colored Box Shader Implementation
 * Creates customizable colored boxes with various effects and animations
 */

export interface ColoredBoxUniforms {
    [uniform: string]: THREE.IUniform<any>;
    boxColor: { value: THREE.Color };
    time: { value: number };
    intensity: { value: number };
    gradient: { value: boolean };
    gradientColor: { value: THREE.Color };
    animate: { value: boolean };
    pattern: { value: number }; // 0: solid, 1: gradient, 2: animated
}

export class ColoredBoxShader {
    public static vertexShader = `
    // Vertex attributes
    attribute vec3 position;
    attribute vec2 uv;
    attribute vec3 normal;

    // Matrices
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    uniform mat3 normalMatrix;

    // Varyings to pass to fragment shader
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vWorldPosition;
    varying vec3 vViewPosition;

    void main() {
        // Pass UV coordinates to fragment shader
        vUv = uv;

        // Transform normal to view space
        vNormal = normalize(normalMatrix * normal);

        // Calculate world position
        vec4 worldPosition = modelViewMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        vViewPosition = worldPosition.xyz;

        // Calculate final vertex position
        gl_Position = projectionMatrix * worldPosition;
    }
  `;

    public static fragmentShader = `
    precision highp float;

    // Uniforms
    uniform vec3 boxColor;
    uniform float time;
    uniform float intensity;
    uniform bool gradient;
    uniform vec3 gradientColor;
    uniform bool animate;
    uniform float pattern;

    // Varyings from vertex shader
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vWorldPosition;
    varying vec3 vViewPosition;

    // Utility functions
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    float noise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
        vec3 finalColor = boxColor;

        // Pattern selection
        if (pattern < 0.5) {
            // Solid color
            finalColor = boxColor;

        } else if (pattern < 1.5) {
            // Gradient pattern
            if (gradient) {
                float gradientFactor = vUv.y; // Vertical gradient
                finalColor = mix(boxColor, gradientColor, gradientFactor);
            }

        } else if (pattern < 2.5) {
            // Animated color pattern
            if (animate) {
                float animationFactor = sin(time * 2.0 + vUv.x * 10.0) * 0.5 + 0.5;
                vec3 animatedColor = hsv2rgb(vec3(
                    fract(time * 0.1 + vUv.x * 0.5),
                    0.7,
                    0.8
                ));
                finalColor = mix(boxColor, animatedColor, animationFactor);
            }

        } else {
            // Procedural noise pattern
            float noiseValue = noise(vUv * 10.0 + time);
            finalColor = mix(boxColor, gradientColor, noiseValue);
        }

        // Apply lighting based on normal
        vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
        float lightIntensity = max(dot(vNormal, lightDirection), 0.3);
        finalColor *= lightIntensity;

        // Apply intensity multiplier
        finalColor *= intensity;

        // Set final fragment color
        gl_FragColor = vec4(finalColor, 1.0);
    }
  `;

    public static createMaterial(options: Partial<ColoredBoxUniforms> = {}): THREE.ShaderMaterial {
        const uniforms: ColoredBoxUniforms = {
            boxColor: { value: options.boxColor?.value || new THREE.Color(0xff0000) },
            time: { value: options.time?.value || 0 },
            intensity: { value: options.intensity?.value || 1.0 },
            gradient: { value: options.gradient?.value || false },
            gradientColor: { value: options.gradientColor?.value || new THREE.Color(0x0000ff) },
            animate: { value: options.animate?.value || false },
            pattern: { value: options.pattern?.value || 0 }
        };

        return new THREE.ShaderMaterial({
            uniforms,
            vertexShader: ColoredBoxShader.vertexShader,
            fragmentShader: ColoredBoxShader.fragmentShader,
            transparent: false,
            side: THREE.FrontSide
        });
    }

    public static createColoredBox(
        size: [number, number, number] = [1, 1, 1],
        color: THREE.Color = new THREE.Color(0xff0000),
        options: Partial<ColoredBoxUniforms> = {}
    ): THREE.Mesh {
        const geometry = new THREE.BoxGeometry(...size);
        const material = ColoredBoxShader.createMaterial({
            boxColor: { value: color },
            ...options
        });

        return new THREE.Mesh(geometry, material);
    }

    // Animation helper
    public static updateTime(material: THREE.ShaderMaterial, deltaTime: number): void {
        if (material.uniforms.time) {
            material.uniforms.time.value += deltaTime;
        }
    }

    // Color change helper
    public static setColor(material: THREE.ShaderMaterial, color: THREE.Color): void {
        if (material.uniforms.boxColor) {
            material.uniforms.boxColor.value.copy(color);
        }
    }

    // Pattern change helper
    public static setPattern(material: THREE.ShaderMaterial, pattern: number): void {
        if (material.uniforms.pattern) {
            material.uniforms.pattern.value = pattern;
        }
    }
}

// Export utility types
export type ColoredBoxMaterial = THREE.ShaderMaterial & {
    uniforms: ColoredBoxUniforms;
};

// Preset configurations
export const ColoredBoxPresets = {
    solid: (color: THREE.Color) => ({
        boxColor: { value: color },
        pattern: { value: 0 },
        intensity: { value: 1.0 }
    }),

    gradient: (color1: THREE.Color, color2: THREE.Color) => ({
        boxColor: { value: color1 },
        gradientColor: { value: color2 },
        gradient: { value: true },
        pattern: { value: 1 },
        intensity: { value: 1.0 }
    }),

    animated: (baseColor: THREE.Color) => ({
        boxColor: { value: baseColor },
        animate: { value: true },
        pattern: { value: 2 },
        intensity: { value: 1.2 }
    }),

    noise: (color1: THREE.Color, color2: THREE.Color) => ({
        boxColor: { value: color1 },
        gradientColor: { value: color2 },
        pattern: { value: 3 },
        intensity: { value: 1.0 }
    })
};
