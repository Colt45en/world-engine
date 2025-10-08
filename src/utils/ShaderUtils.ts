import * as THREE from 'three';
import { ColoredBoxShader, ColoredBoxUniforms } from '../shaders/ColoredBoxShader';

/**
 * Utility functions for working with colored box shaders
 * Provides simplified APIs for common operations
 */

export class ShaderUtils {
    /**
     * Creates a simple colored box with minimal configuration
     */
    static createSimpleBox(
        color: string | number = 0xff0000,
        size: number = 1,
        position: [number, number, number] = [0, 0, 0]
    ): THREE.Mesh {
        const colorObj = new THREE.Color(color);
        const box = ColoredBoxShader.createColoredBox([size, size, size], colorObj);
        box.position.set(...position);
        return box;
    }

    /**
     * Creates a gradient box
     */
    static createGradientBox(
        color1: string | number,
        color2: string | number,
        size: number = 1,
        position: [number, number, number] = [0, 0, 0]
    ): THREE.Mesh {
        const box = ColoredBoxShader.createColoredBox(
            [size, size, size],
            new THREE.Color(color1),
            {
                gradientColor: { value: new THREE.Color(color2) },
                gradient: { value: true },
                pattern: { value: 1 }
            }
        );
        box.position.set(...position);
        return box;
    }

    /**
     * Creates an animated color-changing box
     */
    static createAnimatedBox(
        baseColor: string | number = 0xff0000,
        size: number = 1,
        position: [number, number, number] = [0, 0, 0]
    ): THREE.Mesh {
        const box = ColoredBoxShader.createColoredBox(
            [size, size, size],
            new THREE.Color(baseColor),
            {
                animate: { value: true },
                pattern: { value: 2 },
                intensity: { value: 1.2 }
            }
        );
        box.position.set(...position);
        return box;
    }

    /**
     * Animates a collection of shader materials
     */
    static animateShaders(materials: THREE.ShaderMaterial[], deltaTime: number): void {
        materials.forEach(material => {
            if (material.uniforms.time) {
                material.uniforms.time.value += deltaTime;
            }
        });
    }

    /**
     * Changes the color of a shader material with smooth transition
     */
    static changeColorSmooth(
        material: THREE.ShaderMaterial,
        targetColor: THREE.Color,
        speed: number = 0.02
    ): void {
        if (material.uniforms.boxColor) {
            const currentColor = material.uniforms.boxColor.value;
            currentColor.lerp(targetColor, speed);
        }
    }

    /**
     * Creates a rainbow effect by cycling through hue values
     */
    static createRainbowEffect(
        material: THREE.ShaderMaterial,
        time: number,
        speed: number = 1
    ): void {
        if (material.uniforms.boxColor) {
            const hue = (time * speed) % 1;
            const rainbowColor = new THREE.Color().setHSL(hue, 0.7, 0.5);
            material.uniforms.boxColor.value.copy(rainbowColor);
        }
    }

    /**
     * Applies a pulsing intensity effect
     */
    static applyPulseEffect(
        material: THREE.ShaderMaterial,
        time: number,
        minIntensity: number = 0.5,
        maxIntensity: number = 1.5,
        speed: number = 2
    ): void {
        if (material.uniforms.intensity) {
            const pulse = Math.sin(time * speed) * 0.5 + 0.5;
            material.uniforms.intensity.value = minIntensity + pulse * (maxIntensity - minIntensity);
        }
    }

    /**
     * Randomizes the color of a material
     */
    static randomizeColor(material: THREE.ShaderMaterial): void {
        if (material.uniforms.boxColor) {
            const randomColor = new THREE.Color().setHSL(
                Math.random(),
                0.5 + Math.random() * 0.5,
                0.4 + Math.random() * 0.4
            );
            material.uniforms.boxColor.value.copy(randomColor);
        }
    }

    /**
     * Sets up a color palette and cycles through it
     */
    static cyclePalette(
        material: THREE.ShaderMaterial,
        palette: THREE.Color[],
        time: number,
        cycleSpeed: number = 1
    ): void {
        if (!palette.length || !material.uniforms.boxColor) return;

        const cycle = (time * cycleSpeed) % palette.length;
        const index = Math.floor(cycle);
        const nextIndex = (index + 1) % palette.length;
        const lerpFactor = cycle - index;

        const color = palette[index].clone().lerp(palette[nextIndex], lerpFactor);
        material.uniforms.boxColor.value.copy(color);
    }

    /**
     * Creates a grid of colored boxes
     */
    static createBoxGrid(
        scene: THREE.Scene,
        gridSize: number = 3,
        spacing: number = 2,
        randomColors: boolean = true
    ): THREE.Mesh[] {
        const boxes: THREE.Mesh[] = [];
        const offset = (gridSize - 1) * spacing * 0.5;

        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                for (let z = 0; z < gridSize; z++) {
                    const color = randomColors
                        ? new THREE.Color().setHSL(Math.random(), 0.7, 0.5)
                        : new THREE.Color(0xff0000);

                    const box = this.createSimpleBox(
                        color.getHex(),
                        1,
                        [
                            x * spacing - offset,
                            y * spacing - offset,
                            z * spacing - offset
                        ]
                    );

                    scene.add(box);
                    boxes.push(box);
                }
            }
        }

        return boxes;
    }

    /**
     * Applies wave motion to an array of boxes
     */
    static applyWaveMotion(
        boxes: THREE.Mesh[],
        time: number,
        amplitude: number = 1,
        frequency: number = 1
    ): void {
        boxes.forEach((box, index) => {
            const wave = Math.sin(time * frequency + index * 0.5) * amplitude;
            box.position.y += wave * 0.01; // Small wave motion

            // Also apply wave to material if it's a shader material
            if (box.material instanceof THREE.ShaderMaterial && box.material.uniforms.intensity) {
                box.material.uniforms.intensity.value = 1 + wave * 0.3;
            }
        });
    }

    /**
     * Disposes of shader materials properly
     */
    static disposeShaderMaterials(materials: THREE.ShaderMaterial[]): void {
        materials.forEach(material => {
            material.dispose();
        });
    }
}

// Predefined color palettes
export const ColorPalettes = {
    sunset: [
        new THREE.Color(0xff6b35),
        new THREE.Color(0xf7931e),
        new THREE.Color(0xffd23f),
        new THREE.Color(0xff4757)
    ],

    ocean: [
        new THREE.Color(0x003f5c),
        new THREE.Color(0x2f4b7c),
        new THREE.Color(0x665191),
        new THREE.Color(0xa05195)
    ],

    forest: [
        new THREE.Color(0x2d5016),
        new THREE.Color(0x3e7b27),
        new THREE.Color(0x4f9d2b),
        new THREE.Color(0x6bb93f)
    ],

    neon: [
        new THREE.Color(0xff0080),
        new THREE.Color(0x8000ff),
        new THREE.Color(0x0080ff),
        new THREE.Color(0x00ff80)
    ],

    fire: [
        new THREE.Color(0xff4500),
        new THREE.Color(0xff6600),
        new THREE.Color(0xff8800),
        new THREE.Color(0xffaa00)
    ]
};

export default ShaderUtils;
