// Advanced silhouette processing algorithms from C++ game integration
import * as THREE from 'three';

export interface SilhouetteProcessingOptions {
    auto: boolean;
    dual: boolean;
    thr: number;
    thrF: number; // Front threshold for dual mode
    thrS: number; // Side threshold for dual mode
    kern: number; // Morphological kernel size
    res: number;  // Field resolution (N×N×N)
    iso: number;  // Isosurface value
    height: number; // Output mesh height
    subs: number; // Subdivision iterations
    lap: number;  // Laplacian smoothing iterations
    dec: number;  // Decimation percentage
    flip: boolean; // Flip side silhouette
}

export const BUILTIN_PRESETS: Record<string, SilhouetteProcessingOptions> = {
    'Auto (Otsu)': {
        auto: true, dual: false, thr: 190, thrF: 90, thrS: 103,
        kern: 2, res: 128, iso: 0.50, height: 1.70, subs: 1,
        lap: 0, dec: 0, flip: false
    },
    'Baked: F90 / S103': {
        auto: false, dual: true, thr: 190, thrF: 90, thrS: 103,
        kern: 2, res: 128, iso: 0.50, height: 1.70, subs: 1,
        lap: 0, dec: 0, flip: false
    },
    'High Detail': {
        auto: true, dual: false, thr: 180, thrF: 80, thrS: 95,
        kern: 1, res: 256, iso: 0.45, height: 1.80, subs: 2,
        lap: 1, dec: 0, flip: false
    },
    'Performance': {
        auto: true, dual: false, thr: 200, thrF: 100, thrS: 110,
        kern: 3, res: 64, iso: 0.55, height: 1.60, subs: 0,
        lap: 0, dec: 15, flip: false
    }
};

/**
 * Otsu's method for automatic threshold detection
 */
export function otsu(imageData: ImageData): number {
    const data = imageData.data;
    const hist = new Array(256).fill(0);

    // Build histogram
    for (let i = 0; i < data.length; i += 4) {
        const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
        hist[gray]++;
    }

    const total = imageData.width * imageData.height;
    let sum = 0;
    for (let i = 0; i < 256; i++) {
        sum += i * hist[i];
    }

    let sumB = 0, wB = 0, wF = 0, mB = 0, mF = 0;
    let varMax = 0, threshold = 0;

    for (let i = 0; i < 256; i++) {
        wB += hist[i];
        if (wB === 0) continue;

        wF = total - wB;
        if (wF === 0) break;

        sumB += i * hist[i];
        mB = sumB / wB;
        mF = (sum - sumB) / wF;

        const varBetween = wB * wF * (mB - mF) * (mB - mF);

        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = i;
        }
    }

    return threshold;
}

/**
 * Morphological operations (erosion/dilation)
 */
export function morph(imageData: ImageData, kernelSize: number, expand: boolean): ImageData {
    if (kernelSize <= 0) return imageData;

    const { width, height, data } = imageData;
    const result = new ImageData(width, height);
    const newData = result.data;

    const radius = Math.floor(kernelSize / 2);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;

            let extremeValue = expand ? 0 : 255;

            // Sample kernel area
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    const nx = Math.max(0, Math.min(width - 1, x + kx));
                    const ny = Math.max(0, Math.min(height - 1, y + ky));
                    const nIdx = (ny * width + nx) * 4;

                    const alpha = data[nIdx + 3];

                    if (expand) {
                        extremeValue = Math.max(extremeValue, alpha);
                    } else {
                        extremeValue = Math.min(extremeValue, alpha);
                    }
                }
            }

            newData[idx] = data[idx];
            newData[idx + 1] = data[idx + 1];
            newData[idx + 2] = data[idx + 2];
            newData[idx + 3] = extremeValue;
        }
    }

    return result;
}

/**
 * Calculate bounding box of non-transparent pixels
 */
export function bbox(imageData: ImageData): { x: number, y: number, w: number, h: number } {
    const { width, height, data } = imageData;

    let minX = width, minY = height, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const alpha = data[(y * width + x) * 4 + 3];
            if (alpha > 0) {
                if (!found) found = true;
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }

    return found
        ? { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 }
        : { x: 0, y: 0, w: 0, h: 0 };
}

/**
 * Clean and process silhouette mask
 */
export function cleanMask(canvas: HTMLCanvasElement, options: {
    auto?: boolean;
    threshold?: number;
    kernel?: number;
    maxSide?: number;
    flipX?: boolean;
}): HTMLCanvasElement {
    const { auto = true, threshold = 190, kernel = 2, maxSide = 640, flipX = false } = options;

    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Apply threshold
    const finalThreshold = auto ? otsu(imageData) : threshold;

    for (let i = 0; i < data.length; i += 4) {
        const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
        const alpha = gray > finalThreshold ? 255 : 0;
        data[i + 3] = alpha;
    }

    // Apply morphological operations
    let processedData = imageData;
    if (kernel > 0) {
        processedData = morph(processedData, kernel, true);  // Dilation
        processedData = morph(processedData, kernel, false); // Erosion
    }

    // Calculate bounding box with margin
    const rect = bbox(processedData);
    const margin = Math.round(Math.max(rect.w, rect.h) * 0.04);

    rect.x = Math.max(0, rect.x - margin);
    rect.y = Math.max(0, rect.y - margin);
    rect.w = Math.min(canvas.width - rect.x, rect.w + margin * 2);
    rect.h = Math.min(canvas.height - rect.y, rect.h + margin * 2);

    // Create temporary canvas with processed data
    const tmpCanvas = document.createElement('canvas');
    const tmpCtx = tmpCanvas.getContext('2d')!;
    tmpCanvas.width = canvas.width;
    tmpCanvas.height = canvas.height;
    tmpCtx.putImageData(processedData, 0, 0);

    // Create output canvas with proper scaling
    const outputCanvas = document.createElement('canvas');
    const outputCtx = outputCanvas.getContext('2d')!;
    const scale = maxSide / Math.max(rect.w, rect.h);
    outputCanvas.width = Math.round(rect.w * scale);
    outputCanvas.height = Math.round(rect.h * scale);

    if (flipX) {
        outputCtx.translate(outputCanvas.width, 0);
        outputCtx.scale(-1, 1);
    }

    outputCtx.imageSmoothingEnabled = false;
    outputCtx.drawImage(
        tmpCanvas,
        rect.x, rect.y, rect.w, rect.h,
        0, 0, outputCanvas.width, outputCanvas.height
    );

    return outputCanvas;
}

/**
 * Build 3D volume field from front and side silhouettes
 */
export function buildField(frontMask: HTMLCanvasElement, sideMask: HTMLCanvasElement, N: number): Float32Array {
    const fctx = frontMask.getContext('2d')!;
    const sctx = sideMask.getContext('2d')!;

    const fw = frontMask.width, fh = frontMask.height;
    const sw = sideMask.width, sh = sideMask.height;

    const frontData = fctx.getImageData(0, 0, fw, fh).data;
    const sideData = sctx.getImageData(0, 0, sw, sh).data;

    const field = new Float32Array(N * N * N);
    let i = 0;

    for (let z = 0; z < N; z++) {
        const uz = z / (N - 1);
        for (let y = 0; y < N; y++) {
            const v = 1 - y / (N - 1);
            for (let x = 0; x < N; x++, i++) {
                const u = x / (N - 1);

                // Sample front silhouette
                const fx = Math.min(fw - 1, Math.floor(u * fw));
                const fy = Math.min(fh - 1, Math.floor(v * fh));
                const frontAlpha = frontData[(fy * fw + fx) * 4 + 3] / 255;

                // Sample side silhouette
                const sx = Math.min(sw - 1, Math.floor(uz * sw));
                const sy = Math.min(sh - 1, Math.floor(v * sh));
                const sideAlpha = sideData[(sy * sw + sx) * 4 + 3] / 255;

                // Intersection of both silhouettes
                field[i] = Math.min(frontAlpha, sideAlpha);
            }
        }
    }

    return field;
}

/**
 * Laplacian smoothing for geometry
 */
export function laplacianSmoothGeometry(geometry: THREE.BufferGeometry, iterations: number = 1): THREE.BufferGeometry {
    const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    const positions = positionAttr.array as Float32Array;
    const indexAttr = geometry.getIndex();

    if (!indexAttr) return geometry;

    const nVerts = positionAttr.count;
    const indices = indexAttr.array;

    // Build adjacency list
    const adjacency: number[][] = Array.from({ length: nVerts }, () => []);

    for (let i = 0; i < indices.length; i += 3) {
        const a = indices[i], b = indices[i + 1], c = indices[i + 2];
        adjacency[a].push(b, c);
        adjacency[b].push(a, c);
        adjacency[c].push(a, b);
    }

    // Remove duplicates and self-references
    for (let i = 0; i < nVerts; i++) {
        adjacency[i] = [...new Set(adjacency[i])].filter(j => j !== i);
    }

    // Apply smoothing iterations
    for (let iter = 0; iter < iterations; iter++) {
        const newPositions = new Float32Array(positions.length);

        for (let i = 0; i < nVerts; i++) {
            const neighbors = adjacency[i];
            if (neighbors.length === 0) {
                // No neighbors, keep original position
                newPositions[i * 3] = positions[i * 3];
                newPositions[i * 3 + 1] = positions[i * 3 + 1];
                newPositions[i * 3 + 2] = positions[i * 3 + 2];
                continue;
            }

            // Calculate average neighbor position
            let sx = 0, sy = 0, sz = 0;
            for (const j of neighbors) {
                sx += positions[j * 3];
                sy += positions[j * 3 + 1];
                sz += positions[j * 3 + 2];
            }

            const inv = 1 / neighbors.length;
            // Blend 50% original position with 50% average neighbor position
            newPositions[i * 3] = positions[i * 3] * 0.5 + sx * inv * 0.5;
            newPositions[i * 3 + 1] = positions[i * 3 + 1] * 0.5 + sy * inv * 0.5;
            newPositions[i * 3 + 2] = positions[i * 3 + 2] * 0.5 + sz * inv * 0.5;
        }

        positions.set(newPositions);
    }

    positionAttr.needsUpdate = true;
    geometry.computeVertexNormals();

    return geometry;
}

/**
 * Generate mesh from volume field using marching cubes
 * Note: This requires a marching cubes implementation to be loaded
 */
export function meshFromField(field: Float32Array, N: number, options: {
    iso?: number;
    height?: number;
    subs?: number;
    lap?: number;
    dec?: number;
    wireframe?: boolean;
    color?: string;
}): THREE.Mesh {
    const {
        iso = 0.5,
        height = 1.7,
        subs = 1,
        lap = 0,
        dec = 0,
        wireframe = false,
        color = '#4ecdc4'
    } = options;

    // This would use a proper marching cubes implementation
    // For now, create a placeholder that represents the volume structure
    const geometry = createVolumeGeometry(field, N, iso);

    // Scale to desired height
    const scale = 0.5 * height;
    geometry.scale(scale, scale, scale);
    geometry.translate(0, 0.5 * height, 0);

    // Apply post-processing
    if (lap > 0) {
        laplacianSmoothGeometry(geometry, Math.min(3, Math.floor(lap)));
    }

    // Subdivision and decimation would be applied here in full implementation

    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();

    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(color),
        roughness: 0.55,
        metalness: 0.15,
        wireframe
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'SilhouetteMesh';
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    return mesh;
}

/**
 * Create approximate geometry from volume field
 * This is a simplified version - real implementation would use marching cubes
 */
function createVolumeGeometry(field: Float32Array, N: number, isoValue: number): THREE.BufferGeometry {
    // For demo purposes, create a geometry that approximates the volume
    // In production, this would use the marching cubes algorithm

    const geometry = new THREE.SphereGeometry(0.5, 32, 16);

    // Sample the field to determine approximate shape characteristics
    let maxDensity = 0;
    let centerOfMass = new THREE.Vector3();
    let totalMass = 0;

    for (let z = 0; z < N; z++) {
        for (let y = 0; y < N; y++) {
            for (let x = 0; x < N; x++) {
                const idx = z * N * N + y * N + x;
                const value = field[idx];

                if (value > isoValue) {
                    maxDensity = Math.max(maxDensity, value);
                    const worldPos = new THREE.Vector3(
                        (x / (N - 1) - 0.5),
                        (y / (N - 1) - 0.5),
                        (z / (N - 1) - 0.5)
                    );
                    centerOfMass.add(worldPos.multiplyScalar(value));
                    totalMass += value;
                }
            }
        }
    }

    if (totalMass > 0) {
        centerOfMass.divideScalar(totalMass);

        // Deform sphere based on center of mass and density distribution
        const positions = geometry.getAttribute('position') as THREE.BufferAttribute;
        const posArray = positions.array as Float32Array;

        for (let i = 0; i < positions.count; i++) {
            const x = posArray[i * 3];
            const y = posArray[i * 3 + 1];
            const z = posArray[i * 3 + 2];

            // Apply subtle deformation based on center of mass
            const offset = centerOfMass.clone().multiplyScalar(0.3);
            posArray[i * 3] += offset.x;
            posArray[i * 3 + 1] += offset.y;
            posArray[i * 3 + 2] += offset.z;
        }

        positions.needsUpdate = true;
    }

    return geometry;
}
