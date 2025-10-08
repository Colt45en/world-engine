import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { ColoredBoxShader, ColoredBoxPresets } from '../shaders/ColoredBoxShader';

/**
 * Colored Box Demo Component
 * Demonstrates various colored box shader effects
 */
export const ColoredBoxDemo: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const animationIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 5);

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    rendererRef.current = renderer;
    mountRef.current.appendChild(renderer.domElement);

    // Create different colored boxes with various effects
    const boxes: Array<{ mesh: THREE.Mesh; material: THREE.ShaderMaterial }> = [];

    // 1. Solid red box
    const solidBox = ColoredBoxShader.createColoredBox(
      [1, 1, 1],
      new THREE.Color(0xff0000),
      ColoredBoxPresets.solid(new THREE.Color(0xff0000))
    );
    solidBox.position.set(-3, 1, 0);
    scene.add(solidBox);
    boxes.push({ mesh: solidBox, material: solidBox.material as THREE.ShaderMaterial });

    // 2. Gradient box
    const gradientBox = ColoredBoxShader.createColoredBox(
      [1, 1, 1],
      new THREE.Color(0x00ff00),
      ColoredBoxPresets.gradient(
        new THREE.Color(0x00ff00),
        new THREE.Color(0x0000ff)
      )
    );
    gradientBox.position.set(-1, 1, 0);
    scene.add(gradientBox);
    boxes.push({ mesh: gradientBox, material: gradientBox.material as THREE.ShaderMaterial });

    // 3. Animated color box
    const animatedBox = ColoredBoxShader.createColoredBox(
      [1, 1, 1],
      new THREE.Color(0xff00ff),
      ColoredBoxPresets.animated(new THREE.Color(0xff00ff))
    );
    animatedBox.position.set(1, 1, 0);
    scene.add(animatedBox);
    boxes.push({ mesh: animatedBox, material: animatedBox.material as THREE.ShaderMaterial });

    // 4. Noise pattern box
    const noiseBox = ColoredBoxShader.createColoredBox(
      [1, 1, 1],
      new THREE.Color(0xffff00),
      ColoredBoxPresets.noise(
        new THREE.Color(0xffff00),
        new THREE.Color(0xff0000)
      )
    );
    noiseBox.position.set(3, 1, 0);
    scene.add(noiseBox);
    boxes.push({ mesh: noiseBox, material: noiseBox.material as THREE.ShaderMaterial });

    // 5. Custom pattern box
    const customMaterial = ColoredBoxShader.createMaterial({
      boxColor: { value: new THREE.Color(0x00ffff) },
      gradientColor: { value: new THREE.Color(0xff8800) },
      gradient: { value: true },
      animate: { value: true },
      pattern: { value: 1.5 },
      intensity: { value: 1.3 }
    });
    const customBox = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      customMaterial
    );
    customBox.position.set(0, -1, 0);
    scene.add(customBox);
    boxes.push({ mesh: customBox, material: customMaterial });

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Animation loop
    const clock = new THREE.Clock();

    const animate = () => {
      const deltaTime = clock.getDelta();

      // Update shader time uniforms
      boxes.forEach(({ mesh, material }) => {
        ColoredBoxShader.updateTime(material, deltaTime);

        // Rotate boxes
        mesh.rotation.x += deltaTime * 0.5;
        mesh.rotation.y += deltaTime * 0.3;
      });

      renderer.render(scene, camera);
      animationIdRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    // Interactive color changing
    const handleClick = (event: MouseEvent) => {
      const randomColor = new THREE.Color().setHSL(
        Math.random(),
        0.7,
        0.5
      );

      // Change color of a random box
      const randomBox = boxes[Math.floor(Math.random() * boxes.length)];
      ColoredBoxShader.setColor(randomBox.material, randomColor);
    };

    window.addEventListener('click', handleClick);

    // Cleanup
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }

      window.removeEventListener('resize', handleResize);
      window.removeEventListener('click', handleClick);

      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }

      // Dispose of resources
      boxes.forEach(({ mesh, material }) => {
        mesh.geometry.dispose();
        material.dispose();
      });

      renderer.dispose();
    };
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* Instructions overlay */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        color: 'white',
        background: 'rgba(0, 0, 0, 0.7)',
        padding: '15px',
        borderRadius: '8px',
        fontFamily: 'monospace',
        fontSize: '14px',
        zIndex: 1000
      }}>
        <h3 style={{ margin: '0 0 10px 0' }}>🎨 Colored Box Shader Demo</h3>
        <div>• Solid Color (Red)</div>
        <div>• Gradient Effect (Green → Blue)</div>
        <div>• Animated Colors (Purple)</div>
        <div>• Noise Pattern (Yellow → Red)</div>
        <div>• Custom Pattern (Cyan → Orange)</div>
        <div style={{ marginTop: '10px', fontStyle: 'italic' }}>
          Click anywhere to change colors randomly!
        </div>
      </div>
    </div>
  );
};

export default ColoredBoxDemo;