// Modular glyphs and hologram embodiment update
import React, { useRef, useState } from 'react';
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber';
import * as THREE from 'three';
import { OrbitControls, Stats } from '@react-three/drei';

const glyphPresets = {
    Cross: [
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
    ],
    Square: [
        [1, 1],
        [1, 1],
    ],
    Star: [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
};

const behaviors = {
    Orbit: (mesh, t) => {
        mesh.position.x = Math.cos(t + mesh.userData.offset) * 2;
        mesh.position.z = Math.sin(t + mesh.userData.offset) * 2;
    },
    Pulse: (mesh, t) => {
        const scale = 1 + 0.2 * Math.sin(t * 3);
        mesh.scale.set(scale, scale, scale);
    },
    Gravity: (mesh, t) => {
        mesh.position.y -= 0.01;
        if (mesh.position.y < -2) mesh.position.y = 2;
    },
};

function PrefabCube({ prefab, index, onClick }) {
    const mesh = useRef();
    const flash = useRef(1);
    const animStart = useRef(performance.now());

    useFrame(({ clock }) => {
        if (!mesh.current) return;

        const elapsed = performance.now() - animStart.current;
        const animProgress = Math.min(elapsed / 500, 1);
        const targetScale = prefab.size || 1;
        mesh.current.scale.setScalar(targetScale * animProgress);
        mesh.current.material.opacity = animProgress;
        mesh.current.material.transparent = true;

        if (flash.current > 0) {
            flash.current -= 0.05;
            mesh.current.material.emissive.setRGB(flash.current, flash.current, flash.current);
        }

        if (prefab.behaviors.length > 0 && !prefab.stopped) {
            const t = clock.getElapsedTime();
            prefab.behaviors.forEach((b) => behaviors[b]?.(mesh.current, t));
        }
    });

    return (
        <mesh
            ref={mesh}
            position={prefab.position}
            userData={{ offset: Math.random() * 10 }}
            castShadow
            onClick={(e) => {
                e.stopPropagation();
                onClick(index);
            }}
        >
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial
                color={prefab.selected ? 'orange' : 'skyblue'}
                emissive={new THREE.Color('white')}
                emissiveIntensity={0.5}
                transparent
                opacity={0}
            />
        </mesh>
    );
}

function SceneContent({ prefabs, onCubeClick }) {
    return (
        <>
            <ambientLight intensity={0.4} />
            <pointLight position={[5, 10, 5]} intensity={1} castShadow />
            <OrbitControls />
            <gridHelper args={[20, 20]} />
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
                <planeGeometry args={[100, 100]} />
                <shadowMaterial opacity={0.2} />
            </mesh>
            <mesh position={[0, 0.7, 0]} castShadow>
                <cylinderGeometry args={[1.5, 1.5, 1.2, 32]} />
                <meshStandardMaterial color="#555" />
            </mesh>
            {prefabs.map((p, i) => (
                <PrefabCube key={i} index={i} prefab={p} onClick={onCubeClick} />
            ))}
            <Stats />
        </>
    );
}

function Scene({ prefabs, onClick, onCubeClick }) {
    return (
        <Canvas shadows camera={{ position: [10, 10, 10], fov: 60 }} onClick={onClick}>
            <SceneContent prefabs={prefabs} onCubeClick={onCubeClick} />
        </Canvas>
    );
}

export default function NexusSceneLab() {
    const [current, setCurrent] = useState('A');
    const [prefabs, setPrefabs] = useState([]);
    const [spawnMode, setSpawnMode] = useState(false);
    const [gridSize] = useState(1);

    const handleCanvasClick = (event) => {
        if (!spawnMode) return;
        const size = 1;
        const x = Math.floor(((event.clientX / window.innerWidth - 0.5) * 20) / gridSize) * gridSize + size / 2;
        const z = Math.floor((-(event.clientY / window.innerHeight - 0.5) * 20) / gridSize) * gridSize + size / 2;
        setPrefabs([...prefabs.map(p => ({ ...p, selected: false })), { position: [x, size / 2, z], behaviors: [], stopped: true, size, selected: false }]);
    };

    const handleCubeClick = (index) => {
        setPrefabs(prefabs.map((p, i) => ({ ...p, selected: i === index })));
    };

    const updateSelectedSize = (newSize) => {
        setPrefabs(prefabs.map(p => p.selected ? {
            ...p,
            size: newSize,
            position: [
                Math.floor((p.position[0] - newSize / 2) / gridSize) * gridSize + newSize / 2,
                newSize / 2,
                Math.floor((p.position[2] - newSize / 2) / gridSize) * gridSize + newSize / 2,
            ]
        } : p));
    };

    const spawnGlyphFromMatrix = (matrix) => {
        const size = 1;
        const spacing = 1.5;
        const offsetX = -Math.floor(matrix[0].length / 2) * spacing;
        const offsetZ = -Math.floor(matrix.length / 2) * spacing;
        const newCubes = [];
        for (let z = 0; z < matrix.length; z++) {
            for (let x = 0; x < matrix[z].length; x++) {
                if (matrix[z][x]) {
                    newCubes.push({
                        position: [x * spacing + offsetX, size / 2 + 1.2, z * spacing + offsetZ],
                        behaviors: [],
                        stopped: true,
                        size,
                        selected: false,
                    });
                }
            }
        }
        setPrefabs([...prefabs, ...newCubes]);
    };

    const toggleScene = () => {
        setPrefabs([]);
        setCurrent(current === 'A' ? 'B' : 'A');
    };

    return (
        <div className="w-full h-screen">
            <div className="absolute top-4 left-4 z-10 space-y-2">
                <button className="bg-cyan-600 text-white px-4 py-2 rounded" onClick={toggleScene}>
                    Toggle Scene
                </button>
                <button
                    className={`px-4 py-2 rounded ${spawnMode ? 'bg-red-600' : 'bg-green-600'} text-white`}
                    onClick={() => setSpawnMode(!spawnMode)}
                >
                    {spawnMode ? 'Disable Spawn Mode' : 'Enable Spawn Mode'}
                </button>
                <div className="space-x-2">
                    {[1, 2, 3].map(size => (
                        <button
                            key={size}
                            className="bg-gray-800 text-white px-2 py-1 rounded"
                            onClick={() => updateSelectedSize(size)}
                        >
                            Size {size}
                        </button>
                    ))}
                </div>
                {Object.keys(glyphPresets).map(name => (
                    <button key={name} className="bg-indigo-600 text-white px-4 py-2 rounded" onClick={() => spawnGlyphFromMatrix(glyphPresets[name])}>
                        Load {name} Glyph
                    </button>
                ))}
            </div>
            <Scene prefabs={prefabs} onClick={handleCanvasClick} onCubeClick={handleCubeClick} />
        </div>
    );
}
