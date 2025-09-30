// Visual Bleedway integration for World Engine - ADVANCED silhouette processing
import * as React from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Environment, Center } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { SensoryOverlay } from '../sensory/SensoryOverlay';
import { useSensoryMoment } from '../sensory/useSensoryMoment';
import { useSharedMesh, BridgeUtils } from '../bridge/WorldEngineBridge';
import {
  cleanMask,
  buildField,
  meshFromField,
  BUILTIN_PRESETS,
  type SilhouetteProcessingOptions
} from './SilhouetteProcessing';type ImgPayload = { file: File; url: string; tex?: THREE.Texture };

type BleedwayStore = {
  front?: ImgPayload;
  side?: ImgPayload;
  mesh?: THREE.Mesh | null;
  setMesh: (m: THREE.Mesh | null) => void;
};

const bleedwayStore: BleedwayStore = {
  front: undefined,
  side: undefined,
  mesh: null,
  setMesh: (m) => (bleedwayStore.mesh = m),
};

export function VisualBleedway() {
  return (
    <div className="w-full h-screen relative bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <HeaderBar />
      <div className="grid grid-cols-12 gap-3 px-3 pb-3 pt-16 h-[calc(100vh-0px)]">
        <LeftPanel className="col-span-3" />
        <Viewport className="col-span-6" />
        <RightPanel className="col-span-3" />
      </div>
    </div>
  );
}

function HeaderBar() {
  return (
    <div className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-4 py-3 bg-black/30 backdrop-blur border-b border-white/10">
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
        <h1 className="text-lg font-semibold tracking-wide">
          Visual <span className="text-emerald-400">Bleedway</span>
        </h1>
      </div>
      <div className="text-xs opacity-70 select-none">
        World Engine Integration: Visual-first mask → mesh → sensory overlay
      </div>
    </div>
  );
}

function LeftPanel({ className = '' }: { className?: string }) {
  const [dragOverFront, setDragOverFront] = React.useState(false);
  const [dragOverSide, setDragOverSide] = React.useState(false);

  async function loadTexture(file: File) {
    const url = URL.createObjectURL(file);
    const tex = await new THREE.TextureLoader().loadAsync(url);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return { file, url, tex } as ImgPayload;
  }

  async function handleFiles(kind: 'front' | 'side', files?: FileList | null) {
    if (!files || !files[0]) return;
    const img = await loadTexture(files[0]);
    if (kind === 'front') bleedwayStore.front = img;
    if (kind === 'side') bleedwayStore.side = img;
  }

  return (
    <div className={`${className} rounded-2xl p-3 bg-white/5 border border-white/10 backdrop-blur overflow-auto`}>
      <Section title="Silhouettes">
        <DropZone
          label="Front PNG silhouette"
          hint="Drag & drop or click"
          active={dragOverFront}
          onDrag={setDragOverFront}
          onFiles={(f) => handleFiles('front', f)}
        />
        <DropZone
          label="Side PNG silhouette (optional)"
          hint="Improves volume accuracy"
          active={dragOverSide}
          onDrag={setDragOverSide}
          onFiles={(f) => handleFiles('side', f)}
        />
      </Section>

      <MeshingControls />

      <Section title="World Engine Export">
        <ExportControls />
      </Section>
    </div>
  );
}

function RightPanel({ className = '' }: { className?: string }) {
  return (
    <div className={`${className} rounded-2xl p-3 bg-white/5 border border-white/10 backdrop-blur overflow-auto`}>
      <Section title="Sensory Integration">
        <SensoryControls />
      </Section>

      <Section title="Scene Optics">
        <SceneControls />
      </Section>

      <Section title="Integration Notes">
        <ul className="text-sm opacity-80 list-disc pl-5 space-y-1">
          <li>Visual → Sensory: Mesh generates procedural moments</li>
          <li>Proximity overlays adapt to mesh geometry</li>
          <li>Export integrates with World Engine storage</li>
          <li>Front silhouette drives base volume structure</li>
        </ul>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <div className="text-emerald-300/90 font-semibold text-sm mb-2 tracking-wide">{title}</div>
      <div className="rounded-xl border border-white/10 bg-white/5 p-3">{children}</div>
    </div>
  );
}

function DropZone({ label, hint, active, onDrag, onFiles }: {
  label: string;
  hint?: string;
  active?: boolean;
  onDrag?: (b: boolean) => void;
  onFiles: (files?: FileList) => void;
}) {
  return (
    <label
      className={`block rounded-lg border border-dashed p-4 cursor-pointer transition mb-3 ${
        active ? 'border-emerald-400 bg-emerald-400/10' : 'border-white/20 hover:border-white/40'
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        onDrag?.(true);
      }}
      onDragLeave={() => onDrag?.(false)}
      onDrop={(e) => {
        e.preventDefault();
        onDrag?.(false);
        onFiles(e.dataTransfer?.files ?? undefined);
      }}
    >
      <input
        type="file"
        accept="image/png,image/webp,image/jpg,image/jpeg"
        className="hidden"
        onChange={(e) => onFiles(e.currentTarget.files ?? undefined)}
      />
      <div className="text-sm font-medium">{label}</div>
      {hint && <div className="text-xs opacity-70 mt-1">{hint}</div>}
    </label>
  );
}

function MeshingControls() {
  const [building, setBuilding] = React.useState(false);
  const [preset, setPreset] = React.useState<keyof typeof BUILTIN_PRESETS>('Auto (Otsu)');
  const [settings, setSettings] = React.useState<SilhouetteProcessingOptions>(BUILTIN_PRESETS['Auto (Otsu)']);
  const [showAdvanced, setShowAdvanced] = React.useState(false);

  // Update settings when preset changes
  React.useEffect(() => {
    setSettings({ ...BUILTIN_PRESETS[preset] });
  }, [preset]);

  async function build() {
    if (!bleedwayStore.front) {
      alert('Drop a front silhouette first.');
      return;
    }

    if (!bleedwayStore.side) {
      // Create a simple side silhouette from front if missing
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      canvas.width = bleedwayStore.front.width;
      canvas.height = bleedwayStore.front.height;
      ctx.drawImage(bleedwayStore.front, 0, 0);
      bleedwayStore.side = { file: new File([], 'side.png'), url: '', tex: undefined };
      bleedwayStore.side.url = canvas.toDataURL();
    }

    setBuilding(true);
    try {
      // Clean masks using advanced processing
      const frontMask = cleanMask(bleedwayStore.front as any, {
        auto: settings.auto,
        threshold: settings.dual ? settings.thrF : settings.thr,
        kernel: settings.kern,
        maxSide: 640,
        flipX: false
      });

      const sideMask = cleanMask(bleedwayStore.side as any, {
        auto: settings.auto,
        threshold: settings.dual ? settings.thrS : settings.thr,
        kernel: settings.kern,
        maxSide: 640,
        flipX: settings.flip
      });

      // Build 3D volume field
      const field = buildField(frontMask, sideMask, settings.res);

      // Generate mesh using advanced algorithm
      const mesh = meshFromField(field, settings.res, {
        iso: settings.iso,
        height: settings.height,
        subs: settings.subs,
        lap: settings.lap,
        dec: settings.dec,
        color: '#4ecdc4'
      });

      bleedwayStore.setMesh(mesh);
    } catch (error) {
      console.error('Mesh generation failed:', error);
      alert('Mesh generation failed. Check console for details.');
    } finally {
      setBuilding(false);
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs opacity-70 mb-2">Processing Preset</label>
        <select
          value={preset}
          onChange={(e) => setPreset(e.target.value as any)}
          className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-sm"
        >
          {Object.keys(BUILTIN_PRESETS).map(name => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      <div className="flex items-center justify-between">
        <button
          onClick={build}
          disabled={building}
          className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg text-white font-medium disabled:opacity-50 flex-1"
        >
          {building ? 'Processing...' : 'Build Mesh'}
        </button>

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="ml-2 px-3 py-2 bg-white/10 border border-white/20 rounded text-sm hover:bg-white/20 transition-colors"
        >
          {showAdvanced ? '−' : '+'}
        </button>
      </div>

      {showAdvanced && (
        <div className="space-y-3 pt-2 border-t border-white/10">
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="block opacity-70 mb-1">Resolution</label>
              <input
                type="range"
                min={64}
                max={512}
                step={32}
                value={settings.res}
                onChange={(e) => setSettings(prev => ({ ...prev, res: parseInt(e.target.value) }))}
                className="w-full accent-emerald-400"
              />
              <span className="opacity-60">{settings.res}</span>
            </div>

            <div>
              <label className="block opacity-70 mb-1">Isosurface</label>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={settings.iso}
                onChange={(e) => setSettings(prev => ({ ...prev, iso: parseFloat(e.target.value) }))}
                className="w-full accent-emerald-400"
              />
              <span className="opacity-60">{settings.iso.toFixed(2)}</span>
            </div>

            <div>
              <label className="block opacity-70 mb-1">Height</label>
              <input
                type="range"
                min={0.5}
                max={3.0}
                step={0.1}
                value={settings.height}
                onChange={(e) => setSettings(prev => ({ ...prev, height: parseFloat(e.target.value) }))}
                className="w-full accent-emerald-400"
              />
              <span className="opacity-60">{settings.height.toFixed(1)}</span>
            </div>

            <div>
              <label className="block opacity-70 mb-1">Smooth</label>
              <input
                type="range"
                min={0}
                max={5}
                step={1}
                value={settings.lap}
                onChange={(e) => setSettings(prev => ({ ...prev, lap: parseInt(e.target.value) }))}
                className="w-full accent-emerald-400"
              />
              <span className="opacity-60">{settings.lap}</span>
            </div>
          </div>

          <div className="flex items-center space-x-4 text-xs">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={settings.auto}
                onChange={(e) => setSettings(prev => ({ ...prev, auto: e.target.checked }))}
                className="mr-1 accent-emerald-400"
              />
              Auto Threshold
            </label>

            <label className="flex items-center">
              <input
                type="checkbox"
                checked={settings.dual}
                onChange={(e) => setSettings(prev => ({ ...prev, dual: e.target.checked }))}
                className="mr-1 accent-emerald-400"
              />
              Dual Threshold
            </label>

            <label className="flex items-center">
              <input
                type="checkbox"
                checked={settings.flip}
                onChange={(e) => setSettings(prev => ({ ...prev, flip: e.target.checked }))}
                className="mr-1 accent-emerald-400"
              />
              Flip Side
            </label>
          </div>
        </div>
      )}

      {bleedwayStore.mesh && (
        <div className="text-xs text-green-300 bg-green-500/10 border border-green-500/30 rounded p-2">
          ✓ Mesh generated successfully
        </div>
      )}
    </div>
  );
}

function SensoryControls() {
  const { moment, setPreset, setPerspective, availablePresets } = useSensoryMoment();

  return (
    <div className="space-y-3">
      <div>
        <label className="block text-xs opacity-70 mb-1">Sensory Preset</label>
        <select
          value={moment.id.split('-')[0]}
          onChange={(e) => setPreset(e.target.value as any)}
          className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-sm"
        >
          {availablePresets.map(preset => (
            <option key={preset} value={preset}>{preset}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-xs opacity-70 mb-1">Perspective</label>
        <select
          value={moment.perspective}
          onChange={(e) => setPerspective(e.target.value as any)}
          className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-sm"
        >
          <option value="attuned">Attuned</option>
          <option value="oblivious">Oblivious</option>
          <option value="object">Object</option>
        </select>
      </div>

      <div className="text-xs opacity-70">
        Current: {moment.label}
      </div>
    </div>
  );
}

function ExportControls() {
  const [settings, setSettings] = React.useState({
    targetMaxDim: 1.0,
    center: true,
    rotateY: 0
  });

  async function exportGLB() {
    if (!bleedwayStore.mesh) {
      alert('Build a mesh first');
      return;
    }

    const prepared = normalizeGeometry(bleedwayStore.mesh, settings);
    const blob = await exportObjectToGLB(prepared);
    downloadBlob(blob, 'visual-bleedway-export.glb');
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-xs opacity-70 mb-1">Size</label>
          <input
            type="range"
            min={0.1}
            max={3}
            step={0.1}
            value={settings.targetMaxDim}
            onChange={(e) => setSettings(prev => ({ ...prev, targetMaxDim: parseFloat(e.target.value) }))}
            className="w-full accent-emerald-400"
          />
        </div>
        <div>
          <label className="block text-xs opacity-70 mb-1">Rotation</label>
          <input
            type="range"
            min={-180}
            max={180}
            step={15}
            value={settings.rotateY}
            onChange={(e) => setSettings(prev => ({ ...prev, rotateY: parseInt(e.target.value) }))}
            className="w-full accent-emerald-400"
          />
        </div>
      </div>

      <div className="flex gap-2">
        <button
          onClick={exportGLB}
          className="px-3 py-2 bg-white/10 border border-white/20 rounded text-sm hover:bg-white/20 transition-colors flex-1"
        >
          Export GLB
        </button>
        <button
          onClick={() => bleedwayStore.setMesh(null)}
          className="px-3 py-2 bg-red-500/20 border border-red-500/30 rounded text-sm hover:bg-red-500/30 transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
  );
}

function SceneControls() {
  const [settings, setSettings] = React.useState({
    bloom: true,
    bloomIntensity: 0.9,
    grid: false,
    sensoryOverlay: true
  });

  // Bridge to viewport
  React.useEffect(() => {
    (window as any).__VB__ = settings;
  }, [settings]);

  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={settings.bloom}
          onChange={(e) => setSettings(prev => ({ ...prev, bloom: e.target.checked }))}
          className="accent-emerald-400"
        />
        Bloom Effect
      </label>

      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={settings.sensoryOverlay}
          onChange={(e) => setSettings(prev => ({ ...prev, sensoryOverlay: e.target.checked }))}
          className="accent-emerald-400"
        />
        Sensory Overlay
      </label>

      {settings.bloom && (
        <div>
          <label className="block text-xs opacity-70 mb-1">Bloom Intensity</label>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={settings.bloomIntensity}
            onChange={(e) => setSettings(prev => ({ ...prev, bloomIntensity: parseFloat(e.target.value) }))}
            className="w-full accent-emerald-400"
          />
        </div>
      )}
    </div>
  );
}

function Viewport({ className = '' }: { className?: string }) {
  return (
    <div className={`${className} relative rounded-2xl border border-white/10 overflow-hidden`}>
      <Canvas
        shadows
        camera={{ position: [2.2, 1.2, 2.2], fov: 45 }}
        className="absolute inset-0"
      >
        <color attach="background" args={['#0f1419']} />
        <hemisphereLight intensity={0.4} groundColor="#1a2030" />
        <directionalLight
          castShadow
          intensity={1.0}
          position={[3, 4, 2]}
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <SceneContent />
        <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
        <EffectSwitch />
      </Canvas>
    </div>
  );
}

function EffectSwitch() {
  const settings = (window as any).__VB__ ?? { bloom: true, bloomIntensity: 0.9 };

  return (
    <>
      {settings.bloom && (
        <EffectComposer>
          <Bloom intensity={settings.bloomIntensity} mipmapBlur luminanceThreshold={0.2} />
        </EffectComposer>
      )}
    </>
  );
}

function SceneContent() {
  const meshRef = React.useRef<THREE.Mesh>(null);
  const [proxy, setProxy] = React.useState<THREE.Mesh | null>(null);
  const { moment } = useSensoryMoment();
  const settings = (window as any).__VB__ ?? { grid: false, sensoryOverlay: true };

  React.useEffect(() => {
    const interval = setInterval(() => {
      if (bleedwayStore.mesh !== meshRef.current) {
        setProxy(bleedwayStore.mesh ?? null);
      }
    }, 200);
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <Environment preset="city" />
      {settings.grid && <gridHelper args={[10, 10]} />}

      <Center top>
        {proxy ? (
          <>
            <primitive ref={meshRef} object={proxy} />
            {settings.sensoryOverlay && (
              <SensoryOverlay
                moment={moment}
                attachTo={meshRef.current ?? undefined}
                visible={settings.sensoryOverlay}
              />
            )}
          </>
        ) : (
          <PlaceholderGhost />
        )}
      </Center>
    </>
  );
}

function PlaceholderGhost() {
  const ref = React.useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime();
    ref.current.position.y = Math.sin(t * 1.2) * 0.05;
  });

  return (
    <group ref={ref}>
      <mesh castShadow receiveShadow>
        <capsuleGeometry args={[0.4, 0.8, 4, 8]} />
        <meshStandardMaterial color="#4ecdc4" roughness={0.35} metalness={0.1} />
      </mesh>
    </group>
  );
}

function downloadBlob(blob: Blob, filename: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
