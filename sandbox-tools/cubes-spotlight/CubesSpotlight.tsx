// R3F Conversion of the C++ GLFW/GLEW/ImGui cube scene
// Tech: React, @react-three/fiber, @react-three/drei, zustand, shadcn/ui (for simple UI)
// Features parity:
// - Add/Delete cubes
// - Select cube
// - Toggle spotlight; drag to reposition on ground with mouse (hold while moving)
// - Simple bob animation per-cube (offset)
// - Scene control panel (replaces ImGui)

import React, { useMemo, useRef, useState, useEffect, Suspense } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import { create } from "zustand";

// Fallback UI components for standalone usage (shadcn/ui compatible)
const Button = ({ children, onClick, size = "default", variant = "default", disabled = false, ...props }) => {
  const sizeClasses = {
    sm: "px-3 py-1.5 text-sm",
    default: "px-4 py-2",
    lg: "px-8 py-3 text-lg"
  };

  const variantClasses = {
    default: "bg-blue-600 hover:bg-blue-700 text-white",
    secondary: "bg-gray-600 hover:bg-gray-700 text-white",
    destructive: "bg-red-600 hover:bg-red-700 text-white"
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        rounded-md font-medium transition-colors
        ${sizeClasses[size]}
        ${variantClasses[variant]}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      `}
      {...props}
    >
      {children}
    </button>
  );
};

const Card = ({ children, className = "", ...props }) => (
  <div className={`rounded-lg border shadow-sm ${className}`} {...props}>
    {children}
  </div>
);

const CardHeader = ({ children, className = "", ...props }) => (
  <div className={`flex flex-col space-y-1.5 p-6 ${className}`} {...props}>
    {children}
  </div>
);

const CardTitle = ({ children, className = "", ...props }) => (
  <h3 className={`text-lg font-semibold leading-none tracking-tight ${className}`} {...props}>
    {children}
  </h3>
);

const CardContent = ({ children, className = "", ...props }) => (
  <div className={`p-6 pt-0 ${className}`} {...props}>
    {children}
  </div>
);

const Slider = ({ value, min, max, step, onValueChange, className = "", ...props }) => {
  const handleChange = (e) => {
    onValueChange([parseFloat(e.target.value)]);
  };

  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value[0]}
      onChange={handleChange}
      className={`w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider ${className}`}
      {...props}
    />
  );
};

const Select = ({ children, value, onValueChange, ...props }) => {
  const handleChange = (e) => {
    onValueChange(e.target.value);
  };

  return (
    <select
      value={value}
      onChange={handleChange}
      className="w-full p-2 bg-gray-800 border border-gray-700 rounded-md text-white"
      {...props}
    >
      {children}
    </select>
  );
};

const SelectTrigger = ({ children, className = "", ...props }) => (
  <div className={className} {...props}>{children}</div>
);

const SelectValue = ({ placeholder }) => null; // Handled by select element

const SelectContent = ({ children }) => children;

const SelectItem = ({ children, value, ...props }) => (
  <option value={value} {...props}>
    {children}
  </option>
);

// ────────────────────────────────────────────────────────────────────────────────
// State
const useStore = create((set, get) => ({
  cubes: [
    { id: 1, position: [-1, 0, -1], offset: 0, color: "#b3c0ff" },
    { id: 2, position: [ 1, 0, -1], offset: 1, color: "#b3ffe1" },
  ],
  selected: -1,
  animationSpeed: 1,
  spotlight: { enabled: false, position: [0, 1, 0] },
  // actions
  addCube: () => set(s => ({ cubes: [...s.cubes, { id: Date.now(), position: [0,0,0], offset: s.cubes.length, color: `hsl(${(Math.random()*360)|0},70%,70%)` }] })),
  deleteSelected: () => set(s => ({ cubes: s.selected === -1 ? s.cubes : s.cubes.filter((_, i) => i !== s.selected), selected: -1 })),
  setSelected: (i) => set({ selected: i }),
  setAnimationSpeed: (v) => set({ animationSpeed: v }),
  toggleSpot: () => set(s => ({ spotlight: { ...s.spotlight, enabled: !s.spotlight.enabled } })),
  setSpotPos: (p) => set(s => ({ spotlight: { ...s.spotlight, position: p } })),
  // programmatic, pointerless setters
  setSpotPosDirect: (x, y, z) => set(s => ({ spotlight: { ...s.spotlight, position: [x, y, z] } })),
  nudgeSpot: (dx, dz) => set(s => ({ spotlight: { ...s.spotlight, position: [s.spotlight.position[0] + dx, s.spotlight.position[1], s.spotlight.position[2] + dz] } })),

  moveCube: (i, pos) => set(s => ({ cubes: s.cubes.map((c, idx) => idx===i ? { ...c, position: pos } : c) })),
}));

// ────────────────────────────────────────────────────────────────────────────────
// Helpers
function useGroundPointer(onPoint) {
  const { camera, gl } = useThree();
  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), []); // y=0
  const ndc = new THREE.Vector2();

  return (event) => {
    if (!event) return; // defensive

    // Prefer R3F's event.ray when available (most reliable)
    let ray = event.ray;

    if (!ray) {
      // Fallback to building a ray from client coords
      const rect = gl.domElement.getBoundingClientRect();
      const cx = (event.clientX ?? event?.nativeEvent?.clientX);
      const cy = (event.clientY ?? event?.nativeEvent?.clientY);
      if (typeof cx !== 'number' || typeof cy !== 'number') return; // nothing we can do
      ndc.x = ((cx - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((cy - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);
      ray = raycaster.ray;
    }

    const hit = new THREE.Vector3();
    ray.intersectPlane(plane, hit);
    if (Number.isFinite(hit.x) && Number.isFinite(hit.y) && Number.isFinite(hit.z)) {
      onPoint([hit.x, 0, hit.z]);
    }
  };
}

function Cube({ index, position, offset, color }) {
  const ref = useRef();
  const speed = useStore(s => s.animationSpeed);
  const selected = useStore(s => s.selected);
  const setSelected = useStore(s => s.setSelected);

  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.getElapsedTime();
    const bob = Math.sin(t + offset) * 0.1 * speed;
    ref.current.position.set(position[0], position[1] + bob, position[2]);
    ref.current.rotation.x += 0.01 * 0.3 * speed;
    ref.current.rotation.y += 0.01 * 0.5 * speed;
  });

  return (
    <group>
      <mesh
        ref={ref}
        onPointerDown={(e) => { e.stopPropagation(); setSelected(index); }}
        castShadow
      >
        <boxGeometry args={[1,1,1]} />
        <meshStandardMaterial color={selected===index? "#66ccff" : color} />
      </mesh>
      <Text position={[position[0], -0.8, position[2]]} fontSize={0.18} color="#ffffff" anchorX="center" anchorY="middle">
        {`Cube ${index}`}
      </Text>
    </group>
  );
}

function SpotlightMarker() {
  const { enabled, position } = useStore(s => s.spotlight);
  if (!enabled) return null;
  return (
    <mesh position={position} castShadow>
      <sphereGeometry args={[0.12, 16, 16]} />
      <meshStandardMaterial emissive="#ffff99" color="#ffff99" />
    </mesh>
  );
}

function Scene() {
  const cubes = useStore(s => s.cubes);
  const spotlight = useStore(s => s.spotlight);
  const setSpotPos = useStore(s => s.setSpotPos);
  const handlePointer = useGroundPointer((p) => setSpotPos(p));
  const [draggingSpot, setDraggingSpot] = useState(false);

  return (
    <>
      <color attach="background" args={["#0b0e12"]} />
      <hemisphereLight args={[0xffffff, 0x223344, 0.4]} />
      <directionalLight position={[5,6,5]} intensity={0.8} castShadow />

      <gridHelper args={[20, 20, "#444", "#222"]} position={[0,-0.001,0]} />
      <mesh rotation={[-Math.PI/2,0,0]} receiveShadow>
        <planeGeometry args={[200,200]} />
        <meshStandardMaterial color="#111" />
      </mesh>

      {spotlight.enabled && (
        <pointLight position={spotlight.position} intensity={1.2} distance={30} decay={2} color="#fff6cc" castShadow />
      )}
      <SpotlightMarker />

      {cubes.map((c, i) => (
        <Cube key={c.id} index={i} position={c.position} offset={c.offset} color={c.color} />
      ))}

      {/* drag spotlight across ground when pressed */}
      <mesh
        rotation={[-Math.PI/2,0,0]}
        onPointerDown={(e)=>{ if (spotlight.enabled){ setDraggingSpot(true); handlePointer(e);} }}
        onPointerMove={(e)=>{ if (spotlight.enabled && draggingSpot){ handlePointer(e);} }}
        onPointerUp={()=> setDraggingSpot(false)}
      >
        <planeGeometry args={[200,200]} />
        <meshBasicMaterial visible={false} />
      </mesh>

      <OrbitControls enableDamping dampingFactor={0.08} />
    </>
  );
}

// ────────────────────────────────────────────────────────────────────────────────
// UI
function Panel() {
  const cubes = useStore(s => s.cubes);
  const selected = useStore(s => s.selected);
  const addCube = useStore(s => s.addCube);
  const deleteSelected = useStore(s => s.deleteSelected);
  const animationSpeed = useStore(s => s.animationSpeed);
  const setAnimationSpeed = useStore(s => s.setAnimationSpeed);
  const spotlight = useStore(s => s.spotlight);
  const toggleSpot = useStore(s => s.toggleSpot);
  const nudgeSpot = useStore(s => s.nudgeSpot);
  const setSpotPosDirect = useStore(s => s.setSpotPosDirect);
  const setSelected = useStore(s => s.setSelected);
  const [step, setStep] = useState(0.25);
  const [sx, setSx] = useState(0);
  const [sz, setSz] = useState(0);

  useEffect(()=>{
    if (spotlight?.position) {
      setSx(Number(spotlight.position[0].toFixed(2)));
      setSz(Number(spotlight.position[2].toFixed(2)));
    }
  }, [spotlight.position[0], spotlight.position[2]]);

  return (
    <Card className="bg-gray-900 text-white border-gray-800">
      <CardHeader className="py-2"><CardTitle className="text-sm">Scene Control</CardTitle></CardHeader>
      <CardContent className="space-y-3 text-xs">
        <div className="flex gap-2">
          <Button size="sm" onClick={addCube}>Add Cube</Button>
          <Button size="sm" variant="destructive" onClick={deleteSelected} disabled={selected===-1}>Delete Selected</Button>
        </div>
        <div>
          <div className="mb-1">Animation Speed: {animationSpeed.toFixed(2)}</div>
          <Slider value={[animationSpeed]} min={0} max={2} step={0.01} onValueChange={([v])=> setAnimationSpeed(v)} />
        </div>
        <div className="flex items-center justify-between">
          <div>Spotlight: {spotlight.enabled? 'On':'Off'}</div>
          <Button size="sm" variant={spotlight.enabled? 'secondary':'default'} onClick={toggleSpot}>
            Toggle
          </Button>
        </div>
        {/* Programmatic spotlight controls (no pointer needed) */}
        <div className="space-y-2 p-2 rounded-md bg-gray-800/60">
          <div className="flex items-center justify-between">
            <div className="opacity-80">Spot pos (X,Z)</div>
            <div className="flex gap-2">
              <input className="w-16 bg-black/40 border border-gray-700 rounded px-2 py-1 text-right" value={sx}
                     onChange={(e)=> setSx(parseFloat(e.target.value)||0)} />
              <input className="w-16 bg-black/40 border border-gray-700 rounded px-2 py-1 text-right" value={sz}
                     onChange={(e)=> setSz(parseFloat(e.target.value)||0)} />
              <Button size="sm" variant="secondary" onClick={()=> setSpotPosDirect(Number(sx)||0, 1, Number(sz)||0)}>Set</Button>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="opacity-80">Nudge step: {step.toFixed(2)}</div>
            <Slider value={[step]} min={0.05} max={1} step={0.05} onValueChange={([v])=> setStep(v)} />
          </div>
          <div className="grid grid-cols-3 gap-1 place-items-center">
            <div />
            <Button size="sm" onClick={()=> nudgeSpot(0, -step)}>↑</Button>
            <div />
            <Button size="sm" onClick={()=> nudgeSpot(-step, 0)}>←</Button>
            <Button size="sm" variant="secondary" onClick={()=> setSpotPosDirect(0,1,0)}>Center</Button>
            <Button size="sm" onClick={()=> nudgeSpot(step, 0)}>→</Button>
            <div />
            <Button size="sm" onClick={()=> nudgeSpot(0, step)}>↓</Button>
            <div />
          </div>
        </div>
        <div>
          <div className="mb-1">Select Cube</div>
          <Select value={selected===-1? "none" : String(selected)} onValueChange={(v)=> setSelected(v === "none" ? -1 : parseInt(v, 10))}>
            <SelectTrigger className="w-full"><SelectValue placeholder="None" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              {cubes.map((_, i)=> (<SelectItem key={i} value={String(i)}>{`Cube ${i}`}</SelectItem>))}
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  );
}


export default function CubesSpotlight() {
  return (
    <div className="grid grid-cols-4 h-screen">
      <div className="col-span-3">
        <Canvas shadows camera={{ position:[0,1.5,6], fov:50 }}>
          <Suspense fallback={null}>
            <Scene />
          </Suspense>
        </Canvas>
      </div>
      <div className="p-4 bg-black/80 overflow-y-auto">
        <Panel />
      </div>
    </div>
  );
}

// Programmatic control from outside this module:
// import { setSpotlightPosition, nudgeSpotlight } from '...';
export const setSpotlightPosition = (x, y, z) => useStore.getState().setSpotPosDirect(x, y, z);
export const nudgeSpotlight = (dx, dz) => useStore.getState().nudgeSpot(dx, dz);

// Export the store for external access
export { useStore };
