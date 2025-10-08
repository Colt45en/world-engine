// Nexus Forge v2.1 — R3F‑only (Integrated Slideshow + Media)
// React + @react-three/fiber + drei + zustand + shadcn/ui
// Key changes from your draft:
// 1) Removed the separate vanilla three.js HTML app — everything is React.
// 2) Slideshow/Video are React components; no document.getElementById.
// 3) Scene selection uses R3F pointer events; spotlight/particles are preserved.
// 4) Media uploads create textures and wire to a VideoScreen + SlideshowStrip in 3D.

import React, { useMemo, useRef, useState, useEffect, Suspense } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text, useTexture, Html } from "@react-three/drei";
import { create } from "zustand";

// Note: For standalone usage, you'll need these shadcn/ui components
// For sandbox integration, we'll create fallback components
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'destructive' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({
  variant = 'default',
  size = 'md',
  className = '',
  children,
  ...props
}) => {
  const baseClass = "px-3 py-2 rounded-md font-medium transition-all duration-200 cursor-pointer";
  const variantClass = {
    default: "bg-blue-600 hover:bg-blue-700 text-white",
    outline: "border border-gray-300 hover:bg-gray-100 text-gray-700",
    destructive: "bg-red-600 hover:bg-red-700 text-white",
    secondary: "bg-gray-600 hover:bg-gray-700 text-white"
  }[variant];

  const sizeClass = {
    sm: "text-xs px-2 py-1",
    md: "text-sm",
    lg: "text-base px-4 py-3"
  }[size];

  return (
    <button
      className={`${baseClass} ${variantClass} ${sizeClass} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};

const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <div className={`bg-gray-800 border border-gray-700 rounded-lg ${className}`}>
    {children}
  </div>
);

const CardHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <div className={`px-4 py-3 border-b border-gray-700 ${className}`}>
    {children}
  </div>
);

const CardTitle: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <h3 className={`font-semibold text-white ${className}`}>
    {children}
  </h3>
);

const CardDescription: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <p className={`text-gray-400 ${className}`}>
    {children}
  </p>
);

const CardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <div className={`p-4 ${className}`}>
    {children}
  </div>
);

const Select: React.FC<{
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}> = ({ value, onValueChange, children }) => {
  return (
    <select
      value={value}
      onChange={(e) => onValueChange(e.target.value)}
      className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white"
    >
      {children}
    </select>
  );
};

const SelectTrigger: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => <>{children}</>;

const SelectValue: React.FC<{ placeholder: string }> = ({ placeholder }) => <>{placeholder}</>;

const SelectContent: React.FC<{ children: React.ReactNode }> = ({ children }) => <>{children}</>;

const SelectItem: React.FC<{ value: string; children: React.ReactNode }> = ({
  value,
  children
}) => (
  <option value={value}>{children}</option>
);

const Slider: React.FC<{
  value: number[];
  min: number;
  max: number;
  step: number;
  onValueChange: (values: number[]) => void;
}> = ({ value, min, max, step, onValueChange }) => (
  <input
    type="range"
    min={min}
    max={max}
    step={step}
    value={value[0]}
    onChange={(e) => onValueChange([parseFloat(e.target.value)])}
    className="w-full"
  />
);

const Toggle: React.FC<{
  pressed: boolean;
  onPressedChange: (pressed: boolean) => void;
  children: React.ReactNode;
  className?: string;
}> = ({ pressed, onPressedChange, children, className = '' }) => (
  <button
    onClick={() => onPressedChange(!pressed)}
    className={`${className} px-3 py-2 rounded-md transition-all ${
      pressed ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
    }`}
  >
    {children}
  </button>
);

const Tabs: React.FC<{
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
  className?: string;
}> = ({ value, onValueChange, children, className = '' }) => (
  <div className={className}>
    {React.Children.map(children, child =>
      React.cloneElement(child as React.ReactElement, { activeTab: value, setActiveTab: onValueChange })
    )}
  </div>
);

const TabsList: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <div className={`flex gap-1 ${className}`}>
    {children}
  </div>
);

const TabsTrigger: React.FC<{
  value: string;
  children: React.ReactNode;
  className?: string;
  activeTab?: string;
  setActiveTab?: (tab: string) => void;
}> = ({ value, children, className = '', activeTab, setActiveTab }) => (
  <button
    onClick={() => setActiveTab?.(value)}
    className={`px-3 py-2 rounded-md text-sm transition-all ${className} ${
      activeTab === value ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
    }`}
  >
    {children}
  </button>
);

const TabsContent: React.FC<{
  value: string;
  children: React.ReactNode;
  className?: string;
  activeTab?: string;
}> = ({ value, children, className = '', activeTab }) =>
  activeTab === value ? <div className={className}>{children}</div> : null;

const Accordion: React.FC<{
  type: string;
  children: React.ReactNode;
  className?: string;
}> = ({ children, className = '' }) => (
  <div className={className}>
    {children}
  </div>
);

const AccordionItem: React.FC<{
  value: string;
  children: React.ReactNode;
}> = ({ children }) => (
  <div className="border-b border-gray-700">
    {children}
  </div>
);

const AccordionTrigger: React.FC<{
  children: React.ReactNode;
  className?: string;
}> = ({ children, className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full text-left py-2 px-3 hover:bg-gray-700 transition-colors ${className}`}
      >
        {children} <span className="float-right">{isOpen ? '−' : '+'}</span>
      </button>
      <div style={{ display: isOpen ? 'block' : 'none' }}>
        {/* Content will be rendered by AccordionContent */}
      </div>
    </div>
  );
};

const AccordionContent: React.FC<{
  children: React.ReactNode;
  className?: string;
}> = ({ children, className = '' }) => (
  <div className={`px-3 pb-3 text-gray-400 ${className}`}>
    {children}
  </div>
);

// ────────────────────────────────────────────────────────────────────────────────
// Store
const useStore = create((set, get) => ({
  cubes: [
    { id: 1, position: [-1.5, 0, -1], tag: "Seeker", rotation: [0, 0, 0], scale: 1, color: "#4080ff" },
    { id: 2, position: [1.5, 0, -1], tag: "Weaver", rotation: [0, 0, 0], scale: 1, color: "#ff40a0" },
  ],
  spotlight: { enabled: false, position: [0, 5, 0], intensity: 1, color: "#ffffff" },
  ambientLight: { intensity: 0.3, color: "#ffffff" },
  selectedCube: null,
  theme: "default",
  animationSpeed: 0.5,
  fogDensity: 0.2,
  viewMode: "orbit",
  particles: false,

  // Media state
  imageTextures: [], // array of THREE.Texture
  currentSlide: 0,
  slideshowSpeed: 3000,
  slideshowActive: false,
  videoURL: "",

  // Actions
  toggleSpotlight: () => set(s => ({ spotlight: { ...s.spotlight, enabled: !s.spotlight.enabled } })),
  setSpotlightIntensity: (intensity) => set(s => ({ spotlight: { ...s.spotlight, intensity } })),
  setSpotlightColor: (color) => set(s => ({ spotlight: { ...s.spotlight, color } })),
  setAmbientIntensity: (intensity) => set(s => ({ ambientLight: { ...s.ambientLight, intensity } })),

  addCube: () => set(s => {
    const position = [(Math.random() - 0.5) * 5, (Math.random() - 0.5) * 3, (Math.random() - 0.5) * 5];
    const color = `hsl(${Math.random() * 360}, 70%, 60%)`;
    return { cubes: [...s.cubes, { id: Date.now(), position, rotation: [0,0,0], tag: "Newborn", scale: 0.8, color }] };
  }),
  removeCube: (id) => set(s => ({ cubes: s.cubes.filter(c => c.id !== id), selectedCube: s.selectedCube === id ? null : s.selectedCube })),
  selectCube: (id) => set(s => ({ selectedCube: id, theme: s.cubes.find(c => c.id === id)?.tag ?? "default" })),
  updateCube: (id, updates) => set(s => ({ cubes: s.cubes.map(c => (c.id === id ? { ...c, ...updates } : c)) })),

  setTheme: (theme) => set({ theme }),
  setAnimationSpeed: (speed) => set({ animationSpeed: speed }),
  setFogDensity: (density) => set({ fogDensity: density }),
  setViewMode: (mode) => set({ viewMode: mode }),
  toggleParticles: () => set(s => ({ particles: !s.particles })),

  // Media actions
  addImages: (textures) => set(s => ({ imageTextures: [...s.imageTextures, ...textures] })),
  nextSlide: () => set(s => ({ currentSlide: s.imageTextures.length ? (s.currentSlide + 1) % s.imageTextures.length : 0 })),
  prevSlide: () => set(s => ({ currentSlide: s.imageTextures.length ? (s.currentSlide - 1 + s.imageTextures.length) % s.imageTextures.length : 0 })),
  setSlideSpeed: (ms) => set({ slideshowSpeed: ms }),
  toggleSlideshow: () => set(s => ({ slideshowActive: !s.slideshowActive })),
  setVideoURL: (url) => set({ videoURL: url }),

  resetScene: () => set({
    cubes: [
      { id: 1, position: [-1.5, 0, -1], tag: "Seeker", rotation: [0, 0, 0], scale: 1, color: "#4080ff" },
      { id: 2, position: [1.5, 0, -1], tag: "Weaver", rotation: [0, 0, 0], scale: 1, color: "#ff40a0" },
    ],
    spotlight: { enabled: false, position: [0, 5, 0], intensity: 1, color: "#ffffff" },
    ambientLight: { intensity: 0.3, color: "#ffffff" },
    selectedCube: null,
    theme: "default",
    animationSpeed: 0.5,
    fogDensity: 0.2,
    viewMode: "orbit",
    imageTextures: [],
    currentSlide: 0,
    slideshowActive: false,
    videoURL: "",
  })
}));

// ────────────────────────────────────────────────────────────────────────────────
// Themes
const themeMap = {
  default: { bg: "#000000", fog: "#222222", title: "Neutral Void", ambience: "#222222", description: "The beginning and the end - pure potential in perfect balance." },
  Seeker:  { bg: "#001122", fog: "#223344", title: "Karma Echoes", ambience: "#113355", description: "Those who search through the echoes of past actions find wisdom." },
  Weaver:  { bg: "#331144", fog: "#442255", title: "Threading the Dao", ambience: "#662277", description: "Patterns interlace through the fabric of reality." },
  Newborn: { bg: "#0a0a0a", fog: "#1a1a1a", title: "Blank Origin", ambience: "#333333", description: "A clean slate of pure potential." },
  Ascendant:{ bg: "#112233", fog: "#334455", title: "Celestial Ascension", ambience: "#446688", description: "Rising into realms of pure thought." },
  Twilight: { bg: "#221133", fog: "#332244", title: "Ethereal Twilight", ambience: "#553366", description: "The liminal space where mysteries reveal themselves." },
};

// ────────────────────────────────────────────────────────────────────────────────
// 3D Components
function Cube({ position, rotation, id, tag, scale, color }) {
  const meshRef = useRef();
  const animationSpeed = useStore(s => s.animationSpeed);
  const selectedCube = useStore(s => s.selectedCube);
  const selectCube = useStore(s => s.selectCube);
  const isSelected = selectedCube === id;

  useFrame((state) => {
    if (!meshRef.current) return;
    meshRef.current.rotation.x += 0.003 * animationSpeed;
    meshRef.current.rotation.y += 0.005 * animationSpeed;
    const pulse = isSelected ? 1 + Math.sin(state.clock.elapsedTime * 2) * 0.05 : 1;
    meshRef.current.scale.set(scale * pulse, scale * pulse, scale * pulse);
  });

  return (
    <group position={position}>
      <mesh ref={meshRef} onPointerDown={(e) => { e.stopPropagation(); selectCube(id); }}>
        <boxGeometry args={[1,1,1]} />
        <meshStandardMaterial color={color} metalness={0.5} roughness={0.2} emissive={isSelected ? color : "#000000"} emissiveIntensity={isSelected ? 0.3 : 0}/>
      </mesh>
      <Text position={[0, -0.8, 0]} fontSize={0.2} color="white" anchorX="center" anchorY="middle">{tag}</Text>
    </group>
  );
}

function EnhancedSpotlight() {
  const spotlight = useStore(s => s.spotlight);
  if (!spotlight.enabled) return null;
  return (
    <group>
      <pointLight position={spotlight.position} intensity={spotlight.intensity} color={spotlight.color} castShadow shadow-mapSize-width={1024} shadow-mapSize-height={1024} />
      <mesh position={spotlight.position}><sphereGeometry args={[0.1,16,16]} /><meshBasicMaterial color={spotlight.color} transparent opacity={0.7}/></mesh>
    </group>
  );
}

function ParticleSystem() {
  const particlesEnabled = useStore(s => s.particles);
  const theme = useStore(s => s.theme);
  const activeTheme = themeMap[theme] || themeMap.default;
  const particlesRef = useRef();
  const particleCount = 500;

  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const base = new THREE.Color(activeTheme.ambience);
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      const r = 10 * Math.cbrt(Math.random()) + 1;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      positions[i3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i3+1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i3+2] = r * Math.cos(phi);
      const jitter = new THREE.Color().setHSL(base.getHSL({h:0,s:0,l:0}).h, base.getHSL({h:0,s:0,l:0}).s, base.getHSL({h:0,s:0,l:0}).l);
      colors[i3] = base.r + (Math.random() * 0.2 - 0.1);
      colors[i3+1] = base.g + (Math.random() * 0.2 - 0.1);
      colors[i3+2] = base.b + (Math.random() * 0.2 - 0.1);
    }
    return { positions, colors };
  }, [theme]);

  useFrame(() => { if (particlesRef.current && particlesEnabled) particlesRef.current.rotation.y += 0.0003; });
  if (!particlesEnabled) return null;

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.05} vertexColors transparent opacity={0.7} sizeAttenuation />
    </points>
  );
}

// Video screen in 3D (accepts a HTMLVideoElement)
function VideoScreen({ video }) {
  const meshRef = useRef();
  const texture = useMemo(() => (video ? new THREE.VideoTexture(video) : null), [video]);
  useEffect(() => { if (texture) { texture.minFilter = THREE.LinearFilter; texture.magFilter = THREE.LinearFilter; texture.needsUpdate = true; } }, [texture]);
  return (
    <group position={[0, 3, -5]}>
      <mesh><boxGeometry args={[6,4,0.2]} /><meshStandardMaterial color="#333" /></mesh>
      <mesh position={[0,0,0.11]} ref={meshRef}>
        <planeGeometry args={[5.5,3.5]} />
        <meshBasicMaterial color="#fff" map={texture ?? null} />
      </mesh>
    </group>
  );
}

// Slideshow strip of up to 8 images
function SlideshowStrip() {
  const imageTextures = useStore(s => s.imageTextures);
  const currentSlide = useStore(s => s.currentSlide);
  const slideshowActive = useStore(s => s.slideshowActive);
  const slideshowSpeed = useStore(s => s.slideshowSpeed);
  const nextSlide = useStore(s => s.nextSlide);

  useEffect(() => {
    if (!slideshowActive || imageTextures.length === 0) return;
    const id = setInterval(() => nextSlide(), slideshowSpeed);
    return () => clearInterval(id);
  }, [slideshowActive, slideshowSpeed, imageTextures.length, nextSlide]);

  const count = Math.min(8, imageTextures.length);
  const spacing = 1.2;
  const size = 0.8;

  return (
    <group position={[0, 0.5, -5]}>
      <mesh><boxGeometry args={[6,1,0.2]} /><meshStandardMaterial color="#222" /></mesh>
      {Array.from({length: count}).map((_, i) => {
        const tex = imageTextures[i];
        const isActive = i === (currentSlide % Math.max(1, count));
        return (
          <mesh key={i} position={[-1.8 + i*spacing, isActive ? 0.1 : 0, 0.11]} scale={isActive ? [1.2,1.2,1] : [1,1,1]}>
            <planeGeometry args={[size, size]} />
            <meshBasicMaterial color={tex ? "#fff" : "#aaa"} map={tex ?? null} />
          </mesh>
        );
      })}
    </group>
  );
}

function Scene() {
  const { cubes, theme, fogDensity, viewMode, ambientLight } = useStore.getState();
  const themeState = useStore(s => s.theme);
  const activeTheme = themeMap[themeState] || themeMap.default;

  return (
    <>
      <color attach="background" args={[activeTheme.bg]} />
      <fog attach="fog" args={[activeTheme.fog, 2, 10 / Math.max(0.05, useStore.getState().fogDensity)]} />
      <ambientLight intensity={ambientLight.intensity} color={ambientLight.color} />
      <EnhancedSpotlight />
      {useStore.getState().viewMode === 'orbit' ? <OrbitControls enableDamping dampingFactor={0.1} /> : null}
      {useStore.getState().cubes.map(c => <Cube key={c.id} {...c} />)}
      <ParticleSystem />
      <gridHelper args={[20,20,"#444","#222"]} position={[0,-1,0]} />
    </>
  );
}

// ────────────────────────────────────────────────────────────────────────────────
// UI Panels
function MediaPanel() {
  const addImages = useStore(s => s.addImages);
  const setVideoURL = useStore(s => s.setVideoURL);
  const videoURL = useStore(s => s.videoURL);
  const [videoEl, setVideoEl] = useState(null);
  const nextSlide = useStore(s => s.nextSlide);
  const prevSlide = useStore(s => s.prevSlide);
  const toggleSlideshow = useStore(s => s.toggleSlideshow);
  const setSlideSpeed = useStore(s => s.setSlideSpeed);
  const slideshowActive = useStore(s => s.slideshowActive);
  const slideshowSpeed = useStore(s => s.slideshowSpeed);

  // Create textures from uploaded images
  function onImages(e) {
    const files = Array.from(e.target.files || []);
    const loader = new THREE.TextureLoader();
    const texes = files.map(f => loader.load(URL.createObjectURL(f)));
    addImages(texes);
  }

  function onVideo(e){
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setVideoURL(url);
  }

  return (
    <Card className="mb-4 bg-gray-800 border-gray-700">
      <CardHeader className="py-3"><CardTitle className="text-sm">🎞 Media</CardTitle></CardHeader>
      <CardContent className="space-y-3 text-xs">
        <div className="space-y-1">
          <label className="block">Load images</label>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={onImages}
            className="w-full text-xs bg-gray-700 border border-gray-600 rounded p-1"
          />
          <div className="flex gap-2 mt-2">
            <Button size="sm" variant="outline" onClick={prevSlide}>◀ Prev</Button>
            <Button size="sm" variant="outline" onClick={nextSlide}>Next ▶</Button>
            <Button size="sm" onClick={toggleSlideshow}>{slideshowActive ? 'Stop' : 'Start'} Slideshow</Button>
          </div>
          <div className="flex items-center gap-2 mt-2">
            <span>Speed</span>
            <input
              type="range"
              min={1}
              max={10}
              defaultValue={Math.round(slideshowSpeed/1000)}
              onChange={(e)=> setSlideSpeed(Number(e.target.value)*1000)}
              className="flex-1"
            />
            <span>{Math.round(slideshowSpeed/1000)}s</span>
          </div>
        </div>
        <div className="space-y-1">
          <label className="block">Load video</label>
          <input
            type="file"
            accept="video/*"
            onChange={onVideo}
            className="w-full text-xs bg-gray-700 border border-gray-600 rounded p-1"
          />
          {videoURL && (
            <video
              src={videoURL}
              controls
              className="mt-2 w-full"
              ref={setVideoEl}
              style={{ maxHeight: '120px' }}
            />
          )}
        </div>
        <div className="text-gray-400">Images appear in the 3D strip; video plays on the screen.</div>
      </CardContent>
    </Card>
  );
}

function ThemeSelector() {
  const theme = useStore(s => s.theme);
  const setTheme = useStore(s => s.setTheme);
  const activeTheme = themeMap[theme] || themeMap.default;
  return (
    <div className="mb-4">
      <p className="text-xs mb-2 text-white">Current Theme: {activeTheme.title}</p>
      <p className="text-xs italic mb-2 opacity-70 text-gray-400">{activeTheme.description}</p>
      <Select value={theme} onValueChange={setTheme}>
        {Object.entries(themeMap).map(([key, t]) => (
          <SelectItem key={key} value={key}>{t.title}</SelectItem>
        ))}
      </Select>
    </div>
  );
}

function CubeControls() {
  const selectedCube = useStore(s => s.selectedCube);
  const cubes = useStore(s => s.cubes);
  const updateCube = useStore(s => s.updateCube);
  const removeCube = useStore(s => s.removeCube);
  const c = useMemo(() => cubes.find(x => x.id === selectedCube), [cubes, selectedCube]);

  if (!c) {
    return (
      <div className="p-3 text-center border border-gray-700 rounded-md bg-gray-800 text-xs text-gray-300">
        Select a cube to edit
      </div>
    );
  }

  return (
    <Card className="mb-4">
      <CardHeader className="py-2"><CardTitle className="text-sm">Editing: {c.tag}</CardTitle></CardHeader>
      <CardContent className="space-y-4 text-xs">
        <div>
          <p className="mb-1 text-gray-300">Tag</p>
          <Select value={c.tag} onValueChange={v => updateCube(c.id, { tag: v })}>
            {Object.keys(themeMap).filter(k=>k!=="default").map(k => (
              <SelectItem key={k} value={k}>{k}</SelectItem>
            ))}
          </Select>
        </div>
        <div>
          <p className="mb-1 text-gray-300">Scale: {c.scale.toFixed(1)}</p>
          <Slider value={[c.scale]} min={0.5} max={2} step={0.1} onValueChange={([v]) => updateCube(c.id, { scale: v })} />
        </div>
        <div>
          <p className="mb-1 text-gray-300">Color</p>
          <div className="grid grid-cols-6 gap-1">
            {["#4080ff","#ff40a0","#40ff80","#ff8040","#8040ff","#40ffff"].map(col => (
              <div
                key={col}
                className="w-full aspect-square rounded-md cursor-pointer border-2 hover:scale-110 transition-transform"
                style={{
                  backgroundColor: col,
                  borderColor: c.color===col? 'white':'transparent'
                }}
                onClick={() => updateCube(c.id, { color: col })}
              />
            ))}
          </div>
        </div>
        <Button variant="destructive" size="sm" className="w-full" onClick={() => removeCube(c.id)}>
          Remove Cube
        </Button>
      </CardContent>
    </Card>
  );
}

function LightControls() {
  const spotlight = useStore(s => s.spotlight);
  const ambientLight = useStore(s => s.ambientLight);
  const toggleSpotlight = useStore(s => s.toggleSpotlight);
  const setSpotlightIntensity = useStore(s => s.setSpotlightIntensity);
  const setSpotlightColor = useStore(s => s.setSpotlightColor);
  const setAmbientIntensity = useStore(s => s.setAmbientIntensity);

  return (
    <Card className="mb-4">
      <CardHeader className="py-2"><CardTitle className="text-sm">Light Settings</CardTitle></CardHeader>
      <CardContent className="space-y-4 text-xs">
        <Toggle pressed={spotlight.enabled} onPressedChange={toggleSpotlight} className="w-full justify-between">
          Spotlight {spotlight.enabled? 'On':'Off'}
        </Toggle>
        {spotlight.enabled && (
          <>
            <div>
              <p className="mb-1 text-gray-300">Intensity: {spotlight.intensity.toFixed(1)}</p>
              <Slider value={[spotlight.intensity]} min={0.1} max={3} step={0.1} onValueChange={([v]) => setSpotlightIntensity(v)} />
            </div>
            <div>
              <p className="mb-1 text-gray-300">Color</p>
              <div className="grid grid-cols-6 gap-1">
                {["#ffffff","#ffeecc","#ccffee","#eeccff","#ffccee","#ccddff"].map(c => (
                  <div
                    key={c}
                    className="w-full aspect-square rounded-md cursor-pointer border-2 hover:scale-110 transition-transform"
                    style={{
                      backgroundColor: c,
                      borderColor: spotlight.color===c? 'white':'transparent'
                    }}
                    onClick={() => setSpotlightColor(c)}
                  />
                ))}
              </div>
            </div>
          </>
        )}
        <div>
          <p className="mb-1 text-gray-300">Ambient: {ambientLight.intensity.toFixed(1)}</p>
          <Slider value={[ambientLight.intensity]} min={0} max={1} step={0.05} onValueChange={([v]) => setAmbientIntensity(v)} />
        </div>
      </CardContent>
    </Card>
  );
}

function SceneControls() {
  const animationSpeed = useStore(s => s.animationSpeed);
  const fogDensity = useStore(s => s.fogDensity);
  const viewMode = useStore(s => s.viewMode);
  const particles = useStore(s => s.particles);
  const setAnimationSpeed = useStore(s => s.setAnimationSpeed);
  const setFogDensity = useStore(s => s.setFogDensity);
  const setViewMode = useStore(s => s.setViewMode);
  const toggleParticles = useStore(s => s.toggleParticles);
  const resetScene = useStore(s => s.resetScene);
  const addCube = useStore(s => s.addCube);

  return (
    <Card className="mb-4">
      <CardHeader className="py-2"><CardTitle className="text-sm">Scene Settings</CardTitle></CardHeader>
      <CardContent className="space-y-4 text-xs">
        <div>
          <p className="mb-1 text-gray-300">Animation Speed: {animationSpeed.toFixed(1)}</p>
          <Slider value={[animationSpeed]} min={0} max={2} step={0.1} onValueChange={([v]) => setAnimationSpeed(v)} />
        </div>
        <div>
          <p className="mb-1 text-gray-300">Fog Density: {fogDensity.toFixed(2)}</p>
          <Slider value={[fogDensity]} min={0.05} max={1} step={0.05} onValueChange={([v]) => setFogDensity(v)} />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Button variant={viewMode==='orbit'?'default':'outline'} onClick={()=>setViewMode('orbit')} size="sm">
            Orbit
          </Button>
          <Button variant={viewMode==='fixed'?'default':'outline'} onClick={()=>setViewMode('fixed')} size="sm">
            Fixed
          </Button>
        </div>
        <Toggle pressed={particles} onPressedChange={toggleParticles} className="w-full justify-between">
          Particle Effects
        </Toggle>
        <div className="grid grid-cols-2 gap-2">
          <Button variant="secondary" size="sm" onClick={resetScene}>Reset Scene</Button>
          <Button size="sm" onClick={addCube}>Add Cube</Button>
        </div>
      </CardContent>
    </Card>
  );
}

function MythicCodex() {
  return (
    <Accordion type="multiple" className="text-sm">
      {[
        ["🔸 The Oars of Karma", "Each stroke of the oar determines the flow of fate…"],
        ["🔸 The Drowned Monastery", "A sunken temple hidden within the river…"],
        ["🔸 The Whirlpool of Forgotten Names", "A celestial vortex said to erase the past entirely…"],
        ["🔸 The Void of Echoes", "Where all thoughts become reality before dissolving…"],
        ["🔸 The Crystal Nexus", "A convergence point where all timelines intersect…"],
      ].map(([t, d], i) => (
        <AccordionItem key={i} value={`item-${i}`}>
          <AccordionTrigger className="text-sm">{t}</AccordionTrigger>
          <AccordionContent className="text-xs">{d}</AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  );
}

// Glue that renders the video/slideshow meshes inside the same Canvas
function MediaBridges() {
  const videoURL = useStore(s => s.videoURL);
  const [videoEl, setVideoEl] = useState(null);

  useEffect(() => {
    if (!videoURL) {
      setVideoEl(null);
      return;
    }

    const el = document.createElement('video');
    el.src = videoURL;
    el.crossOrigin = 'anonymous';
    el.loop = true;
    el.muted = true;
    el.playsInline = true;
    el.autoplay = true;
    el.play().catch(console.warn);
    setVideoEl(el);

    return () => {
      el.pause();
      el.src = '';
    };
  }, [videoURL]);

  return (
    <>
      {videoEl && <VideoScreen video={videoEl} />}
      <SlideshowStrip />
    </>
  );
}

export default function NexusForge() {
  const [activeTab, setActiveTab] = useState('controls');

  return (
    <div className="flex h-screen bg-gray-900 text-white" style={{ fontFamily: 'system-ui, sans-serif' }}>
      <div className="flex-1 relative">
        <Canvas shadows camera={{ position:[0,2,6], fov:60 }}>
          <Suspense fallback={
            <Html center>
              <div className="text-white">Loading Nexus Forge...</div>
            </Html>
          }>
            <Scene />
            <MediaBridges />
          </Suspense>
        </Canvas>
      </div>

      <div className="w-80 p-4 bg-gray-900 text-white overflow-y-auto border-l border-gray-700">
        <Card className="mb-4 bg-gray-800 border-gray-700">
          <CardHeader className="py-3">
            <CardTitle>🧱 Nexus Forge</CardTitle>
            <CardDescription className="text-gray-400">
              Visual Reality Constructor v2.1
            </CardDescription>
          </CardHeader>
        </Card>

        <ThemeSelector />

        <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-4">
          <TabsList className="grid grid-cols-3 mb-2">
            <TabsTrigger value="controls" className="text-xs">Controls</TabsTrigger>
            <TabsTrigger value="cubes" className="text-xs">Cubes</TabsTrigger>
            <TabsTrigger value="media" className="text-xs">Media</TabsTrigger>
          </TabsList>
          <TabsContent value="controls" className="mt-0">
            <SceneControls />
            <LightControls />
          </TabsContent>
          <TabsContent value="cubes" className="mt-0">
            <CubeControls />
          </TabsContent>
          <TabsContent value="media" className="mt-0">
            <MediaPanel />
          </TabsContent>
        </Tabs>

        <Card>
          <CardHeader className="py-2">
            <CardTitle className="text-sm">📜 Mythic Symbol Codex</CardTitle>
          </CardHeader>
          <CardContent>
            <MythicCodex />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
