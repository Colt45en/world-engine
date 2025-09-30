// NEXUS 3D Iframe Room Integration for World Engine
import * as React from 'react';

export function NexusRoom() {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const cssRef = React.useRef<HTMLDivElement>(null);
  const [fps, setFps] = React.useState(60);
  const [showHUD, setShowHUD] = React.useState(false);
  const [panels, setPanels] = React.useState<PanelData[]>([]);
  const [selectedPanel, setSelectedPanel] = React.useState<PanelData | null>(null);
  const [showDialog, setShowDialog] = React.useState(false);
  const [urlInput, setUrlInput] = React.useState('');

  React.useEffect(() => {
    initializeNexusRoom();
  }, []);

  const initializeNexusRoom = async () => {
    if (!canvasRef.current || !cssRef.current) return;

    // Import Three.js modules
    const THREE = await import('three');
    const { CSS3DRenderer, CSS3DObject } = await import('three/examples/jsm/renderers/CSS3DRenderer.js');

    // Scene setup
    const scene = new THREE.Scene();
    const sceneCSS = new THREE.Scene();
    const ROOM = 10;
    const HALF = ROOM / 2;
    const FLOOR_Y = -HALF;

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.05, 200);
    const EYE_Y = FLOOR_Y + 0.32;
    camera.position.set(0, EYE_Y, 0);
    camera.rotation.order = 'YXZ';
    scene.add(camera);

    // WebGL renderer
    const targetDPR = Math.min(devicePixelRatio || 1, 1.5);
    const gl = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: true });
    gl.setPixelRatio(targetDPR);
    gl.setSize(window.innerWidth, window.innerHeight);

    // CSS3D renderer
    const css = new CSS3DRenderer({ element: cssRef.current });
    css.setSize(window.innerWidth, window.innerHeight);

    // Lighting
    scene.add(new THREE.AmbientLight(0xbaffea, 0.22));
    const key = new THREE.DirectionalLight(0x9ff7dc, 0.5);
    key.position.set(3, 4, 2);
    scene.add(key);

    // Room geometry
    const room = new THREE.Mesh(
      new THREE.BoxGeometry(ROOM, ROOM, ROOM),
      new THREE.MeshPhongMaterial({ color: 0x0a1612, side: THREE.BackSide, shininess: 10, specular: 0x0a1f17 })
    );
    scene.add(room);

    const grid = new THREE.GridHelper(ROOM, ROOM, 0x29584a, 0x12362c);
    grid.position.y = FLOOR_Y + 0.01;
    scene.add(grid);

    // Panel creation system
    const createPanel = (config: PanelConfig): PanelData => {
      const { name, pos, rot, w = 2.2, h = 1.6, url = 'about:blank' } = config;

      const geo = new THREE.PlaneGeometry(w, h);
      const mat = new THREE.MeshBasicMaterial({
        color: 0x0d1512,
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide
      });

      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(pos);
      mesh.rotation.set(rot.x, rot.y, rot.z);
      mesh.updateMatrix();
      mesh.matrixAutoUpdate = false;
      scene.add(mesh);

      const wire = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0x56ffd2 })
      );
      wire.position.copy(mesh.position);
      wire.rotation.copy(mesh.rotation);
      wire.updateMatrix();
      wire.matrixAutoUpdate = false;
      scene.add(wire);

      // CSS panel
      const div = document.createElement('div');
      div.className = 'nexus-panel';
      div.style.width = `${(w - 2 * 0.06) * 100}px`;
      div.style.height = `${(h - 2 * 0.06) * 100}px`;
      div.style.borderRadius = '12px';
      div.style.overflow = 'hidden';
      div.style.boxShadow = '0 10px 34px rgba(0,0,0,.55)';

      const iframe = document.createElement('iframe');
      iframe.allow = 'clipboard-write; fullscreen';
      iframe.loading = 'lazy';
      iframe.referrerPolicy = 'no-referrer';
      iframe.src = url;
      iframe.style.width = '100%';
      iframe.style.height = '100%';
      iframe.style.border = '0';
      iframe.style.background = '#0b1216';
      iframe.style.filter = 'saturate(1.02) contrast(1.01)';
      div.appendChild(iframe);

      const cssObj = new CSS3DObject(div);
      cssObj.position.copy(pos);
      cssObj.rotation.copy(mesh.rotation);
      const s = 1 / 100;
      cssObj.scale.set(s, s, 1);

      const forward = new THREE.Vector3(0, 0, 1)
        .applyEuler(cssObj.rotation)
        .multiplyScalar(0.002);
      cssObj.position.add(forward);
      cssObj.updateMatrix();
      cssObj.matrixAutoUpdate = false;
      sceneCSS.add(cssObj);

      return {
        id: Date.now() + Math.random(),
        name,
        mesh,
        wire,
        cssObj,
        iframe,
        w,
        h,
        url
      };
    };

    // Create initial panels
    const WIDTH = 2.2;
    const HEIGHT = 1.6;
    const GAP = 3.0;
    const PANEL_Y = FLOOR_Y + 0.4572 + HEIGHT * 0.5;
    const EPS = 0.06;

    const initialPanels: PanelData[] = [];

    // Front wall panels
    const frontZ = -HALF + EPS;
    const frontRot = new THREE.Euler(0, 0, 0);
    initialPanels.push(
      createPanel({ name: 'F-L', pos: new THREE.Vector3(-GAP, PANEL_Y, frontZ), rot: frontRot, url: 'about:blank' }),
      createPanel({ name: 'F-C', pos: new THREE.Vector3(0, PANEL_Y, frontZ), rot: frontRot, url: '/visual-bleedway' }),
      createPanel({ name: 'F-R', pos: new THREE.Vector3(GAP, PANEL_Y, frontZ), rot: frontRot, url: '/sensory-demo' })
    );

    // Right wall panels
    const rightX = HALF - EPS;
    const rightRot = new THREE.Euler(0, -Math.PI / 2, 0);
    initialPanels.push(
      createPanel({ name: 'R-F', pos: new THREE.Vector3(rightX, PANEL_Y, -GAP), rot: rightRot, url: 'about:blank' }),
      createPanel({ name: 'R-C', pos: new THREE.Vector3(rightX, PANEL_Y, 0), rot: rightRot, url: '/crypto-dashboard' }),
      createPanel({ name: 'R-B', pos: new THREE.Vector3(rightX, PANEL_Y, GAP), rot: rightRot, url: 'about:blank' })
    );

    // Back wall panels
    const backZ = HALF - EPS;
    const backRot = new THREE.Euler(0, Math.PI, 0);
    initialPanels.push(
      createPanel({ name: 'B-L', pos: new THREE.Vector3(-GAP, PANEL_Y, backZ), rot: backRot, url: 'about:blank' }),
      createPanel({ name: 'B-C', pos: new THREE.Vector3(0, PANEL_Y, backZ), rot: backRot, url: '/dashboard' }),
      createPanel({ name: 'B-R', pos: new THREE.Vector3(GAP, PANEL_Y, backZ), rot: backRot, url: 'about:blank' })
    );

    // Left wall panels
    const leftX = -HALF + EPS;
    const leftRot = new THREE.Euler(0, Math.PI / 2, 0);
    initialPanels.push(
      createPanel({ name: 'L-F', pos: new THREE.Vector3(leftX, PANEL_Y, -GAP), rot: leftRot, url: 'about:blank' }),
      createPanel({ name: 'L-C', pos: new THREE.Vector3(leftX, PANEL_Y, 0), rot: leftRot, url: '/documentation' }),
      createPanel({ name: 'L-B', pos: new THREE.Vector3(leftX, PANEL_Y, GAP), rot: leftRot, url: 'about:blank' })
    );

    setPanels(initialPanels);

    // Camera controls
    let dragging = false;
    let yaw = 0;
    let pitch = 0;
    const PITCH_MIN = -0.05;
    const PITCH_MAX = 0.28;

    const applyView = () => {
      camera.rotation.set(pitch, yaw, 0);
      camera.updateProjectionMatrix();
    };

    const handlePointerDown = (e: PointerEvent) => {
      if (e.button === 0) {
        dragging = true;
      }
    };

    const handlePointerMove = (e: PointerEvent) => {
      if (!dragging) return;

      yaw -= e.movementX * 0.006;
      pitch = Math.max(PITCH_MIN, Math.min(PITCH_MAX, pitch - e.movementY * 0.0018));
      applyView();
    };

    const handlePointerUp = () => {
      dragging = false;
    };

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      camera.fov = Math.max(35, Math.min(85, camera.fov + (e.deltaY > 0 ? 3 : -3)));
      applyView();
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'r' || e.key === 'R') {
        yaw = 0;
        pitch = 0;
        camera.fov = 55;
        applyView();
      }
    };

    // Event listeners
    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('wheel', handleWheel, { passive: false });
    window.addEventListener('keydown', handleKeyDown);

    // Animation loop
    let lastFPS = 60;
    let lastTime = performance.now();
    let frames = 0;

    const animate = (now: number) => {
      frames++;
      if (now - lastTime > 600) {
        lastFPS = frames * 1000 / (now - lastTime);
        setFps(Math.round(lastFPS));
        lastTime = now;
        frames = 0;
      }

      gl.render(scene, camera);
      css.render(sceneCSS, camera);
      requestAnimationFrame(animate);
    };

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      gl.setSize(window.innerWidth, window.innerHeight);
      css.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    applyView();
    animate(performance.now());

    // Cleanup function
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('wheel', handleWheel);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('resize', handleResize);

      scene.clear();
      sceneCSS.clear();
      gl.dispose();
    };
  };

  const openDialog = (panel: PanelData) => {
    setSelectedPanel(panel);
    setUrlInput(panel.url);
    setShowDialog(true);
  };

  const closeDialog = () => {
    setShowDialog(false);
    setSelectedPanel(null);
    setUrlInput('');
  };

  const applyEmbed = () => {
    if (!selectedPanel) return;

    selectedPanel.iframe.src = urlInput;
    selectedPanel.url = urlInput;
    closeDialog();
  };

  const saveSnapshot = () => {
    if (!canvasRef.current) return;

    try {
      const url = canvasRef.current.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = 'nexus-room-snapshot.png';
      a.click();
    } catch (err) {
      console.error('Snapshot failed:', err);
    }
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* WebGL Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ background: 'radial-gradient(1200px 700px at 60% -10%, #0e241d 0%, #08120f 60%)' }}
      />

      {/* CSS3D Container */}
      <div
        ref={cssRef}
        className="absolute inset-0 w-full h-full pointer-events-auto"
      />

      {/* HUD Toggle */}
      <div className="fixed left-3 bottom-3 z-50">
        <button
          onClick={() => setShowHUD(!showHUD)}
          className="px-3 py-2 bg-emerald-500/20 border border-emerald-400/60 text-emerald-100 rounded-lg font-bold text-sm backdrop-blur hover:bg-emerald-500/30 transition-colors"
        >
          HUD
        </button>
      </div>

      {/* HUD Panel */}
      {showHUD && (
        <div className="fixed left-3 bottom-14 z-40 flex flex-col gap-2 max-w-md">
          <div className="bg-black/40 border border-white/10 rounded-lg backdrop-blur p-3">
            <div className="text-emerald-400 font-bold mb-2">
              NEXUS • 3D Iframe Room
            </div>
            <div className="text-gray-300 text-sm">
              Grounded FPV • Axis-locked look • Mouse wheel: FOV • <kbd className="px-1 py-0.5 bg-white/10 rounded text-xs">R</kbd> reset • <kbd className="px-1 py-0.5 bg-white/10 rounded text-xs">S</kbd> snapshot
            </div>
          </div>

          <div className="bg-black/40 border border-white/10 rounded-lg backdrop-blur p-3">
            <div className="text-gray-300 text-sm">
              FPS: <span className="font-mono">{fps}</span> • Panels: {panels.length}
            </div>
          </div>

          <div className="bg-black/40 border border-white/10 rounded-lg backdrop-blur p-3">
            <button
              onClick={() => setShowDialog(true)}
              className="px-3 py-2 bg-cyan-500/20 border border-cyan-400/60 text-cyan-100 rounded-lg font-bold text-sm hover:bg-cyan-500/30 transition-colors"
            >
              Configure Panel URLs
            </button>
          </div>
        </div>
      )}

      {/* Panel Configuration Dialog */}
      {showDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur"
            onClick={closeDialog}
          />
          <div className="relative bg-gradient-to-b from-gray-800/90 to-gray-900/85 border border-white/10 rounded-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-emerald-400 font-bold mb-4">Configure Panel</h3>

            <div className="mb-4">
              <label className="block text-gray-300 text-sm mb-2">
                Selected Panel: {selectedPanel?.name || 'None'}
              </label>
              <input
                type="url"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="Enter URL or route (e.g., /visual-bleedway)"
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-gray-400"
              />
            </div>

            <div className="flex gap-2 mb-4">
              <button
                onClick={applyEmbed}
                className="px-4 py-2 bg-emerald-500/20 border border-emerald-400/60 text-emerald-100 rounded-lg font-bold hover:bg-emerald-500/30 transition-colors"
              >
                Apply
              </button>
              <button
                onClick={closeDialog}
                className="px-4 py-2 bg-gray-500/20 border border-gray-400/60 text-gray-100 rounded-lg font-bold hover:bg-gray-500/30 transition-colors"
              >
                Cancel
              </button>
            </div>

            <div className="flex gap-2 text-xs">
              <button
                onClick={() => setUrlInput('/visual-bleedway')}
                className="px-2 py-1 bg-cyan-500/20 border border-cyan-400/40 text-cyan-200 rounded"
              >
                Visual Bleedway
              </button>
              <button
                onClick={() => setUrlInput('/sensory-demo')}
                className="px-2 py-1 bg-purple-500/20 border border-purple-400/40 text-purple-200 rounded"
              >
                Sensory Demo
              </button>
              <button
                onClick={() => setUrlInput('/dashboard')}
                className="px-2 py-1 bg-blue-500/20 border border-blue-400/40 text-blue-200 rounded"
              >
                Dashboard
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Snapshot Button */}
      <button
        onClick={saveSnapshot}
        className="fixed right-3 bottom-3 z-50 px-3 py-2 bg-blue-500/20 border border-blue-400/60 text-blue-100 rounded-lg font-bold text-sm backdrop-blur hover:bg-blue-500/30 transition-colors"
      >
        📸 Snapshot
      </button>
    </div>
  );
}

// Types
interface PanelConfig {
  name: string;
  pos: THREE.Vector3;
  rot: THREE.Euler;
  w?: number;
  h?: number;
  url?: string;
}

interface PanelData {
  id: number;
  name: string;
  mesh: THREE.Mesh;
  wire: THREE.LineSegments;
  cssObj: any; // CSS3DObject
  iframe: HTMLIFrameElement;
  w: number;
  h: number;
  url: string;
}
