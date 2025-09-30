// Crypto Dashboard - Advanced Trading & Predictions (React Integration)
import * as React from 'react';

export function CryptoDashboard() {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [fps, setFps] = React.useState(60);
  const [apiConfig, setApiConfig] = React.useState({
    provider: 'coingecko',
    key: '',
    secret: '',
    customEndpoint: '',
    currentCrypto: 'BTC'
  });
  const [apiStatus, setApiStatus] = React.useState('Not configured');
  const [loading, setLoading] = React.useState(false);
  const [data, setData] = React.useState({
    candles: [],
    rsi: [],
    macd: [],
    obv: [],
    trend: []
  });

  const [camera, setCamera] = React.useState({
    pos: { x: 0, y: 84, z: 0 },
    yaw: 180,
    pitch: 0,
    fov: 520
  });

  const [showHUD, setShowHUD] = React.useState(true);
  const [showApiConfig, setShowApiConfig] = React.useState(true);
  const [paused, setPaused] = React.useState(false);
  const [visible, setVisible] = React.useState(120);
  const [start, setStart] = React.useState(0);

  React.useEffect(() => {
    initializeCrypto3D();
    seedData(); // Start with demo data
  }, []);

  const initializeCrypto3D = async () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;

    // Canvas setup
    const DPR = Math.min(3, devicePixelRatio || 1);
    const resize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.floor(width * DPR);
      canvas.height = Math.floor(height * DPR);
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.imageSmoothingEnabled = true;
    };

    window.addEventListener('resize', resize);
    resize();

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // 3D Projection system
    const toCam = (p: Point3D) => {
      const px = p.x - camera.pos.x;
      const py = p.y - camera.pos.y;
      const pz = p.z - camera.pos.z;

      const cy = Math.cos(-camera.yaw * Math.PI / 180);
      const sy = Math.sin(-camera.yaw * Math.PI / 180);

      const x = px * cy - pz * sy;
      const z = px * sy + pz * cy;

      return { x, y: py, z };
    };

    const proj = (p: Point3D) => {
      const s = camera.fov / p.z;
      return {
        x: canvas.width / (2 * DPR) + p.x * s,
        y: canvas.height / (2 * DPR) - p.y * s,
        s
      };
    };

    // Room geometry
    const GRID = { size: 2000, lines: 21, wallH: 520 };
    const gridLines: Array<{ a: Point3D; b: Point3D; type: string }> = [];

    // Generate grid lines
    const half = GRID.size / 2;
    const step = GRID.size / (GRID.lines - 1);
    const H = GRID.wallH;

    for (let i = 0; i < GRID.lines; i++) {
      const v = -half + i * step;

      // Floor grid
      gridLines.push({
        a: { x: v, y: 0, z: -half },
        b: { x: v, y: 0, z: half },
        type: 'floor'
      });
      gridLines.push({
        a: { x: -half, y: 0, z: v },
        b: { x: half, y: 0, z: v },
        type: 'floor'
      });

      // Ceiling grid
      gridLines.push({
        a: { x: v, y: H, z: -half },
        b: { x: v, y: H, z: half },
        type: 'ceil'
      });
      gridLines.push({
        a: { x: -half, y: H, z: v },
        b: { x: half, y: H, z: v },
        type: 'ceil'
      });
    }

    // Pillars
    gridLines.push({ a: { x: -half, y: 0, z: -half }, b: { x: -half, y: H, z: -half }, type: 'pillar' });
    gridLines.push({ a: { x: half, y: 0, z: -half }, b: { x: half, y: H, z: -half }, type: 'pillar' });
    gridLines.push({ a: { x: -half, y: 0, z: half }, b: { x: -half, y: H, z: half }, type: 'pillar' });
    gridLines.push({ a: { x: half, y: 0, z: half }, b: { x: half, y: H, z: half }, type: 'pillar' });

    // Draw functions
    const drawLine3D = (p1: Point3D, p2: Point3D, w = 1, strokeStyle?: string) => {
      const a = toCam(p1);
      const b = toCam(p2);

      if (a.z <= 1 && b.z <= 1) return;

      const A = proj(a);
      const B = proj(b);

      ctx.lineWidth = Math.max(0.6, Math.min(8, w));
      if (strokeStyle) ctx.strokeStyle = strokeStyle;
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x, B.y);
    };

    // Wall panels for charts
    const walls = [
      { pos: { x: 0, y: 0, z: -half + 5 }, rot: 0, panels: ['BTC/USD', 'Volume', 'RSI'] },
      { pos: { x: -half + 5, y: 0, z: 0 }, rot: -90, panels: ['MACD', 'RSI', 'Trend'] },
      { pos: { x: half - 5, y: 0, z: 0 }, rot: 90, panels: ['OBV', 'Trend', 'MACD'] },
      { pos: { x: 0, y: 0, z: half - 5 }, rot: 180, panels: ['Market', 'OBV', 'RSI'] }
    ];

    // Animation loop
    let lastTime = performance.now();
    let frames = 0;

    const animate = (now: number) => {
      frames++;
      if (now - lastTime > 600) {
        setFps(Math.round(frames * 1000 / (now - lastTime)));
        lastTime = now;
        frames = 0;
      }

      // Clear canvas
      ctx.fillStyle = 'rgba(10, 15, 23, 0.95)';
      ctx.fillRect(0, 0, canvas.width / DPR, canvas.height / DPR);

      // Draw grid
      ctx.beginPath();
      for (const line of gridLines) {
        const color = line.type === 'floor' ? '#1a4a3a' :
                     line.type === 'ceil' ? '#0f2d22' : '#2a5a4a';
        drawLine3D(line.a, line.b, 1, color);
      }
      ctx.stroke();

      // Draw chart panels
      drawChartPanels(ctx, walls);

      requestAnimationFrame(animate);
    };

    const drawChartPanels = (ctx: CanvasRenderingContext2D, walls: any[]) => {
      const PANEL_WIDTH = GRID.size * 0.25;
      const PANEL_HEIGHT = GRID.wallH * 0.7;

      walls.forEach((wall, wallIndex) => {
        wall.panels.forEach((panelName: string, panelIndex: number) => {
          const panelX = (panelIndex - 1) * PANEL_WIDTH * 1.2;
          const panelY = GRID.wallH * 0.15;

          // Calculate panel corners in 3D space
          const rot = (wall.rot || 0) * Math.PI / 180;
          const corners = [
            { x: panelX - PANEL_WIDTH/2, y: panelY },
            { x: panelX + PANEL_WIDTH/2, y: panelY },
            { x: panelX + PANEL_WIDTH/2, y: panelY + PANEL_HEIGHT },
            { x: panelX - PANEL_WIDTH/2, y: panelY + PANEL_HEIGHT }
          ].map(p => ({
            x: wall.pos.x + p.x * Math.cos(rot),
            y: p.y,
            z: wall.pos.z - p.x * Math.sin(rot)
          }));

          // Draw panel background
          drawQuad(ctx, corners, 'rgba(20, 40, 35, 0.6)');

          // Draw chart data
          if (data.candles.length > 0) {
            drawChartInPanel(ctx, corners, panelName, wallIndex * 3 + panelIndex);
          }
        });
      });
    };

    const drawQuad = (ctx: CanvasRenderingContext2D, corners: Point3D[], fillStyle: string) => {
      const projectedCorners = corners.map(c => {
        const cam = toCam(c);
        if (cam.z <= 1) return null;
        return proj(cam);
      }).filter(p => p !== null);

      if (projectedCorners.length < 4) return;

      ctx.save();
      ctx.fillStyle = fillStyle;
      ctx.beginPath();
      ctx.moveTo(projectedCorners[0].x, projectedCorners[0].y);
      for (let i = 1; i < projectedCorners.length; i++) {
        ctx.lineTo(projectedCorners[i].x, projectedCorners[i].y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    };

    const drawChartInPanel = (ctx: CanvasRenderingContext2D, corners: Point3D[], chartType: string, index: number) => {
      // Sample chart drawing - would be expanded based on chart type
      const color = ['#38bdf8', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#f97316'][index % 6];

      // Draw simple line chart representation
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const sampleData = data.candles.slice(start, start + visible);
      if (sampleData.length > 1) {
        sampleData.forEach((candle, i) => {
          const t = i / (sampleData.length - 1);
          const x = corners[0].x + t * (corners[1].x - corners[0].x);
          const y = corners[0].y + 0.5 * (corners[3].y - corners[0].y);

          const cam = toCam({ x, y, z: corners[0].z });
          if (cam.z > 1) {
            const proj_pt = proj(cam);
            if (i === 0) {
              ctx.moveTo(proj_pt.x, proj_pt.y);
            } else {
              ctx.lineTo(proj_pt.x, proj_pt.y);
            }
          }
        });
      }

      ctx.stroke();
    };

    // Controls
    let dragging = false;

    const handleMouseDown = (e: MouseEvent) => {
      if (e.button === 0) {
        dragging = true;
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!dragging) return;

      setCamera(prev => ({
        ...prev,
        yaw: prev.yaw - e.movementX * 0.4,
        pos: {
          ...prev.pos,
          y: Math.max(12, Math.min(GRID.wallH - 12, prev.pos.y - e.movementY * 1.5))
        }
      }));
    };

    const handleMouseUp = () => {
      dragging = false;
    };

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      setVisible(prev => Math.max(10, Math.min(300, prev + (e.deltaY > 0 ? 5 : -5))));
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'r' || e.key === 'R') {
        setCamera({
          pos: { x: 0, y: 84, z: 0 },
          yaw: 180,
          pitch: 0,
          fov: 520
        });
      }
      if (e.key === ' ') {
        e.preventDefault();
        setPaused(!paused);
      }
    };

    canvas.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('wheel', handleWheel, { passive: false });
    window.addEventListener('keydown', handleKeyDown);

    animate(performance.now());

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('wheel', handleWheel);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('resize', resize);
    };
  };

  const seedData = (n = 220) => {
    const newData = {
      candles: [] as any[],
      trend: [] as number[],
      obv: [] as number[],
      rsi: [] as number[],
      macd: [] as any[]
    };

    let last = 500;
    let obv = 10000;

    for (let i = 0; i < n; i++) {
      const o = last;
      const c = last + (Math.random() - 0.5) * 18 + Math.sin(i * 0.07) * 0.8;
      const h = Math.max(o, c) + Math.random() * 10;
      const l = Math.min(o, c) - Math.random() * 10;
      const v = 100 + Math.random() * 80;

      newData.candles.push({ o, h, l, c, v });
      newData.trend.push(c);

      obv += (c > o ? 1 : -1) * v;
      newData.obv.push(obv);

      last = c;
    }

    // Compute indicators
    recomputeIndicators(newData);
    setData(newData);
    setStart(Math.max(0, newData.candles.length - visible));
  };

  const recomputeIndicators = (data: any) => {
    const closes = data.candles.map((c: any) => c.c);

    // RSI calculation
    const rsi = [];
    let g = 0, L = 0;
    for (let i = 1; i < closes.length; i++) {
      const diff = closes[i] - closes[i - 1];
      g = (g * 13 + Math.max(0, diff)) / 14;
      L = (L * 13 + Math.max(0, -diff)) / 14;
      rsi.push(L === 0 ? 100 : 100 - 100 / (1 + g / L));
    }
    data.rsi = rsi;

    // MACD calculation (simplified)
    const ema = (arr: number[], n: number) => {
      const k = 2 / (n + 1);
      let e = arr[0];
      const out = [e];
      for (let i = 1; i < arr.length; i++) {
        e = arr[i] * k + e * (1 - k);
        out.push(e);
      }
      return out;
    };

    const ema12 = ema(closes, 12);
    const ema26 = ema(closes, 26);
    const macd = ema12.map((v, i) => v - (ema26[i] || v));
    const signal = ema(macd, 9);
    const histogram = macd.map((m, i) => m - (signal[i] || 0));

    data.macd = macd.map((m, i) => ({
      macd: m,
      signal: signal[i] || 0,
      histogram: histogram[i] || 0
    }));
  };

  const fetchDataFromAPI = async () => {
    setLoading(true);
    try {
      // Mock API call - replace with real implementation
      const response = await fetch(`/api/crypto/${apiConfig.currentCrypto}`);
      const apiData = await response.json();

      // Process API data
      processAPIData(apiConfig.currentCrypto, apiData);
      setApiStatus('Connected');
    } catch (error) {
      console.error('API fetch failed:', error);
      setApiStatus('Failed');
      seedData(); // Fallback to demo data
    } finally {
      setLoading(false);
    }
  };

  const processAPIData = (symbol: string, apiData: any) => {
    // Convert API data to internal format
    const newData = {
      candles: [],
      trend: [],
      obv: [],
      rsi: [],
      macd: []
    };

    // Process different API response formats
    if (apiData.prices && Array.isArray(apiData.prices)) {
      // CoinGecko format
      apiData.prices.forEach(([timestamp, price]: [number, number]) => {
        const o = price;
        const c = price * (1 + (Math.random() - 0.5) * 0.02); // Add some variance
        const h = Math.max(o, c) * (1 + Math.random() * 0.01);
        const l = Math.min(o, c) * (1 - Math.random() * 0.01);
        const v = 100 + Math.random() * 50;

        (newData.candles as any).push({ o, h, l, c, v, timestamp });
        newData.trend.push(c);
      });
    }

    recomputeIndicators(newData);
    setData(newData);
  };

  const saveSnapshot = () => {
    if (!canvasRef.current) return;

    const url = canvasRef.current.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = 'crypto-dashboard-snapshot.png';
    a.click();
  };

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 bg-gray-900/80 flex items-center justify-center z-50">
          <div className="w-12 h-12 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
        </div>
      )}

      {/* Main Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ background: 'radial-gradient(circle at 60% 10%, #0a0f17 0%, #050a0f 100%)' }}
      />

      {/* API Configuration */}
      {showApiConfig && (
        <div className="absolute top-3 right-3 z-40 bg-black/60 border border-white/20 backdrop-blur rounded-lg p-4 max-w-sm">
          <h3 className="text-blue-400 font-bold mb-3">API Configuration</h3>

          <select
            value={apiConfig.provider}
            onChange={(e) => setApiConfig(prev => ({ ...prev, provider: e.target.value }))}
            className="w-full mb-2 px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-sm"
          >
            <option value="coingecko">CoinGecko</option>
            <option value="coinbase">Coinbase</option>
            <option value="binance">Binance</option>
            <option value="custom">Custom</option>
          </select>

          <select
            value={apiConfig.currentCrypto}
            onChange={(e) => setApiConfig(prev => ({ ...prev, currentCrypto: e.target.value }))}
            className="w-full mb-2 px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-sm"
          >
            <option value="BTC">Bitcoin</option>
            <option value="ETH">Ethereum</option>
            <option value="ADA">Cardano</option>
            <option value="SOL">Solana</option>
          </select>

          <button
            onClick={fetchDataFromAPI}
            className="w-full px-3 py-2 bg-blue-500/30 border border-blue-500/50 rounded text-white hover:bg-blue-500/40 transition-colors"
          >
            Apply Configuration
          </button>

          <div className="text-xs text-gray-400 mt-2">
            Status: {apiStatus}
          </div>

          <button
            onClick={() => setShowApiConfig(false)}
            className="absolute top-2 right-2 w-6 h-6 flex items-center justify-center text-gray-400 hover:text-white"
          >
            ×
          </button>
        </div>
      )}

      {/* HUD */}
      {showHUD && (
        <div className="absolute left-3 bottom-3 z-40 bg-black/40 border border-white/10 backdrop-blur rounded-lg p-3">
          <div className="text-blue-400 font-mono text-sm mb-1">
            FPS: {fps} • Panels: 12
          </div>
          <div className="text-gray-300 text-xs mb-2">
            Drag: Look around • Wheel: Zoom timeline • <kbd className="px-1 bg-white/10 rounded">R</kbd> Reset • <kbd className="px-1 bg-white/10 rounded">Space</kbd> Pause
          </div>
          <div className="text-gray-400 text-xs">
            Visible: {visible} bars • Start: {start}
          </div>
        </div>
      )}

      {/* Control Buttons */}
      <div className="absolute right-3 bottom-3 z-40 flex gap-2">
        <button
          onClick={() => setShowHUD(!showHUD)}
          className="px-3 py-2 bg-gray-500/20 border border-gray-400/60 text-gray-100 rounded-lg font-bold text-sm backdrop-blur hover:bg-gray-500/30 transition-colors"
        >
          HUD
        </button>
        <button
          onClick={() => setShowApiConfig(!showApiConfig)}
          className="px-3 py-2 bg-blue-500/20 border border-blue-400/60 text-blue-100 rounded-lg font-bold text-sm backdrop-blur hover:bg-blue-500/30 transition-colors"
        >
          API
        </button>
        <button
          onClick={saveSnapshot}
          className="px-3 py-2 bg-emerald-500/20 border border-emerald-400/60 text-emerald-100 rounded-lg font-bold text-sm backdrop-blur hover:bg-emerald-500/30 transition-colors"
        >
          📸
        </button>
      </div>

      {/* Pause Indicator */}
      {paused && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 bg-black/60 border border-yellow-400/60 backdrop-blur rounded-lg px-4 py-2">
          <div className="text-yellow-400 font-bold">⏸️ PAUSED</div>
        </div>
      )}
    </div>
  );
}

// Types
interface Point3D {
  x: number;
  y: number;
  z: number;
}

interface CandleData {
  o: number; // open
  h: number; // high
  l: number; // low
  c: number; // close
  v: number; // volume
  timestamp?: number;
}

interface MACDData {
  macd: number;
  signal: number;
  histogram: number;
}
