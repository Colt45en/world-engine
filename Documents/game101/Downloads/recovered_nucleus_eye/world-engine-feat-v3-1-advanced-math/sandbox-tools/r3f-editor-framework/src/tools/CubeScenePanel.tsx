import { useScene } from '../state/store';
import { useState, useEffect } from 'react';

// Fallback UI components
const Button = ({ children, onClick, size = "default", variant = "default", disabled = false, className = "", ...props }) => {
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
        ${className}
      `}
      {...props}
    >
      {children}
    </button>
  );
};

const Card = ({ children, className = "", ...props }) => (
  <div className={`rounded-lg border border-gray-700 bg-gray-800 shadow-sm ${className}`} {...props}>
    {children}
  </div>
);

const CardHeader = ({ children, className = "", ...props }) => (
  <div className={`flex flex-col space-y-1.5 p-4 ${className}`} {...props}>
    {children}
  </div>
);

const CardTitle = ({ children, className = "", ...props }) => (
  <h3 className={`text-lg font-semibold leading-none tracking-tight text-white ${className}`} {...props}>
    {children}
  </h3>
);

const CardContent = ({ children, className = "", ...props }) => (
  <div className={`p-4 pt-0 ${className}`} {...props}>
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
      className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white"
      {...props}
    >
      {children}
    </select>
  );
};

const SelectTrigger = ({ children, className = "", ...props }) => (
  <div className={className} {...props}>{children}</div>
);

const SelectValue = ({ placeholder }) => null;

const SelectContent = ({ children }) => children;

const SelectItem = ({ children, value, ...props }) => (
  <option value={value} {...props}>
    {children}
  </option>
);

export default function CubeScenePanel() {
  const {
    entities,
    selected,
    animationSpeed,
    spotlight,
    addCube,
    deleteSelected,
    setAnimationSpeed,
    toggleSpotlight,
    setSpotlightPositionDirect,
    nudgeSpotlight,
    selectOnly,
    undo,
    redo,
    history,
    future
  } = useScene();

  const [step, setStep] = useState(0.25);
  const [sx, setSx] = useState(0);
  const [sz, setSz] = useState(0);

  useEffect(() => {
    if (spotlight?.position) {
      setSx(Number(spotlight.position[0].toFixed(2)));
      setSz(Number(spotlight.position[2].toFixed(2)));
    }
  }, [spotlight.position]);

  const cubes = Object.values(entities).filter(e => e.kind === 'cube');
  const selectedIndex = selected.length > 0 ? cubes.findIndex(cube => cube.id === selected[0]) : -1;

  return (
    <Card className="bg-gray-900 text-white border-gray-800">
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Cube Scene Control</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        {/* History Controls */}
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="secondary"
            onClick={undo}
            disabled={history.length === 0}
          >
            Undo ({history.length})
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={redo}
            disabled={future.length === 0}
          >
            Redo ({future.length})
          </Button>
        </div>

        {/* Entity Management */}
        <div className="flex gap-2">
          <Button size="sm" onClick={addCube}>Add Cube</Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={deleteSelected}
            disabled={selectedIndex === -1}
          >
            Delete Selected
          </Button>
        </div>

        {/* Animation Speed */}
        <div>
          <div className="mb-1">Animation Speed: {animationSpeed.toFixed(2)}</div>
          <Slider
            value={[animationSpeed]}
            min={0}
            max={2}
            step={0.01}
            onValueChange={([v]) => setAnimationSpeed(v)}
          />
        </div>

        {/* Spotlight Toggle */}
        <div className="flex items-center justify-between">
          <div>Spotlight: {spotlight.enabled ? 'On' : 'Off'}</div>
          <Button
            size="sm"
            variant={spotlight.enabled ? 'secondary' : 'default'}
            onClick={toggleSpotlight}
          >
            Toggle
          </Button>
        </div>

        {/* Programmatic spotlight controls */}
        <div className="space-y-2 p-2 rounded-md bg-gray-800/60">
          <div className="flex items-center justify-between">
            <div className="opacity-80">Spot pos (X,Z)</div>
            <div className="flex gap-2">
              <input
                className="w-16 bg-black/40 border border-gray-700 rounded px-2 py-1 text-right text-white"
                value={sx}
                onChange={(e) => setSx(parseFloat(e.target.value) || 0)}
              />
              <input
                className="w-16 bg-black/40 border border-gray-700 rounded px-2 py-1 text-right text-white"
                value={sz}
                onChange={(e) => setSz(parseFloat(e.target.value) || 0)}
              />
              <Button
                size="sm"
                variant="secondary"
                onClick={() => setSpotlightPositionDirect(Number(sx) || 0, 1, Number(sz) || 0)}
              >
                Set
              </Button>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="opacity-80">Nudge step: {step.toFixed(2)}</div>
            <Slider
              value={[step]}
              min={0.05}
              max={1}
              step={0.05}
              onValueChange={([v]) => setStep(v)}
            />
          </div>

          <div className="grid grid-cols-3 gap-1 place-items-center">
            <div />
            <Button size="sm" onClick={() => nudgeSpotlight(0, -step)}>↑</Button>
            <div />
            <Button size="sm" onClick={() => nudgeSpotlight(-step, 0)}>←</Button>
            <Button size="sm" variant="secondary" onClick={() => setSpotlightPositionDirect(0, 1, 0)}>Center</Button>
            <Button size="sm" onClick={() => nudgeSpotlight(step, 0)}>→</Button>
            <div />
            <Button size="sm" onClick={() => nudgeSpotlight(0, step)}>↓</Button>
            <div />
          </div>
        </div>

        {/* Cube Selection */}
        <div>
          <div className="mb-1">Select Cube</div>
          <Select
            value={selectedIndex === -1 ? "none" : String(selectedIndex)}
            onValueChange={(v) => {
              const index = v === "none" ? -1 : parseInt(v, 10);
              const entityId = index === -1 ? undefined : cubes[index]?.id;
              selectOnly(entityId);
            }}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="None" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              {cubes.map((cube, i) => (
                <SelectItem key={cube.id} value={String(i)}>
                  {cube.name || `Cube ${i}`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Entity Count Display */}
        <div className="text-xs opacity-70 pt-2 border-t border-gray-700">
          Entities: {Object.keys(entities).length} | Selected: {selected.length}
        </div>
      </CardContent>
    </Card>
  );
}
