import React, { useState, useEffect } from 'react';
import { Tool } from './types';
import CubeScenePanel from './CubeScenePanel';
import SceneGraphInspector from './SceneGraphInspector';
import TransformTool from './TransformTool';
import CommandPalette from './CommandPalette';

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
    outline: "border border-gray-600 bg-transparent hover:bg-gray-700 text-white",
    ghost: "bg-transparent hover:bg-gray-700 text-white"
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

// Built-in tools
const builtInTools: Tool[] = [
  {
    id: 'transform',
    label: 'Transform',
    description: 'Transform selected entities',
    icon: '🔧',
    shortcut: 'T',
    category: 'edit',
    enable: () => console.log('Transform tool enabled'),
    disable: () => console.log('Transform tool disabled'),
    Panel: TransformTool
  },
  {
    id: 'scene-graph',
    label: 'Scene Graph',
    description: 'Inspect and manage scene hierarchy',
    icon: '🌳',
    shortcut: 'G',
    category: 'scene',
    enable: () => console.log('Scene Graph enabled'),
    disable: () => console.log('Scene Graph disabled'),
    Panel: SceneGraphInspector
  },
  {
    id: 'cube-scene',
    label: 'Cube Scene',
    description: 'Control cube scene parameters',
    icon: '🎮',
    shortcut: 'C',
    category: 'scene',
    enable: () => console.log('Cube Scene panel enabled'),
    disable: () => console.log('Cube Scene panel disabled'),
    Panel: CubeScenePanel
  }
];

export function ToolManager() {
  const [enabledTools, setEnabledTools] = useState<Set<string>>(new Set(['cube-scene']));
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Command palette
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(true);
        return;
      }

      // Tool shortcuts
      if (e.altKey) {
        const tool = builtInTools.find(t => t.shortcut?.toLowerCase() === e.key.toLowerCase());
        if (tool) {
          e.preventDefault();
          toggleTool(tool.id);
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const toggleTool = (toolId: string) => {
    const tool = builtInTools.find(t => t.id === toolId);
    if (!tool) return;

    setEnabledTools(prev => {
      const newSet = new Set(prev);
      if (newSet.has(toolId)) {
        newSet.delete(toolId);
        tool.disable();
      } else {
        newSet.add(toolId);
        tool.enable();
      }
      return newSet;
    });
  };

  const enabledToolsList = Array.from(enabledTools)
    .map(id => builtInTools.find(t => t.id === id))
    .filter(Boolean);

  return (
    <div className="space-y-4">
      {/* Tool Bar */}
      <Card className="bg-gray-900 border-gray-800">
        <CardHeader className="py-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Tools</CardTitle>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setCommandPaletteOpen(true)}
              className="text-xs"
            >
              Ctrl+K
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="grid grid-cols-1 gap-1">
            {builtInTools.map(tool => {
              const isEnabled = enabledTools.has(tool.id);
              return (
                <Button
                  key={tool.id}
                  size="sm"
                  variant={isEnabled ? 'default' : 'outline'}
                  onClick={() => toggleTool(tool.id)}
                  className="justify-start text-xs"
                >
                  {tool.icon && <span className="mr-2">{tool.icon}</span>}
                  {tool.label}
                  {tool.shortcut && (
                    <span className="ml-auto text-xs opacity-50">
                      Alt+{tool.shortcut}
                    </span>
                  )}
                </Button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Tool Panels */}
      {enabledToolsList.map(tool => {
        if (!tool?.Panel) return null;
        return (
          <div key={tool.id}>
            <tool.Panel />
          </div>
        );
      })}

      {/* Command Palette */}
      <CommandPalette
        open={commandPaletteOpen}
        onOpenChange={setCommandPaletteOpen}
      />
    </div>
  );
}
