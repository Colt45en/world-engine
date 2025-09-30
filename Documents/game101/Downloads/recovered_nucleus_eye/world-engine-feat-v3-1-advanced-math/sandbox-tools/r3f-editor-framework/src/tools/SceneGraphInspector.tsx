import React from 'react';
import { useScene } from '../state/store';

// Fallback UI components
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

const Button = ({ children, onClick, size = "default", variant = "default", disabled = false, className = "", ...props }) => {
  const sizeClasses = {
    sm: "px-3 py-1.5 text-sm",
    default: "px-4 py-2",
    lg: "px-8 py-3 text-lg"
  };

  const variantClasses = {
    default: "bg-blue-600 hover:bg-blue-700 text-white",
    secondary: "bg-gray-600 hover:bg-gray-700 text-white",
    destructive: "bg-red-600 hover:bg-red-700 text-white",
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

export default function SceneGraphInspector() {
  const { entities, selected, selectOnly, updateEntity, removeEntity } = useScene();

  const entityList = Object.values(entities);

  return (
    <Card className="bg-gray-900 text-white border-gray-800">
      <CardHeader className="py-2">
        <CardTitle className="text-sm">Scene Graph</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        {entityList.length === 0 ? (
          <div className="opacity-50 text-center py-4">No entities in scene</div>
        ) : (
          <div className="space-y-1">
            {entityList.map((entity) => {
              const isSelected = selected.includes(entity.id);
              return (
                <div
                  key={entity.id}
                  className={`
                    p-2 rounded border cursor-pointer transition-colors
                    ${isSelected
                      ? 'border-blue-500 bg-blue-500/20'
                      : 'border-gray-600 hover:border-gray-500'
                    }
                  `}
                  onClick={() => selectOnly(entity.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`
                        w-2 h-2 rounded-full
                        ${entity.kind === 'cube' ? 'bg-blue-400' : 'bg-gray-400'}
                      `} />
                      <span className="font-medium">
                        {entity.name || entity.id}
                      </span>
                      <span className="opacity-50 text-xs">
                        ({entity.kind})
                      </span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeEntity(entity.id);
                      }}
                      className="text-red-400 hover:text-red-300"
                    >
                      ×
                    </Button>
                  </div>

                  {isSelected && (
                    <div className="mt-2 space-y-1 text-xs opacity-75">
                      <div>Position: {entity.position.map(v => v.toFixed(2)).join(', ')}</div>
                      <div>Rotation: {entity.rotation.map(v => v.toFixed(2)).join(', ')}</div>
                      <div>Scale: {entity.scale.map(v => v.toFixed(2)).join(', ')}</div>
                      {entity.color && <div>Color: {entity.color}</div>}
                      <div>Visible: {entity.visible ? 'Yes' : 'No'}</div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        <div className="pt-2 border-t border-gray-700 text-xs opacity-50">
          Total entities: {entityList.length}
        </div>
      </CardContent>
    </Card>
  );
}
