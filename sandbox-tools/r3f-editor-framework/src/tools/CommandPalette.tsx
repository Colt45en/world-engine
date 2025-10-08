import React, { useState, useEffect } from 'react';
import { useScene } from '../state/store';
import { CommandItem } from './types';

// Fallback UI components
const Dialog = ({ open, onOpenChange, children }) => {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && open) {
        onOpenChange(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [open, onOpenChange]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-start justify-center pt-20 z-50">
      <div className="bg-gray-800 rounded-lg border border-gray-600 min-w-[500px] max-h-[60vh] overflow-hidden">
        {children}
      </div>
    </div>
  );
};

const Input = ({ className = "", ...props }) => (
  <input
    className={`w-full p-3 bg-gray-700 border border-gray-600 rounded-t-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 ${className}`}
    {...props}
  />
);

export default function CommandPalette({ open, onOpenChange }) {
  const [search, setSearch] = useState('');
  const { addCube, deleteSelected, toggleSpotlight, undo, redo, history, future, selected } = useScene();

  const commands: CommandItem[] = [
    {
      id: 'add-cube',
      label: 'Add Cube',
      description: 'Add a new cube to the scene',
      shortcut: 'Ctrl+N',
      category: 'Create',
      action: () => {
        addCube();
        onOpenChange(false);
      }
    },
    {
      id: 'delete-selected',
      label: 'Delete Selected',
      description: 'Delete the currently selected entities',
      shortcut: 'Delete',
      category: 'Edit',
      action: () => {
        deleteSelected();
        onOpenChange(false);
      },
      isEnabled: selected.length > 0
    },
    {
      id: 'toggle-spotlight',
      label: 'Toggle Spotlight',
      description: 'Enable or disable the scene spotlight',
      shortcut: 'L',
      category: 'Lighting',
      action: () => {
        toggleSpotlight();
        onOpenChange(false);
      }
    },
    {
      id: 'undo',
      label: 'Undo',
      description: 'Undo the last action',
      shortcut: 'Ctrl+Z',
      category: 'Edit',
      action: () => {
        undo();
        onOpenChange(false);
      },
      isEnabled: history.length > 0
    },
    {
      id: 'redo',
      label: 'Redo',
      description: 'Redo the last undone action',
      shortcut: 'Ctrl+Shift+Z',
      category: 'Edit',
      action: () => {
        redo();
        onOpenChange(false);
      },
      isEnabled: future.length > 0
    }
  ];

  const filteredCommands = commands.filter(cmd => {
    const searchLower = search.toLowerCase();
    return cmd.label.toLowerCase().includes(searchLower) ||
           cmd.description?.toLowerCase().includes(searchLower) ||
           cmd.category.toLowerCase().includes(searchLower);
  });

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && filteredCommands.length > 0) {
      const firstEnabled = filteredCommands.find(cmd => cmd.isEnabled !== false);
      if (firstEnabled) {
        firstEnabled.action();
      }
    }
  };

  // Reset search when opening
  useEffect(() => {
    if (open) {
      setSearch('');
    }
  }, [open]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <Input
        placeholder="Type a command or search..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        onKeyDown={handleKeyDown}
        autoFocus
      />

      <div className="max-h-80 overflow-y-auto">
        {filteredCommands.length === 0 ? (
          <div className="p-4 text-gray-400 text-center">
            No commands found
          </div>
        ) : (
          <div className="py-2">
            {filteredCommands.map((cmd) => {
              const isDisabled = cmd.isEnabled === false;
              return (
                <div
                  key={cmd.id}
                  className={`
                    px-4 py-2 cursor-pointer flex items-center justify-between
                    ${isDisabled
                      ? 'opacity-50 cursor-not-allowed'
                      : 'hover:bg-gray-700 text-white'
                    }
                  `}
                  onClick={() => !isDisabled && cmd.action()}
                >
                  <div>
                    <div className="font-medium">{cmd.label}</div>
                    {cmd.description && (
                      <div className="text-sm text-gray-400">{cmd.description}</div>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500 bg-gray-700 px-2 py-1 rounded">
                      {cmd.category}
                    </span>
                    {cmd.shortcut && (
                      <span className="text-xs text-gray-400 bg-gray-600 px-2 py-1 rounded">
                        {cmd.shortcut}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="p-2 border-t border-gray-600 text-xs text-gray-400">
        Press <kbd className="bg-gray-600 px-1 rounded">Enter</kbd> to execute, <kbd className="bg-gray-600 px-1 rounded">Esc</kbd> to close
      </div>
    </Dialog>
  );
}
