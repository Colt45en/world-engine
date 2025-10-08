export interface Tool {
    id: string;
    label: string;
    description?: string;
    icon?: string;
    shortcut?: string;
    enable: () => void;
    disable: () => void;
    Panel?: React.ComponentType<any>;
    isActive?: boolean;
    category?: 'scene' | 'edit' | 'view' | 'debug';
}

export interface ToolState {
    activeTool: string | null;
    enabledTools: Set<string>;
    panelVisibility: Record<string, boolean>;
}

export interface CommandItem {
    id: string;
    label: string;
    description?: string;
    shortcut?: string;
    category: string;
    action: () => void;
    isEnabled?: boolean;
}

export type Vec3 = [number, number, number];
export type EntityId = string;

export interface TransformData {
    position: Vec3;
    rotation: Vec3;
    scale: Vec3;
}
