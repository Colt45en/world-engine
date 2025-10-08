// Nexus Widget Integration System
import * as React from 'react';
import { nexusWidgetBridge } from './NexusWidgetBridge';

interface WidgetConfig {
  id: string;
  title: string;
  url: string;
  icon: string;
  description: string;
  category: 'core' | 'tool' | 'demo' | 'analytics';
  size: 'small' | 'medium' | 'large';
  embedded: boolean;
}

interface NexusConnection {
  isConnected: boolean;
  widgets: WidgetConfig[];
  lastSync: Date | null;
  bridge: any;
}

export class NexusWidgetManager {
  private widgets: WidgetConfig[] = [
    {
      id: 'nexus-dashboard',
      title: 'Nexus Dashboard Widget',
      url: './docs/technical/portable_widgets/nexus-dashboard-widget.html',
      icon: '🚀',
      description: 'Main Nexus control center with system monitoring',
      category: 'core',
      size: 'large',
      embedded: true
    },
    {
      id: 'nexus-bot',
      title: 'Nexus AI Bot',
      url: './docs/technical/portable_widgets/nexus-bot-standalone.html',
      icon: '🤖',
      description: 'Standalone AI chat interface',
      category: 'tool',
      size: 'medium',
      embedded: true
    },
    {
      id: 'master-demo',
      title: 'Master Integration Demo',
      url: './src/demos/master-demo.html',
      icon: '🎵',
      description: 'NEXUS Holy Beat Master System',
      category: 'demo',
      size: 'large',
      embedded: false
    },
    {
      id: 'nexus-forge',
      title: 'Nexus Forge',
      url: './public/nexus-forge.html',
      icon: '🔧',
      description: '3D environment builder and configurator',
      category: 'tool',
      size: 'large',
      embedded: false
    },
    {
      id: 'nexus-math-academy',
      title: 'Nexus Math Academy',
      url: './public/nexus-math-academy.html',
      icon: '🧮',
      description: 'Interactive math learning environment',
      category: 'demo',
      size: 'medium',
      embedded: false
    },
    {
      id: 'nexus-math-academy-connected',
      title: 'Math Academy (Connected)',
      url: './public/nexus-math-academy-connected.html',
      icon: '🧮⚡',
      description: 'Math Academy with full Nexus AI bridge integration',
      category: 'demo',
      size: 'medium',
      embedded: true
    },
    {
      id: 'nexus-3d-sculptor',
      title: 'Nexus 3D Sculptor',
      url: './public/nexus-3d-sculptor.html',
      icon: '🎯',
      description: 'AI-enhanced silhouette-to-mesh 3D modeling with marching cubes',
      category: 'tool',
      size: 'large',
      embedded: true
    },
    {
      id: 'vector-forge-complete',
      title: 'Vector Forge Complete',
      url: './public/vector-forge-complete.html',
      icon: '🔥',
      description: 'Complete creative development suite with 3D, audio, video, AI integration',
      category: 'tool',
      size: 'large',
      embedded: true
    },
    {
      id: 'nexus-physics',
      title: 'Nexus Integrated Math & Physics',
      url: './public/nexus-integrated-math-physics.html',
      icon: '⚛️',
      description: 'Physics simulation and visualization',
      category: 'demo',
      size: 'large',
      embedded: false
    },
    {
      id: 'vector-forge-unified',
      title: 'Vector Forge Unified Engine',
      url: './public/vector-forge-unified-engine.html',
      icon: '🎛️',
      description: 'Unified vector engine with Heart Engine, timeline, and advanced rendering',
      category: 'tool',
      size: 'large',
      embedded: true
    },
    {
      id: 'r3f-toolbar',
      title: 'R3F Compact Toolbar',
      url: './src/components/R3FCompactToolbar.tsx',
      icon: '🔧',
      description: 'Compact toggleable toolbar for React Three Fiber with Vector Forge integration',
      category: 'tool',
      size: 'small',
      embedded: true
    }
  ];

  getWidgets(): WidgetConfig[] {
    return [...this.widgets];
  }

  getWidgetsByCategory(category: string): WidgetConfig[] {
    return this.widgets.filter(w => w.category === category);
  }

  getWidget(id: string): WidgetConfig | undefined {
    return this.widgets.find(w => w.id === id);
  }

  // Bridge communication with widgets
  async connectToWidget(widgetId: string, bridge: any): Promise<boolean> {
    const widget = this.getWidget(widgetId);
    if (!widget) return false;

    try {
      // Send bridge reference to widget iframe
      const iframe = document.getElementById(`widget-${widgetId}`) as HTMLIFrameElement;
      if (iframe && iframe.contentWindow) {
        iframe.contentWindow.postMessage({
          type: 'NEXUS_BRIDGE_CONNECT',
          bridge: {
            query: bridge.query.bind(bridge),
            startTraining: bridge.startTraining.bind(bridge),
            stopTraining: bridge.stopTraining.bind(bridge),
            getStatus: () => bridge.trainingState
          }
        }, '*');
      }
      return true;
    } catch (error) {
      console.error('Failed to connect to widget:', error);
      return false;
    }
  }

  // Register new widget
  registerWidget(widget: WidgetConfig): void {
    const existingIndex = this.widgets.findIndex(w => w.id === widget.id);
    if (existingIndex >= 0) {
      this.widgets[existingIndex] = widget;
    } else {
      this.widgets.push(widget);
    }
  }
}

// React Hook for Nexus Integration
export function useNexusIntegration(): {
  widgets: WidgetConfig[];
  connection: NexusConnection;
  openWidget: (widgetId: string) => void;
  embedWidget: (widgetId: string) => React.ReactElement | null;
  connectWidget: (widgetId: string, bridge: any) => Promise<boolean>;
} {
  const [widgetManager] = React.useState(() => new NexusWidgetManager());
  const [widgets, setWidgets] = React.useState<WidgetConfig[]>([]);
  const [connection, setConnection] = React.useState<NexusConnection>({
    isConnected: false,
    widgets: [],
    lastSync: null,
    bridge: null
  });

  React.useEffect(() => {
    setWidgets(widgetManager.getWidgets());
    setConnection(prev => ({
      ...prev,
      isConnected: true,
      widgets: widgetManager.getWidgets(),
      lastSync: new Date()
    }));

    // Initialize bridge connection
    nexusWidgetBridge.setBridge(connection.bridge);

    // Listen for widget messages
    const handleMessage = (event: MessageEvent) => {
      if (event.data.type === 'NEXUS_WIDGET_READY') {
        console.log('Widget ready:', event.data.widgetId);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [widgetManager, connection.bridge]);

  const openWidget = (widgetId: string) => {
    const widget = widgetManager.getWidget(widgetId);
    if (widget) {
      window.open(widget.url, '_blank', 'width=1200,height=800');
    }
  };

  const embedWidget = (widgetId: string): React.ReactElement | null => {
    const widget = widgetManager.getWidget(widgetId);
    if (!widget || !widget.embedded) return null;

    // Returns the height for the widget iframe based on its size
    function getWidgetHeight(size: string): string {
      if (size === 'large') return '600px';
      if (size === 'medium') return '400px';
      return '300px';
    }

    return (
      <iframe
        id={`widget-${widgetId}`}
        src={widget.url}
        title={widget.title}
        className="nexus-widget-embed"
        style={{
          width: '100%',
          height: getWidgetHeight(widget.size),
          border: 'none',
          borderRadius: '12px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }}
        sandbox="allow-scripts allow-same-origin allow-forms"
      />
    );
  };

  const connectWidget = async (widgetId: string, bridge: any): Promise<boolean> => {
    const success = await widgetManager.connectToWidget(widgetId, bridge);
    if (success) {
      setConnection(prev => ({ ...prev, bridge }));
    }
    return success;
  };

  return {
    widgets,
    connection,
    openWidget,
    embedWidget,
    connectWidget
  };
}

// Nexus Widget Dashboard Component
export function NexusWidgetDashboard({ bridge }: { bridge: any }) {
  const { widgets, connection, openWidget, embedWidget, connectWidget } = useNexusIntegration();
  const [activeWidget, setActiveWidget] = React.useState<string | null>(null);
  const [embeddedWidgets, setEmbeddedWidgets] = React.useState<Set<string>>(new Set());


  // Initialize bridge when component mounts
  React.useEffect(() => {
    nexusWidgetBridge.setBridge(bridge);
  }, [bridge]);

  const handleWidgetToggle = (widgetId: string) => {
    const widget = widgets.find(w => w.id === widgetId);
    if (!widget) return;

    if (widget.embedded) {
      setEmbeddedWidgets(prev => {
        const newSet = new Set(prev);
        if (newSet.has(widgetId)) {
          newSet.delete(widgetId);
        } else {
          newSet.add(widgetId);
          // Connect bridge to widget
          connectWidget(widgetId, bridge);
        }
        return newSet;
      });
    } else {
      openWidget(widgetId);
    }
  };

  const categoryGroups = React.useMemo(() => {
    const groups: Record<string, WidgetConfig[]> = {};
    widgets.forEach(widget => {
      if (!groups[widget.category]) {
        groups[widget.category] = [];
      }
      groups[widget.category].push(widget);
    });
    return groups;
  }, [widgets]);

  return (
    <div className="nexus-widget-dashboard">
      {/* Connection Status */}
      <div className="nexus-connection-status">
        <div className={`status-indicator ${connection.isConnected ? 'connected' : 'disconnected'}`}>
          {connection.isConnected ? '🟢' : '🔴'}
        </div>
        <span>Nexus Widgets: {connection.isConnected ? 'Connected' : 'Disconnected'}</span>
        <span className="widget-count">({widgets.length} available)</span>
      </div>

      {/* Widget Categories */}
      <div className="widget-categories">
        {Object.entries(categoryGroups).map(([category, categoryWidgets]) => (
          <div key={category} className="widget-category">
            <h3 className="category-title">
              {category.charAt(0).toUpperCase() + category.slice(1)} Widgets
            </h3>
            <div className="widget-grid">
              {categoryWidgets.map(widget => (
                <button
                  key={widget.id}
                  className={`widget-card ${embeddedWidgets.has(widget.id) ? 'active' : ''}`}
                  onClick={() => handleWidgetToggle(widget.id)}
                  aria-label={`${embeddedWidgets.has(widget.id) ? 'Remove' : 'Add'} ${widget.title} widget - ${widget.description}`}
                >
                  <div className="widget-icon">{widget.icon}</div>
                  <div className="widget-info">
                    <h4 className="widget-title">{widget.title}</h4>
                    <p className="widget-description">{widget.description}</p>
                    <div className="widget-badges">
                      <span className={`badge badge-${widget.category}`}>{widget.category}</span>
                      <span className={`badge badge-${widget.size}`}>{widget.size}</span>
                      {widget.embedded && <span className="badge badge-embedded">embeddable</span>}
                    </div>
                  </div>
                  <div className="widget-actions">
                    {widget.embedded ? (
                      <button className="action-btn">
                        {embeddedWidgets.has(widget.id) ? '✖️ Close' : '🔗 Embed'}
                      </button>
                    ) : (
                      <button className="action-btn">🚀 Open</button>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Embedded Widgets */}
      {embeddedWidgets.size > 0 && (
        <div className="embedded-widgets-container">
          <h3>📱 Embedded Widgets</h3>
          <div className="embedded-widgets">
            {Array.from(embeddedWidgets).map(widgetId => (
              <div key={widgetId} className="embedded-widget">
                <div className="embedded-widget-header">
                  <span>{widgets.find(w => w.id === widgetId)?.title}</span>
                  <button
                    onClick={() => setEmbeddedWidgets(prev => {
                      const newSet = new Set(prev);
                      newSet.delete(widgetId);
                      return newSet;
                    })}
                    className="close-widget-btn"
                  >
                    ✖️
                  </button>
                </div>
                {embedWidget(widgetId)}
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        .nexus-widget-dashboard {
          padding: 20px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .nexus-connection-status {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 15px 20px;
          background: rgba(255,255,255,0.1);
          border-radius: 12px;
          margin-bottom: 20px;
          backdrop-filter: blur(10px);
        }

        .status-indicator {
          font-size: 1.2em;
        }

        .widget-count {
          margin-left: auto;
          color: #666;
          font-size: 0.9em;
        }

        .widget-category {
          margin-bottom: 30px;
        }

        .category-title {
          font-size: 1.5em;
          margin-bottom: 15px;
          color: #333;
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .widget-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
          gap: 20px;
        }

        .widget-card {
          background: rgba(255,255,255,0.95);
          border-radius: 15px;
          padding: 20px;
          cursor: pointer;
          transition: all 0.3s ease;
          border: 2px solid transparent;
          backdrop-filter: blur(10px);
        }

        .widget-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 10px 25px rgba(0,0,0,0.1);
          border-color: #667eea;
        }

        .widget-card.active {
          border-color: #667eea;
          background: rgba(102, 126, 234, 0.1);
        }

        .widget-icon {
          font-size: 2.5em;
          margin-bottom: 15px;
        }

        .widget-title {
          font-size: 1.2em;
          margin-bottom: 8px;
          color: #333;
        }

        .widget-description {
          color: #666;
          font-size: 0.9em;
          margin-bottom: 15px;
          line-height: 1.4;
        }

        .widget-badges {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          margin-bottom: 15px;
        }

        .badge {
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 0.75em;
          font-weight: 600;
          text-transform: uppercase;
        }

        .badge-core { background: #667eea; color: white; }
        .badge-tool { background: #f093fb; color: white; }
        .badge-demo { background: #4facfe; color: white; }
        .badge-analytics { background: #43e97b; color: white; }
        .badge-small { background: #ffeaa7; color: #333; }
        .badge-medium { background: #fab1a0; color: white; }
        .badge-large { background: #fd79a8; color: white; }
        .badge-embedded { background: #00b894; color: white; }

        .widget-actions {
          display: flex;
          justify-content: flex-end;
        }

        .action-btn {
          padding: 8px 16px;
          border: none;
          border-radius: 8px;
          background: #667eea;
          color: white;
          cursor: pointer;
          font-size: 0.9em;
          transition: background 0.3s;
        }

        .action-btn:hover {
          background: #5a67d8;
        }

        .embedded-widgets-container {
          margin-top: 40px;
          padding-top: 30px;
          border-top: 2px solid rgba(255,255,255,0.2);
        }

        .embedded-widgets {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .embedded-widget {
          background: rgba(255,255,255,0.95);
          border-radius: 15px;
          padding: 20px;
          backdrop-filter: blur(10px);
        }

        .embedded-widget-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
          font-weight: 600;
          color: #333;
        }

        .close-widget-btn {
          background: none;
          border: none;
          cursor: pointer;
          font-size: 1.2em;
          padding: 5px;
          border-radius: 5px;
          transition: background 0.3s;
        }

        .close-widget-btn:hover {
          background: rgba(255,0,0,0.1);
        }
      `}</style>
    </div>
  );
}
