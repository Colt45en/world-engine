"""
World Engine Python Package
Multi-language game engine with Python bindings
"""

__version__ = '1.0.0'
__author__ = 'World Engine Team'

# Try to import the C++ core module
try:
    from . import _core
    
    # Export core classes
    Engine = _core.Engine
    Entity = _core.Entity
    Component = _core.Component
    System = _core.System
    Scene = _core.Scene
    
except ImportError:
    # Fallback to pure Python implementation if C++ module is not available
    import warnings
    warnings.warn("C++ core module not available, using pure Python implementation")
    
    class Engine:
        """Pure Python Engine implementation"""
        def __init__(self):
            self.running = False
            self.active_scene = None
        
        def initialize(self):
            self.running = True
            return True
        
        def run(self):
            pass
        
        def shutdown(self):
            self.running = False
    
    class Entity:
        """Pure Python Entity implementation"""
        def __init__(self, entity_id=None):
            self.id = entity_id or 0
            self.name = "Entity"
            self.active = True
            self.components = []
    
    class Component:
        """Pure Python Component implementation"""
        def __init__(self):
            self.id = 0
            self.enabled = True
    
    class System:
        """Pure Python System implementation"""
        def __init__(self):
            self.id = 0
            self.entities = []
    
    class Scene:
        """Pure Python Scene implementation"""
        def __init__(self, name="Scene"):
            self.name = name
            self.entities = []
            self.systems = []
            self.next_entity_id = 1
        
        def create_entity(self, name="Entity"):
            """Create a new entity"""
            entity = Entity(self.next_entity_id)
            entity.name = name
            self.next_entity_id += 1
            self.entities.append(entity)
            return entity

# Utility functions
def get_version():
    """Get World Engine version"""
    return __version__

def print_info():
    """Print World Engine information"""
    print("=" * 40)
    print("World Engine")
    print(f"Version: {__version__}")
    print("Multi-language game engine")
    print("Supports: C++, Python, TypeScript, HTML")
    print("=" * 40)

__all__ = [
    'Engine',
    'Entity',
    'Component',
    'System',
    'Scene',
    'get_version',
    'print_info',
]
