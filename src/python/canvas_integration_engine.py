# CONVERSATIONAL IDE - CANVAS NODE EDITOR INTEGRATION
# Teaching AI to understand and manipulate visual node graphs and 3D shapes
# Extends the conversational engine with canvas-specific capabilities

import json
import os
import webbrowser
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from conversational_ide_engine import ConversationalIDEEngine
from advanced_input_processor import AdvancedInputProcessor

class CanvasNodeEditorEngine:
    """Conversational AI engine specialized for visual node editing and 3D shape creation"""

    def __init__(self):
        # Initialize base conversational components
        self.ide_engine = ConversationalIDEEngine()
        self.input_processor = AdvancedInputProcessor()

        # Canvas-specific capabilities
        self.canvas_commands = self._initialize_canvas_commands()
        self.shape_templates = self._initialize_shape_templates()
        self.current_session = {
            "nodes": [],
            "edges": [],
            "canvas_state": "drawing",  # "drawing" | "editing_nodes"
            "active_tool": "draw",
            "grid_visible": True
        }

    def _initialize_canvas_commands(self) -> Dict[str, Any]:
        """Define natural language commands for canvas operations"""
        return {
            # Node operations
            "create_node": {
                "patterns": ["create node", "add node", "make node", "place node"],
                "params": ["position", "size", "color"],
                "description": "Create a new node at specified position"
            },
            "connect_nodes": {
                "patterns": ["connect", "link", "join", "connect nodes", "draw line between"],
                "params": ["node1", "node2", "line_style"],
                "description": "Create edge between two nodes"
            },
            "move_node": {
                "patterns": ["move", "drag", "reposition", "relocate"],
                "params": ["node_id", "new_position"],
                "description": "Move node to new position"
            },
            "delete_node": {
                "patterns": ["delete", "remove", "erase node"],
                "params": ["node_id"],
                "description": "Remove node and its connections"
            },

            # Shape creation
            "create_3d_shape": {
                "patterns": ["create 3d", "make shape", "build geometry", "construct"],
                "params": ["shape_type", "dimensions", "material"],
                "description": "Generate 3D shape from node connections"
            },

            # Canvas tools
            "switch_tool": {
                "patterns": ["switch to", "use tool", "activate", "select tool"],
                "params": ["tool_name"],
                "description": "Change active drawing tool"
            },
            "toggle_grid": {
                "patterns": ["toggle grid", "show grid", "hide grid", "grid on/off"],
                "params": [],
                "description": "Toggle grid visibility"
            },
            "clear_canvas": {
                "patterns": ["clear", "reset", "erase all", "start over"],
                "params": ["layer"],
                "description": "Clear specified layer or entire canvas"
            },

            # Export/Save
            "save_canvas": {
                "patterns": ["save", "export", "download", "save image"],
                "params": ["format", "filename"],
                "description": "Export canvas as image or data"
            }
        }

    def _initialize_shape_templates(self) -> Dict[str, Any]:
        """Define 3D shape templates that can be created from node patterns"""
        return {
            "cube": {
                "nodes_required": 8,
                "connections": [
                    # Front face
                    [0,1], [1,2], [2,3], [3,0],
                    # Back face
                    [4,5], [5,6], [6,7], [7,4],
                    # Connecting edges
                    [0,4], [1,5], [2,6], [3,7]
                ],
                "description": "8 nodes arranged as cube vertices"
            },
            "pyramid": {
                "nodes_required": 5,
                "connections": [
                    # Base
                    [0,1], [1,2], [2,3], [3,0],
                    # Apex connections
                    [4,0], [4,1], [4,2], [4,3]
                ],
                "description": "4-sided pyramid with apex"
            },
            "tetrahedron": {
                "nodes_required": 4,
                "connections": [
                    [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]
                ],
                "description": "4 nodes, all connected"
            },
            "octahedron": {
                "nodes_required": 6,
                "connections": [
                    # Top pyramid
                    [0,1], [0,2], [0,3], [0,4],
                    # Middle square
                    [1,2], [2,3], [3,4], [4,1],
                    # Bottom pyramid
                    [5,1], [5,2], [5,3], [5,4]
                ],
                "description": "Two pyramids joined at base"
            }
        }

    def process_canvas_request(self, user_input: str) -> Dict[str, Any]:
        """Process natural language requests for canvas operations"""

        # Use the existing IDE engine to understand the request
        base_result = self.ide_engine.plan_and_respond(user_input)

        # Enhance with canvas-specific understanding
        canvas_understanding = self._analyze_canvas_intent(user_input)

        # Generate canvas-specific response
        canvas_response = self._generate_canvas_response(canvas_understanding, base_result)

        return {
            "success": True,
            "input": user_input,
            "base_understanding": base_result["understanding"] if base_result["success"] else {},
            "canvas_understanding": canvas_understanding,
            "canvas_actions": canvas_response.get("actions", []),
            "javascript_code": canvas_response.get("javascript", ""),
            "html_updates": canvas_response.get("html", ""),
            "response": canvas_response.get("description", ""),
            "session_state": self.current_session
        }

    def _analyze_canvas_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze input for canvas-specific operations"""

        input_lower = user_input.lower()
        detected_commands = []
        extracted_params = {}

        # Check for canvas command patterns
        for command_name, command_info in self.canvas_commands.items():
            for pattern in command_info["patterns"]:
                if pattern in input_lower:
                    detected_commands.append({
                        "command": command_name,
                        "confidence": 0.9,
                        "pattern_matched": pattern
                    })

        # Extract parameters and coordinates
        coordinates = self._extract_coordinates(user_input)
        colors = self._extract_colors(user_input)
        numbers = self._extract_numbers(user_input)
        shape_types = self._extract_shape_types(user_input)

        return {
            "detected_commands": detected_commands,
            "coordinates": coordinates,
            "colors": colors,
            "numbers": numbers,
            "shape_types": shape_types,
            "canvas_context": self._infer_canvas_context(user_input),
            "spatial_references": self._extract_spatial_references(user_input)
        }

    def _extract_coordinates(self, text: str) -> List[Tuple[float, float]]:
        """Extract coordinate pairs from text"""
        import re

        # Pattern for coordinates like (100, 200) or 100,200 or at 100 200
        coord_patterns = [
            r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)',  # (x,y)
            r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)',      # x,y
            r'at\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)',     # at x y
            r'position\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)' # position x y
        ]

        coordinates = []
        for pattern in coord_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    x, y = float(match[0]), float(match[1])
                    coordinates.append((x, y))
                except ValueError:
                    continue

        return coordinates

    def _extract_colors(self, text: str) -> List[str]:
        """Extract color names and hex codes from text"""
        import re

        # Common color names
        color_names = ["red", "blue", "green", "yellow", "purple", "orange", "pink",
                      "cyan", "magenta", "black", "white", "gray", "grey", "brown"]

        # Hex color pattern
        hex_pattern = r'#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}'

        colors = []
        text_lower = text.lower()

        for color in color_names:
            if color in text_lower:
                colors.append(color)

        hex_colors = re.findall(hex_pattern, text)
        colors.extend(hex_colors)

        return colors

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text"""
        import re

        # Pattern for numbers (including decimals)
        number_pattern = r'\d+(?:\.\d+)?'
        matches = re.findall(number_pattern, text)

        return [float(match) for match in matches]

    def _extract_shape_types(self, text: str) -> List[str]:
        """Extract 3D shape types mentioned in text"""
        text_lower = text.lower()
        mentioned_shapes = []

        for shape_name in self.shape_templates.keys():
            if shape_name in text_lower:
                mentioned_shapes.append(shape_name)

        # Additional shape terms
        shape_terms = ["sphere", "cylinder", "cone", "torus", "prism", "polyhedron"]
        for term in shape_terms:
            if term in text_lower:
                mentioned_shapes.append(term)

        return mentioned_shapes

    def _infer_canvas_context(self, text: str) -> str:
        """Infer the current canvas operation context"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["node", "vertex", "point", "connect"]):
            return "node_editing"
        elif any(word in text_lower for word in ["draw", "sketch", "paint", "brush"]):
            return "drawing"
        elif any(word in text_lower for word in ["3d", "shape", "geometry", "construct"]):
            return "3d_creation"
        else:
            return "general"

    def _extract_spatial_references(self, text: str) -> List[str]:
        """Extract spatial relationship terms"""
        spatial_terms = [
            "above", "below", "left", "right", "center", "middle", "corner",
            "top", "bottom", "side", "between", "near", "far", "close"
        ]

        text_lower = text.lower()
        found_terms = [term for term in spatial_terms if term in text_lower]
        return found_terms

    def _generate_canvas_response(self, canvas_understanding: Dict, base_result: Dict) -> Dict[str, Any]:
        """Generate canvas-specific response with JavaScript code"""

        detected_commands = canvas_understanding.get("detected_commands", [])

        if not detected_commands:
            return {
                "description": "I understand you want to work with the canvas. Could you be more specific about what you'd like to do?",
                "actions": [],
                "javascript": "",
                "html": ""
            }

        primary_command = detected_commands[0]["command"]

        # Generate response based on primary command
        if primary_command == "create_node":
            return self._generate_create_node_response(canvas_understanding)
        elif primary_command == "connect_nodes":
            return self._generate_connect_nodes_response(canvas_understanding)
        elif primary_command == "create_3d_shape":
            return self._generate_3d_shape_response(canvas_understanding)
        elif primary_command == "switch_tool":
            return self._generate_switch_tool_response(canvas_understanding)
        elif primary_command == "toggle_grid":
            return self._generate_toggle_grid_response()
        elif primary_command == "clear_canvas":
            return self._generate_clear_canvas_response(canvas_understanding)
        elif primary_command == "save_canvas":
            return self._generate_save_canvas_response()
        else:
            return self._generate_generic_canvas_response(canvas_understanding)

    def _generate_create_node_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for creating nodes"""
        coordinates = understanding.get("coordinates", [])
        colors = understanding.get("colors", ["#10e0e0"])  # default cyan

        if coordinates:
            x, y = coordinates[0]
            js_code = f"""
// Create node at specified position
if (typeof nodes !== 'undefined') {{
    nodes.push({{x: {x}, y: {y}}});
    redrawNodes();
    updatePreview();
    console.log('Node created at ({x}, {y})');
}}
"""
        else:
            js_code = """
// Create node at center
if (typeof nodes !== 'undefined') {
    const cx = W * 0.5, cy = H * 0.5;
    nodes.push({x: cx, y: cy});
    redrawNodes();
    updatePreview();
    console.log('Node created at center');
}
"""

        return {
            "description": f"I'll create a new node{'at the specified coordinates' if coordinates else ' at the center of the canvas'}.",
            "actions": ["create_node"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_connect_nodes_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for connecting nodes"""
        numbers = understanding.get("numbers", [])

        if len(numbers) >= 2:
            node1, node2 = int(numbers[0]), int(numbers[1])
            js_code = f"""
// Connect nodes {node1} and {node2}
if (typeof edges !== 'undefined' && typeof nodes !== 'undefined') {{
    if ({node1} < nodes.length && {node2} < nodes.length) {{
        edges.push([{node1}, {node2}]);
        redrawNodes();
        updatePreview();
        console.log('Connected node {node1} to node {node2}');
    }} else {{
        console.log('Node indices out of range');
    }}
}}
"""
        else:
            js_code = """
// Connect last two nodes
if (typeof edges !== 'undefined' && typeof nodes !== 'undefined' && nodes.length >= 2) {
    const last = nodes.length - 1;
    edges.push([last-1, last]);
    redrawNodes();
    updatePreview();
    console.log('Connected last two nodes');
}
"""

        return {
            "description": "I'll connect the specified nodes with an edge.",
            "actions": ["connect_nodes"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_3d_shape_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for creating 3D shapes"""
        shape_types = understanding.get("shape_types", [])

        if shape_types and shape_types[0] in self.shape_templates:
            shape_name = shape_types[0]
            template = self.shape_templates[shape_name]

            js_code = f"""
// Create {shape_name} structure
if (typeof nodes !== 'undefined' && typeof edges !== 'undefined') {{
    // Clear existing nodes/edges
    nodes = [];
    edges = [];

    const cx = W * 0.5, cy = H * 0.5, r = Math.min(W, H) * 0.2;

    // Generate {shape_name} nodes
    {self._generate_shape_nodes_js(shape_name, template)}

    // Generate {shape_name} edges
    const connections = {json.dumps(template["connections"])};
    edges = connections;

    redrawNodes();
    updatePreview();
    console.log('{shape_name.capitalize()} created with {{nodes.length}} nodes and {{edges.length}} edges');
}}
"""
        else:
            js_code = """
// Create generic 3D structure
if (typeof nodes !== 'undefined' && typeof edges !== 'undefined') {
    // Create a simple tetrahedron as default 3D shape
    nodes = [];
    edges = [];

    const cx = W * 0.5, cy = H * 0.5, r = Math.min(W, H) * 0.15;

    // Tetrahedron nodes
    nodes.push({x: cx, y: cy - r});        // top
    nodes.push({x: cx - r, y: cy + r});    // bottom left
    nodes.push({x: cx + r, y: cy + r});    // bottom right
    nodes.push({x: cx, y: cy + r * 0.3});  // center

    // Connect all nodes (tetrahedron)
    edges = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]];

    redrawNodes();
    updatePreview();
    console.log('3D tetrahedron created');
}
"""

        return {
            "description": f"I'll create a 3D {shape_types[0] if shape_types else 'tetrahedron'} structure using connected nodes.",
            "actions": ["create_3d_shape"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_shape_nodes_js(self, shape_name: str, template: Dict) -> str:
        """Generate JavaScript code for positioning shape nodes"""

        if shape_name == "cube":
            return """
    // Cube vertices
    nodes.push({x: cx - r, y: cy - r}); // 0: front bottom left
    nodes.push({x: cx + r, y: cy - r}); // 1: front bottom right
    nodes.push({x: cx + r, y: cy + r}); // 2: front top right
    nodes.push({x: cx - r, y: cy + r}); // 3: front top left
    nodes.push({x: cx - r*0.5, y: cy - r*0.5}); // 4: back bottom left
    nodes.push({x: cx + r*0.5, y: cy - r*0.5}); // 5: back bottom right
    nodes.push({x: cx + r*0.5, y: cy + r*0.5}); // 6: back top right
    nodes.push({x: cx - r*0.5, y: cy + r*0.5}); // 7: back top left
"""
        elif shape_name == "pyramid":
            return """
    // Pyramid base
    nodes.push({x: cx - r, y: cy + r});     // 0: base corner 1
    nodes.push({x: cx + r, y: cy + r});     // 1: base corner 2
    nodes.push({x: cx + r, y: cy + r*0.5}); // 2: base corner 3
    nodes.push({x: cx - r, y: cy + r*0.5}); // 3: base corner 4
    nodes.push({x: cx, y: cy - r});         // 4: apex
"""
        else:
            return """
    // Default positioning
    const angleStep = (Math.PI * 2) / nodeCount;
    for (let i = 0; i < nodeCount; i++) {
        const angle = i * angleStep;
        nodes.push({
            x: cx + Math.cos(angle) * r,
            y: cy + Math.sin(angle) * r
        });
    }
"""

    def _generate_switch_tool_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for switching tools"""
        # Try to infer tool from context
        spatial_refs = understanding.get("spatial_references", [])

        tool_keywords = {
            "draw": ["draw", "sketch", "paint", "brush"],
            "erase": ["erase", "delete", "remove"],
            "node": ["node", "edit", "connect"]
        }

        detected_tool = "draw"  # default
        for tool, keywords in tool_keywords.items():
            if any(keyword in understanding.get("canvas_context", "").lower() for keyword in keywords):
                detected_tool = tool
                break

        js_code = f"""
// Switch to {detected_tool} tool
if (typeof setTool !== 'undefined') {{
    setTool('{detected_tool}');
    console.log('Switched to {detected_tool} tool');
}}
"""

        return {
            "description": f"I'll switch to the {detected_tool} tool for you.",
            "actions": ["switch_tool"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_toggle_grid_response(self) -> Dict[str, Any]:
        """Generate response for toggling grid"""
        js_code = """
// Toggle grid visibility
if (typeof drawGrid !== 'undefined') {
    // Simple grid toggle - you can enhance this
    const gridCanvas = document.getElementById('gridCanvas');
    if (gridCanvas) {
        const isVisible = gridCanvas.style.opacity !== '0';
        gridCanvas.style.opacity = isVisible ? '0' : '1';
        console.log('Grid ' + (isVisible ? 'hidden' : 'shown'));
    }
}
"""

        return {
            "description": "I'll toggle the grid visibility for you.",
            "actions": ["toggle_grid"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_clear_canvas_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for clearing canvas"""
        js_code = """
// Clear canvas layers
if (typeof dctx !== 'undefined') {
    dctx.clearRect(0, 0, W, H);
    console.log('Drawing layer cleared');
}
if (typeof nodes !== 'undefined' && typeof edges !== 'undefined') {
    nodes = [];
    edges = [];
    if (typeof redrawNodes !== 'undefined') {
        redrawNodes();
    }
    console.log('Nodes and edges cleared');
}
if (typeof updatePreview !== 'undefined') {
    updatePreview();
}
"""

        return {
            "description": "I'll clear the canvas for you.",
            "actions": ["clear_canvas"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_save_canvas_response(self) -> Dict[str, Any]:
        """Generate response for saving canvas"""
        js_code = """
// Save canvas composite
if (typeof saveComposite !== 'undefined') {
    saveComposite();
    console.log('Canvas saved as composite image');
} else {
    // Fallback manual save
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');

    // Draw all layers
    const layers = ['gridCanvas', 'drawingCanvas', 'nodeCanvas'];
    layers.forEach(id => {
        const layer = document.getElementById(id);
        if (layer) ctx.drawImage(layer, 0, 0);
    });

    const a = document.createElement('a');
    a.download = 'canvas_export.png';
    a.href = canvas.toDataURL('image/png');
    a.click();
}
"""

        return {
            "description": "I'll save the current canvas as an image file.",
            "actions": ["save_canvas"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_generic_canvas_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate generic helpful response"""
        return {
            "description": "I can help you with canvas operations like creating nodes, connecting them, drawing 3D shapes, and managing the canvas. What would you like to do?",
            "actions": ["help"],
            "javascript": "console.log('Canvas helper ready');",
            "html": ""
        }

    def open_canvas_editor(self):
        """Open the canvas node editor in the browser"""
        html_path = os.path.join(os.path.dirname(__file__), "canvas_node_editor.html")

        if os.path.exists(html_path):
            webbrowser.open(f"file://{html_path}")
            return {"success": True, "message": "Canvas editor opened in browser"}
        else:
            return {"success": False, "message": "Canvas editor file not found"}

    def generate_canvas_code_injection(self, javascript_code: str) -> str:
        """Generate HTML/JS code that can be injected into the canvas editor"""
        return f"""
<script>
// Conversational AI Generated Code
(function() {{
    // Wait for canvas to be ready
    if (typeof W === 'undefined') {{
        setTimeout(arguments.callee, 100);
        return;
    }}

    try {{
        {javascript_code}
    }} catch(e) {{
        console.error('Canvas AI command error:', e);
    }}
}})();
</script>
"""

def demo_canvas_integration():
    """Demonstrate the canvas node editor integration"""

    print("🎨 CANVAS NODE EDITOR - CONVERSATIONAL AI INTEGRATION")
    print("=" * 70)
    print("Teaching AI to understand and manipulate visual node graphs")
    print("=" * 70)

    # Initialize the canvas engine
    canvas_engine = CanvasNodeEditorEngine()

    print("\n✅ Canvas Node Editor Engine initialized")
    print("✅ Conversational AI integration ready")
    print("✅ 3D shape templates loaded")

    # Test conversational commands
    test_commands = [
        "Create a node at position 300, 200",
        "Connect the first two nodes",
        "Make a cube shape with connected nodes",
        "Switch to the drawing tool",
        "Toggle the grid visibility",
        "Save the canvas as an image"
    ]

    print(f"\n🧪 Testing {len(test_commands)} conversational commands...\n")

    for i, command in enumerate(test_commands, 1):
        print(f"📝 Command {i}: '{command}'")
        print("-" * 50)

        result = canvas_engine.process_canvas_request(command)

        if result["success"]:
            print(f"🎯 Understanding: {', '.join([cmd['command'] for cmd in result['canvas_understanding']['detected_commands']])}")
            print(f"💡 Response: {result['response']}")
            print(f"⚡ Actions: {', '.join(result['canvas_actions'])}")

            if result["javascript_code"]:
                print(f"🖥️ Generated JavaScript:")
                print("```javascript")
                print(result["javascript_code"].strip())
                print("```")
        else:
            print(f"❌ Error processing command")

        print("\n" + "─" * 70 + "\n")

    # Show 3D shape capabilities
    print("🔷 AVAILABLE 3D SHAPE TEMPLATES:")
    for shape_name, template in canvas_engine.shape_templates.items():
        print(f"   • {shape_name.capitalize()}: {template['description']} ({template['nodes_required']} nodes)")

    print("\n🌐 Canvas editor integration ready!")
    print("   • Natural language understanding for canvas operations")
    print("   • JavaScript code generation for visual manipulation")
    print("   • 3D shape creation from node patterns")
    print("   • Real-time canvas state tracking")

    return canvas_engine

if __name__ == "__main__":
    demo_canvas_integration()
