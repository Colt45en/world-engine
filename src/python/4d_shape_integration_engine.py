# 4D SHAPE INTEGRATION ENGINE
# Conversational AI for 4D hypershape creation and manipulation
# Extends the canvas integration with inter-dimensional linking capabilities

import json
import os
import re
import webbrowser
import tempfile
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from canvas_integration_engine import CanvasNodeEditorEngine

class FourDimensionalShapeEngine:
    """Advanced conversational AI engine for 4D hypershape visualization and control"""

    def __init__(self):
        # Inherit 3D capabilities
        self.canvas_engine = CanvasNodeEditorEngine()

        # 4D-specific capabilities
        self.hypershape_templates = self._initialize_hypershape_templates()
        self.wall_commands = self._initialize_wall_commands()
        self.dimensional_linking = self._initialize_dimensional_linking()

        # 4D session state
        self.current_4d_session = {
            "back_wall": {"nodes": [], "edges": [], "type": "grid", "params": {}},
            "front_wall": {"nodes": [], "edges": [], "type": "radial", "params": {}},
            "interdimensional_links": [],
            "limbs_back": [],  # Dynamic oscillating limbs on back wall
            "limbs_front": [], # Dynamic oscillating limbs on front wall
            "active_wall": "front",
            "link_mode": False,
            "limb_spawn_armed": False,
            "wall_alphas": {"back": 0.55, "front": 0.75, "links": 0.90},
            "limb_settings": {"speed": 80, "max_angle": 45, "length": 100}
        }

    def _initialize_hypershape_templates(self) -> Dict[str, Any]:
        """Define 4D hypershape templates that can be created from dual-wall structures"""
        return {
            "tesseract": {
                "description": "4D hypercube with 16 vertices and interdimensional connections",
                "back_wall": {
                    "type": "grid",
                    "params": {"rows": 4, "cols": 4},
                    "description": "4x4 grid representing one face of tesseract"
                },
                "front_wall": {
                    "type": "grid",
                    "params": {"rows": 4, "cols": 4},
                    "description": "4x4 grid representing opposite face of tesseract"
                },
                "link_pattern": "orthogonal",  # each back node links to corresponding front node
                "total_nodes": 16,
                "total_edges": 32,
                "interdimensional_links": 16
            },
            "hyperoctahedron": {
                "description": "4D cross-polytope with 8 vertices and radial structure",
                "back_wall": {
                    "type": "radial",
                    "params": {"rings": 2, "per": 4},
                    "description": "Radial structure with 4-fold symmetry"
                },
                "front_wall": {
                    "type": "radial",
                    "params": {"rings": 2, "per": 4},
                    "description": "Mirror radial structure"
                },
                "link_pattern": "center_symmetric",
                "total_nodes": 8,
                "interdimensional_links": 8
            },
            "hyperpyramid": {
                "description": "4D pyramid with pentagonal base extending through dimensions",
                "back_wall": {
                    "type": "radial",
                    "params": {"rings": 2, "per": 5},
                    "description": "Pentagonal base structure"
                },
                "front_wall": {
                    "type": "grid",
                    "params": {"rows": 1, "cols": 1},
                    "description": "Single apex point"
                },
                "link_pattern": "pyramid",
                "total_nodes": 12,
                "interdimensional_links": 10
            },
            "klein_bottle_graph": {
                "description": "4D Klein bottle represented as graph structure",
                "back_wall": {
                    "type": "grid",
                    "params": {"rows": 6, "cols": 8},
                    "description": "Rectangular grid base"
                },
                "front_wall": {
                    "type": "grid",
                    "params": {"rows": 6, "cols": 8},
                    "description": "Twisted grid surface"
                },
                "link_pattern": "twisted",
                "special_properties": ["self_intersecting", "non_orientable"]
            },
            "hypersphere": {
                "description": "4D sphere approximation through radial node networks",
                "back_wall": {
                    "type": "radial",
                    "params": {"rings": 5, "per": 12},
                    "description": "Multi-ring radial structure"
                },
                "front_wall": {
                    "type": "radial",
                    "params": {"rings": 5, "per": 12},
                    "description": "Corresponding radial projection"
                },
                "link_pattern": "spherical",
                "curvature": "positive"
            }
        }

    def _initialize_wall_commands(self) -> Dict[str, Any]:
        """Define natural language commands for wall and dimensional operations"""
        return {
            "create_wall": {
                "patterns": ["create wall", "make wall", "generate wall", "build wall"],
                "params": ["wall_name", "wall_type", "dimensions"],
                "description": "Create front or back wall with specified structure"
            },
            "switch_wall": {
                "patterns": ["switch to", "work on", "select wall", "use wall"],
                "params": ["wall_name"],
                "description": "Switch active wall between front and back"
            },
            "link_walls": {
                "patterns": ["link walls", "connect dimensions", "bridge", "interdimensional link"],
                "params": ["link_pattern", "node_selection"],
                "description": "Create interdimensional links between walls"
            },
            "create_hypershape": {
                "patterns": ["create 4d", "make hypershape", "build tesseract", "4d shape"],
                "params": ["shape_type", "dimensions", "link_pattern"],
                "description": "Generate complete 4D hypershape structure"
            },
            "adjust_transparency": {
                "patterns": ["transparency", "alpha", "opacity", "fade"],
                "params": ["wall_name", "alpha_value"],
                "description": "Adjust wall or link transparency"
            },
            "rotate_4d": {
                "patterns": ["rotate 4d", "spin hypershape", "4d rotation"],
                "params": ["rotation_axis", "angle"],
                "description": "Simulate 4D rotation projection"
            },
            "spawn_limbs": {
                "patterns": ["add limbs", "spawn limbs", "create limbs", "attach limbs"],
                "params": ["wall_name", "limb_count", "limb_properties"],
                "description": "Spawn oscillating limbs on wall nodes"
            },
            "control_limbs": {
                "patterns": ["limb speed", "limb length", "limb angle", "oscillate"],
                "params": ["speed", "length", "max_angle"],
                "description": "Control limb oscillation parameters"
            }
        }

    def _initialize_dimensional_linking(self) -> Dict[str, Any]:
        """Define patterns for linking nodes across dimensions"""
        return {
            "orthogonal": {
                "description": "Direct 1:1 correspondence between wall nodes",
                "method": "position_match"
            },
            "center_symmetric": {
                "description": "Link nodes symmetrically around center points",
                "method": "radial_symmetry"
            },
            "pyramid": {
                "description": "All nodes on one wall link to single apex on other",
                "method": "many_to_one"
            },
            "twisted": {
                "description": "Klein bottle style twisted connections",
                "method": "twisted_topology"
            },
            "spherical": {
                "description": "Spherical coordinate mapping between walls",
                "method": "spherical_projection"
            },
            "fibonacci": {
                "description": "Golden ratio spiral linking pattern",
                "method": "fibonacci_spiral"
            }
        }

    def process_4d_request(self, user_input: str) -> Dict[str, Any]:
        """Process natural language requests for 4D operations"""

        # Use base canvas engine for initial understanding
        base_result = self.canvas_engine.process_canvas_request(user_input)

        # Enhance with 4D-specific analysis
        four_d_understanding = self._analyze_4d_intent(user_input)

        # Generate 4D-specific response
        four_d_response = self._generate_4d_response(four_d_understanding, base_result)

        return {
            "success": True,
            "input": user_input,
            "base_understanding": base_result.get("base_understanding", {}),
            "4d_understanding": four_d_understanding,
            "4d_actions": four_d_response.get("actions", []),
            "javascript_code": four_d_response.get("javascript", ""),
            "html_updates": four_d_response.get("html", ""),
            "response": four_d_response.get("description", ""),
            "session_state": self.current_4d_session
        }

    def _analyze_4d_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze input for 4D-specific operations and concepts"""

        input_lower = user_input.lower()
        detected_commands = []
        extracted_params = {}

        # Check for 4D command patterns
        for command_name, command_info in self.wall_commands.items():
            for pattern in command_info["patterns"]:
                if pattern in input_lower:
                    detected_commands.append({
                        "command": command_name,
                        "confidence": 0.95,
                        "pattern_matched": pattern
                    })

        # Extract 4D-specific parameters
        hypershapes = self._extract_hypershape_types(user_input)
        walls = self._extract_wall_references(user_input)
        dimensions = self._extract_dimensional_terms(user_input)
        linking_patterns = self._extract_linking_patterns(user_input)
        limb_parameters = self._extract_limb_parameters(user_input)

        return {
            "detected_commands": detected_commands,
            "hypershapes": hypershapes,
            "walls": walls,
            "dimensions": dimensions,
            "linking_patterns": linking_patterns,
            "limb_parameters": limb_parameters,
            "4d_context": self._infer_4d_context(user_input),
            "complexity_level": self._assess_4d_complexity(user_input)
        }

    def _extract_hypershape_types(self, text: str) -> List[str]:
        """Extract 4D hypershape type mentions from text"""
        text_lower = text.lower()
        mentioned_shapes = []

        for shape_name in self.hypershape_templates.keys():
            if shape_name in text_lower or shape_name.replace('_', ' ') in text_lower:
                mentioned_shapes.append(shape_name)

        # Additional 4D shape terms
        shape_terms = ["4d", "hypercube", "hyperoctahedron", "4-cube", "tesseract",
                      "klein bottle", "hypersphere", "4-dimensional", "four dimensional"]
        for term in shape_terms:
            if term in text_lower:
                # Map terms to shape types
                if term in ["tesseract", "4-cube", "hypercube"]:
                    mentioned_shapes.append("tesseract")
                elif term in ["hyperoctahedron"]:
                    mentioned_shapes.append("hyperoctahedron")
                elif term in ["klein bottle"]:
                    mentioned_shapes.append("klein_bottle_graph")
                elif term in ["hypersphere"]:
                    mentioned_shapes.append("hypersphere")

        return list(set(mentioned_shapes))  # Remove duplicates

    def _extract_wall_references(self, text: str) -> List[str]:
        """Extract wall references (front/back) from text"""
        text_lower = text.lower()
        walls = []

        if any(term in text_lower for term in ["front", "front wall", "forward"]):
            walls.append("front")
        if any(term in text_lower for term in ["back", "back wall", "rear", "behind"]):
            walls.append("back")

        return walls

    def _extract_dimensional_terms(self, text: str) -> List[str]:
        """Extract dimensional terminology from text"""
        text_lower = text.lower()
        dimensional_terms = []

        terms = ["4d", "4-dimensional", "fourth dimension", "hyperdimensional",
                "interdimensional", "cross-dimensional", "dimensional bridge"]

        for term in terms:
            if term in text_lower:
                dimensional_terms.append(term)

        return dimensional_terms

    def _extract_limb_parameters(self, text: str) -> Dict[str, Any]:
        """Extract limb-related parameters from user input"""
        limb_params = {}
        text_lower = text.lower()

        # Extract speed values
        speed_patterns = [r'speed\s*(\d+)', r'fast\s*(\d+)', r'slow\s*(\d+)', r'(\d+)\s*speed']
        for pattern in speed_patterns:
            match = re.search(pattern, text_lower)
            if match:
                limb_params['speed'] = int(match.group(1))
                break

        # Extract angle values
        angle_patterns = [r'angle\s*(\d+)', r'(\d+)\s*degrees?', r'(\d+)°']
        for pattern in angle_patterns:
            match = re.search(pattern, text_lower)
            if match:
                limb_params['max_angle'] = int(match.group(1))
                break

        # Extract length values
        length_patterns = [r'length\s*(\d+)', r'long\s*(\d+)', r'short\s*(\d+)', r'(\d+)\s*px']
        for pattern in length_patterns:
            match = re.search(pattern, text_lower)
            if match:
                limb_params['length'] = int(match.group(1))
                break

        # Detect qualitative descriptors
        if any(word in text_lower for word in ['fast', 'quick', 'rapid']):
            limb_params['speed_modifier'] = 'fast'
        elif any(word in text_lower for word in ['slow', 'gentle', 'calm']):
            limb_params['speed_modifier'] = 'slow'

        if any(word in text_lower for word in ['wide', 'large', 'big']):
            limb_params['angle_modifier'] = 'wide'
        elif any(word in text_lower for word in ['narrow', 'small', 'tight']):
            limb_params['angle_modifier'] = 'narrow'

        if any(word in text_lower for word in ['long', 'extended', 'stretched']):
            limb_params['length_modifier'] = 'long'
        elif any(word in text_lower for word in ['short', 'compact', 'small']):
            limb_params['length_modifier'] = 'short'

        return limb_params

    def _extract_linking_patterns(self, text: str) -> List[str]:
        """Extract linking pattern descriptions from text"""
        text_lower = text.lower()
        patterns = []

        for pattern_name, pattern_info in self.dimensional_linking.items():
            if pattern_name in text_lower or pattern_name.replace('_', ' ') in text_lower:
                patterns.append(pattern_name)

        # Pattern keywords
        if "twisted" in text_lower or "klein" in text_lower:
            patterns.append("twisted")
        if "symmetric" in text_lower or "center" in text_lower:
            patterns.append("center_symmetric")
        if "pyramid" in text_lower or "apex" in text_lower:
            patterns.append("pyramid")

        return list(set(patterns))

    def _infer_4d_context(self, text: str) -> str:
        """Infer the current 4D operation context"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["tesseract", "hypercube", "4d cube"]):
            return "tesseract_creation"
        elif any(word in text_lower for word in ["wall", "front", "back", "layer"]):
            return "wall_manipulation"
        elif any(word in text_lower for word in ["link", "connect", "bridge", "interdimensional"]):
            return "dimensional_linking"
        elif any(word in text_lower for word in ["rotate", "spin", "4d rotation"]):
            return "4d_transformation"
        else:
            return "general_4d"

    def _assess_4d_complexity(self, text: str) -> str:
        """Assess the complexity level of the 4D request"""
        text_lower = text.lower()

        complex_terms = ["klein bottle", "twisted", "non-orientable", "self-intersecting",
                        "hypersphere", "curvature", "fibonacci"]
        medium_terms = ["tesseract", "hyperoctahedron", "link walls", "interdimensional"]

        if any(term in text_lower for term in complex_terms):
            return "high"
        elif any(term in text_lower for term in medium_terms):
            return "medium"
        else:
            return "low"

    def _generate_4d_response(self, four_d_understanding: Dict, base_result: Dict) -> Dict[str, Any]:
        """Generate 4D-specific response with JavaScript code"""

        detected_commands = four_d_understanding.get("detected_commands", [])
        hypershapes = four_d_understanding.get("hypershapes", [])

        if not detected_commands and not hypershapes:
            return self._generate_generic_4d_response()

        # Handle hypershape creation
        if hypershapes:
            return self._generate_hypershape_response(hypershapes[0], four_d_understanding)

        # Handle specific 4D commands
        if detected_commands:
            primary_command = detected_commands[0]["command"]

            if primary_command == "create_wall":
                return self._generate_wall_creation_response(four_d_understanding)
            elif primary_command == "link_walls":
                return self._generate_wall_linking_response(four_d_understanding)
            elif primary_command == "switch_wall":
                return self._generate_wall_switch_response(four_d_understanding)
            elif primary_command == "adjust_transparency":
                return self._generate_transparency_response(four_d_understanding)
            elif primary_command == "rotate_4d":
                return self._generate_4d_rotation_response(four_d_understanding)
            elif primary_command == "spawn_limbs":
                return self._generate_limb_spawn_response(four_d_understanding)
            elif primary_command == "control_limbs":
                return self._generate_limb_control_response(four_d_understanding)

        return self._generate_generic_4d_response()

    def _generate_hypershape_response(self, shape_type: str, understanding: Dict) -> Dict[str, Any]:
        """Generate response for creating 4D hypershapes"""

        if shape_type not in self.hypershape_templates:
            return self._generate_generic_4d_response()

        template = self.hypershape_templates[shape_type]

        js_code = f"""
// Create {shape_type} hypershape
if (typeof wallBack !== 'undefined' && typeof wallFront !== 'undefined') {{

    // Configure back wall
    const backParams = {json.dumps(template['back_wall']['params'])};
    const backType = '{template['back_wall']['type']}';

    if (backType === 'grid') {{
        wallBack.grid({{
            rows: backParams.rows || 4,
            cols: backParams.cols || 4,
            width: W(),
            height: H()
        }});
    }} else if (backType === 'radial') {{
        wallBack.radial({{
            center: {{x: W()/2, y: H()/2}},
            rings: backParams.rings || 3,
            per: backParams.per || 8
        }});
    }}

    // Configure front wall
    const frontParams = {json.dumps(template['front_wall']['params'])};
    const frontType = '{template['front_wall']['type']}';

    if (frontType === 'grid') {{
        wallFront.grid({{
            rows: frontParams.rows || 4,
            cols: frontParams.cols || 4,
            width: W(),
            height: H()
        }});
    }} else if (frontType === 'radial') {{
        wallFront.radial({{
            center: {{x: W()/2, y: H()/2}},
            rings: frontParams.rings || 3,
            per: frontParams.per || 8
        }});
    }}

    // Create interdimensional links based on pattern
    links.length = 0; // Clear existing links
    const linkPattern = '{template.get('link_pattern', 'orthogonal')}';

    if (linkPattern === 'orthogonal') {{
        // Direct correspondence linking
        const minNodes = Math.min(wallBack.nodes.length, wallFront.nodes.length);
        for (let i = 0; i < minNodes; i++) {{
            links.push({{iBack: i, iFront: i}});
        }}
    }} else if (linkPattern === 'center_symmetric') {{
        // Center-based symmetric linking
        const backCenter = wallBack.nodes.find(n => n.id === 'center');
        const frontCenter = wallFront.nodes.find(n => n.id === 'center');
        if (backCenter && frontCenter) {{
            const backIdx = wallBack.nodes.indexOf(backCenter);
            const frontIdx = wallFront.nodes.indexOf(frontCenter);
            links.push({{iBack: backIdx, iFront: frontIdx}});
        }}

        // Link corresponding ring nodes
        wallBack.nodes.forEach((backNode, bi) => {{
            if (backNode.id.startsWith('r-')) {{
                const [, ring, pos] = backNode.id.split('-').map(Number);
                const frontMatch = wallFront.nodes.find(fn =>
                    fn.id === `r-${{ring}}-${{pos}}`);
                if (frontMatch) {{
                    const fi = wallFront.nodes.indexOf(frontMatch);
                    links.push({{iBack: bi, iFront: fi}});
                }}
            }}
        }});
    }} else if (linkPattern === 'pyramid') {{
        // Many-to-one pyramid linking
        if (wallBack.nodes.length > 0 && wallFront.nodes.length > 0) {{
            const apexIndex = 0; // Assume first node is apex
            wallBack.nodes.forEach((_, bi) => {{
                links.push({{iBack: bi, iFront: apexIndex}});
            }});
        }}
    }}

    // Update display
    if (typeof redrawAll !== 'undefined') {{
        redrawAll();
        updatePreview();
    }}

    console.log('{shape_type.replace('_', ' ').title()} created with {{wallBack.nodes.length}} back nodes, {{wallFront.nodes.length}} front nodes, and {{links.length}} interdimensional links');
}}
"""

        return {
            "description": f"I'll create a {shape_type.replace('_', ' ')} hypershape structure with interdimensional connections. {template['description']}",
            "actions": ["create_hypershape"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_wall_creation_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for wall creation"""

        walls = understanding.get("walls", ["front"])
        wall_name = walls[0] if walls else "front"

        js_code = f"""
// Create {wall_name} wall
if (typeof wallBack !== 'undefined' && typeof wallFront !== 'undefined') {{
    const wall = {wall_name} === 'back' ? wallBack : wallFront;

    // Generate grid structure by default
    wall.grid({{
        rows: 6,
        cols: 8,
        width: W(),
        height: H()
    }});

    if (typeof redrawAll !== 'undefined') {{
        redrawAll();
        updatePreview();
    }}

    console.log('{wall_name.title()} wall created');
}}
"""

        return {
            "description": f"I'll create a {wall_name} wall with a grid structure for you.",
            "actions": ["create_wall"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_wall_linking_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for interdimensional wall linking"""

        linking_patterns = understanding.get("linking_patterns", ["orthogonal"])
        pattern = linking_patterns[0] if linking_patterns else "orthogonal"

        js_code = f"""
// Create interdimensional links between walls
if (typeof wallBack !== 'undefined' && typeof wallFront !== 'undefined' && typeof links !== 'undefined') {{
    links.length = 0; // Clear existing links

    const pattern = '{pattern}';

    if (pattern === 'orthogonal') {{
        // Direct correspondence linking
        const minNodes = Math.min(wallBack.nodes.length, wallFront.nodes.length);
        for (let i = 0; i < minNodes; i++) {{
            links.push({{iBack: i, iFront: i}});
        }}
    }} else if (pattern === 'center_symmetric') {{
        // Symmetric radial linking
        wallBack.nodes.forEach((backNode, bi) => {{
            // Find closest front node by distance
            let closestDist = Infinity;
            let closestIdx = -1;
            wallFront.nodes.forEach((frontNode, fi) => {{
                const dist = Math.hypot(backNode.x - frontNode.x, backNode.y - frontNode.y);
                if (dist < closestDist) {{
                    closestDist = dist;
                    closestIdx = fi;
                }}
            }});
            if (closestIdx !== -1) {{
                links.push({{iBack: bi, iFront: closestIdx}});
            }}
        }});
    }}

    if (typeof drawLinks !== 'undefined') {{
        drawLinks();
        updatePreview();
    }}

    console.log(`Interdimensional links created using ${{pattern}} pattern: ${{links.length}} connections`);
}}
"""

        return {
            "description": f"I'll create interdimensional links between the walls using a {pattern} pattern.",
            "actions": ["link_walls"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_wall_switch_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for switching active walls"""

        walls = understanding.get("walls", [])
        target_wall = walls[0] if walls else "front"

        js_code = f"""
// Switch to {target_wall} wall
if (typeof state !== 'undefined') {{
    state.wall = '{target_wall}';

    // Update UI if available
    const wallSelect = document.getElementById('wall');
    if (wallSelect) {{
        wallSelect.value = '{target_wall}';
    }}

    console.log('Switched to {target_wall} wall');
}}
"""

        return {
            "description": f"I'll switch the active wall to the {target_wall} wall for editing.",
            "actions": ["switch_wall"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_transparency_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for adjusting wall transparency"""

        walls = understanding.get("walls", [])
        # Extract alpha values from numbers in understanding
        alpha_value = 0.75  # default

        js_code = """
// Adjust wall transparency
if (typeof state !== 'undefined') {
    // Set moderate transparency for better 4D visualization
    state.backAlpha = 0.6;
    state.frontAlpha = 0.8;
    state.linkAlpha = 0.9;

    // Update sliders if available
    const backSlider = document.getElementById('backAlpha');
    const frontSlider = document.getElementById('frontAlpha');
    const linkSlider = document.getElementById('linkAlpha');

    if (backSlider) backSlider.value = 60;
    if (frontSlider) frontSlider.value = 80;
    if (linkSlider) linkSlider.value = 90;

    if (typeof redrawAll !== 'undefined') {
        redrawAll();
        updatePreview();
    }

    console.log('Wall transparency adjusted for optimal 4D viewing');
}
"""

        return {
            "description": "I'll adjust the wall transparency to optimize the 4D visualization.",
            "actions": ["adjust_transparency"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_4d_rotation_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for 4D rotation simulation"""

        js_code = """
// Simulate 4D rotation by animating node positions
if (typeof wallBack !== 'undefined' && typeof wallFront !== 'undefined') {
    let rotationAngle = 0;
    const originalPositions = {
        back: wallBack.nodes.map(n => ({x: n.x, y: n.y})),
        front: wallFront.nodes.map(n => ({x: n.x, y: n.y}))
    };

    function animate4DRotation() {
        rotationAngle += 0.02; // Rotation speed

        // Apply 4D rotation transformation to node positions
        wallBack.nodes.forEach((node, i) => {
            if (originalPositions.back[i]) {
                const orig = originalPositions.back[i];
                const centerX = W() / 2;
                const centerY = H() / 2;

                // Rotate around center with 4D projection
                const dx = orig.x - centerX;
                const dy = orig.y - centerY;
                const cos4d = Math.cos(rotationAngle);
                const sin4d = Math.sin(rotationAngle);

                node.x = centerX + dx * cos4d - dy * sin4d * 0.5;
                node.y = centerY + dy * cos4d + dx * sin4d * 0.3;
            }
        });

        wallFront.nodes.forEach((node, i) => {
            if (originalPositions.front[i]) {
                const orig = originalPositions.front[i];
                const centerX = W() / 2;
                const centerY = H() / 2;

                // Counter-rotation for front wall
                const dx = orig.x - centerX;
                const dy = orig.y - centerY;
                const cos4d = Math.cos(-rotationAngle * 0.7);
                const sin4d = Math.sin(-rotationAngle * 0.7);

                node.x = centerX + dx * cos4d - dy * sin4d * 0.3;
                node.y = centerY + dy * cos4d + dx * sin4d * 0.5;
            }
        });

        if (typeof redrawAll !== 'undefined') {
            redrawAll();
        }

        // Continue animation for a few seconds
        if (rotationAngle < Math.PI * 4) {
            requestAnimationFrame(animate4DRotation);
        }
    }

    animate4DRotation();
    console.log('4D rotation animation started');
}
"""

        return {
            "description": "I'll simulate a 4D rotation by projecting the hypershape movement into 3D space.",
            "actions": ["rotate_4d"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_limb_spawn_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for spawning limbs on wall nodes"""

        walls = understanding.get("walls", ["front"])
        wall_name = walls[0] if walls else "front"

        js_code = f"""
// Spawn oscillating limbs on {wall_name} wall
if (typeof wallBack !== 'undefined' && typeof wallFront !== 'undefined' && typeof LimbVector !== 'undefined') {{
    const wall = {wall_name} === 'back' ? wallBack : wallFront;
    const limbBucket = {wall_name} === 'back' ? limbsBack : limbsFront;

    // Auto-spawn limbs on first 3 nodes if they exist
    for (let i = 0; i < Math.min(3, wall.nodes.length); i++) {{
        const node = wall.nodes[i];
        const length = parseInt(document.getElementById('limbLen')?.value, 10) || 100;
        const limb = new LimbVector(node, length);
        limbBucket.push({{ limb }});
    }}

    if (typeof renderLimbs !== 'undefined') {{
        renderLimbs();
        updatePreview();
    }}

    console.log('Limbs spawned on {wall_name} wall: ' + Math.min(3, wall.nodes.length) + ' limbs created');
}}
"""

        return {
            "description": f"I'll spawn oscillating limbs on the {wall_name} wall nodes for dynamic 4D movement.",
            "actions": ["spawn_limbs"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_limb_control_response(self, understanding: Dict) -> Dict[str, Any]:
        """Generate response for controlling limb parameters"""

        # Extract limb control parameters from understanding
        speed = 80  # default
        max_angle = 45  # default
        length = 100  # default

        js_code = f"""
// Control limb oscillation parameters
if (typeof document !== 'undefined') {{
    // Update limb control sliders
    const speedSlider = document.getElementById('limbSpeed');
    const maxSlider = document.getElementById('limbMax');
    const lenSlider = document.getElementById('limbLen');

    if (speedSlider) speedSlider.value = {speed};
    if (maxSlider) maxSlider.value = {max_angle};
    if (lenSlider) lenSlider.value = {length};

    // Update all existing limbs with new length if changed
    if (typeof limbsBack !== 'undefined' && typeof limbsFront !== 'undefined') {{
        const newLength = {length};
        limbsBack.forEach(L => {{ if (L.limb) L.limb.length = newLength; }});
        limbsFront.forEach(L => {{ if (L.limb) L.limb.length = newLength; }});
    }}

    if (typeof renderLimbs !== 'undefined') {{
        renderLimbs();
        updatePreview();
    }}

    console.log('Limb parameters updated: speed={speed}, maxAngle={max_angle}°, length={length}');
}}
"""

        return {
            "description": f"I'll adjust limb oscillation parameters: speed={speed}, max angle={max_angle}°, length={length}.",
            "actions": ["control_limbs"],
            "javascript": js_code,
            "html": ""
        }

    def _generate_generic_4d_response(self) -> Dict[str, Any]:
        """Generate generic helpful 4D response"""
        return {
            "description": "I can help you create 4D hypershapes like tesseracts, link walls across dimensions, adjust transparency, simulate 4D rotations, and spawn oscillating limbs. Try saying 'create tesseract', 'link the walls', or 'spawn limbs'.",
            "actions": ["help"],
            "javascript": "console.log('4D hypershape assistant ready');",
            "html": "",
            "suggestions": [
                "Create a tesseract hypercube",
                "Spawn limbs on the front wall",
                "Link the dimensional walls",
                "Rotate the 4D shape",
                "Adjust limb oscillation speed"
            ]
        }

def demo_4d_integration():
    """Demonstrate the 4D hypershape conversational integration"""

    print("🌌 4D HYPERSHAPE CONVERSATIONAL INTEGRATION")
    print("=" * 70)
    print("Advanced AI for 4-dimensional shape creation and manipulation")
    print("=" * 70)

    # Initialize 4D engine
    four_d_engine = FourDimensionalShapeEngine()

    print("\n✅ 4D Hypershape Engine initialized")
    print("✅ Dimensional linking patterns loaded")
    print("✅ Hypershape templates ready")

    # Test 4D conversational commands
    test_commands = [
        "Create a tesseract hypercube",
        "Make a hyperoctahedron with radial symmetry",
        "Link the back and front walls",
        "Build a Klein bottle graph structure",
        "Rotate the 4D shape slowly",
        "Adjust transparency for better visualization"
    ]

    print(f"\n🧪 Testing {len(test_commands)} 4D conversational commands...\n")

    for i, command in enumerate(test_commands, 1):
        print(f"🌌 Command {i}: '{command}'")
        print("-" * 50)

        result = four_d_engine.process_4d_request(command)

        if result["success"]:
            understanding = result["4d_understanding"]
            hypershapes = understanding.get("hypershapes", [])
            commands = [cmd["command"] for cmd in understanding.get("detected_commands", [])]
            complexity = understanding.get("complexity_level", "unknown")

            print(f"🔍 4D Understanding: {', '.join(hypershapes + commands)}")
            print(f"📊 Complexity: {complexity}")
            print(f"💫 Response: {result['response']}")
            print(f"⚡ Actions: {', '.join(result['4d_actions'])}")

            if result["javascript_code"]:
                print(f"🖥️ JavaScript Generated:")
                preview_lines = result["javascript_code"].strip().split('\n')[:3]
                for line in preview_lines:
                    if line.strip():
                        print(f"   {line.strip()}")
                if len(result["javascript_code"].strip().split('\n')) > 3:
                    print(f"   ... ({len(result['javascript_code'].strip().split('\n')) - 3} more lines)")
        else:
            print(f"❌ Error processing 4D command")

        print("\n" + "─" * 70 + "\n")

    # Show hypershape capabilities
    print("🌌 AVAILABLE 4D HYPERSHAPE TEMPLATES:")
    for shape_name, template in four_d_engine.hypershape_templates.items():
        print(f"   • {shape_name.replace('_', ' ').title()}: {template['description']}")

    print("\n🔗 INTERDIMENSIONAL LINKING PATTERNS:")
    for pattern_name, pattern_info in four_d_engine.dimensional_linking.items():
        print(f"   • {pattern_name.replace('_', ' ').title()}: {pattern_info['description']}")

    print("\n🎯 4D VISUALIZATION READY!")
    print("   • Natural language 4D shape creation")
    print("   • Interdimensional wall linking")
    print("   • Hypershape template library")
    print("   • 4D rotation simulation")
    print("   • Advanced transparency control")

    return four_d_engine

if __name__ == "__main__":
    demo_4d_integration()
