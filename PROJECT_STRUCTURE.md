# World Engine v3.1 Advanced Math - Project Structure

This document describes the organized folder structure of the World Engine project after comprehensive reorganization.

## 📁 Root Directory Structure

```
world-engine-feat-v3-1-advanced-math/
├── archive/                 # Legacy files and old experiments
├── components/              # Reusable components and modules
├── config/                  # Configuration files (ESLint, TypeScript, etc.)
├── controllers/             # Control logic and orchestration
├── data/                    # Data files and databases
├── demos/                   # Demo applications and examples
├── docs/                    # Documentation (organized by category)
├── examples/                # Code examples and samples
├── extensions/              # Project extensions and plugins
├── include/                 # Header files and includes
├── nexus-holy-beat-system/  # Core nexus system components
├── public/                  # Public assets and resources
├── schemas/                 # Data schemas and definitions
├── scripts/                 # Build scripts, deployment scripts
├── server/                  # Server-related files (Node.js, Express)
├── src/                     # Source code (organized by language)
├── tests/                   # Test files and test suites
├── utils/                   # Utility functions and helpers
├── web-apps/                # Web applications and HTML demos
└── websocket/               # WebSocket implementations
```

## 🗂️ Detailed Folder Descriptions

### `/config/` - Configuration Files
- `.eslintrc.json` - ESLint configuration for code linting
- `.eslintignore` - Files to ignore during linting
- `.hintrc` - VS Code hints configuration
- `tsconfig.json` - TypeScript compiler configuration

### `/scripts/` - Automation & Deployment
- `*.bat` - Windows batch scripts for building and deployment
- `*.sh` - Unix shell scripts for cross-platform deployment
- `*.ps1` - PowerShell scripts for Windows automation

### `/server/` - Server Components
- `*.cjs` - CommonJS server modules
- `serve.js` - Development server
- Server configuration and middleware files

### `/src/` - Source Code (by Language)
```
src/
├── cpp/           # C++ source files and implementations
├── javascript/    # JavaScript modules and libraries
├── python/        # Python scripts, engines, and processing tools
├── typescript/    # TypeScript source files and components
└── web/           # Web-specific source files (HTML templates, etc.)
```

### `/web-apps/` - Interactive Web Applications
```
web-apps/
├── 3d-canvas-physics/     # 3D Canvas Physics System
│   └── 3d_canvas_editor.html
├── *.html                 # Other web applications and demos
└── README.md              # Web apps documentation
```

### `/docs/` - Documentation Categories
```
docs/
├── api/          # API documentation and references
├── technical/    # Technical specifications and analysis
├── tutorials/    # Learning materials and progressive tutorials
└── *.md          # General documentation files
```

### `/archive/` - Legacy & Experimental Files
- `old_experiments/` - Previous implementation attempts
- `old_files_2025/` - Files from current year archive
- `development_notes/` - Development notes and research

## 🎯 Key Applications

### 3D Canvas Physics System
**Location**: `/web-apps/3d-canvas-physics/3d_canvas_editor.html`
- **Description**: Interactive 3D visualization with cube physics
- **Features**: Dual-layer canvas, transparency controls, radial graph generation
- **Status**: ⚠️ Currently has rendering issues (black screen with corner cube)

### Python Engines
**Location**: `/src/python/`
- Main processing engines and AI systems
- Machine learning components
- Data processing pipelines

### TypeScript Components
**Location**: `/src/typescript/`
- Type-safe component definitions
- World engine interfaces
- Advanced system integrations

## 🔧 Development Workflow

1. **Configuration**: Check `/config/` for project settings
2. **Building**: Use scripts in `/scripts/` folder
3. **Development**: Source code organized by language in `/src/`
4. **Testing**: Run tests from `/tests/` directory
5. **Deployment**: Server files in `/server/`, web apps in `/web-apps/`

## 📝 Notes

- **Canvas Rendering Issue**: The 3D canvas editor currently shows black screen with small cube in corner - likely canvas layering/z-index conflicts
- **Self-Contained**: All web applications are dependency-free after removing external CDN references
- **Cross-Platform**: Scripts available for both Windows (.bat/.ps1) and Unix (.sh) systems
- **Modular Design**: Code organized by language and functionality for easy maintenance

## 🚀 Quick Start

1. Navigate to `/web-apps/3d-canvas-physics/`
2. Open `3d_canvas_editor.html` in browser with console open
3. Check console for initialization logs and debug information
4. For Python components: `cd src/python && python main.py`
5. For development server: `cd server && node serve.js`

---

*Last Updated: September 28, 2025*
*Project: World Engine v3.1 Advanced Math*
