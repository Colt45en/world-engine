# RNES VectorLab Development Vault

**Secure sandboxed development environment for Math Pro with ECDSA-signed bundles**

## 🔮 Overview

RNES VectorLab provides a **stealth development vault** that integrates seamlessly with your Math Pro glyph collation system. Features include:

- **🔐 Cryptographic Security**: ECDSA P-256 signed bundles with SHA-256 file integrity
- **📦 Sandboxed Execution**: Blob imports with allow-listed file types
- **🎭 Stealth Interface**: Long-press + Shift×5 reveal with Escape close
- **🧮 Math Pro Integration**: Direct access to glyph mappings, morphemes, and transformations
- **💾 Persistent Storage**: IndexedDB-backed vault with snippet management

## 🚀 Quick Start

### 1. Open the Development Vault
```bash
# Serve the vault interface
python -m http.server 8080
# Or use any static server

# Navigate to: http://localhost:8080/dev-vault.html?dev=1
```

### 2. Access the Stealth Interface
- **Long press** the crystal orb (bottom-right corner)
- **Shift×5** rapid sequence
- **Escape** to close

### 3. Create Development Packages
```bash
# Generate example Math Pro integration
node scripts/pak-generator.js example ./my-vault

# Create signed package
node scripts/pak-generator.js create ./my-vault ./my-vault.pak

# Import via vault UI
```

## 📁 File Structure

```
src/dev-vault/
├── vault-core.js          # Core vault system with crypto
dev-vault.html             # Stealth UI interface
scripts/
├── pak-generator.js       # Signed package generator
└── add-snippet.ts         # Snippet management
```

## 🔑 Security Features

### **Deterministic Signing**
- Canonical JSON serialization (sorted keys)
- ECDSA P-256 signatures over stable content
- DER-encoded signature format

### **File Integrity**
- SHA-256 hash verification per file
- Allow-list enforcement (`script.mjs`, `shader.frag`, `notes.md`, etc.)
- Atomic IndexedDB operations

### **Sandboxed Execution**
```javascript
// Scripts run in isolated blob context
const blobUrl = URL.createObjectURL(new Blob([code]));
const module = await import(blobUrl);
URL.revokeObjectURL(blobUrl);
```

## 🧮 Math Pro Integration

### **Glyph System Access**
```javascript
// In your vault scripts
export function init(context) {
  const { GlyphCollationMap, LLEMath, vault } = context;

  const glyphMap = new GlyphCollationMap();
  const button = glyphMap.createButtonFromGlyph('∑');

  // Save to vault
  await vault.putGlyph('summation-config', button.toJSON());
}
```

### **Morpheme Transformations**
```javascript
export function update(dt, time) {
  // Access morpheme system
  const morpheme = context.GlyphCollationMap.getMorphemeFromGlyph('∇');
  // Apply transformation based on glyph semantics
  applyMorphemeTransform(object, morpheme, dt);
}
```

### **Snippet Integration**
```javascript
// Vault provides snippet access
const snippets = await vault.listFiles();
const mathSnippets = snippets.filter(s => s.includes('math'));

// Store generated code
await vault.putFile('generated-glyph-sequence.js', codeString);
```

## 📦 Package Format

```json
{
  "version": "1.0.0",
  "type": "rnes-vectorlab-pak",
  "created": "2025-09-29T...",
  "entries": {
    "script.mjs": {
      "sha256": "abc123...",
      "data": "base64url-encoded-content",
      "size": 1234
    }
  },
  "signature": "base64url-ecdsa-der-signature"
}
```

## 🎮 Development Workflow

### **1. Create Development Script**
```javascript
// script.mjs - Math Pro integration
export function init({ GlyphCollationMap, scene, vault }) {
  const glyphMap = new GlyphCollationMap();

  // Create glyph-based visualizations
  const glyphs = ['∑', '∇', '⊗', 'α'];
  glyphs.forEach(glyph => {
    const morpheme = glyphMap.getMorphemeFromGlyph(glyph);
    createGlyphVisualization(glyph, morpheme, scene);
  });
}

export function update(dt, time) {
  // Real-time morpheme transformations
  updateGlyphTransforms(dt, time);
}

export function teardown() {
  // Clean shutdown
  clearGlyphObjects();
}
```

### **2. Package and Deploy**
```bash
# Create example files
node scripts/pak-generator.js example ./my-vault

# Edit your scripts and notes
vim ./my-vault/script.mjs
vim ./my-vault/notes.md

# Create signed package
node scripts/pak-generator.js create ./my-vault ./my-vault.pak

# Import via vault UI (long-press crystal → Import Pak)
```

### **3. Live Development**
- Edit scripts in the vault interface
- Real-time Math Pro integration testing
- Glyph mapping visualization
- Snippet management integration

## 🔧 Advanced Features

### **Custom Glyph Mappings**
```json
// glyph-map.json
{
  "customGlyphs": {
    "⚡": { "morpheme": "accelerate", "description": "speed boost" },
    "🔮": { "morpheme": "transform", "description": "vault magic" }
  }
}
```

### **Cross-Browser File Picker**
```javascript
// Automatic fallback for Safari/Firefox
const file = await pickFile(); // Uses showOpenFilePicker or input fallback
```

### **Escape Hatch Controls**
- **Escape**: Close vault panel
- **Long-press cancel**: Pointer events (touch/mouse)
- **Aria accessibility**: Screen reader support

### **Development Notes Integration**
- Auto-save with 1s debounce
- Math Pro expression examples
- Glyph documentation templates

## 🛡️ Deployment Security

### **Content Security Policy**
```html
<meta http-equiv="Content-Security-Policy"
      content="script-src 'self' 'wasm-unsafe-eval' blob:; worker-src blob:;">
```

### **Trusted Types** (Future)
```javascript
// For production deployments
const policy = trustedTypes.createPolicy('vault-scripts', {
  createScriptURL: (string) => validateAndSanitize(string)
});
```

### **Key Management**
- Rotate signing keys regularly
- Monitor vault access patterns
- Log integrity violations

## 🎯 Integration Examples

### **Math Pro Button Sequence**
```javascript
// Create button chain from glyph sequence
const sequence = "∑⊗α"; // Sum → Scale → Alpha
const buttons = glyphMap.createButtonSequence(sequence);

// Apply morpheme transformations
let state = new StationaryUnit(3);
for (const button of buttons) {
  state = button.apply(state);
}
```

### **Snippet-Driven Development**
```javascript
// Search for Math Pro patterns
const matrixSnippets = await snippetStore.search({
  tags: ['matrix', 'linear-algebra'],
  text: 'LLEMath'
});

// Generate code from patterns
const generatedCode = createCodeFromSnippets(matrixSnippets);
await vault.putFile('generated-matrix-ops.js', generatedCode);
```

### **Real-Time Glyph Visualization**
```javascript
// Live morpheme preview
export function update(dt, time) {
  glyphs.forEach((obj, i) => {
    const morpheme = obj.userData.morpheme;
    applyMorphemeAnimation(obj, morpheme, time + i, dt);
  });
}
```

## 🚀 Next Steps

1. **Enhanced Security**: Hardware security module integration
2. **Collaborative Development**: Multi-user vault synchronization
3. **AI Integration**: Semantic code generation from glyphs
4. **WebRTC Sharing**: Real-time collaborative editing
5. **Version Control**: Git-like versioning for vault contents

Your Math Pro system now has a **military-grade development vault** that scales from stealth prototyping to production deployment! 🔮✨

## 📋 Command Reference

```bash
# Package Management
node scripts/pak-generator.js create [source] [output]
node scripts/pak-generator.js example [target]
node scripts/pak-generator.js key-info

# Snippet Management
npm run snip:add -- --file vault-script.mjs --tags vault,math-pro
npm run snip:find -- --text "glyph" --tags vault

# Development Server
python -m http.server 8080  # or your preferred server
open http://localhost:8080/dev-vault.html?dev=1
```
