#!/usr/bin/env node
/**
 * RNES VectorLab Pak Generator
 * Creates signed development packages with Math Pro integration
 */

import * as fs from 'fs/promises';
import * as crypto from 'crypto';
import * as path from 'path';

// Configuration
const PRIVATE_KEY_PATH = './dev-keys/private.pem';
const PUBLIC_KEY_PATH = './dev-keys/public.pem';

// Base64URL encoding
const base64urlEscape = (str) => str.replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
const base64urlUnescape = (str) => str.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - str.length % 4) % 4);

// Canonical JSON (stable key sort)
const sortObj = (o) => {
    if (Array.isArray(o)) return o.map(sortObj);
    if (o && typeof o === 'object') {
        return Object.keys(o).sort().reduce((acc, k) => { acc[k] = sortObj(o[k]); return acc; }, {});
    }
    return o;
};
const canonicalJSONString = (o) => JSON.stringify(sortObj(o));

// Generate key pair if not exists
async function ensureKeyPair() {
    try {
        await fs.access(PRIVATE_KEY_PATH);
        await fs.access(PUBLIC_KEY_PATH);
        console.log('✓ Using existing key pair');
        return;
    } catch {
        console.log('⚙️ Generating new ECDSA P-256 key pair...');

        const { publicKey, privateKey } = crypto.generateKeyPairSync('ec', {
            namedCurve: 'prime256v1',
            publicKeyEncoding: { type: 'spki', format: 'pem' },
            privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
        });

        await fs.mkdir('./dev-keys', { recursive: true });
        await fs.writeFile(PRIVATE_KEY_PATH, privateKey);
        await fs.writeFile(PUBLIC_KEY_PATH, publicKey);

        console.log('✓ Generated new key pair');
    }
}

// Get public key JWK
async function getPublicKeyJWK() {
    const publicKeyPem = await fs.readFile(PUBLIC_KEY_PATH, 'utf8');
    const publicKey = crypto.createPublicKey(publicKeyPem);
    const jwk = publicKey.export({ format: 'jwk' });

    return {
        kty: jwk.kty,
        crv: jwk.crv,
        alg: 'ES256',
        ext: true,
        x: jwk.x,
        y: jwk.y
    };
}

// Sign data
async function signData(data) {
    const privateKeyPem = await fs.readFile(PRIVATE_KEY_PATH, 'utf8');
    const privateKey = crypto.createPrivateKey(privateKeyPem);

    const signature = crypto.sign('sha256', Buffer.from(data, 'utf8'), {
        key: privateKey,
        format: 'der',
        type: 'pkcs8'
    });

    return base64urlEscape(signature.toString('base64'));
}

// Calculate SHA-256 hash
function sha256(buffer) {
    return crypto.createHash('sha256').update(buffer).digest('hex');
}

// Create pak from directory
async function createPak(sourceDir, outputPath) {
    const entries = {};
    const allowedFiles = [
        'script.mjs', 'shader.frag', 'notes.md',
        'math-pro.js', 'glyph-map.json', 'morpheme-config.js',
        'scene-config.json', 'vector-lab.js'
    ];

    console.log(`📦 Creating pak from ${sourceDir}...`);

    // Read and process files
    const files = await fs.readdir(sourceDir, { withFileTypes: true });

    for (const file of files) {
        if (!file.isFile()) continue;
        if (!allowedFiles.includes(file.name)) {
            console.log(`⚠️ Skipping ${file.name} (not in allow-list)`);
            continue;
        }

        const filePath = path.join(sourceDir, file.name);
        const content = await fs.readFile(filePath);
        const hash = sha256(content);
        const b64data = base64urlEscape(content.toString('base64'));

        entries[file.name] = {
            sha256: hash,
            data: b64data,
            size: content.length
        };

        console.log(`✓ ${file.name} (${content.length} bytes, ${hash.slice(0, 8)}...)`);
    }

    if (Object.keys(entries).length === 0) {
        throw new Error('No valid files found to pack');
    }

    // Create canonical representation and sign
    const canonical = canonicalJSONString(entries);
    const signature = await signData(canonical);

    // Create pak structure
    const pak = {
        version: "1.0.0",
        type: "rnes-vectorlab-pak",
        created: new Date().toISOString(),
        entries,
        signature
    };

    // Write pak file
    await fs.writeFile(outputPath, JSON.stringify(pak, null, 2));

    console.log(`✅ Pak created: ${outputPath}`);
    console.log(`📊 Files: ${Object.keys(entries).length}, Signature: ${signature.slice(0, 16)}...`);

    return pak;
}

// Create example Math Pro script
async function createExampleScript(outputDir) {
    const exampleScript = `/**
 * RNES VectorLab - Math Pro Integration Example
 * This script demonstrates integration with the Math Pro glyph system
 */

let glyphMap, scene, renderer;
let testObjects = [];

export function init(context) {
  console.log('🚀 Math Pro VectorLab script loaded');

  // Extract context
  ({ scene, renderer } = context);
  glyphMap = new context.GlyphCollationMap();

  // Create test visualization
  createGlyphVisualization();
}

export function update(dt, time) {
  // Animate test objects based on morpheme properties
  testObjects.forEach((obj, i) => {
    const morpheme = obj.userData.morpheme;

    switch(morpheme) {
      case 'move':
        obj.position.x = Math.sin(time + i) * 2;
        break;
      case 'scale':
        obj.scale.setScalar(1 + Math.sin(time * 2 + i) * 0.3);
        break;
      case 'multi':
        obj.rotation.y = time + i;
        break;
    }
  });
}

export function teardown() {
  // Clean up test objects
  testObjects.forEach(obj => scene.remove(obj));
  testObjects.length = 0;
  console.log('🧹 Math Pro script cleaned up');
}

function createGlyphVisualization() {
  if (!window.THREE) return;

  const glyphs = ['∑', '∇', '⊗', 'α', '∫'];

  glyphs.forEach((glyph, i) => {
    const morpheme = glyphMap.getMorphemeFromGlyph(glyph);
    const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const material = new THREE.MeshBasicMaterial({
      color: getMorphemeColor(morpheme)
    });

    const cube = new THREE.Mesh(geometry, material);
    cube.position.set(i * 1.5 - 3, 0, 0);
    cube.userData = { glyph, morpheme };

    scene.add(cube);
    testObjects.push(cube);

    console.log(\`✨ Added \${glyph} → \${morpheme}\`);
  });
}

function getMorphemeColor(morpheme) {
  const colors = {
    'multi': 0xff6b6b,
    'move': 0x4ecdc4,
    'scale': 0x45b7d1,
    'build': 0x96ceb4,
    'ize': 0xfeca57
  };
  return colors[morpheme] || 0x666666;
}`;

    const exampleNotes = `# RNES VectorLab - Math Pro Integration

## Glyph Mappings
- ∑ → multi morpheme (summation, accumulation)
- ∇ → move morpheme (gradient, directional change)
- ⊗ → scale morpheme (tensor product, scaling)
- α → scale morpheme (alpha scaling factor)
- ∫ → build morpheme (integration, construction)

## Morpheme System
Each glyph maps to a morpheme that defines transformation behavior:
- **build**: Construction operations (matrices, geometric forms)
- **move**: Movement and directional transforms
- **scale**: Scaling and proportional operations
- **multi**: Multiplication and accumulation
- **ize**: State transformation operations

## Security Notes
- All scripts run in sandboxed blob imports
- File integrity verified via SHA-256 hashes
- Bundle signatures use ECDSA P-256
- Only allow-listed file types accepted

## Development Workflow
1. Edit script.mjs with Math Pro integrations
2. Pack into signed bundle: \`npm run pak:create\`
3. Import via vault UI or drag-drop
4. Run with full Math Pro context access`;

    const glyphConfig = {
        version: "1.0.0",
        mappings: {
            "∑": { morpheme: "multi", category: "operator" },
            "∇": { morpheme: "move", category: "operator" },
            "⊗": { morpheme: "scale", category: "operator" },
            "α": { morpheme: "scale", category: "parameter" },
            "∫": { morpheme: "build", category: "operator" }
        },
        customGlyphs: {
            "⚡": { morpheme: "accelerate", description: "speed boost" },
            "🔮": { morpheme: "transform", description: "vault access" }
        }
    };

    // Create example directory structure
    await fs.mkdir(outputDir, { recursive: true });
    await fs.writeFile(path.join(outputDir, 'script.mjs'), exampleScript);
    await fs.writeFile(path.join(outputDir, 'notes.md'), exampleNotes);
    await fs.writeFile(path.join(outputDir, 'glyph-map.json'), JSON.stringify(glyphConfig, null, 2));

    console.log(`✓ Created example files in ${outputDir}`);
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);
    const command = args[0];

    try {
        await ensureKeyPair();

        switch (command) {
            case 'create':
                const sourceDir = args[1] || './pak-source';
                const outputPath = args[2] || './math-pro-vault.pak';
                await createPak(sourceDir, outputPath);
                break;

            case 'example':
                const exampleDir = args[1] || './pak-source';
                await createExampleScript(exampleDir);
                console.log('✅ Example files created. Run: npm run pak:create');
                break;

            case 'key-info':
                const jwk = await getPublicKeyJWK();
                console.log('📋 Public Key JWK:');
                console.log(JSON.stringify(jwk, null, 2));
                break;

            default:
                console.log(`
🔮 RNES VectorLab Pak Generator

Usage:
  node pak-generator.js create [source-dir] [output-file]  Create signed pak
  node pak-generator.js example [target-dir]              Generate example files
  node pak-generator.js key-info                          Show public key JWK

Examples:
  node pak-generator.js example ./my-vault
  node pak-generator.js create ./my-vault ./my-vault.pak
        `);
                break;
        }
    } catch (err) {
        console.error('❌ Error:', err.message);
        process.exit(1);
    }
}

if (import.meta.url === \`file://\${process.argv[1]}\`) {
  main();
}
