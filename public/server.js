// server.js
// Combined launcher: Express server (8080) + raw HTTP static server (8085)

const express = require('express');
const http = require('http');
const fs = require('fs');
const path = require('path');

// -------------------------------
// Shared config
// -------------------------------
const PUBLIC_DIR = __dirname;
const INDEX_FILE = 'studio.html';

// -------------------------------
// Server A: Express (port 8080)
// -------------------------------
const app = express();
const EXPRESS_PORT = 8080;

// Serve static files from current directory
app.use(express.static(PUBLIC_DIR));

// Route root -> studio.html (explicit)
app.get('/', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, INDEX_FILE));
});

app.listen(EXPRESS_PORT, () => {
  console.log(`🌍 World Engine Studio (Express) running at http://localhost:${EXPRESS_PORT}`);
  console.log(`🎭 Multimodal interface ready for testing`);
});

// -------------------------------
// Server B: Raw HTTP (port 8085)
// -------------------------------
const HTTP_PORT = 8085;

const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.ogg': 'audio/ogg',
  '.mp4': 'video/mp4',
  '.pdf': 'application/pdf',
  '.wasm': 'application/wasm'
};

const server = http.createServer((req, res) => {
  // Normalize and resolve to prevent traversal outside PUBLIC_DIR
  const urlPath = req.url === '/' ? `/${INDEX_FILE}` : req.url;
  const safePath = path.normalize(urlPath).replace(/^(\.\.[/\\])+/, '');
  const filePath = path.join(PUBLIC_DIR, safePath);

  // Enforce that resolved path remains within PUBLIC_DIR
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  const contentType = mimeTypes[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('File not found');
      } else {
        res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Server error');
      }
      return;
    }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(content);
  });
});

server.listen(HTTP_PORT, () => {
  console.log(`🌍 World Engine Studio with Desktop Integration (raw HTTP) running at:`);
  console.log(`   http://localhost:${HTTP_PORT}`);
  console.log(`   http://localhost:${HTTP_PORT}/${INDEX_FILE}`);
  console.log(`\n🖥️ Desktop Features Available (via your app):`);
  console.log(`   • Auto-save LLEX sessions (every 5 minutes)`);
  console.log(`   • Export/Import LLEX data`);
  console.log(`   • Desktop-style notifications`);
  console.log(`   • Keyboard shortcuts (Ctrl+S, Ctrl+E, Ctrl+N)`);
  console.log(`   • Drag-and-drop file import`);
  console.log(`\n📱 Controls:`);
  console.log(`   💾 Save Session - Save current LLEX state`);
  console.log(`   📤 Export LLEX - Download LLEX data`);
  console.log(`   📥 Import LLEX - Load LLEX data`);
  console.log(`   🔔 Notify - Test desktop notifications`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down servers...');
  process.exit(0);
});
  const contentType = mimeTypes[extname] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404);
        res.end('File not found');
      } else {
        res.writeHead(500);
        res.end('Server error');
      }
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    }
  });
});

server.listen(PORT, () => {
  console.log(`🌍 World Engine Studio with Desktop Integration running at:`);
  console.log(`   http://localhost:${PORT}`);
  console.log(`   http://localhost:${PORT}/studio.html`);
  console.log(`\n🖥️ Desktop Features Available:`);
  console.log(`   • Auto-save LLEX sessions (every 5 minutes)`);
  console.log(`   • Export/Import LLEX data`);
  console.log(`   • Desktop-style notifications`);
  console.log(`   • Keyboard shortcuts (Ctrl+S, Ctrl+E, Ctrl+N)`);
  console.log(`   • Drag-and-drop file import`);
  console.log(`\n📱 Controls:`);
  console.log(`   💾 Save Session - Save current LLEX state`);
  console.log(`   📤 Export LLEX - Download LLEX data`);
  console.log(`   📥 Import LLEX - Load LLEX data`);
  console.log(`   🔔 Notify - Test desktop notifications`);
});
