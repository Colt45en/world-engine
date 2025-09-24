const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8085;

const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf'
};

const server = http.createServer((req, res) => {
  let filePath = path.join(__dirname, req.url === '/' ? 'studio.html' : req.url);

  // Security check
  if (filePath.includes('..')) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const extname = path.extname(filePath).toLowerCase();
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
