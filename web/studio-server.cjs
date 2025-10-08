const http = require('http');
const fs = require('fs').promises;
const path = require('path');

const PORT = 8080;

const server = http.createServer(async (req, res) => {
  try {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);

    const baseDir = __dirname;
    const requestedPath = req.url === '/' ? 'studio.html' : req.url.substring(1);

    // Serve files from current directory (which should be the web directory)
    const filePath = path.join(baseDir, requestedPath);
    const safePath = path.normalize(filePath);

    // Prevent directory traversal
    if (!safePath.startsWith(baseDir + path.sep) && safePath !== baseDir) {
      res.writeHead(403, { 'Content-Type': 'text/plain' });
      res.end('403 Forbidden');
      return;
    }

    try {
      // Check if the requested path is a directory
      const stat = await fs.stat(safePath);
      if (stat.isDirectory()) {
        res.writeHead(403, { 'Content-Type': 'text/plain' });
        res.end('403 Forbidden - Directory access is not allowed');
        return;
      }

      const data = await fs.readFile(safePath);
      const ext = path.extname(safePath).toLowerCase();

      const mimeTypes = {
        '.html': 'text/html',
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.ico': 'image/x-icon'
      };

      const contentType = mimeTypes[ext] || 'application/octet-stream';
      res.writeHead(200, {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
      });
      res.end(data);

    } catch (err) {
      console.error(`File error: ${err.message}`);
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('404 Not Found');
      return;
    }
  } catch (err) {
    console.error(`Server error: ${err.message}`);
    res.writeHead(500, { 'Content-Type': 'text/plain' });
    res.end('500 Internal Server Error');
  }
});

server.listen(PORT, 'localhost', () => {
  console.log(`🚀 Server running at http://localhost:${PORT}/`);
  console.log(`📁 Serving files from: ${__dirname}`);
  console.log(`🎯 Open studio: http://localhost:${PORT}/studio.html`);
  console.log(`🔧 Test enhanced functions: http://localhost:${PORT}/test-enhanced-functions.html`);
});

server.on('error', (err) => {
  console.error('Server error:', err);
});
