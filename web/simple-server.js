const http = require('http');
const fs = require('fs').promises;
const path = require('path');

const PORT = 8080;

const server = http.createServer(async (req, res) => {
  try {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);

    const filePath = req.url === '/' ? 'studio.html' : req.url.substring(1);

    // Security check
    if (filePath.includes('..') || filePath.includes('\\')) {
      res.writeHead(403, { 'Content-Type': 'text/plain' });
      res.end('403 Forbidden');
      return;
    }

    try {
      const data = await fs.readFile(filePath);
      const ext = path.extname(filePath).toLowerCase();

      const mimeTypes = {
        '.html': 'text/html',
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.gif': 'image/gif',
        '.ico': 'image/x-icon',
        '.svg': 'image/svg+xml',
        '.wav': 'audio/wav',
        '.mp4': 'video/mp4',
        '.woff': 'application/font-woff',
        '.ttf': 'application/font-ttf',
        '.eot': 'application/vnd.ms-fontobject',
        '.otf': 'application/font-otf'
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
});

server.on('error', (err) => {
  console.error('Server error:', err);
});
