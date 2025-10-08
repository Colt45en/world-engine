const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const port = 8081;
const webDir = path.join(__dirname, 'web');

// MIME type mapping
const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    let pathname = parsedUrl.pathname;

    // Default to studio.html for root
    if (pathname === '/') {
        pathname = '/studio.html';
    }

    const filePath = path.join(webDir, pathname);
    const ext = path.extname(filePath);
    const contentType = mimeTypes[ext] || 'text/plain';

    // Security check - prevent directory traversal
    if (!filePath.startsWith(webDir)) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
    }

    fs.readFile(filePath, (err, content) => {
        if (err) {
            if (err.code === 'ENOENT') {
                res.writeHead(404);
                res.end(`File not found: ${pathname}`);
            } else {
                res.writeHead(500);
                res.end(`Server error: ${err.code}`);
            }
        } else {
            res.writeHead(200, {
                'Content-Type': contentType,
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            });
            res.end(content);
        }
    });
});

server.listen(port, () => {
    console.log(`🧠 Lexical Logic Engine Server running on http://localhost:${port}`);
    console.log('Available endpoints:');
    console.log(`  → http://localhost:${port}/ (studio.html)`);
    console.log(`  → http://localhost:${port}/lexical-logic-engine.html`);
    console.log(`  → http://localhost:${port}/worldengine.html`);
});

// Handle server shutdown gracefully
process.on('SIGINT', () => {
    console.log('\\n🔄 Shutting down LLE server...');
    server.close(() => {
        console.log('✅ Server closed');
        process.exit(0);
    });
});
