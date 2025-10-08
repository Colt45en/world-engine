const http = require('http');
const fs = require('fs');
const path = require('path');

const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.ico': 'image/x-icon',
    '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
    console.log(`${req.method} ${req.url}`);
    
    let filePath = path.join(__dirname, 'web', req.url === '/' ? 'studio.html' : req.url);
    
    // Security check - prevent directory traversal
    if (filePath.indexOf(__dirname) !== 0) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
    }
    
    fs.readFile(filePath, (err, data) => {
        if (err) {
            console.log(`File not found: ${filePath}`);
            res.writeHead(404);
            res.end('404 - File Not Found');
            return;
        }
        
        const ext = path.extname(filePath).toLowerCase();
        const contentType = MIME_TYPES[ext] || 'text/plain';
        
        res.writeHead(200, { 
            'Content-Type': contentType,
            'Cache-Control': 'no-cache'
        });
        res.end(data);
    });
});

const PORT = 8080;
server.listen(PORT, () => {
    console.log(`🌍 World Engine Studio Server running on http://localhost:${PORT}`);
    console.log('Available endpoints:');
    console.log('  → http://localhost:8080/ (studio.html)');
    console.log('  → http://localhost:8080/layout-test.html');
    console.log('  → http://localhost:8080/worldengine.html');
});

server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.log(`Port ${PORT} is in use. Trying port ${PORT + 1}...`);
        server.listen(PORT + 1);
    } else {
        console.error('Server error:', err);
    }
});