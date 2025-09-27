/**
 * Simple HTTP Server for World Engine Studio with LLEX
 * Serves static files and handles CORS for development
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

class SimpleWebServer {
    constructor(port = 8080, directory = 'web') {
        this.port = port;
        this.directory = directory;
        this.mimeTypes = {
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
    }

    getMimeType(filePath) {
        const ext = path.extname(filePath).toLowerCase();
        return this.mimeTypes[ext] || 'text/plain';
    }

    serveFile(filePath, response) {
        try {
            const fullPath = path.join(__dirname, this.directory, filePath);
            const stats = fs.statSync(fullPath);
            
            if (stats.isFile()) {
                const mimeType = this.getMimeType(fullPath);
                const content = fs.readFileSync(fullPath);
                
                response.writeHead(200, {
                    'Content-Type': mimeType,
                    'Content-Length': content.length,
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                });
                
                response.end(content);
                return true;
            }
        } catch (error) {
            return false;
        }
        
        return false;
    }

    handleRequest(request, response) {
        const parsedUrl = url.parse(request.url);
        let pathname = parsedUrl.pathname;

        // Handle CORS preflight
        if (request.method === 'OPTIONS') {
            response.writeHead(200, {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            });
            response.end();
            return;
        }

        // Default to studio.html
        if (pathname === '/') {
            pathname = '/studio.html';
        }

        // Remove leading slash
        if (pathname.startsWith('/')) {
            pathname = pathname.substring(1);
        }

        console.log(`📁 Serving: ${pathname}`);

        if (this.serveFile(pathname, response)) {
            console.log(`✅ Served: ${pathname}`);
        } else {
            // 404 Not Found
            response.writeHead(404, {
                'Content-Type': 'text/html',
                'Access-Control-Allow-Origin': '*'
            });
            response.end(`
                <html>
                    <body>
                        <h1>404 - File Not Found</h1>
                        <p>Could not find: ${pathname}</p>
                        <p><a href="/studio.html">Go to World Engine Studio</a></p>
                    </body>
                </html>
            `);
            console.log(`❌ Not found: ${pathname}`);
        }
    }

    start() {
        const server = http.createServer((req, res) => this.handleRequest(req, res));
        
        server.listen(this.port, () => {
            console.log(`🌟 World Engine Studio Server running at:`);
            console.log(`   📍 http://localhost:${this.port}`);
            console.log(`   📁 Serving directory: ${this.directory}`);
            console.log(`   🚀 Ready for LLEX content-addressable lexical processing!`);
        });

        // Handle graceful shutdown
        process.on('SIGINT', () => {
            console.log('\n🛑 Shutting down server...');
            server.close(() => {
                console.log('✅ Server closed');
                process.exit(0);
            });
        });

        return server;
    }
}

// Start server if run directly
if (require.main === module) {
    const server = new SimpleWebServer(8080, 'web');
    server.start();
}

module.exports = SimpleWebServer;