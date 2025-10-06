/**
 * Multi-Language Command Bridge - JavaScript/Node.js Side
 * Enables commanding between Python, JavaScript, and C++
 */

import { spawn, exec } from 'child_process';
import fs from 'fs';
import path from 'path';
import http from 'http';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class LanguageCommandBridge {
    constructor() {
        this.activeLlanguages = {
            javascript: true,
            python: false,
            cpp: false
        };
        this.commandHistory = [];
        this.pythonBridgePath = './language_command_bridge.py';
        this.serverPort = 3002;
    }

    logCommand(sourceLanguage, targetLanguage, command, result = null) {
        const entry = {
            timestamp: new Date().toISOString(),
            source: sourceLanguage,
            target: targetLanguage,
            command: command,
            result: result ? JSON.stringify(result).substring(0, 500) : null,
            status: result ? 'success' : 'pending'
        };
        this.commandHistory.push(entry);
        console.log(`[JS-BRIDGE] ${sourceLanguage} -> ${targetLanguage}: ${command.substring(0, 50)}...`);
    }

    async executeJavaScriptCommand(jsCode, context = {}) {
        try {
            // Create execution context
            const execContext = {
                console,
                process,
                Buffer,
                setTimeout,
                clearTimeout,
                setInterval,
                clearInterval,
                ...context
            };

            // Create function to execute code safely
            const func = new Function(...Object.keys(execContext), `
                try {
                    ${jsCode}
                } catch (error) {
                    return { error: error.message, status: 'error' };
                }
            `);

            const result = func(...Object.values(execContext));
            
            const response = {
                status: 'success',
                result: result,
                timestamp: new Date().toISOString()
            };

            this.logCommand('bridge', 'javascript', jsCode, response);
            return response;

        } catch (error) {
            const errorResponse = {
                status: 'error',
                error: error.message,
                errorType: error.constructor.name,
                timestamp: new Date().toISOString()
            };
            this.logCommand('bridge', 'javascript', jsCode, errorResponse);
            return errorResponse;
        }
    }

    async executePythonCommand(pythonCode) {
        return new Promise((resolve, reject) => {
            try {
                // Create temporary Python file
                const tempFile = `temp_py_command_${Date.now()}.py`;
                const wrappedCode = `
import json
import sys
from datetime import datetime

try:
    ${pythonCode}
    
    # Try to get result if it exists
    if 'result' in locals():
        output = result
    else:
        output = "Python code executed successfully"
    
    response = {
        'status': 'success',
        'result': str(output),
        'timestamp': datetime.now().isoformat()
    }
    print(json.dumps(response))
    
except Exception as e:
    error_response = {
        'status': 'error',
        'error': str(e),
        'error_type': type(e).__name__,
        'timestamp': datetime.now().isoformat()
    }
    print(json.dumps(error_response))
`;

                fs.writeFileSync(tempFile, wrappedCode);

                // Execute Python code
                exec(`python ${tempFile}`, (error, stdout, stderr) => {
                    // Clean up temp file
                    if (fs.existsSync(tempFile)) {
                        fs.unlinkSync(tempFile);
                    }

                    if (error) {
                        const errorResponse = {
                            status: 'error',
                            error: stderr || error.message,
                            timestamp: new Date().toISOString()
                        };
                        this.logCommand('bridge', 'python', pythonCode, errorResponse);
                        resolve(errorResponse);
                    } else {
                        try {
                            const result = JSON.parse(stdout.trim());
                            this.logCommand('bridge', 'python', pythonCode, result);
                            resolve(result);
                        } catch (parseError) {
                            const response = {
                                status: 'success',
                                result: stdout.trim(),
                                timestamp: new Date().toISOString()
                            };
                            this.logCommand('bridge', 'python', pythonCode, response);
                            resolve(response);
                        }
                    }
                });

            } catch (error) {
                const errorResponse = {
                    status: 'error',
                    error: error.message,
                    timestamp: new Date().toISOString()
                };
                this.logCommand('bridge', 'python', pythonCode, errorResponse);
                resolve(errorResponse);
            }
        });
    }

    async executeCppSimulation(cppCode) {
        // Simulate C++ execution for now
        const simulationResult = {
            status: 'simulated',
            message: 'C++ simulation - would compile and execute in full implementation',
            code: cppCode,
            simulatedOutput: `C++ code processed: ${cppCode.length} characters`,
            timestamp: new Date().toISOString()
        };

        this.logCommand('bridge', 'cpp', cppCode, simulationResult);
        return simulationResult;
    }

    async commandPythonFromJs(pythonCode) {
        console.log('[JS->PY] Executing Python code from JavaScript');
        return await this.executePythonCommand(pythonCode);
    }

    async commandJsFromPython(jsCode) {
        console.log('[PY->JS] Executing JavaScript code from Python');
        return await this.executeJavaScriptCommand(jsCode);
    }

    crossLanguageDataExchange(data, sourceLanguage, targetLanguage) {
        // Convert data types for cross-language compatibility
        const convertedData = this.convertDataTypes(data, targetLanguage);

        const exchangeResult = {
            status: 'success',
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            originalData: data,
            convertedData: convertedData,
            timestamp: new Date().toISOString()
        };

        this.logCommand(sourceLanguage, targetLanguage, `Data exchange: ${typeof data}`, exchangeResult);
        return exchangeResult;
    }

    convertDataTypes(data, targetLanguage) {
        if (targetLanguage === 'python') {
            // Convert JS types to Python-compatible format
            if (typeof data === 'object' && data !== null) {
                if (Array.isArray(data)) {
                    return data.map(item => this.convertDataTypes(item, targetLanguage));
                } else {
                    const converted = {};
                    for (const [key, value] of Object.entries(data)) {
                        converted[key] = this.convertDataTypes(value, targetLanguage);
                    }
                    return converted;
                }
            }
            return data;
        } else if (targetLanguage === 'cpp') {
            // Convert to C++-compatible format
            if (typeof data === 'object' && data !== null) {
                if (Array.isArray(data)) {
                    return `std::vector with ${data.length} elements`;
                } else {
                    return `std::map with ${Object.keys(data).length} entries`;
                }
            }
            return String(data);
        }
        return data;
    }

    getCommandHistory() {
        return this.commandHistory;
    }

    getSystemStatus() {
        return {
            bridgeActive: true,
            activeLanguages: this.activeLlanguages,
            totalCommandsExecuted: this.commandHistory.length,
            serverPort: this.serverPort,
            timestamp: new Date().toISOString()
        };
    }

    startHttpServer() {
        const server = http.createServer(async (req, res) => {
            // Enable CORS
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

            if (req.method === 'OPTIONS') {
                res.writeHead(200);
                res.end();
                return;
            }

            if (req.method === 'POST' && req.url === '/execute') {
                let body = '';
                req.on('data', chunk => body += chunk);
                req.on('end', async () => {
                    try {
                        const { language, code, context } = JSON.parse(body);
                        let result;

                        switch (language) {
                            case 'javascript':
                                result = await this.executeJavaScriptCommand(code, context);
                                break;
                            case 'python':
                                result = await this.executePythonCommand(code);
                                break;
                            case 'cpp':
                                result = await this.executeCppSimulation(code);
                                break;
                            default:
                                result = { status: 'error', error: 'Unknown language' };
                        }

                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify(result));
                    } catch (error) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ status: 'error', error: error.message }));
                    }
                });
            } else if (req.method === 'GET' && req.url === '/status') {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(this.getSystemStatus()));
            } else if (req.method === 'GET' && req.url === '/history') {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(this.getCommandHistory()));
            } else {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'error', error: 'Not found' }));
            }
        });

        server.listen(this.serverPort, () => {
            console.log(`🚀 JavaScript Language Bridge Server running on http://localhost:${this.serverPort}`);
            console.log(`📊 Status: http://localhost:${this.serverPort}/status`);
            console.log(`📋 History: http://localhost:${this.serverPort}/history`);
            console.log(`⚡ Execute: POST to http://localhost:${this.serverPort}/execute`);
        });

        return server;
    }
}

// Test function
async function testLanguageCommanding() {
    const bridge = new LanguageCommandBridge();
    
    console.log('🌍 NEXUS World Engine - JavaScript Multi-Language Command Bridge');
    console.log('='.repeat(70));

    // Test 1: JavaScript command execution
    console.log('\n⚡ Test 1: JavaScript Command Execution');
    const jsResult = await bridge.executeJavaScriptCommand(`
        const message = "Hello from JavaScript Bridge!";
        const numbers = [10, 20, 30, 40, 50];
        const sum = numbers.reduce((a, b) => a + b, 0);
        const result = { message, sum, numbers, timestamp: new Date().toISOString() };
        return result;
    `);
    console.log('Result:', jsResult);

    // Test 2: Python command execution
    console.log('\n🐍 Test 2: Python Command Execution from JavaScript');
    const pyResult = await bridge.executePythonCommand(`
result = {
    'message': 'Hello from Python via JavaScript!',
    'calculation': sum([100, 200, 300, 400, 500]),
    'language': 'Python 3.13.7'
}
    `);
    console.log('Result:', pyResult);

    // Test 3: C++ simulation
    console.log('\n🔧 Test 3: C++ Command Simulation');
    const cppResult = await bridge.executeCppSimulation(`
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> numbers = {1000, 2000, 3000, 4000, 5000};
    int sum = std::accumulate(numbers.begin(), numbers.end(), 0);
    std::cout << "C++ Sum: " << sum << std::endl;
    return 0;
}
    `);
    console.log('Result:', cppResult);

    // Test 4: Data exchange
    console.log('\n🔄 Test 4: Cross-Language Data Exchange');
    const testData = {
        projectName: "NEXUS World Engine",
        version: "3.1.0",
        languages: ["Python", "JavaScript", "C++"],
        features: {
            ai_integration: true,
            cross_language_commands: true,
            real_time_execution: true
        },
        performance_score: 98.7
    };

    const jsToNodejs = bridge.crossLanguageDataExchange(testData, 'javascript', 'nodejs');
    console.log('JS -> NodeJS:', jsToNodejs);

    const nodejsToCpp = bridge.crossLanguageDataExchange(testData, 'nodejs', 'cpp');
    console.log('NodeJS -> C++:', nodejsToCpp);

    // Test 5: System status
    console.log('\n📊 Test 5: System Status');
    const status = bridge.getSystemStatus();
    console.log('Status:', status);

    console.log('\n🎉 JavaScript Multi-Language Command Bridge Test Complete!');
    
    // Start HTTP server for web integration
    console.log('\n🌐 Starting HTTP Server for Web Integration...');
    bridge.startHttpServer();
    
    return bridge;
}

// Export for use as module
export default LanguageCommandBridge;

// Run test if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    testLanguageCommanding().catch(console.error);
}