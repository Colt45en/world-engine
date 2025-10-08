#!/usr/bin/env node

// Tier-4 WebSocket Integration Setup Script
// This script starts the enhanced WebSocket relay and opens the collaborative demo

const fs = require('fs');
const path = require('path');
const { spawn, exec } = require('child_process');

console.log('🚀 Tier-4 WebSocket Integration Setup');
console.log('=====================================');

// Check if Node.js modules are available
const requiredModules = ['ws'];
const missingModules = [];

requiredModules.forEach(module => {
    try {
        require(module);
    } catch (error) {
        missingModules.push(module);
    }
});

if (missingModules.length > 0) {
    console.log(`📦 Installing missing modules: ${missingModules.join(', ')}`);

    const installProcess = spawn('npm', ['install', ...missingModules], {
        stdio: 'inherit',
        shell: true
    });

    installProcess.on('close', (code) => {
        if (code === 0) {
            startServices();
        } else {
            console.error('❌ Failed to install modules');
            process.exit(1);
        }
    });
} else {
    startServices();
}

function startServices() {
    console.log('\n🌐 Starting Tier-4 WebSocket Relay...');

    // Start the enhanced WebSocket relay
    const relayPath = path.join(__dirname, 'tier4_ws_relay.js');

    if (!fs.existsSync(relayPath)) {
        console.error('❌ tier4_ws_relay.js not found!');
        console.log('Please ensure the relay file is in the same directory as this script.');
        process.exit(1);
    }

    const relay = spawn('node', [relayPath], {
        stdio: 'pipe'
    });

    relay.stdout.on('data', (data) => {
        console.log(`[RELAY] ${data.toString().trim()}`);
    });

    relay.stderr.on('data', (data) => {
        console.error(`[RELAY ERROR] ${data.toString().trim()}`);
    });

    relay.on('close', (code) => {
        console.log(`WebSocket relay exited with code ${code}`);
    });

    // Wait a moment for the server to start
    setTimeout(() => {
        console.log('\n🎨 Opening collaborative demo...');
        openDemo();

        console.log('\n✅ Setup complete!');
        console.log('🔗 WebSocket Relay: ws://localhost:9000');
        console.log('🌐 Demo Interface: tier4_collaborative_demo.html');
        console.log('\nControls:');
        console.log('  - Connect to WebSocket relay');
        console.log('  - Join collaborative sessions');
        console.log('  - Apply Tier-4 operators (ST, UP, PR, CV, RB, RS)');
        console.log('  - Run Three Ides macros (IDE_A, IDE_B, MERGE)');
        console.log('  - Trigger nucleus events (VIBRATE, OPTIMIZE, STATE)');
        console.log('  - Watch real-time distributed state synchronization');
        console.log('\nPress Ctrl+C to stop all services');

    }, 2000);

    // Handle cleanup
    process.on('SIGINT', () => {
        console.log('\n🔄 Shutting down services...');
        relay.kill('SIGINT');
        process.exit(0);
    });
}

function openDemo() {
    const demoPath = path.join(__dirname, 'tier4_collaborative_demo.html');

    if (!fs.existsSync(demoPath)) {
        console.log('⚠️  Demo file not found, but relay is running');
        console.log('You can create your own client to connect to ws://localhost:9000');
        return;
    }

    // Try to open the demo in the default browser
    const platform = process.platform;
    let command;

    if (platform === 'win32') {
        command = `start ${demoPath}`;
    } else if (platform === 'darwin') {
        command = `open ${demoPath}`;
    } else {
        command = `xdg-open ${demoPath}`;
    }

    exec(command, (error) => {
        if (error) {
            console.log('⚠️  Could not open demo automatically');
            console.log(`Please open: ${demoPath}`);
        } else {
            console.log('🎨 Demo opened in browser');
        }
    });
}

function printUsageInfo() {
    console.log('\n📖 Integration Guide:');
    console.log('---------------------');

    console.log('\n1. WebSocket Events → Tier-4 Operators:');
    console.log('   VIBRATE → ST (stabilization)');
    console.log('   OPTIMIZATION → UP (update/progress)');
    console.log('   STATE → CV (convergence)');
    console.log('   SEED → RB (rollback)');

    console.log('\n2. Memory Tags → Operators:');
    console.log('   energy → CH (change)');
    console.log('   refined → PR (progress)');
    console.log('   condition → SL (selection)');
    console.log('   seed → MD (multidimensional)');

    console.log('\n3. Cycle Events → Macros:');
    console.log('   cycle_start → IDE_A, IDE_B, or MERGE_ABC');
    console.log('   Depends on cycle position and context');

    console.log('\n4. Collaborative Features:');
    console.log('   - Multi-client state synchronization');
    console.log('   - Session-based collaboration');
    console.log('   - Real-time operator broadcasting');
    console.log('   - Conflict-free state merging');

    console.log('\n5. Your NDJSON Integration:');
    console.log('   - Place your run.ndjson in this directory');
    console.log('   - The relay will auto-process NDJSON streams');
    console.log('   - Nucleus execution maps to Tier-4 operations');
    console.log('   - Memory events trigger appropriate operators');
}

// Print usage info after a delay
setTimeout(printUsageInfo, 5000);
