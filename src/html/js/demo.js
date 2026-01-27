/**
 * World Engine Interactive Demo
 */

let demoEngine = null;
let demoScene = null;
let animationId = null;
let lastTime = Date.now();
let fps = 0;
let frameCounter = 0;

function startDemo() {
    if (demoEngine) {
        return;
    }

    // Initialize engine
    demoEngine = new WorldEngine();
    demoEngine.initialize();
    
    // Create scene
    demoScene = demoEngine.createScene('DemoScene');
    
    // Create some demo entities
    for (let i = 0; i < 5; i++) {
        demoEngine.createEntity(`DemoEntity_${i + 1}`);
    }
    
    // Start animation loop
    lastTime = Date.now();
    frameCounter = 0;
    animate();
    
    // Update UI
    document.getElementById('start-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
    
    console.log('Demo started');
}

function stopDemo() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    
    console.log('Demo stopped');
}

function resetDemo() {
    stopDemo();
    demoEngine = null;
    demoScene = null;
    fps = 0;
    frameCounter = 0;
    
    updateDemoInfo();
    
    // Clear canvas
    const canvas = document.getElementById('demo-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    document.getElementById('start-btn').disabled = false;
    console.log('Demo reset');
}

function animate() {
    if (!demoEngine) return;
    
    const currentTime = Date.now();
    const deltaTime = (currentTime - lastTime) / 1000.0;
    lastTime = currentTime;
    
    // Update frame counter
    demoEngine.frameCount++;
    frameCounter++;
    
    // Calculate FPS every second
    if (frameCounter >= 60) {
        fps = Math.round(1.0 / deltaTime);
        frameCounter = 0;
    }
    
    // Render
    renderDemo();
    
    // Update info display
    updateDemoInfo();
    
    // Continue animation
    animationId = requestAnimationFrame(animate);
}

function renderDemo() {
    const canvas = document.getElementById('demo-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 50) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    // Draw entities as animated circles
    if (demoEngine && demoEngine.entities) {
        const time = Date.now() / 1000.0;
        demoEngine.entities.forEach((entity, index) => {
            const angle = time + (index * Math.PI * 2 / demoEngine.entities.length);
            const x = canvas.width / 2 + Math.cos(angle) * 200;
            const y = canvas.height / 2 + Math.sin(angle) * 150;
            
            // Draw entity
            ctx.fillStyle = `hsl(${index * 60}, 70%, 50%)`;
            ctx.beginPath();
            ctx.arc(x, y, 20, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw entity name
            ctx.fillStyle = '#fff';
            ctx.font = '12px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(entity.name, x, y - 30);
        });
    }
    
    // Draw title
    ctx.fillStyle = '#3498db';
    ctx.font = 'bold 24px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('World Engine Demo', 20, 40);
}

function updateDemoInfo() {
    document.getElementById('fps').textContent = fps || 0;
    document.getElementById('entity-count').textContent = 
        demoEngine ? demoEngine.entities.length : 0;
    document.getElementById('frame-count').textContent = 
        demoEngine ? demoEngine.frameCount : 0;
}

// Initialize demo info display
document.addEventListener('DOMContentLoaded', function() {
    updateDemoInfo();
});
