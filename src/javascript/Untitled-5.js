// 🔮 Event Visualizer Overlay System + Glyph Labeling
class EventVisualizerOverlay {
    constructor(canvasId, nexus) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.nexus = nexus;
    }

    draw() {
        const ctx = this.ctx;
        const canvas = this.canvas;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (const event of this.nexus.activeEvents) {
            const alpha = 1 - (event.timeElapsed / event.duration);
            const radius = event.radius * 20;
            const { x, y } = event.origin;

            ctx.beginPath();
            ctx.arc(x * 20, y * 20, radius, 0, 2 * Math.PI);

            switch (event.type) {
                case "storm":
                    ctx.strokeStyle = `rgba(100,100,255,${alpha})`;
                    ctx.lineWidth = 2;
                    break;
                case "flux_surge":
                    ctx.strokeStyle = `rgba(255,100,0,${alpha})`;
                    ctx.setLineDash([5, 5]);
                    break;
                case "memory_echo":
                    ctx.strokeStyle = `rgba(200,255,150,${alpha})`;
                    ctx.shadowColor = `rgba(200,255,150,${alpha})`;
                    ctx.shadowBlur = 10;
                    break;
            }

            ctx.stroke();
            ctx.setLineDash([]);
            ctx.shadowBlur = 0;

            // 🧠 Draw glyphs affected
            for (const [id, glyph] of this.nexus.goldGlyphs.entries()) {
                if (event.affects(glyph.position)) {
                    const gx = glyph.position.x * 20;
                    const gy = glyph.position.y * 20;
                    ctx.fillStyle = "white";
                    ctx.font = "10px monospace";
                    ctx.fillText(`${glyph.name || id} (${glyph.energyLevel.toFixed(2)})`, gx + 6, gy - 6);
                }
            }
        }
    }
}

// Usage remains the same
const overlay = new EventVisualizerOverlay("overlayCanvas", nexus);
function updateUI() {
    overlay.draw();
    requestAnimationFrame(updateUI);
}
requestAnimationFrame(updateUI);
