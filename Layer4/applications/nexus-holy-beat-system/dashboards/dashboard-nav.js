/**
 * Universal Dashboard Navigation Component
 * Provides consistent AI, linking, and navigation across all dashboards
 */

class DashboardNav {
    constructor(dashboardName, dashboardType, relatedDashboards = []) {
        this.dashboardName = dashboardName;
        this.dashboardType = dashboardType;
        this.relatedDashboards = relatedDashboards;
        this.aiVisible = false;
        this.init();
    }

    init() {
        this.createNavStyles();
        this.createNavBar();
        this.createAIInterface();
        this.setupEventListeners();
    }

    createNavStyles() {
        const styles = document.createElement('style');
        styles.textContent = `
            .dashboard-nav {
                position: fixed;
                top: 10px;
                right: 10px;
                z-index: 9999;
                display: flex;
                gap: 8px;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }

            .nav-btn {
                background: rgba(15, 22, 38, 0.95);
                border: 2px solid #64ffda;
                border-radius: 12px;
                color: #64ffda;
                padding: 8px 12px;
                cursor: pointer;
                font-weight: bold;
                font-size: 12px;
                transition: all 0.3s ease;
                backdrop-filter: blur(8px);
                display: flex;
                align-items: center;
                gap: 4px;
                text-decoration: none;
            }

            .nav-btn:hover {
                background: #64ffda;
                color: #0b0f1a;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(100, 255, 218, 0.3);
            }

            .nav-btn.active {
                background: #ff00ff;
                border-color: #ff00ff;
                color: white;
            }

            .linked-menu {
                position: absolute;
                top: 100%;
                right: 0;
                margin-top: 4px;
                background: rgba(15, 22, 38, 0.98);
                border: 2px solid #64ffda;
                border-radius: 12px;
                padding: 8px;
                display: none;
                min-width: 200px;
                backdrop-filter: blur(12px);
            }

            .linked-menu.show {
                display: block;
                animation: slideDown 0.3s ease;
            }

            .linked-item {
                display: block;
                color: #64ffda;
                padding: 8px 12px;
                border-radius: 8px;
                text-decoration: none;
                font-size: 12px;
                transition: all 0.2s ease;
            }

            .linked-item:hover {
                background: rgba(100, 255, 218, 0.1);
                transform: translateX(4px);
            }

            .ai-interface {
                position: fixed;
                top: 60px;
                right: 10px;
                width: 350px;
                height: 500px;
                background: rgba(15, 22, 38, 0.98);
                border: 2px solid #ff00ff;
                border-radius: 16px;
                backdrop-filter: blur(12px);
                display: none;
                flex-direction: column;
                z-index: 9998;
            }

            .ai-interface.show {
                display: flex;
                animation: slideIn 0.4s ease;
            }

            .ai-header {
                padding: 16px;
                border-bottom: 1px solid rgba(100, 255, 218, 0.2);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .ai-title {
                color: #ff00ff;
                font-weight: bold;
                font-size: 14px;
            }

            .ai-context {
                color: #64ffda;
                font-size: 11px;
                opacity: 0.8;
            }

            .ai-close {
                background: none;
                border: none;
                color: #ff4444;
                cursor: pointer;
                font-size: 16px;
                padding: 4px;
                border-radius: 4px;
            }

            .ai-close:hover {
                background: rgba(255, 68, 68, 0.1);
            }

            .ai-messages {
                flex: 1;
                padding: 12px;
                overflow-y: auto;
                font-size: 12px;
                line-height: 1.4;
            }

            .ai-message {
                margin-bottom: 12px;
                padding: 8px 12px;
                border-radius: 8px;
                animation: messageIn 0.3s ease;
            }

            .ai-message.user {
                background: rgba(100, 255, 218, 0.1);
                border-left: 3px solid #64ffda;
                color: #64ffda;
            }

            .ai-message.ai {
                background: rgba(255, 0, 255, 0.1);
                border-left: 3px solid #ff00ff;
                color: #ff00ff;
            }

            .ai-message.system {
                background: rgba(255, 255, 255, 0.05);
                color: #888;
                text-align: center;
                font-size: 11px;
            }

            .ai-input-area {
                padding: 12px;
                border-top: 1px solid rgba(100, 255, 218, 0.2);
            }

            .ai-input {
                width: 100%;
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(100, 255, 218, 0.3);
                border-radius: 8px;
                padding: 8px 12px;
                color: #e8f0ff;
                font-size: 12px;
                resize: none;
                min-height: 60px;
                font-family: inherit;
            }

            .ai-input:focus {
                outline: none;
                border-color: #64ffda;
                box-shadow: 0 0 0 2px rgba(100, 255, 218, 0.2);
            }

            .ai-send {
                margin-top: 8px;
                background: linear-gradient(45deg, #64ffda, #ff00ff);
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                color: #000;
                font-weight: bold;
                cursor: pointer;
                font-size: 12px;
                width: 100%;
            }

            .ai-send:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(100, 255, 218, 0.3);
            }

            .ai-send:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            @keyframes slideDown {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @keyframes slideIn {
                from { opacity: 0; transform: translateX(20px) scale(0.9); }
                to { opacity: 1; transform: translateX(0) scale(1); }
            }

            @keyframes messageIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 768px) {
                .dashboard-nav {
                    top: 5px;
                    right: 5px;
                    gap: 4px;
                }

                .nav-btn {
                    padding: 6px 8px;
                    font-size: 11px;
                }

                .ai-interface {
                    top: 50px;
                    right: 5px;
                    left: 5px;
                    width: auto;
                }
            }
        `;
        document.head.appendChild(styles);
    }

    createNavBar() {
        const nav = document.createElement('div');
        nav.className = 'dashboard-nav';
        nav.innerHTML = `
            <button class="nav-btn" id="aiToggle">
                🤖 <span>AI</span>
            </button>
            <div style="position: relative;">
                <button class="nav-btn" id="linkedBtn">
                    🔗 <span>Links</span>
                </button>
                <div class="linked-menu" id="linkedMenu">
                    ${this.relatedDashboards.map(dash =>
            `<a href="${dash.url}" class="linked-item">
                            ${dash.icon} ${dash.name}
                        </a>`
        ).join('')}
                    <a href="dashboard_index.html" class="linked-item">
                        📊 All Dashboards
                    </a>
                </div>
            </div>
            <a href="dashboard_index.html" class="nav-btn" id="homeBtn">
                🏠 <span>Home</span>
            </a>
        `;
        document.body.appendChild(nav);
    }

    createAIInterface() {
        const aiInterface = document.createElement('div');
        aiInterface.className = 'ai-interface';
        aiInterface.id = 'aiInterface';
        aiInterface.innerHTML = `
            <div class="ai-header">
                <div>
                    <div class="ai-title">🤖 NEXUS AI Assistant</div>
                    <div class="ai-context">Context: ${this.dashboardType.toUpperCase()}</div>
                </div>
                <button class="ai-close" id="aiClose">×</button>
            </div>
            <div class="ai-messages" id="aiMessages">
                <div class="ai-message system">
                    AI Assistant initialized for ${this.dashboardName}
                </div>
                <div class="ai-message ai">
                    Hello! I'm your contextual AI assistant for ${this.dashboardName}. I understand this ${this.dashboardType} environment and can help you with navigation, controls, troubleshooting, and optimization. What can I help you with?
                </div>
            </div>
            <div class="ai-input-area">
                <textarea class="ai-input" id="aiInput" placeholder="Ask me about ${this.dashboardName} controls, features, or get help with your workflow..."></textarea>
                <button class="ai-send" id="aiSend">Send Message 🚀</button>
            </div>
        `;
        document.body.appendChild(aiInterface);
    }

    setupEventListeners() {
        const aiToggle = document.getElementById('aiToggle');
        const aiInterface = document.getElementById('aiInterface');
        const aiClose = document.getElementById('aiClose');
        const linkedBtn = document.getElementById('linkedBtn');
        const linkedMenu = document.getElementById('linkedMenu');
        const aiInput = document.getElementById('aiInput');
        const aiSend = document.getElementById('aiSend');

        // AI Toggle
        aiToggle.addEventListener('click', () => {
            this.aiVisible = !this.aiVisible;
            if (this.aiVisible) {
                aiInterface.classList.add('show');
                aiToggle.classList.add('active');
                setTimeout(() => aiInput.focus(), 400);
            } else {
                aiInterface.classList.remove('show');
                aiToggle.classList.remove('active');
            }
        });

        // Close AI
        aiClose.addEventListener('click', () => {
            this.aiVisible = false;
            aiInterface.classList.remove('show');
            aiToggle.classList.remove('active');
        });

        // Linked menu
        linkedBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            linkedMenu.classList.toggle('show');
        });

        // Close linked menu when clicking outside
        document.addEventListener('click', () => {
            linkedMenu.classList.remove('show');
        });

        // AI Input handling
        aiSend.addEventListener('click', () => this.sendAIMessage());
        aiInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendAIMessage();
            }
        });

        // Auto-resize textarea
        aiInput.addEventListener('input', () => {
            aiInput.style.height = 'auto';
            aiInput.style.height = Math.min(aiInput.scrollHeight, 120) + 'px';
        });
    }

    sendAIMessage() {
        const input = document.getElementById('aiInput');
        const sendBtn = document.getElementById('aiSend');
        const message = input.value.trim();

        if (!message) return;

        // Disable send button to prevent spam
        sendBtn.disabled = true;
        sendBtn.textContent = 'Sending...';

        // Add user message
        this.addAIMessage(message, 'user');
        
        // Clear input immediately
        input.value = '';
        input.style.height = 'auto';

        // Process AI response
        this.processAIMessage(message).finally(() => {
            // Re-enable send button
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send Message 🚀';
            // Focus back to input
            setTimeout(() => input.focus(), 100);
        });
    }

    addAIMessage(message, type = 'ai', id = null) {
        const messages = document.getElementById('aiMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-message ${type}`;
        if (id) messageDiv.dataset.messageId = id;
        messageDiv.textContent = message;
        messages.appendChild(messageDiv);
        messages.scrollTop = messages.scrollHeight;
    }

    removeAIMessage(id) {
        if (!id) return;
        const messageDiv = document.querySelector(`[data-message-id="${id}"]`);
        if (messageDiv) {
            messageDiv.remove();
        }
    }

    processAIMessage(message) {
        const lowerMessage = message.toLowerCase();
        
        // Add thinking indicator
        const thinkingId = Date.now();
        this.addAIMessage('Processing your request...', 'system', thinkingId);
        
        return new Promise((resolve) => {
            setTimeout(() => {
                try {
                    let response = '';

                    // Context-specific responses based on dashboard type
                    if (this.dashboardType === 'glyph-forge') {
                        if (lowerMessage.includes('create') || lowerMessage.includes('glyph')) {
                            response = `🔮 To create a new glyph in ${this.dashboardName}:\n1. Enter a name in the Glyph Editor\n2. Select the type (Emotional/Mechanical/Temporal/Worldshift)\n3. Adjust intensity with the slider\n4. Add tags for categorization\n5. Write JavaScript code for the effect\n6. Click "Create Glyph" to save it\n\nThe glyph will appear in your library and animate in the 3D world!`;
                        } else if (lowerMessage.includes('heart') || lowerMessage.includes('pulse')) {
                            response = '💓 The heartbeat system shows the world engine\'s resonance. Current pulse rate indicates system activity. You can trigger heart pulses to sync all glyphs and create resonance cascades through the 3D environment.';
                        } else if (lowerMessage.includes('help') || lowerMessage.includes('what')) {
                            response = '🔮 Glyph Forge Help:\n• Create glyphs with custom JavaScript effects\n• Use the heartbeat system for synchronization\n• Animate glyphs in real-time 3D space\n• Build libraries of reusable magical effects\n• Connect to other dashboards for integration';
                        }
                    } else if (this.dashboardType === 'world-engine') {
                        if (lowerMessage.includes('build') || lowerMessage.includes('compile')) {
                            response = '🏗️ The World Engine build process:\n1. Use "Build Success" to simulate successful compilation\n2. "Build Error" to test error handling\n3. State Vector sliders (p,i,g,c) control system parameters\n4. Parse button processes operator commands\n5. Stack Height (μ) shows processing depth';
                        } else if (lowerMessage.includes('test') || lowerMessage.includes('simulation')) {
                            response = '🧪 IDE Simulation features:\n• Tests Pass/Fail buttons simulate unit testing\n• Build Success/Error for compilation testing\n• State Vector represents system consciousness levels\n• Operators section processes natural language commands';
                        } else if (lowerMessage.includes('help') || lowerMessage.includes('what')) {
                            response = '🧠 World Engine Help:\n• Adjust consciousness parameters with sliders\n• Use IDE simulation for testing\n• Process natural language with operators\n• Monitor consciousness emergence\n• Export and integrate with other systems';
                        }
                    } else if (this.dashboardType === 'lexical-engine') {
                        if (lowerMessage.includes('vector') || lowerMessage.includes('meaning')) {
                            response = '📊 Lexical Vector Processing:\n• Meaning Vector (x) processes semantic content\n• Structure Matrix (Σ) handles grammatical patterns\n• Confidence levels show processing certainty\n• Scaling operations adjust abstraction levels';
                        } else if (lowerMessage.includes('help') || lowerMessage.includes('what')) {
                            response = '🔤 Lexical Logic Engine Help:\n• Process language through mathematical vectors\n• Analyze meaning and structure simultaneously\n• Use scaling operations for abstraction\n• Monitor confidence and processing levels\n• Integrate with other language systems';
                        }
                    } else if (this.dashboardType === 'nexus-beat') {
                        if (lowerMessage.includes('start') || lowerMessage.includes('play')) {
                            response = '🎵 NEXUS Holy Beat controls:\n• START ALL - Begins full synthesis across all modalities\n• Clock/Bus System runs at 120 BPM with φ(t) timing\n• Audio Synthesis generates at 440 Hz with 6 harmonics\n• Cross-Modal Feature Bus connects AUDIO→ART→WORLD→TRAIN\n• All green status lights indicate systems are synchronized';
                        } else if (lowerMessage.includes('train') || lowerMessage.includes('ml')) {
                            response = '🧠 ML Training system:\n• Start Session begins data collection\n• System learns from audio patterns, art generation, and world interactions\n• Export Data saves trained models\n• Cross-modal learning creates unified mathematical synthesis';
                        }
                    }

                    // Universal commands
                    if (!response) {
                        if (lowerMessage.includes('help') || lowerMessage.includes('what can')) {
                            response = '🤖 I can help you with:\n• Navigation and controls\n• Feature explanations\n• Troubleshooting issues\n• Optimization tips\n• Integration with other dashboards\n• Status monitoring\n\nTry asking specific questions about any feature you see on screen!';
                        } else if (lowerMessage.includes('navigate') || lowerMessage.includes('switch')) {
                            response = '🗺️ Navigation options:\n• 🏠 Home button returns to dashboard index\n• 🔗 Links button shows related dashboards\n• Each dashboard connects to the unified ecosystem\n• AI context automatically adapts to your current environment';
                        } else if (lowerMessage.includes('core') || lowerMessage.includes('nexus')) {
                            response = '🌌 NEXUS Core Communication:\n• All dashboards connect through the NEXUS core\n• Cross-dashboard data sharing is active\n• AI context adapts to your current environment\n• Core consciousness monitors all systems\n• Integration protocols are running smoothly';
                        } else {
                            response = `I understand you're asking about "${message}" in the context of ${this.dashboardName}. This ${this.dashboardType} environment has unique capabilities. Could you be more specific about what you'd like to know or do?`;
                        }
                    }

                    this.addAIMessage(response, 'ai');

                    // Remove thinking indicator by ID
                    this.removeAIMessage(thinkingId);
                    
                    resolve();
                } catch (error) {
                    console.error('AI Processing Error:', error);
                    this.addAIMessage('Sorry, I encountered an error processing your request. Please try again.', 'ai');
                    this.removeAIMessage(thinkingId);
                    resolve();
                }
            }, 800); // Reduced timeout for faster responses
        });
    }
}

// Auto-initialize if dashboard info is provided
window.initDashboardNav = (name, type, related = []) => {
    return new DashboardNav(name, type, related);
};

console.log('🚀 Universal Dashboard Navigation loaded');
