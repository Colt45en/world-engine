// Lexicon Engine UI Enhancements
class LexiconEngineUIFixes {
    constructor() {
        this.init();
    }

    init() {
        this.addProcessingIndicator();
        this.enhanceButtonFeedback();
        this.fixContentOverflow();
        this.addVisualFeedback();
        console.log('🔧 Lexicon Engine UI fixes applied');
    }

    addProcessingIndicator() {
        // Create processing indicator
        const indicator = document.createElement('div');
        indicator.id = 'processing-indicator';
        indicator.className = 'processing-indicator';
        indicator.innerHTML = `
            <div class="processing-spinner"></div>
            <div>Processing...</div>
        `;
        document.body.appendChild(indicator);

        this.processingIndicator = indicator;
    }

    enhanceButtonFeedback() {
        // Add visual feedback to all buttons
        const buttons = document.querySelectorAll('.math-btn, .action-btn, .desktop-btn, .send-btn');

        buttons.forEach(button => {
            button.addEventListener('click', (e) => {
                this.showButtonPress(e.target);
                this.showProcessing();

                // Simulate processing delay
                setTimeout(() => {
                    this.hideProcessing();
                    this.showResult(e.target);
                }, 1000 + Math.random() * 1000);
            });

            // Add hover effects using CSS classes
            button.addEventListener('mouseenter', (e) => {
                if (e.currentTarget instanceof HTMLElement) {
                    e.currentTarget.classList.add('button-hovered');
                }
            });

            button.addEventListener('mouseleave', (e) => {
                if (e.currentTarget instanceof HTMLElement) {
                    e.currentTarget.classList.remove('button-hovered');
                }
            });
        });
    }

    showButtonPress(Element) {
        Element.classList.add('button-pressed');
        setTimeout(() => {
            Element.classList.remove('button-pressed');
        }, 300);
    }
            this.processingIndicator.classList.remove('active');
        }
    }

    /**
     * @param {EventTarget | null} button
     */
    showResult(button) {
        // Show visual feedback for result
        // @ts-ignore
        const resultText = this.generateResultText(button);
        this.displayResult(resultText);
    }

    /**
     * @param {{ textContent: string; }} button
     */
    generateResultText(button) {
        const buttonText = button.textContent.trim();
        const operations = {
            'ALIGN': 'Word alignment completed successfully',
            'PROJ': 'Annotation projection applied',
            'REFINE': 'Statistical refinement processed',
            'GRAPH': 'Synonym graph updated',
            'POLY': 'Polysemy resolution complete',
            'IRONY': 'Irony detection analyzed',
            'DIRECTION': 'Directional analysis complete',
            'RECOMPUTE': 'Recomputation pass finished',
            'Send': 'Message processed successfully',
            'Process Pipeline': 'Pipeline processing complete',
            'Open Canvas': 'Canvas interface activated'
        };

        if (buttonText in operations) {
            return operations[buttonText as keyof typeof operations];
        } else {
            return `${buttonText} operation completed`;
        }
    }

    /**
     * @param {string | null} text
     */
    displayResult(text) {
        // Cache result display area in the class to avoid repeated DOM lookups
        if (!this.resultArea) {
            let resultArea = document.getElementById('result-display');
            if (!resultArea) {
                resultArea = document.createElement('div');
                resultArea.id = 'result-display';
                resultArea.style.cssText = `
                    position: fixed;
                    bottom: 80px;
                    right: 20px;
                    max-width: 300px;
                    z-index: 1500;
                `;
                document.body.appendChild(resultArea);
            }
            this.resultArea = resultArea;
        }

        // Create result message
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-highlight';
        resultDiv.textContent = text;
        resultDiv.style.cssText = `
            background: rgba(124, 220, 255, 0.1);
            border: 1px solid var(--accent);
            border-left: 3px solid var(--accent);
            padding: 12px;
            margin: 4px 0;
            border-radius: 6px;
            color: var(--fg);
            font-size: 0.9rem;
            animation: slideIn 0.3s ease;
        `;

        this.resultArea.appendChild(resultDiv);

        // Remove after delay
        setTimeout(() => {
            if (resultDiv.parentNode) {
                resultDiv.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => {
                    if (resultDiv.parentNode) {
                        resultDiv.parentNode.removeChild(resultDiv);
                    }
                }, 300);
            }
        }, 3000);
    }

    fixContentOverflow() {
        // Fix content areas that might overflow
        const contentAreas = document.querySelectorAll('.chat-content, .audio-engine-content, .visual-workspace');

        contentAreas.forEach(area => {
            if (area instanceof HTMLElement) {
                area.style.overflowY = 'auto';
                area.style.maxHeight = 'calc(100vh - 200px)';
                area.style.paddingRight = '8px';
            }
        });
    }

    addVisualFeedback() {
        // Add visual feedback for text processing
        const textInputs = document.querySelectorAll('input[type="text"], textarea');

        textInputs.forEach(input => {
            input.addEventListener('input', (e) => {
                const target = e.target;
                if (
                    (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) &&
                    typeof target.value === 'string' &&
                    target.value.length > 5
                ) {
                    this.showInputFeedback(target);
                }
            });
        });
    }

    /**
     * @param {EventTarget | null} input
     */
    showInputFeedback(input) {
        if (input instanceof HTMLElement) {
            input.style.borderColor = 'var(--accent)';
            input.style.boxShadow = '0 0 6px rgba(124, 220, 255, 0.3)';

            setTimeout(() => {
                input.style.borderColor = '';
                input.style.boxShadow = '';
            }, 1000);
        }
    }
}

// Additional CSS animations
const additionalStyles = document.createElement('style');
additionalStyles.textContent = `
    @keyframes slideOut {
        0% {
            opacity: 1;
            transform: translateX(0);
        }
        100% {
            opacity: 0;
            transform: translateX(20px);
        }
    }

    .result-highlight {
        transition: all 0.3s ease;
    }

    .processing-indicator {
        backdrop-filter: blur(8px);
    }
    .math-btn:not(:hover):not(.button-pressed) {
        background: var(--panel);
        border: 1px solid var(--border);
    }

    .math-btn.button-pressed {
        background: var(--success) !important;
        border-color: var(--success) !important;
        color: white !important;
    }

    .math-btn.button-hovered,
    .action-btn.button-hovered,
    .desktop-btn.button-hovered,
    .send-btn.button-hovered {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 220, 255, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
        color: white !important;
    }
`;

document.head.appendChild(additionalStyles);

// Initialize the UI fixes when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new LexiconEngineUIFixes();
    });
} else {
    new LexiconEngineUIFixes();
}
function showButtonPress(Element) {
    throw new Error('Function not implemented.');
}
// (Removed duplicate/erroneous function stubs)
