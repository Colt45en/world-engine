/**
 * Natural Language to Tier-4 Operations Translator
 * ================================================
 *
 * Translates natural language descriptions into sequences of Tier-4 operators.
 * Uses pattern matching, semantic analysis, and intent recognition.
 */

interface OperatorIntent {
    operators: string[];
    confidence: number;
    reasoning: string;
    category: string;
}

interface LanguagePattern {
    pattern: RegExp;
    operators: string[];
    confidence: number;
    category: string;
    description: string;
}

class NaturalLanguageProcessor {
    private patterns: LanguagePattern[] = [];
    private contextKeywords: Map<string, string[]> = new Map();
    private operatorDescriptions: Map<string, string> = new Map();

    constructor() {
        this.initializePatterns();
        this.initializeContext();
        this.initializeOperatorDescriptions();
    }

    // ============================= Pattern Initialization =============================

    private initializePatterns() {
        // Development workflow patterns
        this.patterns.push(
            {
                pattern: /optimize|improve|enhance|refine/i,
                operators: ['ST', 'SL', 'PR', 'RC'],
                confidence: 0.8,
                category: 'optimization',
                description: 'Snapshot → Select → Prevent → Recompute'
            },
            {
                pattern: /debug|fix|troubleshoot|diagnose/i,
                operators: ['TL', 'SL', 'ED', 'RS'],
                confidence: 0.85,
                category: 'debugging',
                description: 'Make visible → Select → Edit → Restore'
            },
            {
                pattern: /refactor|restructure|reorganize|rebuild/i,
                operators: ['ST', 'CP', 'CV', 'MD'],
                confidence: 0.9,
                category: 'refactoring',
                description: 'Snapshot → Factor → Convert → Package'
            },
            {
                pattern: /component.*structure|structure.*component/i,
                operators: ['ST', 'CP', 'MD', 'TL'],
                confidence: 0.75,
                category: 'architecture',
                description: 'Snapshot → Factor → Package → Visualize'
            },
            {
                pattern: /analyze|examine|investigate|study/i,
                operators: ['ST', 'SL', 'CP'],
                confidence: 0.8,
                category: 'analysis',
                description: 'IDE_A: Analysis path'
            },
            {
                pattern: /constraint|limit|restrict|bound/i,
                operators: ['CV', 'PR', 'RC'],
                confidence: 0.8,
                category: 'constraints',
                description: 'IDE_B: Constraint path'
            },
            {
                pattern: /build|construct|create|assemble/i,
                operators: ['TL', 'RB', 'MD'],
                confidence: 0.8,
                category: 'construction',
                description: 'IDE_C: Build path'
            },
            {
                pattern: /integrate|merge|combine|unify/i,
                operators: ['MERGE_ABC'],
                confidence: 0.9,
                category: 'integration',
                description: 'Three Ides merger with dimension preservation'
            },
            {
                pattern: /reset|restore|revert|undo/i,
                operators: ['RS'],
                confidence: 0.95,
                category: 'restoration',
                description: 'Restore previous state'
            },
            {
                pattern: /save|snapshot|checkpoint|backup/i,
                operators: ['ST'],
                confidence: 0.95,
                category: 'persistence',
                description: 'Save current state'
            },
            {
                pattern: /update|modify|change|alter/i,
                operators: ['UP'],
                confidence: 0.7,
                category: 'modification',
                description: 'Move along current manifold'
            },
            {
                pattern: /select|choose|pick|focus/i,
                operators: ['SL'],
                confidence: 0.8,
                category: 'selection',
                description: 'Focus on specific aspects'
            },
            {
                pattern: /prevent|guard|protect|safeguard/i,
                operators: ['PR'],
                confidence: 0.85,
                category: 'protection',
                description: 'Apply constraints and safeguards'
            },
            {
                pattern: /convert|transform|translate|adapt/i,
                operators: ['CV'],
                confidence: 0.8,
                category: 'transformation',
                description: 'Change representation'
            },
            {
                pattern: /edit|modify|adjust|tweak/i,
                operators: ['ED'],
                confidence: 0.75,
                category: 'editing',
                description: 'Structural modifications'
            },
            {
                pattern: /stabilize|balance|steady|calm/i,
                operators: ['PR', 'RC', 'TL'],
                confidence: 0.8,
                category: 'stabilization',
                description: 'Prevent → Recompute → Visualize'
            }
        );

        // Complex workflow patterns
        this.patterns.push(
            {
                pattern: /clean.*code|code.*clean/i,
                operators: ['ST', 'SL', 'ED', 'PR', 'RC'],
                confidence: 0.8,
                category: 'code-quality',
                description: 'Snapshot → Select → Edit → Prevent → Recompute'
            },
            {
                pattern: /performance.*issue|slow.*code/i,
                operators: ['ST', 'SL', 'CP', 'PR', 'UP'],
                confidence: 0.75,
                category: 'performance',
                description: 'Snapshot → Select → Factor → Prevent → Update'
            },
            {
                pattern: /add.*feature|new.*functionality/i,
                operators: ['ST', 'CP', 'ED', 'TL', 'MD'],
                confidence: 0.8,
                category: 'feature-development',
                description: 'Snapshot → Factor → Edit → Visualize → Package'
            },
            {
                pattern: /remove.*code|delete.*unused/i,
                operators: ['ST', 'SL', 'ED', 'PR'],
                confidence: 0.85,
                category: 'code-removal',
                description: 'Snapshot → Select → Edit → Prevent'
            },
            {
                pattern: /api.*design|design.*api/i,
                operators: ['ST', 'CP', 'CV', 'PR', 'TL'],
                confidence: 0.8,
                category: 'api-design',
                description: 'Snapshot → Factor → Convert → Prevent → Visualize'
            }
        );
    }

    private initializeContext() {
        // Context keywords that modify operator selection
        this.contextKeywords.set('quickly', ['UP', 'ST']);
        this.contextKeywords.set('carefully', ['PR', 'ST', 'SL']);
        this.contextKeywords.set('completely', ['ST', 'CP', 'ED', 'TL']);
        this.contextKeywords.set('safely', ['PR', 'ST', 'RS']);
        this.contextKeywords.set('efficiently', ['ST', 'SL', 'UP']);
        this.contextKeywords.set('thoroughly', ['ST', 'SL', 'CP', 'PR']);
        this.contextKeywords.set('incrementally', ['UP', 'ST']);
        this.contextKeywords.set('systematically', ['ST', 'SL', 'CP', 'PR', 'RC']);
    }

    private initializeOperatorDescriptions() {
        this.operatorDescriptions.set('RB', 'Rebuild - Recompose from parts (concretize)');
        this.operatorDescriptions.set('UP', 'Update - Move along current manifold');
        this.operatorDescriptions.set('ST', 'Snapshot - Save current state');
        this.operatorDescriptions.set('PR', 'Prevent - Apply constraints and safeguards');
        this.operatorDescriptions.set('ED', 'Edit - Structural modifications');
        this.operatorDescriptions.set('RS', 'Restore - Revert to previous state');
        this.operatorDescriptions.set('CV', 'Convert - Change representation');
        this.operatorDescriptions.set('SL', 'Select - Focus on specific aspects');
        this.operatorDescriptions.set('CH', 'Channel - Direct information flow');
        this.operatorDescriptions.set('MD', 'Module - Package into components');
        this.operatorDescriptions.set('TL', 'Telescope - Make visible/transparent');
        this.operatorDescriptions.set('RC', 'Recompute - Recalculate with new parameters');
        this.operatorDescriptions.set('CP', 'Factor - Extract common patterns');

        // Macros
        this.operatorDescriptions.set('IDE_A', 'Analysis Path - ST → SL → CP');
        this.operatorDescriptions.set('IDE_B', 'Constraint Path - CV → PR → RC');
        this.operatorDescriptions.set('IDE_C', 'Build Path - TL → RB → MD');
        this.operatorDescriptions.set('MERGE_ABC', 'Three Ides Integration - All paths combined safely');
        this.operatorDescriptions.set('OPTIMIZE', 'Standard Optimization - ST → SL → PR → RC');
        this.operatorDescriptions.set('DEBUG', 'Debug Workflow - TL → SL → ED → RS');
        this.operatorDescriptions.set('STABILIZE', 'Stabilize System - PR → RC → TL');
    }

    // ============================= Main Processing Function =============================

    public parseNaturalLanguage(input: string): OperatorIntent {
        const cleanInput = input.trim().toLowerCase();

        // Find matching patterns
        const matches: { pattern: LanguagePattern; score: number }[] = [];

        for (const pattern of this.patterns) {
            if (pattern.pattern.test(input)) {
                let score = pattern.confidence;

                // Boost score for exact keyword matches
                const keywords = this.extractKeywords(cleanInput);
                score += this.calculateKeywordBoost(keywords, pattern.category);

                // Context modifiers
                score += this.calculateContextBoost(cleanInput);

                matches.push({ pattern, score: Math.min(1.0, score) });
            }
        }

        // Sort by score
        matches.sort((a, b) => b.score - a.score);

        if (matches.length === 0) {
            return this.getDefaultIntent(cleanInput);
        }

        // Use best match
        const bestMatch = matches[0];
        const operators = this.refineOperators(bestMatch.pattern.operators, cleanInput);

        return {
            operators,
            confidence: bestMatch.score,
            reasoning: this.generateReasoning(bestMatch.pattern, cleanInput, operators),
            category: bestMatch.pattern.category
        };
    }

    // ============================= Helper Functions =============================

    private extractKeywords(input: string): string[] {
        // Simple keyword extraction
        return input.split(/\s+/).filter(word =>
            word.length > 3 &&
            !['the', 'and', 'for', 'with', 'this', 'that', 'from', 'they', 'have', 'been'].includes(word)
        );
    }

    private calculateKeywordBoost(keywords: string[], category: string): number {
        // Boost based on category-specific keywords
        const categoryKeywords: Record<string, string[]> = {
            optimization: ['fast', 'speed', 'performance', 'efficient'],
            debugging: ['error', 'bug', 'issue', 'problem', 'broken'],
            refactoring: ['clean', 'structure', 'organize', 'modular'],
            architecture: ['design', 'pattern', 'component', 'module'],
            analysis: ['understand', 'examine', 'investigate', 'study'],
            constraints: ['safe', 'secure', 'validate', 'check'],
            construction: ['make', 'create', 'build', 'develop'],
            integration: ['connect', 'join', 'link', 'combine']
        };

        const relevantKeywords = categoryKeywords[category] || [];
        const matchCount = keywords.filter(k => relevantKeywords.includes(k)).length;

        return matchCount * 0.1; // 0.1 boost per relevant keyword
    }

    private calculateContextBoost(input: string): number {
        let boost = 0;

        for (const [contextWord, operators] of this.contextKeywords) {
            if (input.includes(contextWord)) {
                boost += 0.05; // Small boost for context words
            }
        }

        return boost;
    }

    private refineOperators(baseOperators: string[], input: string): string[] {
        let operators = [...baseOperators];

        // Add context-specific operators
        for (const [contextWord, contextOps] of this.contextKeywords) {
            if (input.includes(contextWord)) {
                // Insert context operators appropriately
                if (contextWord === 'safely' && !operators.includes('PR')) {
                    operators = ['PR', ...operators];
                }
                if (contextWord === 'quickly' && !operators.includes('ST')) {
                    operators = ['ST', ...operators];
                }
            }
        }

        // Remove duplicates while preserving order
        return [...new Set(operators)];
    }

    private generateReasoning(pattern: LanguagePattern, input: string, operators: string[]): string {
        const operatorNames = operators.map(op => {
            const desc = this.operatorDescriptions.get(op);
            return desc ? `${op} (${desc.split(' - ')[1]})` : op;
        }).join(' → ');

        return `Matched "${pattern.category}" pattern from "${input}". ` +
            `Sequence: ${operatorNames}. ` +
            `Confidence: ${(pattern.confidence * 100).toFixed(0)}%`;
    }

    private getDefaultIntent(input: string): OperatorIntent {
        // Fallback for unrecognized inputs
        const keywords = this.extractKeywords(input);

        // Simple heuristics for unknown inputs
        if (keywords.some(k => ['problem', 'error', 'issue', 'broken'].includes(k))) {
            return {
                operators: ['TL', 'SL', 'ED'],
                confidence: 0.4,
                reasoning: `Detected problem-related keywords. Using basic debugging sequence.`,
                category: 'fallback-debug'
            };
        }

        if (keywords.some(k => ['make', 'create', 'add', 'new'].includes(k))) {
            return {
                operators: ['ST', 'CP', 'ED'],
                confidence: 0.4,
                reasoning: `Detected creation-related keywords. Using basic construction sequence.`,
                category: 'fallback-create'
            };
        }

        // Default general purpose sequence
        return {
            operators: ['ST', 'SL'],
            confidence: 0.2,
            reasoning: `No specific pattern matched. Using safe default: Snapshot → Select.`,
            category: 'fallback-default'
        };
    }

    // ============================= Advanced Features =============================

    public suggestAlternatives(input: string): OperatorIntent[] {
        const primary = this.parseNaturalLanguage(input);
        const alternatives: OperatorIntent[] = [primary];

        // Generate alternative interpretations
        const cleanInput = input.trim().toLowerCase();

        // Try different categories
        const allMatches: { pattern: LanguagePattern; score: number }[] = [];
        for (const pattern of this.patterns) {
            if (pattern.pattern.test(input)) {
                let score = pattern.confidence * 0.8; // Slightly lower for alternatives
                allMatches.push({ pattern, score });
            }
        }

        // Add top 2 alternatives
        allMatches
            .sort((a, b) => b.score - a.score)
            .slice(1, 3) // Skip the primary (first) match
            .forEach(match => {
                const operators = this.refineOperators(match.pattern.operators, cleanInput);
                alternatives.push({
                    operators,
                    confidence: match.score,
                    reasoning: this.generateReasoning(match.pattern, cleanInput, operators),
                    category: match.pattern.category + '-alt'
                });
            });

        return alternatives;
    }

    public explainOperatorSequence(operators: string[]): string {
        const explanations = operators.map(op => {
            const desc = this.operatorDescriptions.get(op);
            return desc || `${op} (Unknown operator)`;
        });

        return `Operator sequence explanation:\n${explanations.map((exp, i) =>
            `${i + 1}. ${exp}`).join('\n')}`;
    }

    public validateSequence(operators: string[]): { valid: boolean; warnings: string[] } {
        const warnings: string[] = [];

        // Check for potentially problematic sequences
        for (let i = 0; i < operators.length - 1; i++) {
            const current = operators[i];
            const next = operators[i + 1];

            // Warning: RS after ED without ST
            if (current === 'ED' && next === 'RS' && !operators.slice(0, i).includes('ST')) {
                warnings.push('Warning: Restoring after Edit without prior Snapshot may lose changes');
            }

            // Warning: Multiple UP without ST
            if (current === 'UP' && next === 'UP') {
                warnings.push('Warning: Multiple Updates without Snapshot may cause drift');
            }

            // Warning: PR without following action
            if (current === 'PR' && i === operators.length - 2 && !['RC', 'UP', 'ST'].includes(next)) {
                warnings.push('Warning: Prevent should typically be followed by Recompute, Update, or Snapshot');
            }
        }

        return {
            valid: warnings.length === 0,
            warnings
        };
    }

    // ============================= Usage Examples =============================

    public getExampleQueries(): { query: string; expectedOperators: string[]; category: string }[] {
        return [
            {
                query: "optimize the component structure",
                expectedOperators: ['ST', 'SL', 'PR', 'RC'],
                category: 'optimization'
            },
            {
                query: "debug this performance issue",
                expectedOperators: ['TL', 'SL', 'ED', 'RS'],
                category: 'debugging'
            },
            {
                query: "refactor the API design carefully",
                expectedOperators: ['PR', 'ST', 'CP', 'CV', 'PR', 'TL'],
                category: 'refactoring'
            },
            {
                query: "integrate the three analysis systems",
                expectedOperators: ['MERGE_ABC'],
                category: 'integration'
            },
            {
                query: "add a new feature systematically",
                expectedOperators: ['ST', 'SL', 'CP', 'PR', 'RC', 'CP', 'ED', 'TL', 'MD'],
                category: 'feature-development'
            },
            {
                query: "safely remove unused code",
                expectedOperators: ['PR', 'ST', 'SL', 'ED', 'PR'],
                category: 'code-removal'
            }
        ];
    }
}

// ============================= Export =============================

export { NaturalLanguageProcessor, OperatorIntent, LanguagePattern };

// Usage example:
/*
const nlp = new NaturalLanguageProcessor();

const intent = nlp.parseNaturalLanguage("optimize the component structure");
console.log(intent);
// Output: {
//   operators: ['ST', 'SL', 'PR', 'RC'],
//   confidence: 0.8,
//   reasoning: 'Matched "optimization" pattern...',
//   category: 'optimization'
// }

const alternatives = nlp.suggestAlternatives("debug this issue");
console.log(alternatives);

const explanation = nlp.explainOperatorSequence(['ST', 'SL', 'ED']);
console.log(explanation);
*/
