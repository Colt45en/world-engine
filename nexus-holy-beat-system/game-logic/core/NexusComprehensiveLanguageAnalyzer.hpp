// NEXUS Comprehensive Language Analysis Engine
// Rebuilt and Enhanced - Complete JS/HTML Language Mapping System
// Integrates all previous lexical analysis, morphology, and pattern detection systems

#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <regex>
#include <functional>
#include <set>
#include <queue>

namespace NEXUS {

// ============ LANGUAGE ELEMENT TYPES ============

enum class LanguageElementType {
    // JavaScript Elements
    JS_KEYWORD,           // for, while, if, function, class, const, let, var
    JS_OPERATOR,          // +, -, *, /, =, ==, ===, &&, ||
    JS_FUNCTION,          // function definitions and calls
    JS_VARIABLE,          // variable names and references
    JS_STRING,            // string literals
    JS_NUMBER,            // numeric literals
    JS_COMMENT,           // comments
    JS_REGEX,             // regular expressions
    JS_OBJECT,            // object literals and properties
    JS_ARRAY,             // array literals and operations
    JS_ASYNC_AWAIT,       // async/await patterns
    JS_PROMISE,           // promise-related code
    JS_ARROW_FUNCTION,    // arrow functions
    JS_DESTRUCTURING,     // destructuring assignments
    JS_TEMPLATE_LITERAL,  // template strings
    JS_MODULE_IMPORT,     // import statements
    JS_MODULE_EXPORT,     // export statements

    // HTML Elements
    HTML_TAG,             // HTML tags
    HTML_ATTRIBUTE,       // HTML attributes
    HTML_TEXT_CONTENT,    // text content
    HTML_COMMENT,         // HTML comments
    HTML_DOCTYPE,         // DOCTYPE declarations
    HTML_SCRIPT_TAG,      // <script> tags
    HTML_STYLE_TAG,       // <style> tags
    HTML_FORM_ELEMENT,    // form-related elements
    HTML_SEMANTIC_TAG,    // semantic HTML5 tags
    HTML_META_TAG,        // meta tags

    // CSS Elements (in <style> or style attributes)
    CSS_SELECTOR,         // CSS selectors
    CSS_PROPERTY,         // CSS properties
    CSS_VALUE,            // CSS values
    CSS_UNIT,             // CSS units (px, em, %)
    CSS_COLOR,            // color values
    CSS_MEDIA_QUERY,      // media queries
    CSS_ANIMATION,        // animations and transitions
    CSS_FLEXBOX,          // flexbox properties
    CSS_GRID,             // grid properties

    // Pattern-based Elements
    PATTERN_LOOP,         // loop patterns
    PATTERN_CONDITION,    // conditional patterns
    PATTERN_EVENT,        // event handling patterns
    PATTERN_AJAX,         // AJAX/fetch patterns
    PATTERN_DOM_MANIPULATION, // DOM manipulation
    PATTERN_ERROR_HANDLING,   // try/catch patterns
    PATTERN_CLOSURE,      // closure patterns
    PATTERN_PROTOTYPE,    // prototype patterns
    PATTERN_CLASS,        // class patterns
    PATTERN_MODULE,       // module patterns

    // Lexical Analysis Elements
    MORPHEME_PREFIX,      // word prefixes
    MORPHEME_ROOT,        // word roots
    MORPHEME_SUFFIX,      // word suffixes
    COMPOUND_WORD,        // compound words
    ABBREVIATION,         // abbreviations and acronyms
    CAMEL_CASE,           // camelCase identifiers
    SNAKE_CASE,           // snake_case identifiers
    KEBAB_CASE,           // kebab-case identifiers

    // Semantic Elements
    SEMANTIC_INTENT,      // what the code intends to do
    SEMANTIC_COMPLEXITY,  // complexity indicators
    SEMANTIC_QUALITY,     // code quality indicators
    SEMANTIC_PERFORMANCE, // performance-related patterns
    SEMANTIC_SECURITY,    // security-related patterns

    // Meta Elements
    UNKNOWN,              // unclassified elements
    COMPOSITE             // elements that combine multiple types
};

// ============ LANGUAGE ELEMENT ============

struct LanguageElement {
    LanguageElementType type;
    std::string content;
    size_t start_position;
    size_t length;
    std::unordered_map<std::string, std::string> attributes;
    std::vector<std::string> morphemes;
    double confidence{1.0};
    std::vector<std::string> related_patterns;

    LanguageElement(LanguageElementType t, const std::string& c, size_t start = 0, size_t len = 0)
        : type(t), content(c), start_position(start), length(len) {}
};

// ============ PATTERN DEFINITIONS ============

struct LanguagePattern {
    std::string name;
    std::regex pattern;
    LanguageElementType element_type;
    std::function<void(LanguageElement&, const std::smatch&)> enhancer;
    double priority{1.0};

    LanguagePattern(const std::string& n, const std::string& p, LanguageElementType t)
        : name(n), pattern(p), element_type(t) {}
};

class LanguagePatternRegistry {
private:
    std::vector<std::unique_ptr<LanguagePattern>> patterns;

public:
    LanguagePatternRegistry() {
        initializeJavaScriptPatterns();
        initializeHTMLPatterns();
        initializeCSSPatterns();
        initializeLexicalPatterns();
        initializeSemanticPatterns();
    }

    void initializeJavaScriptPatterns() {
        // Keywords
        addPattern("js_keywords", R"(\b(function|class|const|let|var|if|else|for|while|do|switch|case|break|continue|return|try|catch|finally|throw|async|await|import|export|from|default|new|this|super|extends|static|get|set|yield|typeof|instanceof|in|of|delete|void|null|undefined|true|false)\b)", LanguageElementType::JS_KEYWORD);

        // Function definitions
        addPattern("js_function_def", R"((?:function\s+(\w+)\s*\(|(\w+)\s*:\s*function\s*\(|(\w+)\s*=\s*function\s*\(|(\w+)\s*=\s*\([^)]*\)\s*=>))", LanguageElementType::JS_FUNCTION);

        // Arrow functions
        addPattern("js_arrow_function", R"(\([^)]*\)\s*=>\s*[{(]|\w+\s*=>\s*[{(])", LanguageElementType::JS_ARROW_FUNCTION);

        // Async/await patterns
        addPattern("js_async_await", R"(\b(async\s+function|await\s+\w+|\.then\s*\(|\.catch\s*\(|new\s+Promise)\b)", LanguageElementType::JS_ASYNC_AWAIT);

        // Template literals
        addPattern("js_template_literal", R"(`[^`]*`)", LanguageElementType::JS_TEMPLATE_LITERAL);

        // Destructuring
        addPattern("js_destructuring", R"(\{[^}]+\}\s*=|\[[^\]]+\]\s*=)", LanguageElementType::JS_DESTRUCTURING);

        // Import/Export
        addPattern("js_import", R"(\bimport\s+.*\s+from\s+['"]\w+['"])", LanguageElementType::JS_MODULE_IMPORT);
        addPattern("js_export", R"(\bexport\s+(default\s+)?(?:class|function|const|let|var|\{))", LanguageElementType::JS_MODULE_EXPORT);

        // Variables
        addPattern("js_variable", R"(\b[a-zA-Z_$][a-zA-Z0-9_$]*)", LanguageElementType::JS_VARIABLE);

        // String literals
        addPattern("js_string", R"(('[^']*'|"[^"]*"))", LanguageElementType::JS_STRING);

        // Numbers
        addPattern("js_number", R"(\b\d+\.?\d*\b)", LanguageElementType::JS_NUMBER);

        // Comments
        addPattern("js_comment_line", R"(//.*$)", LanguageElementType::JS_COMMENT);
        addPattern("js_comment_block", R"(/\*[\s\S]*?\*/)", LanguageElementType::JS_COMMENT);

        // Regular expressions
        addPattern("js_regex", R"(/(?:[^/\\\r\n]|\\.)+/[gimsuvy]*)", LanguageElementType::JS_REGEX);

        // Objects
        addPattern("js_object_literal", R"(\{[^{}]*\})", LanguageElementType::JS_OBJECT);

        // Arrays
        addPattern("js_array_literal", R"(\[[^\[\]]*\])", LanguageElementType::JS_ARRAY);
    }

    void initializeHTMLPatterns() {
        // HTML tags
        addPattern("html_tag", R"(</?[a-zA-Z][a-zA-Z0-9-]*(?:\s+[^>]*)?>)", LanguageElementType::HTML_TAG);

        // HTML attributes
        addPattern("html_attribute", R"(\b[a-zA-Z-]+\s*=\s*("[^"]*"|'[^']*'|\w+))", LanguageElementType::HTML_ATTRIBUTE);

        // HTML comments
        addPattern("html_comment", R"(<!--[\s\S]*?-->)", LanguageElementType::HTML_COMMENT);

        // DOCTYPE
        addPattern("html_doctype", R"(<!DOCTYPE\s+html>)", LanguageElementType::HTML_DOCTYPE);

        // Script tags
        addPattern("html_script", R"(<script[^>]*>[\s\S]*?</script>)", LanguageElementType::HTML_SCRIPT_TAG);

        // Style tags
        addPattern("html_style", R"(<style[^>]*>[\s\S]*?</style>)", LanguageElementType::HTML_STYLE_TAG);

        // Form elements
        addPattern("html_form", R"(<(?:form|input|select|textarea|button|label|fieldset|legend)[^>]*(?:/?>|>[\s\S]*?</(?:form|select|textarea|button|label|fieldset|legend)>))", LanguageElementType::HTML_FORM_ELEMENT);

        // Semantic HTML5 tags
        addPattern("html_semantic", R"(<(?:header|footer|nav|main|section|article|aside|figure|figcaption)[^>]*(?:/?>|>[\s\S]*?</(?:header|footer|nav|main|section|article|aside|figure|figcaption)>))", LanguageElementType::HTML_SEMANTIC_TAG);

        // Meta tags
        addPattern("html_meta", R"(<meta[^>]*>)", LanguageElementType::HTML_META_TAG);
    }

    void initializeCSSPatterns() {
        // CSS selectors
        addPattern("css_selector", R"([.#]?[a-zA-Z][\w-]*(?:\[[^\]]*\])?(?::[\w-]+(?:\([^)]*\))?)*)", LanguageElementType::CSS_SELECTOR);

        // CSS properties
        addPattern("css_property", R"([a-zA-Z-]+(?=\s*:))", LanguageElementType::CSS_PROPERTY);

        // CSS values
        addPattern("css_value", R"(:\s*([^;}]+))", LanguageElementType::CSS_VALUE);

        // CSS units
        addPattern("css_unit", R"(\d+(?:px|em|rem|%|vh|vw|vmin|vmax|ex|ch|cm|mm|in|pt|pc|deg|rad|grad|turn|s|ms))", LanguageElementType::CSS_UNIT);

        // Colors
        addPattern("css_color", R"(#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|rgba\([^)]+\)|hsl\([^)]+\)|hsla\([^)]+\)|\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey)\b)", LanguageElementType::CSS_COLOR);

        // Media queries
        addPattern("css_media_query", R"(@media[^{]+)", LanguageElementType::CSS_MEDIA_QUERY);

        // Animations
        addPattern("css_animation", R"(\b(?:animation|transition|transform|@keyframes)[^{;]*)", LanguageElementType::CSS_ANIMATION);

        // Flexbox
        addPattern("css_flexbox", R"(\b(?:display:\s*flex|flex-direction|flex-wrap|flex-flow|justify-content|align-items|align-content|flex-grow|flex-shrink|flex-basis|flex|align-self)\b)", LanguageElementType::CSS_FLEXBOX);

        // Grid
        addPattern("css_grid", R"(\b(?:display:\s*grid|grid-template|grid-area|grid-column|grid-row|grid-gap|gap)\b)", LanguageElementType::CSS_GRID);
    }

    void initializeLexicalPatterns() {
        // CamelCase
        addPattern("camel_case", R"(\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*)", LanguageElementType::CAMEL_CASE);

        // snake_case
        addPattern("snake_case", R"(\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+)", LanguageElementType::SNAKE_CASE);

        // kebab-case
        addPattern("kebab_case", R"(\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+)", LanguageElementType::KEBAB_CASE);

        // Common prefixes
        addPattern("prefix_un", R"(\bun[a-zA-Z]+)", LanguageElementType::MORPHEME_PREFIX);
        addPattern("prefix_re", R"(\bre[a-zA-Z]+)", LanguageElementType::MORPHEME_PREFIX);
        addPattern("prefix_pre", R"(\bpre[a-zA-Z]+)", LanguageElementType::MORPHEME_PREFIX);
        addPattern("prefix_auto", R"(\bauto[a-zA-Z]+)", LanguageElementType::MORPHEME_PREFIX);
        addPattern("prefix_multi", R"(\bmulti[a-zA-Z]+)", LanguageElementType::MORPHEME_PREFIX);

        // Common suffixes
        addPattern("suffix_ing", R"(\b[a-zA-Z]+ing\b)", LanguageElementType::MORPHEME_SUFFIX);
        addPattern("suffix_er", R"(\b[a-zA-Z]+er\b)", LanguageElementType::MORPHEME_SUFFIX);
        addPattern("suffix_ed", R"(\b[a-zA-Z]+ed\b)", LanguageElementType::MORPHEME_SUFFIX);
        addPattern("suffix_ly", R"(\b[a-zA-Z]+ly\b)", LanguageElementType::MORPHEME_SUFFIX);
        addPattern("suffix_able", R"(\b[a-zA-Z]+able\b)", LanguageElementType::MORPHEME_SUFFIX);

        // Abbreviations and acronyms
        addPattern("abbreviation", R"(\b[A-Z]{2,}\b)", LanguageElementType::ABBREVIATION);

        // Compound words
        addPattern("compound_word", R"(\b[a-zA-Z]+(?:[A-Z][a-z]*|_[a-z]+|-[a-z]+)+)", LanguageElementType::COMPOUND_WORD);
    }

    void initializeSemanticPatterns() {
        // Loop patterns
        addPattern("pattern_loop", R"(\b(?:for|while|do)\s*\()", LanguageElementType::PATTERN_LOOP);

        // Conditional patterns
        addPattern("pattern_condition", R"(\b(?:if|else|switch)\s*\()", LanguageElementType::PATTERN_CONDITION);

        // Event handling
        addPattern("pattern_event", R"(\b(?:addEventListener|onclick|onload|onchange|on\w+\s*=)", LanguageElementType::PATTERN_EVENT);

        // AJAX/Fetch patterns
        addPattern("pattern_ajax", R"(\b(?:fetch|XMLHttpRequest|axios|ajax|\$\.get|\$\.post))", LanguageElementType::PATTERN_AJAX);

        // DOM manipulation
        addPattern("pattern_dom", R"(\b(?:getElementById|querySelector|createElement|appendChild|removeChild|innerHTML|textContent|classList))", LanguageElementType::PATTERN_DOM_MANIPULATION);

        // Error handling
        addPattern("pattern_error", R"(\b(?:try\s*\{|catch\s*\(|finally\s*\{|throw\s+new)", LanguageElementType::PATTERN_ERROR_HANDLING);

        // Closure patterns
        addPattern("pattern_closure", R"(\(\s*function\s*\(|\(\s*\(\)\s*=>\s*\{)", LanguageElementType::PATTERN_CLOSURE);

        // Prototype patterns
        addPattern("pattern_prototype", R"(\.prototype\.\w+|Object\.create|Object\.setPrototypeOf)", LanguageElementType::PATTERN_PROTOTYPE);

        // Class patterns
        addPattern("pattern_class", R"(\bclass\s+\w+(?:\s+extends\s+\w+)?|new\s+\w+\s*\()", LanguageElementType::PATTERN_CLASS);

        // Module patterns
        addPattern("pattern_module", R"(\b(?:module\.exports|exports\.\w+|require\s*\(|import\s+.*from)", LanguageElementType::PATTERN_MODULE);
    }

    void addPattern(const std::string& name, const std::string& regex_str, LanguageElementType type) {
        auto pattern = std::make_unique<LanguagePattern>(name, regex_str, type);
        patterns.push_back(std::move(pattern));
    }

    const std::vector<std::unique_ptr<LanguagePattern>>& getPatterns() const {
        return patterns;
    }
};

// ============ MORPHOLOGICAL ANALYZER ============

class MorphologicalAnalyzer {
private:
    std::unordered_map<std::string, std::vector<std::string>> prefix_map;
    std::unordered_map<std::string, std::vector<std::string>> suffix_map;
    std::unordered_map<std::string, std::vector<std::string>> root_map;
    std::set<std::string> common_words;

public:
    MorphologicalAnalyzer() {
        initializeMorphologyMaps();
        initializeCommonWords();
    }

    void initializeMorphologyMaps() {
        // Prefixes
        prefix_map["un"] = {"not", "reverse", "opposite"};
        prefix_map["re"] = {"again", "back", "anew"};
        prefix_map["pre"] = {"before", "in advance"};
        prefix_map["post"] = {"after", "later"};
        prefix_map["anti"] = {"against", "opposite"};
        prefix_map["auto"] = {"self", "automatic"};
        prefix_map["multi"] = {"many", "multiple"};
        prefix_map["super"] = {"above", "over", "beyond"};
        prefix_map["sub"] = {"under", "below", "secondary"};
        prefix_map["inter"] = {"between", "among"};
        prefix_map["intra"] = {"within", "inside"};
        prefix_map["extra"] = {"outside", "beyond"};
        prefix_map["ultra"] = {"beyond", "extremely"};
        prefix_map["pseudo"] = {"false", "fake"};
        prefix_map["semi"] = {"half", "partial"};
        prefix_map["micro"] = {"small", "tiny"};
        prefix_map["macro"] = {"large", "overall"};
        prefix_map["meta"] = {"about", "beyond", "self-referential"};
        prefix_map["proto"] = {"first", "original"};
        prefix_map["neo"] = {"new", "recent"};
        prefix_map["crypto"] = {"hidden", "secret"};

        // Suffixes
        suffix_map["ing"] = {"action", "ongoing", "present participle"};
        suffix_map["ed"] = {"past", "completed", "past participle"};
        suffix_map["er"] = {"agent", "doer", "comparative"};
        suffix_map["est"] = {"superlative", "most"};
        suffix_map["ly"] = {"manner", "adverb"};
        suffix_map["tion"] = {"action", "process", "state"};
        suffix_map["sion"] = {"action", "process", "state"};
        suffix_map["ness"] = {"state", "quality"};
        suffix_map["ment"] = {"action", "result", "state"};
        suffix_map["able"] = {"capable of", "worthy of"};
        suffix_map["ible"] = {"capable of", "worthy of"};
        suffix_map["less"] = {"without", "lacking"};
        suffix_map["ful"] = {"full of", "having"};
        suffix_map["ish"] = {"having quality of", "somewhat"};
        suffix_map["ous"] = {"full of", "having"};
        suffix_map["eous"] = {"having nature of"};
        suffix_map["ious"] = {"characterized by"};
        suffix_map["ive"] = {"having tendency to", "relating to"};
        suffix_map["ize"] = {"make", "cause to become"};
        suffix_map["ise"] = {"make", "cause to become"};
        suffix_map["fy"] = {"make", "cause"};
        suffix_map["ward"] = {"direction", "toward"};
        suffix_map["wise"] = {"manner", "direction", "respect to"};

        // Common programming roots
        root_map["script"] = {"writing", "code", "instructions"};
        root_map["graph"] = {"writing", "visual", "chart"};
        root_map["log"] = {"record", "study", "logic"};
        root_map["tech"] = {"skill", "craft", "technology"};
        root_map["data"] = {"information", "facts"};
        root_map["base"] = {"foundation", "bottom", "core"};
        root_map["code"] = {"instructions", "cipher", "program"};
        root_map["text"] = {"writing", "words", "content"};
        root_map["form"] = {"shape", "structure", "format"};
        root_map["port"] = {"carry", "transport", "gateway"};
        root_map["load"] = {"burden", "fill", "transfer"};
        root_map["save"] = {"preserve", "store", "rescue"};
        root_map["send"] = {"transmit", "dispatch", "forward"};
        root_map["get"] = {"obtain", "retrieve", "fetch"};
        root_map["set"] = {"place", "establish", "configure"};
        root_map["run"] = {"execute", "operate", "process"};
        root_map["start"] = {"begin", "initialize", "launch"};
        root_map["stop"] = {"halt", "terminate", "pause"};
        root_map["end"] = {"finish", "terminate", "conclusion"};
        root_map["open"] = {"access", "reveal", "activate"};
        root_map["close"] = {"shut", "terminate", "finish"};
        root_map["read"] = {"interpret", "access", "understand"};
        root_map["write"] = {"create", "modify", "record"};
        root_map["parse"] = {"analyze", "interpret", "break down"};
        root_map["build"] = {"construct", "create", "assemble"};
        root_map["make"] = {"create", "produce", "construct"};
        root_map["create"] = {"make", "generate", "establish"};
        root_map["delete"] = {"remove", "destroy", "erase"};
        root_map["update"] = {"modify", "refresh", "change"};
        root_map["insert"] = {"add", "place", "embed"};
        root_map["select"] = {"choose", "pick", "query"};
        root_map["search"] = {"look for", "find", "query"};
        root_map["find"] = {"locate", "discover", "search"};
        root_map["sort"] = {"arrange", "order", "organize"};
        root_map["filter"] = {"screen", "select", "refine"};
        root_map["map"] = {"transform", "convert", "associate"};
        root_map["reduce"] = {"combine", "minimize", "aggregate"};
        root_map["merge"] = {"combine", "join", "unite"};
        root_map["split"] = {"divide", "separate", "break"};
        root_map["join"] = {"connect", "combine", "unite"};
        root_map["bind"] = {"connect", "attach", "associate"};
        root_map["link"] = {"connect", "associate", "reference"};
        root_map["sync"] = {"synchronize", "coordinate", "match"};
        root_map["async"] = {"asynchronous", "independent", "parallel"};
        root_map["call"] = {"invoke", "summon", "execute"};
        root_map["invoke"] = {"call", "trigger", "activate"};
        root_map["trigger"] = {"activate", "initiate", "cause"};
        root_map["handle"] = {"manage", "process", "deal with"};
        root_map["process"] = {"handle", "execute", "transform"};
        root_map["manage"] = {"control", "handle", "oversee"};
        root_map["control"] = {"manage", "direct", "command"};
        root_map["monitor"] = {"watch", "observe", "track"};
        root_map["track"] = {"follow", "monitor", "trace"};
        root_map["trace"] = {"track", "follow", "debug"};
        root_map["debug"] = {"fix", "troubleshoot", "analyze"};
        root_map["test"] = {"verify", "check", "validate"};
        root_map["check"] = {"verify", "test", "examine"};
        root_map["validate"] = {"verify", "check", "confirm"};
        root_map["verify"] = {"confirm", "check", "validate"};
        root_map["confirm"] = {"verify", "affirm", "validate"};
    }

    void initializeCommonWords() {
        // Common JavaScript/HTML words that shouldn't be broken down
        common_words = {
            "function", "class", "const", "var", "let", "if", "else", "for", "while", "do",
            "try", "catch", "throw", "return", "import", "export", "from", "default",
            "async", "await", "new", "this", "super", "extends", "static", "get", "set",
            "typeof", "instanceof", "in", "of", "delete", "void", "null", "undefined",
            "true", "false", "document", "window", "console", "Object", "Array", "String",
            "Number", "Boolean", "Date", "RegExp", "Math", "JSON", "Promise", "Error",
            "div", "span", "body", "head", "html", "meta", "link", "script", "style",
            "title", "header", "footer", "nav", "main", "section", "article", "aside",
            "form", "input", "button", "select", "option", "textarea", "label", "img",
            "table", "tr", "td", "th", "tbody", "thead", "tfoot", "ul", "ol", "li",
            "width", "height", "color", "background", "border", "margin", "padding",
            "display", "position", "top", "left", "right", "bottom", "font", "text"
        };
    }

    std::vector<std::string> analyzeMorphemes(const std::string& word) {
        std::vector<std::string> morphemes;

        // Skip common words
        if (common_words.find(word) != common_words.end()) {
            morphemes.push_back(word);
            return morphemes;
        }

        std::string remaining = word;

        // Check for prefixes
        for (const auto& prefix : prefix_map) {
            if (remaining.substr(0, prefix.first.length()) == prefix.first &&
                remaining.length() > prefix.first.length()) {
                morphemes.push_back(prefix.first);
                remaining = remaining.substr(prefix.first.length());
                break;
            }
        }

        // Check for suffixes
        for (const auto& suffix : suffix_map) {
            if (remaining.length() > suffix.first.length() &&
                remaining.substr(remaining.length() - suffix.first.length()) == suffix.first) {
                std::string root = remaining.substr(0, remaining.length() - suffix.first.length());
                if (!root.empty()) {
                    morphemes.push_back(root);
                    morphemes.push_back(suffix.first);
                    return morphemes;
                }
            }
        }

        // If no suffix found, check if entire remaining part is a known root
        if (root_map.find(remaining) != root_map.end()) {
            morphemes.push_back(remaining);
        } else if (!remaining.empty()) {
            // Add as unknown morpheme
            morphemes.push_back(remaining);
        }

        return morphemes;
    }

    std::vector<std::string> getMeanings(const std::string& morpheme) {
        auto prefix_it = prefix_map.find(morpheme);
        if (prefix_it != prefix_map.end()) {
            return prefix_it->second;
        }

        auto suffix_it = suffix_map.find(morpheme);
        if (suffix_it != suffix_map.end()) {
            return suffix_it->second;
        }

        auto root_it = root_map.find(morpheme);
        if (root_it != root_map.end()) {
            return root_it->second;
        }

        return {"unknown"};
    }
};

// ============ SEMANTIC ANALYZER ============

class SemanticAnalyzer {
private:
    std::unordered_map<std::string, double> complexity_weights;
    std::unordered_map<std::string, double> quality_indicators;
    std::unordered_map<std::string, std::vector<std::string>> intent_patterns;

public:
    SemanticAnalyzer() {
        initializeComplexityWeights();
        initializeQualityIndicators();
        initializeIntentPatterns();
    }

    void initializeComplexityWeights() {
        // Complexity indicators
        complexity_weights["nested_loop"] = 3.0;
        complexity_weights["recursive_call"] = 2.5;
        complexity_weights["async_await"] = 2.0;
        complexity_weights["promise_chain"] = 2.0;
        complexity_weights["try_catch"] = 1.5;
        complexity_weights["switch_statement"] = 1.5;
        complexity_weights["conditional"] = 1.0;
        complexity_weights["loop"] = 1.0;
        complexity_weights["function_call"] = 0.5;
        complexity_weights["variable_assignment"] = 0.1;
    }

    void initializeQualityIndicators() {
        // Code quality indicators (positive values = good, negative = bad)
        quality_indicators["descriptive_naming"] = 1.0;
        quality_indicators["comments_present"] = 1.0;
        quality_indicators["error_handling"] = 1.0;
        quality_indicators["consistent_indentation"] = 0.5;
        quality_indicators["semantic_html"] = 1.0;
        quality_indicators["accessibility_attributes"] = 1.0;
        quality_indicators["css_organization"] = 0.5;

        // Negative indicators
        quality_indicators["magic_numbers"] = -0.5;
        quality_indicators["deeply_nested"] = -1.0;
        quality_indicators["unused_variables"] = -0.5;
        quality_indicators["inline_styles"] = -0.3;
        quality_indicators["missing_alt_text"] = -0.5;
        quality_indicators["no_error_handling"] = -1.0;
    }

    void initializeIntentPatterns() {
        // What the code is trying to accomplish
        intent_patterns["data_processing"] = {"map", "filter", "reduce", "transform", "parse", "process"};
        intent_patterns["ui_interaction"] = {"click", "hover", "focus", "scroll", "resize", "load"};
        intent_patterns["data_retrieval"] = {"fetch", "get", "load", "request", "query", "select"};
        intent_patterns["data_storage"] = {"save", "store", "persist", "cache", "set", "write"};
        intent_patterns["validation"] = {"validate", "check", "verify", "test", "assert", "ensure"};
        intent_patterns["manipulation"] = {"create", "update", "delete", "modify", "change", "edit"};
        intent_patterns["navigation"] = {"route", "navigate", "redirect", "link", "goto", "switch"};
        intent_patterns["animation"] = {"animate", "transition", "fade", "slide", "rotate", "scale"};
        intent_patterns["communication"] = {"send", "receive", "broadcast", "notify", "emit", "listen"};
        intent_patterns["authentication"] = {"login", "logout", "authenticate", "authorize", "session", "token"};
        intent_patterns["optimization"] = {"cache", "lazy", "throttle", "debounce", "compress", "minify"};
        intent_patterns["debugging"] = {"log", "trace", "debug", "inspect", "monitor", "profile"};
    }

    double calculateComplexity(const std::vector<LanguageElement>& elements) {
        double complexity = 0.0;
        int nesting_level = 0;

        for (const auto& element : elements) {
            if (element.type == LanguageElementType::PATTERN_LOOP) {
                complexity += complexity_weights["loop"] * (1.0 + nesting_level * 0.5);
                nesting_level++;
            } else if (element.type == LanguageElementType::PATTERN_CONDITION) {
                complexity += complexity_weights["conditional"] * (1.0 + nesting_level * 0.3);
                nesting_level++;
            } else if (element.type == LanguageElementType::JS_ASYNC_AWAIT) {
                complexity += complexity_weights["async_await"];
            } else if (element.type == LanguageElementType::PATTERN_ERROR_HANDLING) {
                complexity += complexity_weights["try_catch"];
            }

            // Closing braces decrease nesting (simplified)
            if (element.content == "}") {
                nesting_level = std::max(0, nesting_level - 1);
            }
        }

        return complexity;
    }

    double calculateQuality(const std::vector<LanguageElement>& elements, const std::string& full_text) {
        double quality = 0.0;

        // Check for good practices
        bool has_comments = false;
        bool has_error_handling = false;
        bool has_semantic_html = false;

        for (const auto& element : elements) {
            if (element.type == LanguageElementType::JS_COMMENT ||
                element.type == LanguageElementType::HTML_COMMENT) {
                has_comments = true;
            }
            if (element.type == LanguageElementType::PATTERN_ERROR_HANDLING) {
                has_error_handling = true;
            }
            if (element.type == LanguageElementType::HTML_SEMANTIC_TAG) {
                has_semantic_html = true;
            }
        }

        if (has_comments) quality += quality_indicators["comments_present"];
        if (has_error_handling) quality += quality_indicators["error_handling"];
        if (has_semantic_html) quality += quality_indicators["semantic_html"];

        // Check for bad practices
        std::regex magic_number_regex(R"(\b\d{2,}\b)"); // Numbers with 2+ digits might be magic
        if (std::regex_search(full_text, magic_number_regex)) {
            quality += quality_indicators["magic_numbers"];
        }

        // Check nesting depth
        int max_nesting = 0;
        int current_nesting = 0;
        for (char c : full_text) {
            if (c == '{') {
                current_nesting++;
                max_nesting = std::max(max_nesting, current_nesting);
            } else if (c == '}') {
                current_nesting--;
            }
        }
        if (max_nesting > 4) {
            quality += quality_indicators["deeply_nested"];
        }

        return quality;
    }

    std::vector<std::string> identifyIntents(const std::vector<LanguageElement>& elements) {
        std::vector<std::string> intents;
        std::unordered_map<std::string, int> intent_scores;

        for (const auto& element : elements) {
            std::string lower_content = element.content;
            std::transform(lower_content.begin(), lower_content.end(), lower_content.begin(), ::tolower);

            for (const auto& intent_pair : intent_patterns) {
                const std::string& intent = intent_pair.first;
                const std::vector<std::string>& keywords = intent_pair.second;

                for (const std::string& keyword : keywords) {
                    if (lower_content.find(keyword) != std::string::npos) {
                        intent_scores[intent]++;
                    }
                }
            }
        }

        // Return intents with scores above threshold
        for (const auto& score_pair : intent_scores) {
            if (score_pair.second >= 2) { // Threshold
                intents.push_back(score_pair.first);
            }
        }

        return intents;
    }
};

// ============ MAIN LANGUAGE ANALYSIS ENGINE ============

class ComprehensiveLanguageAnalyzer {
private:
    std::unique_ptr<LanguagePatternRegistry> pattern_registry;
    std::unique_ptr<MorphologicalAnalyzer> morphology_analyzer;
    std::unique_ptr<SemanticAnalyzer> semantic_analyzer;

public:
    ComprehensiveLanguageAnalyzer() {
        pattern_registry = std::make_unique<LanguagePatternRegistry>();
        morphology_analyzer = std::make_unique<MorphologicalAnalyzer>();
        semantic_analyzer = std::make_unique<SemanticAnalyzer>();
    }

    std::vector<LanguageElement> analyzeText(const std::string& text) {
        std::vector<LanguageElement> elements;

        // Apply all patterns
        for (const auto& pattern : pattern_registry->getPatterns()) {
            std::regex_iterator<std::string::const_iterator> iter(text.begin(), text.end(), pattern->pattern);
            std::regex_iterator<std::string::const_iterator> end;

            for (; iter != end; ++iter) {
                const std::smatch& match = *iter;
                LanguageElement element(pattern->element_type, match.str(),
                                      match.position(), match.length());

                // Enhance element with morphological analysis
                if (pattern->element_type == LanguageElementType::JS_VARIABLE ||
                    pattern->element_type == LanguageElementType::JS_FUNCTION) {
                    element.morphemes = morphology_analyzer->analyzeMorphemes(match.str());
                }

                // Add pattern-specific attributes
                if (pattern->enhancer) {
                    pattern->enhancer(element, match);
                }

                elements.push_back(element);
            }
        }

        // Sort elements by position
        std::sort(elements.begin(), elements.end(),
                 [](const LanguageElement& a, const LanguageElement& b) {
                     return a.start_position < b.start_position;
                 });

        return elements;
    }

    struct AnalysisResult {
        std::vector<LanguageElement> elements;
        double complexity_score;
        double quality_score;
        std::vector<std::string> identified_intents;
        std::unordered_map<LanguageElementType, size_t> element_counts;
        std::unordered_map<std::string, std::vector<std::string>> morpheme_meanings;
    };

    AnalysisResult performComprehensiveAnalysis(const std::string& text) {
        AnalysisResult result;

        // Basic element extraction
        result.elements = analyzeText(text);

        // Count elements by type
        for (const auto& element : result.elements) {
            result.element_counts[element.type]++;
        }

        // Semantic analysis
        result.complexity_score = semantic_analyzer->calculateComplexity(result.elements);
        result.quality_score = semantic_analyzer->calculateQuality(result.elements, text);
        result.identified_intents = semantic_analyzer->identifyIntents(result.elements);

        // Morphological meanings
        for (const auto& element : result.elements) {
            for (const std::string& morpheme : element.morphemes) {
                result.morpheme_meanings[morpheme] = morphology_analyzer->getMeanings(morpheme);
            }
        }

        return result;
    }

    // Export analysis as JSON-like structure for dashboard integration
    std::string exportAnalysis(const AnalysisResult& result) {
        std::ostringstream json;
        json << "{\n";
        json << "  \"summary\": {\n";
        json << "    \"total_elements\": " << result.elements.size() << ",\n";
        json << "    \"complexity_score\": " << result.complexity_score << ",\n";
        json << "    \"quality_score\": " << result.quality_score << ",\n";
        json << "    \"unique_morphemes\": " << result.morpheme_meanings.size() << "\n";
        json << "  },\n";
        json << "  \"element_types\": {\n";

        bool first = true;
        for (const auto& count : result.element_counts) {
            if (!first) json << ",\n";
            first = false;
            json << "    \"" << static_cast<int>(count.first) << "\": " << count.second;
        }

        json << "\n  },\n";
        json << "  \"identified_intents\": [";

        first = true;
        for (const std::string& intent : result.identified_intents) {
            if (!first) json << ", ";
            first = false;
            json << "\"" << intent << "\"";
        }

        json << "],\n";
        json << "  \"morpheme_count\": " << result.morpheme_meanings.size() << "\n";
        json << "}\n";

        return json.str();
    }

    // Get detailed element information for debugging
    std::string getElementDetails(const AnalysisResult& result) {
        std::ostringstream details;
        details << "=== COMPREHENSIVE LANGUAGE ANALYSIS ===\n\n";

        details << "Elements found: " << result.elements.size() << "\n";
        details << "Complexity score: " << result.complexity_score << "\n";
        details << "Quality score: " << result.quality_score << "\n\n";

        details << "=== ELEMENT BREAKDOWN ===\n";
        for (const auto& element : result.elements) {
            details << "Type: " << static_cast<int>(element.type) << " | ";
            details << "Content: \"" << element.content << "\" | ";
            details << "Position: " << element.start_position << "-"
                   << (element.start_position + element.length) << "\n";

            if (!element.morphemes.empty()) {
                details << "  Morphemes: ";
                for (const std::string& morpheme : element.morphemes) {
                    details << morpheme << " ";
                }
                details << "\n";
            }
            details << "\n";
        }

        return details.str();
    }
};

} // namespace NEXUS
