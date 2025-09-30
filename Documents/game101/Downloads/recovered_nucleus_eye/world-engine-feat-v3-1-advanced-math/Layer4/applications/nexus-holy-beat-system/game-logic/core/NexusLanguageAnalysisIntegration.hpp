// NEXUS Language Analysis Integration
// Bridges C++ comprehensive language analyzer with JavaScript dashboard systems
// Recovers and enhances all your lost language mapping functionality

#pragma once
#include "NexusComprehensiveLanguageAnalyzer.hpp"
#include "NexusDashboardIntegration.hpp"
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace NEXUS {

// ============ ANALYSIS JOB QUEUE ============

struct AnalysisJob {
    std::string id;
    std::string text;
    std::string language_hint; // "javascript", "html", "css", "mixed"
    std::chrono::steady_clock::time_point submitted;
    std::function<void(const std::string&)> callback;

    AnalysisJob(const std::string& job_id, const std::string& content, const std::string& hint = "mixed")
        : id(job_id), text(content), language_hint(hint), submitted(std::chrono::steady_clock::now()) {}
};

struct AnalysisResult {
    std::string job_id;
    std::string analysis_json;
    std::string detailed_breakdown;
    double processing_time_ms;
    std::chrono::steady_clock::time_point completed;
    bool success{true};
    std::string error_message;
};

// ============ LANGUAGE ANALYSIS SERVICE ============

class LanguageAnalysisService {
private:
    std::unique_ptr<ComprehensiveLanguageAnalyzer> analyzer;

    // Job processing
    std::queue<std::unique_ptr<AnalysisJob>> job_queue;
    std::queue<std::unique_ptr<AnalysisResult>> result_queue;
    std::mutex job_mutex;
    std::mutex result_mutex;
    std::condition_variable job_condition;

    // Worker threads
    std::vector<std::thread> worker_threads;
    std::atomic<bool> should_stop{false};
    int thread_count{3};

    // Statistics
    std::atomic<size_t> jobs_processed{0};
    std::atomic<size_t> total_processing_time_ms{0};
    std::atomic<size_t> jobs_failed{0};

public:
    LanguageAnalysisService() {
        analyzer = std::make_unique<ComprehensiveLanguageAnalyzer>();
        startWorkerThreads();
    }

    ~LanguageAnalysisService() {
        stop();
    }

    void startWorkerThreads() {
        for (int i = 0; i < thread_count; ++i) {
            worker_threads.emplace_back([this]() {
                workerLoop();
            });
        }
    }

    void stop() {
        should_stop.store(true);
        job_condition.notify_all();

        for (auto& thread : worker_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads.clear();
    }

    std::string submitAnalysisJob(const std::string& text,
                                 const std::string& language_hint = "mixed",
                                 std::function<void(const std::string&)> callback = nullptr) {
        std::string job_id = generateJobId();

        auto job = std::make_unique<AnalysisJob>(job_id, text, language_hint);
        if (callback) {
            job->callback = callback;
        }

        {
            std::lock_guard<std::mutex> lock(job_mutex);
            job_queue.push(std::move(job));
        }

        job_condition.notify_one();
        return job_id;
    }

    std::unique_ptr<AnalysisResult> getResult() {
        std::lock_guard<std::mutex> lock(result_mutex);
        if (result_queue.empty()) {
            return nullptr;
        }

        auto result = std::move(result_queue.front());
        result_queue.pop();
        return result;
    }

    std::vector<std::unique_ptr<AnalysisResult>> getAllResults() {
        std::vector<std::unique_ptr<AnalysisResult>> results;
        std::lock_guard<std::mutex> lock(result_mutex);

        while (!result_queue.empty()) {
            results.push_back(std::move(result_queue.front()));
            result_queue.pop();
        }

        return results;
    }

    struct ServiceStats {
        size_t jobs_processed;
        size_t jobs_failed;
        double average_processing_time_ms;
        size_t queue_size;
        size_t result_queue_size;
        bool is_running;
    };

    ServiceStats getStats() {
        ServiceStats stats;
        stats.jobs_processed = jobs_processed.load();
        stats.jobs_failed = jobs_failed.load();
        stats.average_processing_time_ms = stats.jobs_processed > 0
            ? static_cast<double>(total_processing_time_ms.load()) / stats.jobs_processed
            : 0.0;

        {
            std::lock_guard<std::mutex> lock(job_mutex);
            stats.queue_size = job_queue.size();
        }

        {
            std::lock_guard<std::mutex> lock(result_mutex);
            stats.result_queue_size = result_queue.size();
        }

        stats.is_running = !should_stop.load();
        return stats;
    }

private:
    void workerLoop() {
        while (!should_stop.load()) {
            std::unique_ptr<AnalysisJob> job;

            // Get next job
            {
                std::unique_lock<std::mutex> lock(job_mutex);
                job_condition.wait(lock, [this]() {
                    return !job_queue.empty() || should_stop.load();
                });

                if (should_stop.load()) {
                    break;
                }

                if (!job_queue.empty()) {
                    job = std::move(job_queue.front());
                    job_queue.pop();
                }
            }

            if (job) {
                processJob(std::move(job));
            }
        }
    }

    void processJob(std::unique_ptr<AnalysisJob> job) {
        auto start_time = std::chrono::steady_clock::now();
        auto result = std::make_unique<AnalysisResult>();
        result->job_id = job->id;

        try {
            // Perform comprehensive analysis
            auto analysis = analyzer->performComprehensiveAnalysis(job->text);

            // Generate JSON output
            result->analysis_json = analyzer->exportAnalysis(analysis);
            result->detailed_breakdown = analyzer->getElementDetails(analysis);
            result->success = true;

            // Call callback if provided
            if (job->callback) {
                job->callback(result->analysis_json);
            }

            jobs_processed.fetch_add(1);

        } catch (const std::exception& e) {
            result->success = false;
            result->error_message = e.what();
            result->analysis_json = "{\"error\": \"" + std::string(e.what()) + "\"}";
            jobs_failed.fetch_add(1);
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result->processing_time_ms = duration.count();
        result->completed = end_time;

        total_processing_time_ms.fetch_add(result->processing_time_ms);

        // Store result
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            result_queue.push(std::move(result));
        }
    }

    std::string generateJobId() {
        static std::atomic<size_t> counter{0};
        auto now = std::chrono::steady_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        return "lang_analysis_" + std::to_string(timestamp) + "_" + std::to_string(counter.fetch_add(1));
    }
};

// ============ DASHBOARD INTEGRATION ============

class LanguageAnalysisDashboard {
private:
    std::unique_ptr<LanguageAnalysisService> analysis_service;
    DashboardIntegrationManager* integration_manager;

    // Cache for recent analyses
    std::unordered_map<std::string, std::string> analysis_cache;
    std::mutex cache_mutex;
    static constexpr size_t max_cache_size = 100;

    // Live analysis tracking
    std::unordered_map<std::string, std::string> active_jobs; // job_id -> dashboard_id
    std::mutex jobs_mutex;

public:
    LanguageAnalysisDashboard(DashboardIntegrationManager* manager)
        : integration_manager(manager) {
        analysis_service = std::make_unique<LanguageAnalysisService>();
    }

    // Process text from dashboard
    std::string analyzeText(const std::string& dashboard_id, const std::string& text, const std::string& language_hint = "mixed") {
        // Check cache first
        std::string cache_key = generateCacheKey(text, language_hint);
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = analysis_cache.find(cache_key);
            if (it != analysis_cache.end()) {
                return it->second;
            }
        }

        // Submit for analysis
        std::string job_id = analysis_service->submitAnalysisJob(text, language_hint);

        // Track job
        {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            active_jobs[job_id] = dashboard_id;
        }

        return "{\"status\": \"processing\", \"job_id\": \"" + job_id + "\"}";
    }

    // Get analysis results
    std::string getAnalysisResults() {
        auto results = analysis_service->getAllResults();

        if (results.empty()) {
            return "{\"results\": []}";
        }

        std::ostringstream json;
        json << "{\"results\": [";

        bool first = true;
        for (const auto& result : results) {
            if (!first) json << ",";
            first = false;

            json << "{\n";
            json << "  \"job_id\": \"" << result->job_id << "\",\n";
            json << "  \"success\": " << (result->success ? "true" : "false") << ",\n";
            json << "  \"processing_time_ms\": " << result->processing_time_ms << ",\n";

            if (result->success) {
                json << "  \"analysis\": " << result->analysis_json << ",\n";
                json << "  \"detailed_breakdown\": \"" << escapeJsonString(result->detailed_breakdown) << "\"\n";
            } else {
                json << "  \"error\": \"" << result->error_message << "\"\n";
            }

            json << "}";

            // Cache successful results
            if (result->success) {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (analysis_cache.size() >= max_cache_size) {
                    analysis_cache.clear(); // Simple cache eviction
                }
                analysis_cache[result->job_id] = result->analysis_json;
            }

            // Remove from active jobs
            {
                std::lock_guard<std::mutex> lock(jobs_mutex);
                active_jobs.erase(result->job_id);
            }
        }

        json << "]}";
        return json.str();
    }

    // Get service statistics
    std::string getServiceStats() {
        auto stats = analysis_service->getStats();

        std::ostringstream json;
        json << "{\n";
        json << "  \"jobs_processed\": " << stats.jobs_processed << ",\n";
        json << "  \"jobs_failed\": " << stats.jobs_failed << ",\n";
        json << "  \"average_processing_time_ms\": " << stats.average_processing_time_ms << ",\n";
        json << "  \"queue_size\": " << stats.queue_size << ",\n";
        json << "  \"result_queue_size\": " << stats.result_queue_size << ",\n";
        json << "  \"is_running\": " << (stats.is_running ? "true" : "false") << ",\n";
        json << "  \"cache_size\": " << analysis_cache.size() << ",\n";
        json << "  \"active_jobs\": " << active_jobs.size() << "\n";
        json << "}\n";

        return json.str();
    }

    // Process batch of code samples (for training data generation)
    std::string processBatch(const std::vector<std::string>& code_samples) {
        std::vector<std::string> job_ids;

        for (const auto& sample : code_samples) {
            std::string job_id = analysis_service->submitAnalysisJob(sample);
            job_ids.push_back(job_id);
        }

        std::ostringstream json;
        json << "{\"batch_submitted\": " << job_ids.size() << ", \"job_ids\": [";

        bool first = true;
        for (const std::string& job_id : job_ids) {
            if (!first) json << ",";
            first = false;
            json << "\"" << job_id << "\"";
        }

        json << "]}";
        return json.str();
    }

    // Export analysis data for machine learning
    std::string exportTrainingData() {
        std::ostringstream training_data;
        training_data << "[\n";

        bool first = true;
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            for (const auto& cached : analysis_cache) {
                if (!first) training_data << ",\n";
                first = false;

                training_data << "  {\n";
                training_data << "    \"id\": \"" << cached.first << "\",\n";
                training_data << "    \"analysis\": " << cached.second << "\n";
                training_data << "  }";
            }
        }

        training_data << "\n]\n";
        return training_data.str();
    }

private:
    std::string generateCacheKey(const std::string& text, const std::string& language_hint) {
        // Simple hash-based cache key
        std::hash<std::string> hasher;
        size_t text_hash = hasher(text);
        size_t hint_hash = hasher(language_hint);
        return std::to_string(text_hash ^ (hint_hash << 1));
    }

    std::string escapeJsonString(const std::string& str) {
        std::string escaped;
        for (char c : str) {
            switch (c) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default: escaped += c; break;
            }
        }
        return escaped;
    }
};

// ============ JAVASCRIPT RECOVERY SYSTEM ============

const char* LANGUAGE_ANALYSIS_JS_CLIENT = R"(
// NEXUS Language Analysis Client - Recovers your lost JS/HTML mapping functionality
class NexusLanguageAnalyzer {
    constructor(websocketClient) {
        this.ws = websocketClient;
        this.analysisResults = new Map();
        this.activeJobs = new Set();
        this.analysisHistory = [];

        // Recreate your original mapping structures
        this.jsPatterns = this.initializeJSPatterns();
        this.htmlPatterns = this.initializeHTMLPatterns();
        this.morphologyRules = this.initializeMorphologyRules();

        this.setupEventHandlers();
    }

    initializeJSPatterns() {
        return {
            keywords: /\b(function|class|const|let|var|if|else|for|while|do|switch|case|break|continue|return|try|catch|finally|throw|async|await|import|export|from|default|new|this|super|extends|static|get|set|yield|typeof|instanceof|in|of|delete|void|null|undefined|true|false)\b/g,
            functions: /(?:function\s+(\w+)\s*\(|(\w+)\s*:\s*function\s*\(|(\w+)\s*=\s*function\s*\(|(\w+)\s*=\s*\([^)]*\)\s*=>)/g,
            variables: /\b[a-zA-Z_$][a-zA-Z0-9_$]*\b/g,
            strings: /(["'`])(?:(?!\1)[^\\]|\\.)*/g,
            comments: /(\/\/.*$|\/\*[\s\S]*?\*\/)/gm,
            numbers: /\b\d+\.?\d*\b/g,
            regex: /\/(?:[^\/\\\r\n]|\\.)+\/[gimsuvy]*/g
        };
    }

    initializeHTMLPatterns() {
        return {
            tags: /<\/?[a-zA-Z][a-zA-Z0-9-]*(?:\s+[^>]*)?\/?>/g,
            attributes: /\s([a-zA-Z-]+)\s*=\s*(["'][^"']*["']|\w+)/g,
            comments: /<!--[\s\S]*?-->/g,
            doctype: /<!DOCTYPE\s+html>/i,
            scripts: /<script[^>]*>[\s\S]*?<\/script>/gi,
            styles: /<style[^>]*>[\s\S]*?<\/style>/gi
        };
    }

    initializeMorphologyRules() {
        return {
            prefixes: {
                'un': 'not, reverse',
                're': 'again, back',
                'pre': 'before',
                'post': 'after',
                'auto': 'self',
                'multi': 'many',
                'super': 'above',
                'sub': 'under',
                'meta': 'beyond, about',
                'proto': 'first',
                'pseudo': 'false'
            },
            suffixes: {
                'ing': 'ongoing action',
                'ed': 'past action',
                'er': 'agent, doer',
                'ly': 'manner',
                'tion': 'action, state',
                'able': 'capable of',
                'ness': 'state, quality',
                'ment': 'result, action'
            },
            roots: {
                'script': 'writing, code',
                'graph': 'visual, chart',
                'log': 'record, logic',
                'data': 'information',
                'code': 'instructions',
                'text': 'words, content',
                'load': 'transfer, fill',
                'save': 'preserve, store',
                'get': 'retrieve, obtain',
                'set': 'establish, place',
                'run': 'execute, operate',
                'parse': 'analyze, break down',
                'build': 'construct, create',
                'test': 'verify, check',
                'handle': 'manage, process',
                'process': 'transform, execute',
                'render': 'display, show',
                'update': 'modify, refresh',
                'create': 'make, generate',
                'delete': 'remove, destroy'
            }
        };
    }

    setupEventHandlers() {
        this.ws.onMessage(this.ws.MessageType.UTILITY_DATA, (data) => {
            if (data.utility === 'language_analysis') {
                this.handleAnalysisResult(data);
            }
        });
    }

    // Main analysis function - reconstructed from your original system
    async analyzeCode(code, options = {}) {
        const analysisOptions = {
            language: options.language || this.detectLanguage(code),
            includePatterns: options.includePatterns !== false,
            includeMorphology: options.includeMorphology !== false,
            includeSemantics: options.includeSemantics !== false,
            includeComplexity: options.includeComplexity !== false,
            ...options
        };

        // Immediate client-side analysis (faster response)
        const clientAnalysis = this.performClientAnalysis(code, analysisOptions);

        // Also send to C++ backend for comprehensive analysis
        const jobId = await this.submitToBackend(code, analysisOptions);

        return {
            immediate: clientAnalysis,
            jobId: jobId,
            comprehensive: this.waitForResult(jobId)
        };
    }

    performClientAnalysis(code, options) {
        const analysis = {
            timestamp: Date.now(),
            language: options.language,
            elements: {},
            patterns: {},
            morphology: {},
            statistics: {}
        };

        // Pattern matching (reconstructed from your original system)
        if (options.includePatterns) {
            if (options.language === 'javascript' || options.language === 'mixed') {
                analysis.patterns.javascript = this.analyzeJSPatterns(code);
            }
            if (options.language === 'html' || options.language === 'mixed') {
                analysis.patterns.html = this.analyzeHTMLPatterns(code);
            }
        }

        // Morphological analysis
        if (options.includeMorphology) {
            analysis.morphology = this.analyzeMorphology(code);
        }

        // Basic statistics
        analysis.statistics = {
            total_chars: code.length,
            lines: code.split('\n').length,
            words: code.split(/\s+/).filter(w => w.length > 0).length,
            complexity_estimate: this.estimateComplexity(code)
        };

        return analysis;
    }

    analyzeJSPatterns(code) {
        const patterns = {};

        for (const [name, regex] of Object.entries(this.jsPatterns)) {
            const matches = [...code.matchAll(regex)];
            patterns[name] = {
                count: matches.length,
                matches: matches.map(m => ({
                    text: m[0],
                    index: m.index,
                    groups: m.slice(1)
                }))
            };
        }

        return patterns;
    }

    analyzeHTMLPatterns(code) {
        const patterns = {};

        for (const [name, regex] of Object.entries(this.htmlPatterns)) {
            const matches = [...code.matchAll(regex)];
            patterns[name] = {
                count: matches.length,
                matches: matches.map(m => ({
                    text: m[0],
                    index: m.index,
                    groups: m.slice(1)
                }))
            };
        }

        return patterns;
    }

    analyzeMorphology(code) {
        const words = code.match(/\b[a-zA-Z_$][a-zA-Z0-9_$]*\b/g) || [];
        const morphology = {
            words_analyzed: 0,
            morphemes_found: {},
            word_formations: {},
            meanings: {}
        };

        const uniqueWords = [...new Set(words)];

        for (const word of uniqueWords) {
            if (word.length < 3) continue; // Skip short words

            morphology.words_analyzed++;
            const analysis = this.analyzeWordMorphology(word);

            if (analysis.morphemes.length > 1) {
                morphology.word_formations[word] = analysis;

                for (const morpheme of analysis.morphemes) {
                    if (!morphology.morphemes_found[morpheme]) {
                        morphology.morphemes_found[morpheme] = 0;
                    }
                    morphology.morphemes_found[morpheme]++;

                    // Add meanings
                    const meaning = this.getMorphemeMeaning(morpheme);
                    if (meaning) {
                        morphology.meanings[morpheme] = meaning;
                    }
                }
            }
        }

        return morphology;
    }

    analyzeWordMorphology(word) {
        const morphemes = [];
        let remaining = word.toLowerCase();

        // Check prefixes
        for (const [prefix, meaning] of Object.entries(this.morphologyRules.prefixes)) {
            if (remaining.startsWith(prefix) && remaining.length > prefix.length) {
                morphemes.push(prefix);
                remaining = remaining.substring(prefix.length);
                break;
            }
        }

        // Check suffixes
        for (const [suffix, meaning] of Object.entries(this.morphologyRules.suffixes)) {
            if (remaining.endsWith(suffix) && remaining.length > suffix.length) {
                const root = remaining.substring(0, remaining.length - suffix.length);
                if (root.length > 1) {
                    morphemes.push(root);
                    morphemes.push(suffix);
                    remaining = '';
                    break;
                }
            }
        }

        // If no suffix found, add remaining as root
        if (remaining) {
            morphemes.push(remaining);
        }

        return {
            original: word,
            morphemes: morphemes,
            formation_type: this.classifyWordFormation(word)
        };
    }

    getMorphemeMeaning(morpheme) {
        return this.morphologyRules.prefixes[morpheme] ||
               this.morphologyRules.suffixes[morpheme] ||
               this.morphologyRules.roots[morpheme] ||
               null;
    }

    classifyWordFormation(word) {
        if (/^[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*/.test(word)) {
            return 'camelCase';
        } else if (/_/.test(word)) {
            return 'snake_case';
        } else if (/-/.test(word)) {
            return 'kebab-case';
        } else if (/^[A-Z]{2,}$/.test(word)) {
            return 'ACRONYM';
        } else {
            return 'simple';
        }
    }

    detectLanguage(code) {
        const jsScore = (code.match(/\b(function|const|let|var|=\>|\{|\})\b/g) || []).length;
        const htmlScore = (code.match(/<\/?[a-zA-Z][^>]*>/g) || []).length;
        const cssScore = (code.match(/[a-zA-Z-]+\s*:\s*[^;}]+[;}]/g) || []).length;

        if (htmlScore > jsScore && htmlScore > cssScore) {
            return 'html';
        } else if (cssScore > jsScore && cssScore > htmlScore) {
            return 'css';
        } else if (jsScore > 0) {
            return 'javascript';
        } else {
            return 'mixed';
        }
    }

    estimateComplexity(code) {
        let complexity = 0;

        // Control structures add complexity
        complexity += (code.match(/\b(if|else|for|while|switch|case)\b/g) || []).length;
        complexity += (code.match(/\b(function|class)\b/g) || []).length * 0.5;
        complexity += (code.match(/\btry\b/g) || []).length * 1.5;
        complexity += (code.match(/\basync\b/g) || []).length * 2;

        // Nesting adds exponential complexity
        const braces = code.match(/[{}]/g) || [];
        let nesting = 0, maxNesting = 0;
        for (const brace of braces) {
            if (brace === '{') {
                nesting++;
                maxNesting = Math.max(maxNesting, nesting);
            } else {
                nesting--;
            }
        }
        complexity += Math.pow(maxNesting, 1.5);

        return complexity;
    }

    async submitToBackend(code, options) {
        const data = {
            text: code,
            language_hint: options.language,
            options: options
        };

        const success = this.ws.sendUtilityData('language_analysis', data);

        if (success) {
            // Generate job ID (will be replaced by actual backend job ID)
            const jobId = 'job_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            this.activeJobs.add(jobId);
            return jobId;
        } else {
            throw new Error('Failed to submit analysis job to backend');
        }
    }

    handleAnalysisResult(data) {
        if (data.job_id && this.activeJobs.has(data.job_id)) {
            this.analysisResults.set(data.job_id, data);
            this.activeJobs.delete(data.job_id);

            // Add to history
            this.analysisHistory.push({
                jobId: data.job_id,
                timestamp: Date.now(),
                result: data
            });

            // Limit history size
            if (this.analysisHistory.length > 100) {
                this.analysisHistory.shift();
            }
        }
    }

    async waitForResult(jobId, timeout = 30000) {
        const startTime = Date.now();

        while (Date.now() - startTime < timeout) {
            if (this.analysisResults.has(jobId)) {
                const result = this.analysisResults.get(jobId);
                this.analysisResults.delete(jobId);
                return result;
            }

            await new Promise(resolve => setTimeout(resolve, 100));
        }

        throw new Error(`Analysis job ${jobId} timed out`);
    }

    // Utility methods for dashboard integration
    getAnalysisHistory() {
        return this.analysisHistory.slice(-20); // Last 20 analyses
    }

    getActiveJobs() {
        return Array.from(this.activeJobs);
    }

    clearCache() {
        this.analysisResults.clear();
        this.analysisHistory.length = 0;
        this.activeJobs.clear();
    }

    exportTrainingData() {
        return this.analysisHistory.map(entry => ({
            input: entry.result.original_text || '',
            output: entry.result.analysis || {},
            timestamp: entry.timestamp
        }));
    }
}

// Auto-initialize when WebSocket client is available
if (typeof NexusWebSocketClient !== 'undefined') {
    window.NexusLanguageAnalyzer = NexusLanguageAnalyzer;

    // Auto-setup for dashboards
    document.addEventListener('DOMContentLoaded', () => {
        if (window.nexusWebSocket) {
            window.languageAnalyzer = new NexusLanguageAnalyzer(window.nexusWebSocket);
            console.log('🔤 NEXUS Language Analyzer initialized - Your JS/HTML mapping system is restored!');
        }
    });
}
)";

} // namespace NEXUS
