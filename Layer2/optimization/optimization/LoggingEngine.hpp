/**
 * Logging Engine - Comprehensive structured logging and event analysis
 * ==================================================================
 *
 * Features:
 * - Structured event logging with multiple levels
 * - Real-time log streaming and filtering
 * - Event correlation and pattern detection
 * - Performance metrics and timing analysis
 * - Configurable output formats (JSON, structured text)
 * - Log aggregation and search capabilities
 */

#pragma once

#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace LoggingEngine {

enum class LogLevel : int {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

struct LogEvent {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string category;
    std::string message;
    std::string thread_id;
    std::string source_file;
    int source_line;
    std::unordered_map<std::string, std::string> metadata;
    uint64_t sequence_id;

    // Performance tracking
    std::chrono::nanoseconds duration;
    std::string correlation_id;
};

struct LogFilter {
    LogLevel min_level = LogLevel::TRACE;
    std::vector<std::string> categories;
    std::vector<std::string> excluded_categories;
    std::function<bool(const LogEvent&)> custom_filter;
    bool include_performance = true;
    bool include_metadata = true;
};

struct LogMetrics {
    uint64_t total_events = 0;
    std::unordered_map<LogLevel, uint64_t> events_by_level;
    std::unordered_map<std::string, uint64_t> events_by_category;
    std::chrono::nanoseconds total_processing_time{0};
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point last_event_time;
};

class LogFormatter {
public:
    virtual ~LogFormatter() = default;
    virtual std::string format(const LogEvent& event) const = 0;
};

class JSONFormatter : public LogFormatter {
public:
    std::string format(const LogEvent& event) const override;
};

class TextFormatter : public LogFormatter {
public:
    TextFormatter(bool include_metadata = true, bool colorize = false);
    std::string format(const LogEvent& event) const override;

private:
    bool include_metadata_;
    bool colorize_;
};

class LogSink {
public:
    virtual ~LogSink() = default;
    virtual void write(const LogEvent& event, const std::string& formatted) = 0;
    virtual void flush() = 0;
};

class FileSink : public LogSink {
public:
    explicit FileSink(const std::string& filename, bool rotate_daily = false);
    void write(const LogEvent& event, const std::string& formatted) override;
    void flush() override;

private:
    std::string filename_;
    std::ofstream file_;
    bool rotate_daily_;
    std::string current_date_;
    std::mutex write_mutex_;

    void check_rotation();
};

class ConsoleSink : public LogSink {
public:
    explicit ConsoleSink(bool use_stderr_for_errors = true);
    void write(const LogEvent& event, const std::string& formatted) override;
    void flush() override;

private:
    bool use_stderr_for_errors_;
    std::mutex write_mutex_;
};

class MemorySink : public LogSink {
public:
    explicit MemorySink(size_t max_events = 10000);
    void write(const LogEvent& event, const std::string& formatted) override;
    void flush() override;

    std::vector<LogEvent> get_events(const LogFilter& filter = {}) const;
    void clear();

private:
    mutable std::mutex events_mutex_;
    std::queue<LogEvent> events_;
    size_t max_events_;
};

class LoggingCore {
public:
    static LoggingCore& instance();

    // Configuration
    void set_level(LogLevel level);
    void add_sink(std::unique_ptr<LogSink> sink, std::unique_ptr<LogFormatter> formatter = nullptr);
    void set_filter(const LogFilter& filter);

    // Logging methods
    void log(LogLevel level, const std::string& category, const std::string& message,
             const std::string& file = "", int line = 0,
             const std::unordered_map<std::string, std::string>& metadata = {});

    template<typename... Args>
    void log_formatted(LogLevel level, const std::string& category,
                      const std::string& format, Args&&... args);

    // Performance logging
    class PerformanceTimer {
    public:
        PerformanceTimer(const std::string& category, const std::string& operation);
        ~PerformanceTimer();

        void add_metadata(const std::string& key, const std::string& value);

    private:
        std::string category_;
        std::string operation_;
        std::chrono::high_resolution_clock::time_point start_time_;
        std::unordered_map<std::string, std::string> metadata_;
    };

    // Analysis and metrics
    LogMetrics get_metrics() const;
    std::vector<LogEvent> search_events(const std::string& query, const LogFilter& filter = {}) const;
    std::vector<LogEvent> get_correlated_events(const std::string& correlation_id) const;

    // Event streaming
    using StreamCallback = std::function<void(const LogEvent&)>;
    void add_stream_callback(StreamCallback callback);
    void remove_stream_callback(StreamCallback callback);

    // Control
    void flush_all();
    void shutdown();

private:
    LoggingCore();
    ~LoggingCore();

    struct SinkPair {
        std::unique_ptr<LogSink> sink;
        std::unique_ptr<LogFormatter> formatter;
    };

    LogLevel current_level_;
    LogFilter current_filter_;
    std::vector<SinkPair> sinks_;
    std::vector<StreamCallback> stream_callbacks_;

    // Threading
    std::thread worker_thread_;
    std::queue<LogEvent> event_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_requested_{false};

    // Metrics
    LogMetrics metrics_;
    mutable std::mutex metrics_mutex_;
    std::atomic<uint64_t> sequence_counter_{0};

    void worker_loop();
    void process_event(const LogEvent& event);
    bool passes_filter(const LogEvent& event) const;
    std::string generate_correlation_id();
};

// Convenience macros
#define LOG_TRACE(category, message, ...) \
    LoggingEngine::LoggingCore::instance().log(LoggingEngine::LogLevel::TRACE, category, message, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_DEBUG(category, message, ...) \
    LoggingEngine::LoggingCore::instance().log(LoggingEngine::LogLevel::DEBUG, category, message, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_INFO(category, message, ...) \
    LoggingEngine::LoggingCore::instance().log(LoggingEngine::LogLevel::INFO, category, message, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_WARN(category, message, ...) \
    LoggingEngine::LoggingCore::instance().log(LoggingEngine::LogLevel::WARN, category, message, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_ERROR(category, message, ...) \
    LoggingEngine::LoggingCore::instance().log(LoggingEngine::LogLevel::ERROR, category, message, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_FATAL(category, message, ...) \
    LoggingEngine::LoggingCore::instance().log(LoggingEngine::LogLevel::FATAL, category, message, __FILE__, __LINE__, ##__VA_ARGS__)

#define PERF_TIMER(category, operation) \
    LoggingEngine::LoggingCore::PerformanceTimer timer(category, operation)

} // namespace LoggingEngine
