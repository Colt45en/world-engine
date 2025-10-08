#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include <mutex>
#include <memory>
#include <iomanip>
#include <ctime>

namespace NEXUS {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    CRITICAL = 5
};

struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string category;
    std::string message;
    std::string file;
    int line;
    std::string thread_id;

    LogEntry(LogLevel l, const std::string& cat, const std::string& msg,
             const std::string& f = "", int ln = 0)
        : timestamp(std::chrono::system_clock::now()), level(l),
          category(cat), message(msg), file(f), line(ln) {

        std::ostringstream oss;
        oss << std::this_thread::get_id();
        thread_id = oss.str();
    }
};

class LogFormatter {
public:
    virtual ~LogFormatter() = default;
    virtual std::string format(const LogEntry& entry) = 0;
};

class StandardFormatter : public LogFormatter {
public:
    std::string format(const LogEntry& entry) override {
        std::ostringstream oss;

        // Timestamp
        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp.time_since_epoch()) % 1000;

        std::tm* tm_info = std::localtime(&time_t);
        oss << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();

        // Level
        oss << " [" << levelToString(entry.level) << "]";

        // Category
        if (!entry.category.empty()) {
            oss << " [" << entry.category << "]";
        }

        // Thread ID (shortened)
        oss << " [T:" << entry.thread_id.substr(0, 8) << "]";

        // Message
        oss << " " << entry.message;

        // File and line (if available)
        if (!entry.file.empty() && entry.line > 0) {
            size_t pos = entry.file.find_last_of("/\\");
            std::string filename = (pos != std::string::npos)
                ? entry.file.substr(pos + 1) : entry.file;
            oss << " (" << filename << ":" << entry.line << ")";
        }

        return oss.str();
    }

private:
    std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return " INFO";
            case LogLevel::WARN:  return " WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return " CRIT";
            default: return "UNKNW";
        }
    }
};

class ColoredFormatter : public LogFormatter {
public:
    std::string format(const LogEntry& entry) override {
        std::ostringstream oss;

        // Color codes for different log levels
        std::string color = getColorCode(entry.level);
        std::string reset = "\033[0m";

        // Timestamp
        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp.time_since_epoch()) % 1000;

        std::tm* tm_info = std::localtime(&time_t);
        oss << "\033[90m" << std::put_time(tm_info, "%H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count() << reset;

        // Level with color
        oss << " " << color << "[" << levelToString(entry.level) << "]" << reset;

        // Category with color
        if (!entry.category.empty()) {
            oss << " \033[96m[" << entry.category << "]" << reset;
        }

        // Message
        oss << " " << entry.message;

        // File info in dim color
        if (!entry.file.empty() && entry.line > 0) {
            size_t pos = entry.file.find_last_of("/\\");
            std::string filename = (pos != std::string::npos)
                ? entry.file.substr(pos + 1) : entry.file;
            oss << " \033[90m(" << filename << ":" << entry.line << ")" << reset;
        }

        return oss.str();
    }

private:
    std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return " INFO";
            case LogLevel::WARN:  return " WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return " CRIT";
            default: return "UNKNW";
        }
    }

    std::string getColorCode(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "\033[37m";    // Light gray
            case LogLevel::DEBUG: return "\033[36m";    // Cyan
            case LogLevel::INFO:  return "\033[32m";    // Green
            case LogLevel::WARN:  return "\033[33m";    // Yellow
            case LogLevel::ERROR: return "\033[31m";    // Red
            case LogLevel::CRITICAL: return "\033[35m"; // Magenta
            default: return "\033[0m";                  // Reset
        }
    }
};

class LogSink {
public:
    virtual ~LogSink() = default;
    virtual void write(const LogEntry& entry, const std::string& formatted) = 0;
    virtual void flush() = 0;
};

class ConsoleSink : public LogSink {
private:
    std::unique_ptr<LogFormatter> formatter;
    LogLevel min_level;
    mutable std::mutex mutex;

public:
    ConsoleSink(LogLevel min_level = LogLevel::INFO, bool colored = true)
        : min_level(min_level) {
        if (colored) {
            formatter = std::make_unique<ColoredFormatter>();
        } else {
            formatter = std::make_unique<StandardFormatter>();
        }
    }

    void write(const LogEntry& entry, const std::string& formatted) override {
        if (entry.level < min_level) return;

        std::lock_guard<std::mutex> lock(mutex);

        if (entry.level >= LogLevel::ERROR) {
            std::cerr << formatted << std::endl;
        } else {
            std::cout << formatted << std::endl;
        }
    }

    void flush() override {
        std::lock_guard<std::mutex> lock(mutex);
        std::cout.flush();
        std::cerr.flush();
    }
};

class FileSink : public LogSink {
private:
    std::ofstream file;
    std::unique_ptr<LogFormatter> formatter;
    LogLevel min_level;
    mutable std::mutex mutex;
    size_t max_file_size;
    size_t current_file_size;
    std::string base_filename;

public:
    FileSink(const std::string& filename, LogLevel min_level = LogLevel::TRACE,
             size_t max_size_mb = 50)
        : min_level(min_level), max_file_size(max_size_mb * 1024 * 1024),
          current_file_size(0), base_filename(filename) {
        formatter = std::make_unique<StandardFormatter>();
        openFile(filename);
    }

    ~FileSink() {
        if (file.is_open()) {
            file.close();
        }
    }

    void write(const LogEntry& entry, const std::string& formatted) override {
        if (entry.level < min_level) return;

        std::lock_guard<std::mutex> lock(mutex);

        if (!file.is_open()) return;

        std::string output = formatted + "\n";
        file << output;
        current_file_size += output.length();

        // Rotate file if too large
        if (current_file_size > max_file_size) {
            rotateFile();
        }
    }

    void flush() override {
        std::lock_guard<std::mutex> lock(mutex);
        if (file.is_open()) {
            file.flush();
        }
    }

private:
    void openFile(const std::string& filename) {
        file.open(filename, std::ios::app);
        if (file.is_open()) {
            file.seekp(0, std::ios::end);
            current_file_size = file.tellp();
        }
    }

    void rotateFile() {
        if (file.is_open()) {
            file.close();
        }

        // Rename current file with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm* tm_info = std::localtime(&time_t);

        std::ostringstream oss;
        oss << base_filename << "."
            << std::put_time(tm_info, "%Y%m%d_%H%M%S") << ".old";

        std::rename(base_filename.c_str(), oss.str().c_str());

        // Open new file
        openFile(base_filename);
        current_file_size = 0;
    }
};

class MemorySink : public LogSink {
private:
    std::vector<LogEntry> entries;
    std::unique_ptr<LogFormatter> formatter;
    size_t max_entries;
    LogLevel min_level;
    mutable std::mutex mutex;

public:
    MemorySink(size_t max_entries = 1000, LogLevel min_level = LogLevel::TRACE)
        : max_entries(max_entries), min_level(min_level) {
        formatter = std::make_unique<StandardFormatter>();
        entries.reserve(max_entries);
    }

    void write(const LogEntry& entry, const std::string& formatted) override {
        if (entry.level < min_level) return;

        std::lock_guard<std::mutex> lock(mutex);

        entries.push_back(entry);

        // Remove oldest entries if we exceed max
        if (entries.size() > max_entries) {
            entries.erase(entries.begin(), entries.begin() + (entries.size() - max_entries));
        }
    }

    void flush() override {
        // Memory sink doesn't need flushing
    }

    std::vector<LogEntry> getEntries() const {
        std::lock_guard<std::mutex> lock(mutex);
        return entries;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        entries.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return entries.size();
    }
};

class NexusLogger {
private:
    std::vector<std::unique_ptr<LogSink>> sinks;
    std::unique_ptr<LogFormatter> formatter;
    LogLevel global_min_level;
    mutable std::mutex mutex;
    static std::unique_ptr<NexusLogger> instance;

public:
    static NexusLogger& getInstance() {
        static std::once_flag once_flag;
        std::call_once(once_flag, []() {
            instance = std::unique_ptr<NexusLogger>(new NexusLogger());
        });
        return *instance;
    }

    void addSink(std::unique_ptr<LogSink> sink) {
        std::lock_guard<std::mutex> lock(mutex);
        sinks.push_back(std::move(sink));
    }

    void setGlobalMinLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex);
        global_min_level = level;
    }

    void log(LogLevel level, const std::string& category, const std::string& message,
             const std::string& file = "", int line = 0) {
        if (level < global_min_level) return;

        LogEntry entry(level, category, message, file, line);
        std::string formatted = formatter->format(entry);

        std::lock_guard<std::mutex> lock(mutex);
        for (auto& sink : sinks) {
            sink->write(entry, formatted);
        }
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& sink : sinks) {
            sink->flush();
        }
    }

    void dumpSystemState() {
        log(LogLevel::INFO, "SYSTEM", "=== NEXUS System State Dump ===");

        // Memory usage
        log(LogLevel::INFO, "MEMORY", "Current memory usage: [Implementation needed]");

        // Active entities
        log(LogLevel::INFO, "ENTITIES", "Active entities: [Implementation needed]");

        // Quantum processing state
        log(LogLevel::INFO, "QUANTUM", "Active processing modes: [Implementation needed]");

        // Cognitive engine state
        log(LogLevel::INFO, "COGNITIVE", "Active thoughts and memories: [Implementation needed]");

        log(LogLevel::INFO, "SYSTEM", "=== End System State Dump ===");
        flush();
    }

    void createDebugDump(const std::string& filename) {
        std::ofstream dump(filename);
        if (!dump.is_open()) {
            log(LogLevel::ERROR, "LOGGER", "Failed to create debug dump: " + filename);
            return;
        }

        dump << "NEXUS Debug Dump\n";
        dump << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
        dump << "=====================================\n\n";

        // Find memory sink and dump its contents
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& sink : sinks) {
            auto* memory_sink = dynamic_cast<MemorySink*>(sink.get());
            if (memory_sink) {
                auto entries = memory_sink->getEntries();
                dump << "Recent Log Entries (" << entries.size() << "):\n";
                dump << "----------------------------------------\n";
                for (const auto& entry : entries) {
                    dump << formatter->format(entry) << "\n";
                }
                dump << "\n";
                break;
            }
        }

        dump << "System State:\n";
        dump << "-------------\n";
        dump << "Global min level: " << static_cast<int>(global_min_level) << "\n";
        dump << "Active sinks: " << sinks.size() << "\n";

        dump.close();
        log(LogLevel::INFO, "LOGGER", "Debug dump created: " + filename);
    }

private:
    NexusLogger() : global_min_level(LogLevel::TRACE) {
        formatter = std::make_unique<StandardFormatter>();

        // Add default console sink
        addSink(std::make_unique<ConsoleSink>(LogLevel::INFO, true));
    }
};

std::unique_ptr<NexusLogger> NexusLogger::instance = nullptr;

// Convenience macros
#define NEXUS_LOG_TRACE(category, message) \
    NEXUS::NexusLogger::getInstance().log(NEXUS::LogLevel::TRACE, category, message, __FILE__, __LINE__)

#define NEXUS_LOG_DEBUG(category, message) \
    NEXUS::NexusLogger::getInstance().log(NEXUS::LogLevel::DEBUG, category, message, __FILE__, __LINE__)

#define NEXUS_LOG_INFO(category, message) \
    NEXUS::NexusLogger::getInstance().log(NEXUS::LogLevel::INFO, category, message, __FILE__, __LINE__)

#define NEXUS_LOG_WARN(category, message) \
    NEXUS::NexusLogger::getInstance().log(NEXUS::LogLevel::WARN, category, message, __FILE__, __LINE__)

#define NEXUS_LOG_ERROR(category, message) \
    NEXUS::NexusLogger::getInstance().log(NEXUS::LogLevel::ERROR, category, message, __FILE__, __LINE__)

#define NEXUS_LOG_CRITICAL(category, message) \
    NEXUS::NexusLogger::getInstance().log(NEXUS::LogLevel::CRITICAL, category, message, __FILE__, __LINE__)

// Scoped logging for function entry/exit
class ScopedLogger {
private:
    std::string function_name;
    std::string category;
    std::chrono::high_resolution_clock::time_point start_time;

public:
    ScopedLogger(const std::string& func, const std::string& cat = "SCOPE")
        : function_name(func), category(cat) {
        start_time = std::chrono::high_resolution_clock::now();
        NEXUS_LOG_TRACE(category, "Entering " + function_name);
    }

    ~ScopedLogger() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);

        std::ostringstream oss;
        oss << "Exiting " << function_name << " (took " << duration.count() << "μs)";
        NEXUS_LOG_TRACE(category, oss.str());
    }
};

#define NEXUS_SCOPED_LOG(function_name) \
    NEXUS::ScopedLogger _scoped_logger(function_name)

#define NEXUS_SCOPED_LOG_CAT(function_name, category) \
    NEXUS::ScopedLogger _scoped_logger(function_name, category)

} // namespace NEXUS
