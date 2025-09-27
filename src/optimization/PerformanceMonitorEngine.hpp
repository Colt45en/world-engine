/**
 * Performance Monitor Engine - System metrics, profiling, and optimization
 * =======================================================================
 *
 * Features:
 * - Real-time performance metrics collection
 * - CPU, Memory, I/O, and Network monitoring
 * - Application-level profiling and hotspot detection
 * - Performance regression analysis
 * - Adaptive optimization suggestions
 * - Resource usage forecasting
 * - Performance alerting and notifications
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <condition_variable>

namespace PerformanceMonitor {

using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::nanoseconds;

enum class MetricType {
    COUNTER,        // Monotonically increasing value
    GAUGE,          // Current value that can go up/down
    HISTOGRAM,      // Distribution of values
    TIMER,          // Timing measurements
    RATE           // Events per unit time
};

enum class AlertLevel {
    INFO,
    WARNING,
    CRITICAL,
    FATAL
};

struct MetricValue {
    double value;
    TimePoint timestamp;
    std::unordered_map<std::string, std::string> tags;
    std::unordered_map<std::string, double> attributes;
};

struct PerformanceMetric {
    std::string name;
    std::string category;
    MetricType type;
    std::string unit;
    std::string description;
    std::vector<MetricValue> values;

    // Statistical summaries
    double min_value = std::numeric_limits<double>::max();
    double max_value = std::numeric_limits<double>::lowest();
    double sum_value = 0.0;
    size_t count = 0;

    // For histograms
    std::vector<double> buckets;
    std::vector<size_t> bucket_counts;

    // Configuration
    Duration retention_period = std::chrono::hours(24);
    size_t max_values = 10000;
    bool enabled = true;
};

struct SystemMetrics {
    // CPU metrics
    double cpu_usage_percent = 0.0;
    double cpu_user_percent = 0.0;
    double cpu_system_percent = 0.0;
    double cpu_idle_percent = 0.0;
    double load_average_1min = 0.0;
    double load_average_5min = 0.0;
    double load_average_15min = 0.0;

    // Memory metrics
    uint64_t memory_total_bytes = 0;
    uint64_t memory_used_bytes = 0;
    uint64_t memory_available_bytes = 0;
    double memory_usage_percent = 0.0;
    uint64_t memory_cached_bytes = 0;
    uint64_t memory_buffers_bytes = 0;

    // Disk I/O metrics
    uint64_t disk_read_bytes_per_sec = 0;
    uint64_t disk_write_bytes_per_sec = 0;
    uint64_t disk_read_ops_per_sec = 0;
    uint64_t disk_write_ops_per_sec = 0;
    double disk_usage_percent = 0.0;

    // Network metrics
    uint64_t network_rx_bytes_per_sec = 0;
    uint64_t network_tx_bytes_per_sec = 0;
    uint64_t network_rx_packets_per_sec = 0;
    uint64_t network_tx_packets_per_sec = 0;

    TimePoint collection_time;
};

struct ApplicationMetrics {
    // Process metrics
    uint64_t process_memory_rss_bytes = 0;
    uint64_t process_memory_vms_bytes = 0;
    double process_cpu_percent = 0.0;
    uint64_t process_threads = 0;
    uint64_t process_open_files = 0;

    // Application-specific metrics
    uint64_t requests_per_second = 0;
    double average_response_time_ms = 0.0;
    uint64_t active_connections = 0;
    uint64_t error_count = 0;
    double error_rate_percent = 0.0;

    // Resource pools
    std::unordered_map<std::string, size_t> pool_sizes;
    std::unordered_map<std::string, size_t> pool_usage;

    TimePoint collection_time;
};

struct ProfileSample {
    std::string function_name;
    std::string file_name;
    int line_number;
    Duration execution_time;
    Duration self_time;
    uint64_t call_count;
    std::vector<ProfileSample> child_samples;

    // Memory profiling
    uint64_t allocated_bytes = 0;
    uint64_t deallocated_bytes = 0;
    int64_t net_allocation = 0;

    TimePoint sample_time;
};

struct PerformanceAlert {
    std::string metric_name;
    AlertLevel level;
    std::string message;
    double threshold_value;
    double current_value;
    TimePoint alert_time;
    Duration duration; // How long the condition has persisted
    std::unordered_map<std::string, std::string> context;
};

class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    // Control
    void start_monitoring(Duration collection_interval = std::chrono::seconds(1));
    void stop_monitoring();
    bool is_monitoring() const;

    // Data access
    SystemMetrics get_current_metrics() const;
    std::vector<SystemMetrics> get_metrics_history(Duration window = std::chrono::minutes(5)) const;

    // Configuration
    void set_collection_interval(Duration interval);
    void enable_metric_collection(const std::string& metric_category, bool enabled);

    // Callbacks
    using MetricsCallback = std::function<void(const SystemMetrics&)>;
    void on_metrics_collected(MetricsCallback callback);

private:
    std::atomic<bool> monitoring_active_{false};
    Duration collection_interval_{std::chrono::seconds(1)};
    std::thread monitor_thread_;

    mutable std::mutex metrics_mutex_;
    std::queue<SystemMetrics> metrics_history_;
    size_t max_history_size_ = 3600; // 1 hour at 1-second intervals

    std::vector<MetricsCallback> callbacks_;
    std::unordered_set<std::string> disabled_categories_;

    void monitor_loop();
    SystemMetrics collect_system_metrics();

    // Platform-specific implementations
#ifdef _WIN32
    SystemMetrics collect_windows_metrics();
#elif defined(__linux__)
    SystemMetrics collect_linux_metrics();
#elif defined(__APPLE__)
    SystemMetrics collect_macos_metrics();
#endif
};

class ApplicationProfiler {
public:
    ApplicationProfiler();
    ~ApplicationProfiler();

    // Profiling control
    void start_profiling();
    void stop_profiling();
    void pause_profiling();
    void resume_profiling();
    bool is_profiling() const;

    // Sample collection
    void begin_sample(const std::string& function_name,
                     const std::string& file_name = "", int line_number = 0);
    void end_sample();

    // Memory tracking
    void track_allocation(size_t bytes, const std::string& category = "");
    void track_deallocation(size_t bytes, const std::string& category = "");

    // Custom metrics
    void record_metric(const std::string& name, double value,
                      const std::unordered_map<std::string, std::string>& tags = {});

    // Analysis
    std::vector<ProfileSample> get_profile_tree() const;
    std::vector<ProfileSample> get_hotspots(size_t top_n = 10) const;
    ApplicationMetrics get_application_metrics() const;

    // Reporting
    std::string generate_report(bool include_call_tree = true) const;
    void export_profile(const std::string& filename, const std::string& format = "json") const;

private:
    std::atomic<bool> profiling_active_{false};
    std::atomic<bool> profiling_paused_{false};

    struct SampleFrame {
        std::string function_name;
        std::string file_name;
        int line_number;
        TimePoint start_time;
        uint64_t allocated_bytes;
        uint64_t initial_memory;
    };

    thread_local static std::vector<SampleFrame> sample_stack_;

    mutable std::mutex profile_mutex_;
    std::vector<ProfileSample> profile_samples_;
    std::unordered_map<std::string, PerformanceMetric> custom_metrics_;

    ApplicationMetrics current_app_metrics_;

    ProfileSample build_sample_tree(const std::vector<ProfileSample>& samples) const;
    void update_application_metrics();
};

class MetricsCollector {
public:
    MetricsCollector();
    ~MetricsCollector();

    // Metric registration
    void register_metric(const std::string& name, MetricType type,
                        const std::string& unit = "", const std::string& description = "");
    void unregister_metric(const std::string& name);

    // Data recording
    void record_counter(const std::string& name, double value = 1.0,
                       const std::unordered_map<std::string, std::string>& tags = {});
    void record_gauge(const std::string& name, double value,
                     const std::unordered_map<std::string, std::string>& tags = {});
    void record_histogram(const std::string& name, double value,
                         const std::unordered_map<std::string, std::string>& tags = {});
    void record_timer(const std::string& name, Duration duration,
                     const std::unordered_map<std::string, std::string>& tags = {});

    // Batch recording
    void record_metrics(const std::vector<std::pair<std::string, MetricValue>>& metrics);

    // Data access
    std::vector<PerformanceMetric> get_all_metrics() const;
    PerformanceMetric get_metric(const std::string& name) const;
    bool has_metric(const std::string& name) const;

    // Data management
    void clear_metric_data(const std::string& name);
    void clear_all_data();
    void trim_old_data(Duration max_age = std::chrono::hours(24));

    // Export
    std::string export_metrics(const std::string& format = "prometheus") const;
    void export_to_file(const std::string& filename, const std::string& format = "json") const;

private:
    mutable std::shared_mutex metrics_mutex_;
    std::unordered_map<std::string, PerformanceMetric> metrics_;

    void ensure_metric_exists(const std::string& name, MetricType type);
    void update_metric_statistics(PerformanceMetric& metric, double value);
    void trim_metric_data(PerformanceMetric& metric, Duration max_age);
};

class PerformanceAnalyzer {
public:
    PerformanceAnalyzer();

    // Threshold-based alerts
    void set_alert_threshold(const std::string& metric_name, AlertLevel level,
                           double threshold, Duration min_duration = std::chrono::seconds(30));
    void remove_alert_threshold(const std::string& metric_name, AlertLevel level);

    // Analysis
    std::vector<PerformanceAlert> analyze_metrics(const std::vector<PerformanceMetric>& metrics);
    std::vector<PerformanceAlert> get_active_alerts() const;

    // Regression detection
    struct RegressionReport {
        std::string metric_name;
        double baseline_value;
        double current_value;
        double regression_percent;
        TimePoint detection_time;
        std::string analysis;
    };

    std::vector<RegressionReport> detect_regressions(
        const std::vector<PerformanceMetric>& current_metrics,
        const std::vector<PerformanceMetric>& baseline_metrics) const;

    // Optimization suggestions
    struct OptimizationSuggestion {
        std::string category;
        std::string suggestion;
        double potential_improvement;
        std::string rationale;
        int priority; // 1-10, 10 being highest
    };

    std::vector<OptimizationSuggestion> generate_optimization_suggestions(
        const SystemMetrics& system_metrics,
        const ApplicationMetrics& app_metrics,
        const std::vector<ProfileSample>& profile_samples) const;

    // Callbacks
    using AlertCallback = std::function<void(const PerformanceAlert&)>;
    using RegressionCallback = std::function<void(const RegressionReport&)>;

    void on_alert_triggered(AlertCallback callback);
    void on_regression_detected(RegressionCallback callback);

private:
    struct AlertThreshold {
        AlertLevel level;
        double threshold;
        Duration min_duration;
        TimePoint first_triggered;
        bool currently_active;
    };

    std::unordered_map<std::string, std::vector<AlertThreshold>> alert_thresholds_;
    std::vector<PerformanceAlert> active_alerts_;

    std::vector<AlertCallback> alert_callbacks_;
    std::vector<RegressionCallback> regression_callbacks_;

    mutable std::mutex analyzer_mutex_;

    bool check_threshold(const PerformanceMetric& metric, AlertThreshold& threshold);
    void update_alert_state(const std::string& metric_name, AlertThreshold& threshold,
                           double current_value);
};

class PerformanceMonitorCore {
public:
    static PerformanceMonitorCore& instance();

    // Component access
    SystemMonitor& get_system_monitor() { return system_monitor_; }
    ApplicationProfiler& get_profiler() { return profiler_; }
    MetricsCollector& get_metrics_collector() { return metrics_collector_; }
    PerformanceAnalyzer& get_analyzer() { return analyzer_; }

    // Unified control
    void start_all_monitoring();
    void stop_all_monitoring();

    // Comprehensive reporting
    struct PerformanceReport {
        SystemMetrics system_metrics;
        ApplicationMetrics app_metrics;
        std::vector<ProfileSample> profile_samples;
        std::vector<PerformanceAlert> alerts;
        std::vector<PerformanceAnalyzer::OptimizationSuggestion> suggestions;
        TimePoint report_time;
    };

    PerformanceReport generate_comprehensive_report();
    void export_report(const PerformanceReport& report, const std::string& filename);

private:
    PerformanceMonitorCore();
    ~PerformanceMonitorCore();

    SystemMonitor system_monitor_;
    ApplicationProfiler profiler_;
    MetricsCollector metrics_collector_;
    PerformanceAnalyzer analyzer_;
};

// RAII profiling helper
class ScopedProfiler {
public:
    ScopedProfiler(const std::string& function_name, const std::string& file = "", int line = 0);
    ~ScopedProfiler();

    void add_metadata(const std::string& key, const std::string& value);

private:
    bool was_active_;
};

// Convenience macros
#define PERF_MONITOR_START() \
    PerformanceMonitor::PerformanceMonitorCore::instance().start_all_monitoring()

#define PERF_MONITOR_STOP() \
    PerformanceMonitor::PerformanceMonitorCore::instance().stop_all_monitoring()

#define PROFILE_FUNCTION() \
    PerformanceMonitor::ScopedProfiler profiler(__FUNCTION__, __FILE__, __LINE__)

#define PROFILE_SCOPE(name) \
    PerformanceMonitor::ScopedProfiler profiler(name, __FILE__, __LINE__)

#define RECORD_METRIC(name, value) \
    PerformanceMonitor::PerformanceMonitorCore::instance().get_metrics_collector().record_gauge(name, value)

#define RECORD_COUNTER(name, value) \
    PerformanceMonitor::PerformanceMonitorCore::instance().get_metrics_collector().record_counter(name, value)

#define RECORD_TIMER(name, duration) \
    PerformanceMonitor::PerformanceMonitorCore::instance().get_metrics_collector().record_timer(name, duration)

} // namespace PerformanceMonitor
