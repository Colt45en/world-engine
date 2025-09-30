// NEXUS Metrics System
// Real-time telemetry and performance monitoring for Studio integration
// Comprehensive metrics collection with efficient aggregation and export

#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>
#include <functional>
#include <algorithm>
#include <numeric>
#include <queue>
#include <thread>

namespace NEXUS {

// ============ METRIC DATA TYPES ============

enum class MetricType {
    Counter,      // Monotonically increasing value (e.g., total events processed)
    Gauge,        // Point-in-time value (e.g., current memory usage)
    Histogram,    // Distribution of values (e.g., frame times)
    Timer,        // Time-based measurements
    Rate          // Rate of change over time
};

struct MetricValue {
    double value{0.0};
    std::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> labels;

    MetricValue(double v = 0.0) : value(v), timestamp(std::chrono::steady_clock::now()) {}
};

struct HistogramBucket {
    double upperBound;
    std::atomic<uint64_t> count{0};

    HistogramBucket(double bound) : upperBound(bound) {}
};

// ============ BASE METRIC CLASS ============

class BaseMetric {
protected:
    std::string name;
    std::string description;
    MetricType type;
    std::unordered_map<std::string, std::string> constantLabels;
    mutable std::mutex mutex;

public:
    BaseMetric(const std::string& n, const std::string& desc, MetricType t)
        : name(n), description(desc), type(t) {}

    virtual ~BaseMetric() = default;

    const std::string& getName() const { return name; }
    const std::string& getDescription() const { return description; }
    MetricType getType() const { return type; }

    void addConstantLabel(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex);
        constantLabels[key] = value;
    }

    virtual std::vector<MetricValue> collect() const = 0;
    virtual void reset() = 0;
};

// ============ COUNTER METRIC ============

class Counter : public BaseMetric {
private:
    std::atomic<double> value{0.0};

public:
    Counter(const std::string& name, const std::string& description = "")
        : BaseMetric(name, description, MetricType::Counter) {}

    void increment(double amount = 1.0) {
        value.fetch_add(amount, std::memory_order_relaxed);
    }

    double getValue() const {
        return value.load(std::memory_order_relaxed);
    }

    std::vector<MetricValue> collect() const override {
        MetricValue mv(getValue());
        mv.labels = constantLabels;
        return {mv};
    }

    void reset() override {
        value.store(0.0, std::memory_order_relaxed);
    }
};

// ============ GAUGE METRIC ============

class Gauge : public BaseMetric {
private:
    std::atomic<double> value{0.0};

public:
    Gauge(const std::string& name, const std::string& description = "")
        : BaseMetric(name, description, MetricType::Gauge) {}

    void setValue(double v) {
        value.store(v, std::memory_order_relaxed);
    }

    void increment(double amount = 1.0) {
        value.fetch_add(amount, std::memory_order_relaxed);
    }

    void decrement(double amount = 1.0) {
        value.fetch_sub(amount, std::memory_order_relaxed);
    }

    double getValue() const {
        return value.load(std::memory_order_relaxed);
    }

    std::vector<MetricValue> collect() const override {
        MetricValue mv(getValue());
        mv.labels = constantLabels;
        return {mv};
    }

    void reset() override {
        value.store(0.0, std::memory_order_relaxed);
    }
};

// ============ HISTOGRAM METRIC ============

class Histogram : public BaseMetric {
private:
    std::vector<std::unique_ptr<HistogramBucket>> buckets;
    std::atomic<uint64_t> totalCount{0};
    std::atomic<double> totalSum{0.0};

    void initializeDefaultBuckets() {
        // Default buckets for common use cases (in milliseconds for timings)
        std::vector<double> defaultBounds = {
            0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
            2500.0, 5000.0, 10000.0, std::numeric_limits<double>::infinity()
        };

        for (double bound : defaultBounds) {
            buckets.push_back(std::make_unique<HistogramBucket>(bound));
        }
    }

public:
    Histogram(const std::string& name, const std::string& description = "")
        : BaseMetric(name, description, MetricType::Histogram) {
        initializeDefaultBuckets();
    }

    Histogram(const std::string& name, const std::vector<double>& bounds, const std::string& description = "")
        : BaseMetric(name, description, MetricType::Histogram) {
        auto sortedBounds = bounds;
        std::sort(sortedBounds.begin(), sortedBounds.end());

        for (double bound : sortedBounds) {
            buckets.push_back(std::make_unique<HistogramBucket>(bound));
        }

        // Ensure we have an infinity bucket
        if (buckets.empty() || buckets.back()->upperBound != std::numeric_limits<double>::infinity()) {
            buckets.push_back(std::make_unique<HistogramBucket>(std::numeric_limits<double>::infinity()));
        }
    }

    void observe(double value) {
        totalCount.fetch_add(1, std::memory_order_relaxed);
        totalSum.fetch_add(value, std::memory_order_relaxed);

        // Find appropriate bucket and increment
        for (auto& bucket : buckets) {
            if (value <= bucket->upperBound) {
                bucket->count.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
    }

    std::vector<MetricValue> collect() const override {
        std::vector<MetricValue> values;

        // Collect bucket counts
        uint64_t cumulativeCount = 0;
        for (const auto& bucket : buckets) {
            cumulativeCount += bucket->count.load(std::memory_order_relaxed);

            MetricValue mv(static_cast<double>(cumulativeCount));
            mv.labels = constantLabels;
            mv.labels["le"] = (bucket->upperBound == std::numeric_limits<double>::infinity())
                             ? "+Inf" : std::to_string(bucket->upperBound);
            values.push_back(mv);
        }

        // Add sum and count
        MetricValue sumValue(totalSum.load(std::memory_order_relaxed));
        sumValue.labels = constantLabels;
        sumValue.labels["type"] = "sum";
        values.push_back(sumValue);

        MetricValue countValue(static_cast<double>(totalCount.load(std::memory_order_relaxed)));
        countValue.labels = constantLabels;
        countValue.labels["type"] = "count";
        values.push_back(countValue);

        return values;
    }

    void reset() override {
        totalCount.store(0, std::memory_order_relaxed);
        totalSum.store(0.0, std::memory_order_relaxed);
        for (auto& bucket : buckets) {
            bucket->count.store(0, std::memory_order_relaxed);
        }
    }

    // Statistics
    double getSum() const { return totalSum.load(std::memory_order_relaxed); }
    uint64_t getCount() const { return totalCount.load(std::memory_order_relaxed); }
    double getMean() const {
        uint64_t count = getCount();
        return count > 0 ? getSum() / count : 0.0;
    }
};

// ============ TIMER UTILITY ============

class ScopedTimer {
private:
    Histogram* histogram;
    std::chrono::steady_clock::time_point startTime;

public:
    ScopedTimer(Histogram* hist) : histogram(hist), startTime(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        if (histogram) {
            histogram->observe(duration.count() / 1000.0); // Convert to milliseconds
        }
    }
};

// ============ METRICS REGISTRY ============

class MetricsRegistry {
private:
    std::unordered_map<std::string, std::unique_ptr<BaseMetric>> metrics;
    mutable std::mutex registryMutex;

    // Background collection
    std::thread collectionThread;
    std::atomic<bool> shouldStop{false};
    std::chrono::milliseconds collectionInterval{1000};

    // Metric history for rates and trends
    struct MetricHistory {
        std::queue<std::pair<std::chrono::steady_clock::time_point, double>> values;
        static constexpr size_t maxHistorySize = 300; // 5 minutes at 1Hz
    };
    std::unordered_map<std::string, MetricHistory> metricHistory;

public:
    MetricsRegistry() = default;

    ~MetricsRegistry() {
        stopCollection();
    }

    // Register metrics
    template<typename T, typename... Args>
    T* registerMetric(const std::string& name, Args&&... args) {
        static_assert(std::is_base_of_v<BaseMetric, T>, "Type must inherit from BaseMetric");

        std::lock_guard<std::mutex> lock(registryMutex);
        auto metric = std::make_unique<T>(name, std::forward<Args>(args)...);
        T* ptr = metric.get();
        metrics[name] = std::move(metric);
        return ptr;
    }

    // Get existing metric
    template<typename T>
    T* getMetric(const std::string& name) {
        static_assert(std::is_base_of_v<BaseMetric, T>, "Type must inherit from BaseMetric");

        std::lock_guard<std::mutex> lock(registryMutex);
        auto it = metrics.find(name);
        if (it != metrics.end()) {
            return dynamic_cast<T*>(it->second.get());
        }
        return nullptr;
    }

    // Collect all metrics
    std::unordered_map<std::string, std::vector<MetricValue>> collectAll() const {
        std::lock_guard<std::mutex> lock(registryMutex);
        std::unordered_map<std::string, std::vector<MetricValue>> result;

        for (const auto& pair : metrics) {
            result[pair.first] = pair.second->collect();
        }

        return result;
    }

    // Reset all metrics
    void resetAll() {
        std::lock_guard<std::mutex> lock(registryMutex);
        for (auto& pair : metrics) {
            pair.second->reset();
        }
    }

    // Remove metric
    bool removeMetric(const std::string& name) {
        std::lock_guard<std::mutex> lock(registryMutex);
        return metrics.erase(name) > 0;
    }

    // Background collection
    void startCollection(std::chrono::milliseconds interval = std::chrono::milliseconds(1000)) {
        if (collectionThread.joinable()) return; // Already started

        collectionInterval = interval;
        shouldStop.store(false);
        collectionThread = std::thread([this]() {
            while (!shouldStop.load()) {
                collectAndStore();
                std::this_thread::sleep_for(collectionInterval);
            }
        });
    }

    void stopCollection() {
        if (collectionThread.joinable()) {
            shouldStop.store(true);
            collectionThread.join();
        }
    }

    // Get metric statistics
    std::unordered_map<std::string, double> getMetricRates() const {
        std::unordered_map<std::string, double> rates;

        std::lock_guard<std::mutex> lock(registryMutex);
        for (const auto& pair : metricHistory) {
            const auto& history = pair.second;
            if (history.values.size() >= 2) {
                auto latest = history.values.back();
                auto oldest = history.values.front();

                double timeDiff = std::chrono::duration_cast<std::chrono::seconds>(
                    latest.first - oldest.first).count();
                double valueDiff = latest.second - oldest.second;

                if (timeDiff > 0) {
                    rates[pair.first] = valueDiff / timeDiff;
                }
            }
        }

        return rates;
    }

    // Query functions
    std::vector<std::string> getMetricNames() const {
        std::lock_guard<std::mutex> lock(registryMutex);
        std::vector<std::string> names;
        for (const auto& pair : metrics) {
            names.push_back(pair.first);
        }
        return names;
    }

    size_t getMetricCount() const {
        std::lock_guard<std::mutex> lock(registryMutex);
        return metrics.size();
    }

private:
    void collectAndStore() {
        auto timestamp = std::chrono::steady_clock::now();
        auto allMetrics = collectAll();

        std::lock_guard<std::mutex> lock(registryMutex);
        for (const auto& metric : allMetrics) {
            if (!metric.second.empty()) {
                auto& history = metricHistory[metric.first];
                double value = metric.second[0].value; // Use first value for history

                history.values.emplace(timestamp, value);

                // Trim history if too large
                while (history.values.size() > MetricHistory::maxHistorySize) {
                    history.values.pop();
                }
            }
        }
    }
};

// ============ PREDEFINED SYSTEM METRICS ============

class SystemMetrics {
public:
    struct NexusMetrics {
        Counter* totalFrames;
        Counter* totalEvents;
        Counter* totalReactions;

        Gauge* currentFPS;
        Gauge* memoryUsage;
        Gauge* activeTimers;
        Gauge* queuedEvents;

        Histogram* frameTime;
        Histogram* eventProcessingTime;
        Histogram* reactionLatency;

        NexusMetrics(MetricsRegistry& registry) {
            // Counters
            totalFrames = registry.registerMetric<Counter>("nexus_total_frames", "Total frames processed");
            totalEvents = registry.registerMetric<Counter>("nexus_total_events", "Total events processed");
            totalReactions = registry.registerMetric<Counter>("nexus_total_reactions", "Total reactions triggered");

            // Gauges
            currentFPS = registry.registerMetric<Gauge>("nexus_fps", "Current frames per second");
            memoryUsage = registry.registerMetric<Gauge>("nexus_memory_mb", "Memory usage in MB");
            activeTimers = registry.registerMetric<Gauge>("nexus_active_timers", "Number of active timers");
            queuedEvents = registry.registerMetric<Gauge>("nexus_queued_events", "Number of queued events");

            // Histograms
            frameTime = registry.registerMetric<Histogram>("nexus_frame_time_ms", "Frame processing time in milliseconds");
            eventProcessingTime = registry.registerMetric<Histogram>("nexus_event_processing_ms", "Event processing time");
            reactionLatency = registry.registerMetric<Histogram>("nexus_reaction_latency_ms", "Reaction trigger latency");
        }
    };

    static std::unique_ptr<NexusMetrics> createStandardMetrics(MetricsRegistry& registry) {
        return std::make_unique<NexusMetrics>(registry);
    }
};

// ============ METRICS EXPORT ============

class MetricsExporter {
public:
    // Export to JSON format for Studio integration
    static std::string exportToJson(const MetricsRegistry& registry) {
        auto allMetrics = registry.collectAll();
        std::ostringstream json;

        json << "{\n";
        json << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << ",\n";
        json << "  \"metrics\": {\n";

        bool first = true;
        for (const auto& metric : allMetrics) {
            if (!first) json << ",\n";
            first = false;

            json << "    \"" << metric.first << "\": {\n";
            json << "      \"values\": [\n";

            bool firstValue = true;
            for (const auto& value : metric.second) {
                if (!firstValue) json << ",\n";
                firstValue = false;

                json << "        {\n";
                json << "          \"value\": " << value.value << ",\n";
                json << "          \"timestamp\": " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    value.timestamp.time_since_epoch()).count() << ",\n";
                json << "          \"labels\": {\n";

                bool firstLabel = true;
                for (const auto& label : value.labels) {
                    if (!firstLabel) json << ",\n";
                    firstLabel = false;
                    json << "            \"" << label.first << "\": \"" << label.second << "\"";
                }

                json << "\n          }\n";
                json << "        }";
            }

            json << "\n      ]\n";
            json << "    }";
        }

        json << "\n  }\n";
        json << "}\n";

        return json.str();
    }

    // Export to Prometheus format
    static std::string exportToPrometheus(const MetricsRegistry& registry) {
        auto allMetrics = registry.collectAll();
        std::ostringstream prometheus;

        for (const auto& metric : allMetrics) {
            prometheus << "# TYPE " << metric.first << " gauge\n";

            for (const auto& value : metric.second) {
                prometheus << metric.first;

                if (!value.labels.empty()) {
                    prometheus << "{";
                    bool first = true;
                    for (const auto& label : value.labels) {
                        if (!first) prometheus << ",";
                        first = false;
                        prometheus << label.first << "=\"" << label.second << "\"";
                    }
                    prometheus << "}";
                }

                prometheus << " " << value.value << " "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                              value.timestamp.time_since_epoch()).count() << "\n";
            }
        }

        return prometheus.str();
    }
};

// ============ PERFORMANCE MACROS ============

#define NEXUS_TIMER(histogram) ScopedTimer _timer(histogram)
#define NEXUS_COUNTER_INC(counter, amount) if (counter) (counter)->increment(amount)
#define NEXUS_GAUGE_SET(gauge, value) if (gauge) (gauge)->setValue(value)

} // namespace NEXUS
