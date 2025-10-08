#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <fstream>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#endif

namespace NEXUS {

// Performance metrics structure
struct PerformanceMetrics {
    double cpu_usage_percent = 0.0;
    size_t memory_usage_bytes = 0;
    size_t peak_memory_bytes = 0;
    double fps = 0.0;
    double frame_time_ms = 0.0;
    uint64_t total_frames = 0;

    // NEXUS-specific metrics
    uint32_t cognitive_analyses_per_second = 0;
    uint32_t processing_mode_changes = 0;
    uint32_t trail_updates_per_second = 0;
    uint32_t audio_updates_per_second = 0;
    uint32_t art_updates_per_second = 0;

    // Timing metrics
    double cognitive_analysis_time_ms = 0.0;
    double palette_generation_time_ms = 0.0;
    double trail_render_time_ms = 0.0;
    double memory_management_time_ms = 0.0;
};

// Individual timing measurement
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, double* result = nullptr)
        : name_(name), result_(result), start_time_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time_);
        double ms = duration.count();

        if (result_) {
            *result_ = ms;
        }

        // Auto-register with profiler if available
        if (profiler_instance_) {
            profiler_instance_->RecordTiming(name_, ms);
        }
    }

    static void SetProfilerInstance(class NexusProfiler* profiler) {
        profiler_instance_ = profiler;
    }

private:
    std::string name_;
    double* result_;
    std::chrono::high_resolution_clock::time_point start_time_;
    static thread_local class NexusProfiler* profiler_instance_;
};

thread_local class NexusProfiler* ScopedTimer::profiler_instance_ = nullptr;

// Comprehensive performance profiler
class NexusProfiler {
public:
    static NexusProfiler& Instance() {
        static NexusProfiler instance;
        return instance;
    }

    void Initialize() {
        running_ = true;
        start_time_ = std::chrono::high_resolution_clock::now();
        ScopedTimer::SetProfilerInstance(this);

        // Start monitoring thread
        monitor_thread_ = std::thread([this]() { MonitoringLoop(); });
    }

    void Shutdown() {
        running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }

    // Timing measurements
    void RecordTiming(const std::string& category, double time_ms) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        timing_samples_[category].push_back(time_ms);

        // Keep only recent samples (last 1000)
        if (timing_samples_[category].size() > 1000) {
            timing_samples_[category].erase(timing_samples_[category].begin());
        }
    }

    void RecordFrameTime(double frame_time_ms) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        current_metrics_.frame_time_ms = frame_time_ms;
        current_metrics_.fps = (frame_time_ms > 0.0) ? 1000.0 / frame_time_ms : 0.0;
        current_metrics_.total_frames++;

        frame_times_.push_back(frame_time_ms);
        if (frame_times_.size() > 300) { // Keep 5 seconds at 60fps
            frame_times_.erase(frame_times_.begin());
        }
    }

    // NEXUS-specific counters
    void IncrementCognitiveAnalyses() { cognitive_analysis_count_++; }
    void IncrementProcessingModeChanges() { mode_change_count_++; }
    void IncrementTrailUpdates() { trail_update_count_++; }
    void IncrementAudioUpdates() { audio_update_count_++; }
    void IncrementArtUpdates() { art_update_count_++; }

    // Get current metrics
    PerformanceMetrics GetCurrentMetrics() const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        return current_metrics_;
    }

    // Get timing statistics
    struct TimingStats {
        double min_ms = 0.0;
        double max_ms = 0.0;
        double avg_ms = 0.0;
        double p95_ms = 0.0;  // 95th percentile
        double p99_ms = 0.0;  // 99th percentile
        size_t sample_count = 0;
    };

    TimingStats GetTimingStats(const std::string& category) const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        TimingStats stats;
        auto it = timing_samples_.find(category);
        if (it == timing_samples_.end() || it->second.empty()) {
            return stats;
        }

        const auto& samples = it->second;
        stats.sample_count = samples.size();

        // Calculate basic stats
        double sum = 0.0;
        stats.min_ms = *std::min_element(samples.begin(), samples.end());
        stats.max_ms = *std::max_element(samples.begin(), samples.end());

        for (double sample : samples) {
            sum += sample;
        }
        stats.avg_ms = sum / static_cast<double>(samples.size());

        // Calculate percentiles
        auto sorted_samples = samples;
        std::sort(sorted_samples.begin(), sorted_samples.end());

        size_t p95_idx = static_cast<size_t>(sorted_samples.size() * 0.95);
        size_t p99_idx = static_cast<size_t>(sorted_samples.size() * 0.99);

        if (p95_idx < sorted_samples.size()) stats.p95_ms = sorted_samples[p95_idx];
        if (p99_idx < sorted_samples.size()) stats.p99_ms = sorted_samples[p99_idx];

        return stats;
    }

    // System monitoring dashboard
    void PrintDashboard() const {
        auto metrics = GetCurrentMetrics();

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "🚀 NEXUS PERFORMANCE DASHBOARD\n";
        std::cout << std::string(80, '=') << "\n";

        // System metrics
        std::cout << "💻 SYSTEM METRICS:\n";
        std::cout << "  CPU Usage:     " << std::fixed << std::setprecision(1)
                  << metrics.cpu_usage_percent << "%\n";
        std::cout << "  Memory Usage:  " << (metrics.memory_usage_bytes / 1024 / 1024) << " MB\n";
        std::cout << "  Peak Memory:   " << (metrics.peak_memory_bytes / 1024 / 1024) << " MB\n";

        // Performance metrics
        std::cout << "\n📊 PERFORMANCE METRICS:\n";
        std::cout << "  FPS:           " << std::fixed << std::setprecision(1) << metrics.fps << "\n";
        std::cout << "  Frame Time:    " << std::fixed << std::setprecision(2)
                  << metrics.frame_time_ms << " ms\n";
        std::cout << "  Total Frames:  " << metrics.total_frames << "\n";

        // NEXUS-specific metrics
        std::cout << "\n🧠 NEXUS ACTIVITY (per second):\n";
        std::cout << "  Cognitive Analyses:  " << metrics.cognitive_analyses_per_second << "\n";
        std::cout << "  Mode Changes:        " << metrics.processing_mode_changes << "\n";
        std::cout << "  Trail Updates:       " << metrics.trail_updates_per_second << "\n";
        std::cout << "  Audio Updates:       " << metrics.audio_updates_per_second << "\n";
        std::cout << "  Art Updates:         " << metrics.art_updates_per_second << "\n";

        // Timing breakdown
        std::cout << "\n⏱️  TIMING BREAKDOWN (ms):\n";
        PrintTimingCategory("Cognitive Analysis", metrics.cognitive_analysis_time_ms);
        PrintTimingCategory("Palette Generation", metrics.palette_generation_time_ms);
        PrintTimingCategory("Trail Rendering", metrics.trail_render_time_ms);
        PrintTimingCategory("Memory Management", metrics.memory_management_time_ms);

        // Frame rate analysis
        std::cout << "\n📈 FRAME RATE ANALYSIS:\n";
        if (!frame_times_.empty()) {
            double min_frame_time = *std::min_element(frame_times_.begin(), frame_times_.end());
            double max_frame_time = *std::max_element(frame_times_.begin(), frame_times_.end());
            double max_fps = (min_frame_time > 0.0) ? 1000.0 / min_frame_time : 0.0;
            double min_fps = (max_frame_time > 0.0) ? 1000.0 / max_frame_time : 0.0;

            std::cout << "  Peak FPS:      " << std::fixed << std::setprecision(1) << max_fps << "\n";
            std::cout << "  Lowest FPS:    " << std::fixed << std::setprecision(1) << min_fps << "\n";
            std::cout << "  Frame Samples: " << frame_times_.size() << "\n";
        }

        std::cout << std::string(80, '=') << "\n";
    }

    // Generate detailed performance report
    void GenerateReport(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        auto metrics = GetCurrentMetrics();
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        file << "NEXUS Performance Report\n";
        file << "Generated: " << std::ctime(&time_t) << "\n";
        file << std::string(50, '=') << "\n\n";

        // System information
        file << "System Metrics:\n";
        file << "  CPU Usage: " << metrics.cpu_usage_percent << "%\n";
        file << "  Memory Usage: " << (metrics.memory_usage_bytes / 1024 / 1024) << " MB\n";
        file << "  Peak Memory: " << (metrics.peak_memory_bytes / 1024 / 1024) << " MB\n\n";

        // Performance metrics
        file << "Performance Metrics:\n";
        file << "  Average FPS: " << metrics.fps << "\n";
        file << "  Average Frame Time: " << metrics.frame_time_ms << " ms\n";
        file << "  Total Frames: " << metrics.total_frames << "\n\n";

        // NEXUS activity
        file << "NEXUS Activity (per second):\n";
        file << "  Cognitive Analyses: " << metrics.cognitive_analyses_per_second << "\n";
        file << "  Processing Mode Changes: " << metrics.processing_mode_changes << "\n";
        file << "  Trail Updates: " << metrics.trail_updates_per_second << "\n";
        file << "  Audio Updates: " << metrics.audio_updates_per_second << "\n";
        file << "  Art Updates: " << metrics.art_updates_per_second << "\n\n";

        // Detailed timing statistics
        file << "Detailed Timing Statistics:\n";
        for (const auto& category : {"Cognitive Analysis", "Palette Generation", "Trail Rendering", "Memory Management"}) {
            auto stats = GetTimingStats(category);
            if (stats.sample_count > 0) {
                file << "  " << category << ":\n";
                file << "    Samples: " << stats.sample_count << "\n";
                file << "    Average: " << stats.avg_ms << " ms\n";
                file << "    Min: " << stats.min_ms << " ms\n";
                file << "    Max: " << stats.max_ms << " ms\n";
                file << "    95th percentile: " << stats.p95_ms << " ms\n";
                file << "    99th percentile: " << stats.p99_ms << " ms\n\n";
            }
        }

        file.close();
        std::cout << "📄 Performance report saved to: " << filename << "\n";
    }

    // Benchmark system
    void RunBenchmark(int iterations = 1000) {
        std::cout << "🏁 Running NEXUS Performance Benchmark...\n";
        std::cout << "Iterations: " << iterations << "\n\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        // Simulate workload
        for (int i = 0; i < iterations; ++i) {
            {
                ScopedTimer timer("Benchmark_CognitiveAnalysis");
                SimulateCognitiveAnalysis();
            }
            {
                ScopedTimer timer("Benchmark_PaletteGeneration");
                SimulatePaletteGeneration();
            }
            {
                ScopedTimer timer("Benchmark_TrailUpdate");
                SimulateTrailUpdate();
            }

            if (i % 100 == 0) {
                std::cout << "Progress: " << (i * 100 / iterations) << "%\r" << std::flush;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);

        std::cout << "\n✅ Benchmark Complete!\n";
        std::cout << "Total Time: " << std::fixed << std::setprecision(2) << duration.count() << " ms\n";
        std::cout << "Operations/sec: " << std::fixed << std::setprecision(0)
                  << (iterations * 3 * 1000.0 / duration.count()) << "\n\n";

        // Print benchmark results
        PrintBenchmarkResults();
    }

private:
    mutable std::mutex metrics_mutex_;
    std::atomic<bool> running_{false};
    std::thread monitor_thread_;
    std::chrono::high_resolution_clock::time_point start_time_;

    PerformanceMetrics current_metrics_;
    std::unordered_map<std::string, std::vector<double>> timing_samples_;
    std::vector<double> frame_times_;

    // Activity counters
    std::atomic<uint32_t> cognitive_analysis_count_{0};
    std::atomic<uint32_t> mode_change_count_{0};
    std::atomic<uint32_t> trail_update_count_{0};
    std::atomic<uint32_t> audio_update_count_{0};
    std::atomic<uint32_t> art_update_count_{0};

    void MonitoringLoop() {
        auto last_update = std::chrono::high_resolution_clock::now();

        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(now - last_update).count();

            if (elapsed >= 1.0) {
                UpdatePerSecondMetrics(elapsed);
                UpdateSystemMetrics();
                last_update = now;
            }
        }
    }

    void UpdatePerSecondMetrics(double elapsed_seconds) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        current_metrics_.cognitive_analyses_per_second =
            static_cast<uint32_t>(cognitive_analysis_count_.exchange(0) / elapsed_seconds);
        current_metrics_.processing_mode_changes =
            static_cast<uint32_t>(mode_change_count_.exchange(0) / elapsed_seconds);
        current_metrics_.trail_updates_per_second =
            static_cast<uint32_t>(trail_update_count_.exchange(0) / elapsed_seconds);
        current_metrics_.audio_updates_per_second =
            static_cast<uint32_t>(audio_update_count_.exchange(0) / elapsed_seconds);
        current_metrics_.art_updates_per_second =
            static_cast<uint32_t>(art_update_count_.exchange(0) / elapsed_seconds);
    }

    void UpdateSystemMetrics() {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        // Update memory usage
        current_metrics_.memory_usage_bytes = GetCurrentMemoryUsage();
        current_metrics_.peak_memory_bytes = std::max(
            current_metrics_.peak_memory_bytes,
            current_metrics_.memory_usage_bytes
        );

        // Update CPU usage
        current_metrics_.cpu_usage_percent = GetCPUUsage();

        // Update timing averages from samples
        UpdateTimingAverages();
    }

    void UpdateTimingAverages() {
        const std::vector<std::string> categories = {
            "Cognitive Analysis", "Palette Generation", "Trail Rendering", "Memory Management"
        };

        for (const auto& category : categories) {
            auto it = timing_samples_.find(category);
            if (it != timing_samples_.end() && !it->second.empty()) {
                double sum = 0.0;
                for (double sample : it->second) {
                    sum += sample;
                }
                double avg = sum / static_cast<double>(it->second.size());

                if (category == "Cognitive Analysis") current_metrics_.cognitive_analysis_time_ms = avg;
                else if (category == "Palette Generation") current_metrics_.palette_generation_time_ms = avg;
                else if (category == "Trail Rendering") current_metrics_.trail_render_time_ms = avg;
                else if (category == "Memory Management") current_metrics_.memory_management_time_ms = avg;
            }
        }
    }

    size_t GetCurrentMemoryUsage() const {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return pmc.WorkingSetSize;
        }
#elif defined(__linux__)
        std::ifstream file("/proc/self/status");
        std::string line;
        while (std::getline(file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line.substr(6));
                size_t kb;
                iss >> kb;
                return kb * 1024; // Convert KB to bytes
            }
        }
#elif defined(__APPLE__)
        struct mach_task_basic_info info;
        mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                     (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
            return info.resident_size;
        }
#endif
        return 0;
    }

    double GetCPUUsage() const {
        // Simplified CPU usage - would need platform-specific implementation for accuracy
        static auto last_time = std::chrono::high_resolution_clock::now();
        static clock_t last_cpu = clock();

        auto now = std::chrono::high_resolution_clock::now();
        clock_t now_cpu = clock();

        auto wall_time = std::chrono::duration<double>(now - last_time).count();
        double cpu_time = static_cast<double>(now_cpu - last_cpu) / CLOCKS_PER_SEC;

        last_time = now;
        last_cpu = now_cpu;

        if (wall_time > 0.0) {
            return (cpu_time / wall_time) * 100.0;
        }
        return 0.0;
    }

    void PrintTimingCategory(const std::string& name, double avg_ms) const {
        std::cout << "  " << std::setw(20) << std::left << name << ": "
                  << std::fixed << std::setprecision(3) << avg_ms << " ms\n";
    }

    void PrintBenchmarkResults() {
        std::cout << "📊 Benchmark Results:\n";
        for (const auto& category : {"Benchmark_CognitiveAnalysis", "Benchmark_PaletteGeneration", "Benchmark_TrailUpdate"}) {
            auto stats = GetTimingStats(category);
            if (stats.sample_count > 0) {
                std::cout << "  " << category << ":\n";
                std::cout << "    Average: " << std::fixed << std::setprecision(3) << stats.avg_ms << " ms\n";
                std::cout << "    Min: " << std::fixed << std::setprecision(3) << stats.min_ms << " ms\n";
                std::cout << "    Max: " << std::fixed << std::setprecision(3) << stats.max_ms << " ms\n";
                std::cout << "    95th percentile: " << std::fixed << std::setprecision(3) << stats.p95_ms << " ms\n\n";
            }
        }
    }

    // Benchmark simulation functions
    void SimulateCognitiveAnalysis() {
        // Simulate cognitive processing work
        std::string dummy = "Cognitive analysis simulation";
        for (int i = 0; i < 1000; ++i) {
            dummy += std::to_string(i);
        }
        volatile size_t result = dummy.length(); // Prevent optimization
        (void)result;
    }

    void SimulatePaletteGeneration() {
        // Simulate palette generation work
        float r = 0.0f, g = 0.0f, b = 0.0f;
        for (int i = 0; i < 500; ++i) {
            r += std::sin(i * 0.1f);
            g += std::cos(i * 0.1f);
            b += std::tan(i * 0.05f);
        }
        volatile float result = r + g + b;
        (void)result;
    }

    void SimulateTrailUpdate() {
        // Simulate trail update work
        std::vector<float> positions;
        for (int i = 0; i < 100; ++i) {
            positions.push_back(static_cast<float>(i));
            positions.push_back(std::sin(i * 0.1f));
            positions.push_back(std::cos(i * 0.1f));
        }
        volatile size_t result = positions.size();
        (void)result;
    }
};

// Convenience macros for profiling
#define NEXUS_PROFILE(name) ScopedTimer _timer(name)
#define NEXUS_PROFILE_RESULT(name, result) ScopedTimer _timer(name, result)
#define NEXUS_PROFILER NEXUS::NexusProfiler::Instance()

// Frame rate limiter utility
class FrameRateLimiter {
public:
    FrameRateLimiter(double target_fps = 60.0) : target_fps_(target_fps) {
        target_frame_time_ = std::chrono::duration<double, std::milli>(1000.0 / target_fps);
        last_frame_ = std::chrono::high_resolution_clock::now();
    }

    void WaitForNextFrame() {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = now - last_frame_;

        if (elapsed < target_frame_time_) {
            auto sleep_duration = target_frame_time_ - elapsed;
            std::this_thread::sleep_for(sleep_duration);
        }

        now = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration<double, std::milli>(now - last_frame_);

        // Record frame time in profiler
        NEXUS_PROFILER.RecordFrameTime(frame_time.count());

        last_frame_ = now;
    }

    void SetTargetFPS(double fps) {
        target_fps_ = fps;
        target_frame_time_ = std::chrono::duration<double, std::milli>(1000.0 / fps);
    }

    double GetTargetFPS() const { return target_fps_; }

private:
    double target_fps_;
    std::chrono::duration<double, std::milli> target_frame_time_;
    std::chrono::high_resolution_clock::time_point last_frame_;
};

} // namespace NEXUS
