/**
 * Timekeeping Engine - Precise timing, scheduling, and temporal coordination
 * ========================================================================
 *
 * Features:
 * - High-resolution timing and clock synchronization
 * - Scheduled task execution with cron-like syntax
 * - Timer management and recurring events
 * - Time-based event correlation and analysis
 * - Frame rate control for real-time applications
 * - Time dilation and simulation speed control
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace TimekeepingEngine {

using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::nanoseconds;
using TaskId = uint64_t;

enum class TimerType {
    ONE_SHOT,
    REPEATING,
    FRAME_BASED,
    CRON_BASED
};

enum class ClockType {
    SYSTEM_TIME,        // Wall clock time
    MONOTONIC_TIME,     // Monotonic system time
    SIMULATION_TIME,    // Controllable simulation time
    FRAME_TIME         // Frame-based timing
};

struct TimerSpec {
    TimerId id;
    std::string name;
    TimerType type;
    ClockType clock_type;
    Duration interval;
    Duration initial_delay;
    int max_executions = -1; // -1 for infinite
    std::function<void()> callback;
    std::string cron_expression; // For CRON_BASED timers
    bool enabled = true;

    // Execution tracking
    int execution_count = 0;
    TimePoint last_execution;
    TimePoint next_execution;
    Duration total_execution_time{0};
    Duration max_execution_time{0};
    Duration min_execution_time{Duration::max()};
};

struct ScheduledTask {
    TaskId id;
    std::string name;
    TimePoint scheduled_time;
    std::function<void()> task;
    std::string category;
    int priority = 0; // Higher priority executes first
    Duration max_allowed_delay{std::chrono::seconds(1)};
    bool allow_concurrent = false;

    bool operator<(const ScheduledTask& other) const {
        if (scheduled_time != other.scheduled_time) {
            return scheduled_time > other.scheduled_time; // Min-heap by time
        }
        return priority < other.priority; // Max-heap by priority
    }
};

struct TimeMetrics {
    Duration total_uptime{0};
    uint64_t total_tasks_executed = 0;
    uint64_t total_timers_fired = 0;
    Duration average_task_execution_time{0};
    Duration max_task_execution_time{0};
    uint64_t missed_deadlines = 0;
    double current_frame_rate = 0.0;
    double average_frame_rate = 0.0;

    // Clock drift tracking
    Duration system_clock_drift{0};
    Duration simulation_time_offset{0};
};

class CronParser {
public:
    struct CronFields {
        std::set<int> minutes;    // 0-59
        std::set<int> hours;      // 0-23
        std::set<int> days;       // 1-31
        std::set<int> months;     // 1-12
        std::set<int> weekdays;   // 0-6 (Sunday = 0)
    };

    static CronFields parse(const std::string& cron_expression);
    static TimePoint next_execution(const CronFields& fields, TimePoint from = TimePoint{});
    static bool matches(const CronFields& fields, TimePoint time);

private:
    static std::set<int> parse_field(const std::string& field, int min_val, int max_val);
};

class TimekeepingCore {
public:
    static TimekeepingCore& instance();

    // Timer management
    TimerId create_timer(const std::string& name, Duration interval,
                        std::function<void()> callback, TimerType type = TimerType::REPEATING);
    TimerId create_cron_timer(const std::string& name, const std::string& cron_expression,
                             std::function<void()> callback);
    TimerId create_frame_timer(const std::string& name, double target_fps,
                              std::function<void()> callback);

    void enable_timer(TimerId id);
    void disable_timer(TimerId id);
    void remove_timer(TimerId id);
    void modify_timer_interval(TimerId id, Duration new_interval);

    // Task scheduling
    TaskId schedule_task(const std::string& name, TimePoint when,
                        std::function<void()> task, int priority = 0);
    TaskId schedule_task_in(const std::string& name, Duration delay,
                           std::function<void()> task, int priority = 0);
    TaskId schedule_recurring_task(const std::string& name, Duration interval,
                                  std::function<void()> task, int max_executions = -1);

    void cancel_task(TaskId id);
    void modify_task_time(TaskId id, TimePoint new_time);

    // Clock management
    void set_clock_type(ClockType type);
    ClockType get_clock_type() const;

    // Simulation time control (for SIMULATION_TIME clock)
    void set_simulation_speed(double speed_multiplier);
    void pause_simulation();
    void resume_simulation();
    void reset_simulation_time();

    // Frame rate control
    void set_target_frame_rate(double fps);
    double get_current_frame_rate() const;
    void begin_frame();
    void end_frame();
    Duration get_frame_delta_time() const;

    // Time queries
    TimePoint now() const;
    Duration uptime() const;
    uint64_t get_frame_count() const;
    std::chrono::system_clock::time_point get_wall_time() const;

    // Synchronization
    void wait_until(TimePoint time);
    void wait_for(Duration duration);
    void sync_frame(); // Waits until next frame boundary

    // Metrics and analysis
    TimeMetrics get_metrics() const;
    std::vector<TimerSpec> get_active_timers() const;
    std::vector<ScheduledTask> get_pending_tasks() const;

    // Event callbacks
    using TimerCallback = std::function<void(TimerId, const std::string&)>;
    using TaskCallback = std::function<void(TaskId, const std::string&)>;

    void on_timer_fired(TimerCallback callback);
    void on_task_executed(TaskCallback callback);
    void on_missed_deadline(TaskCallback callback);

    // Control
    void start();
    void stop();
    void pause();
    void resume();
    bool is_running() const;

private:
    TimekeepingCore();
    ~TimekeepingCore();

    // Core timing state
    ClockType clock_type_ = ClockType::SYSTEM_TIME;
    TimePoint start_time_;
    TimePoint simulation_start_time_;
    Duration simulation_time_offset_{0};
    std::atomic<double> simulation_speed_{1.0};
    std::atomic<bool> simulation_paused_{false};

    // Frame timing
    std::atomic<double> target_frame_rate_{60.0};
    std::atomic<uint64_t> frame_count_{0};
    TimePoint last_frame_time_;
    Duration last_frame_delta_{0};
    std::vector<Duration> frame_time_history_;

    // Timer and task storage
    std::map<TimerId, TimerSpec> timers_;
    std::priority_queue<ScheduledTask> task_queue_;
    std::atomic<TimerId> next_timer_id_{1};
    std::atomic<TaskId> next_task_id_{1};

    // Threading
    std::thread worker_thread_;
    std::mutex timers_mutex_;
    std::mutex tasks_mutex_;
    std::condition_variable worker_cv_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};

    // Metrics
    mutable std::mutex metrics_mutex_;
    TimeMetrics metrics_;

    // Callbacks
    std::vector<TimerCallback> timer_callbacks_;
    std::vector<TaskCallback> task_callbacks_;
    std::vector<TaskCallback> deadline_callbacks_;

    void worker_loop();
    void process_timers();
    void process_tasks();
    void update_frame_timing();
    void update_metrics();

    TimePoint get_current_time() const;
    Duration get_simulation_time_offset() const;

    void fire_timer(TimerId id);
    void execute_task(const ScheduledTask& task);

    TimerId generate_timer_id();
    TaskId generate_task_id();
};

// Utility classes for common timing patterns
class FrameRateController {
public:
    explicit FrameRateController(double target_fps);

    void begin_frame();
    void end_frame();
    bool should_skip_frame() const;
    Duration get_frame_delta() const;
    double get_actual_fps() const;

private:
    double target_fps_;
    Duration target_frame_time_;
    TimePoint last_frame_start_;
    Duration last_frame_delta_;
    std::vector<Duration> frame_history_;
};

class PerformanceProfiler {
public:
    explicit PerformanceProfiler(const std::string& category);
    ~PerformanceProfiler();

    void checkpoint(const std::string& name);
    void add_metadata(const std::string& key, const std::string& value);

private:
    std::string category_;
    TimePoint start_time_;
    std::vector<std::pair<std::string, TimePoint>> checkpoints_;
    std::map<std::string, std::string> metadata_;
};

// Convenience macros
#define SCHEDULE_TASK(name, delay, task) \
    TimekeepingEngine::TimekeepingCore::instance().schedule_task_in(name, delay, task)

#define CREATE_TIMER(name, interval, callback) \
    TimekeepingEngine::TimekeepingCore::instance().create_timer(name, interval, callback)

#define PROFILE_SCOPE(category) \
    TimekeepingEngine::PerformanceProfiler profiler(category)

#define PROFILE_CHECKPOINT(name) \
    profiler.checkpoint(name)

} // namespace TimekeepingEngine
