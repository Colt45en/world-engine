#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <map>

#include "NexusLogger.hpp"
#include "NexusConfig.hpp"
#include "NexusProfiler.hpp"

namespace NEXUS {

class ASCIIArt {
public:
    static std::string getNexusLogo() {
        return R"(
    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
    ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝

        ♦ Quantum-Enhanced Game Engine ♦
    )";
    }

    static std::string getQuantumPortal() {
        return R"(
         ╭─────────────────────────╮
       ╱                           ╲
      ╱      ◊    QUANTUM    ◊      ╲
     ╱         ◦  PROTOCOL  ◦        ╲
    ╱            ·  CORE  ·           ╲
    ╲               ∞                 ╱
     ╲             ◦ ◦ ◦             ╱
      ╲          ◊  ∞  ◊           ╱
       ╲                          ╱
         ╰─────────────────────────╯
    )";
    }

    static std::string getCognitiveWeb() {
        return R"(
          ┌───┐     ┌───┐     ┌───┐
          │ ∞ │◄────┤ ψ │────►│ Ω │
          └─┬─┘     └─┬─┘     └─┬─┘
            │         │         │
            ▼         ▼         ▼
          ┌───┐     ┌───┐     ┌───┐
          │ Θ │◄────┤ Λ │────►│ Φ │
          └─┬─┘     └─┬─┘     └─┬─┘
            │         │         │
            └─────────┼─────────┘
                      ▼
                   【NEXUS】
          ~ Recursive Keeper Engine ~
    )";
    }

    static std::string getStatusBorder(const std::string& title, int width = 80) {
        if (width < title.length() + 6) width = title.length() + 6;

        std::string top = "╔" + std::string(width - 2, '═') + "╗\n";
        std::string middle = "║ " + title + std::string(width - title.length() - 3, ' ') + "║\n";
        std::string separator = "╠" + std::string(width - 2, '═') + "╣\n";

        return top + middle + separator;
    }

    static std::string getStatusFooter(int width = 80) {
        return "╚" + std::string(width - 2, '═') + "╝";
    }

    static std::string createProgressBar(float percentage, int width = 40, bool colored = true) {
        int filled = static_cast<int>(percentage * width / 100.0f);
        filled = std::max(0, std::min(width, filled));

        std::string bar = "[";

        if (colored) {
            // Color based on percentage
            std::string color;
            if (percentage >= 80) color = "\033[32m";      // Green
            else if (percentage >= 60) color = "\033[33m"; // Yellow
            else if (percentage >= 40) color = "\033[31m"; // Red
            else color = "\033[35m";                       // Magenta

            bar += color;
            bar += std::string(filled, '█');
            bar += "\033[0m";
            bar += std::string(width - filled, '░');
        } else {
            bar += std::string(filled, '█');
            bar += std::string(width - filled, '░');
        }

        bar += "] " + std::to_string(static_cast<int>(percentage)) + "%";
        return bar;
    }

    static std::string createSparkline(const std::vector<float>& data, int width = 20) {
        if (data.empty()) return std::string(width, '_');

        float min_val = *std::min_element(data.begin(), data.end());
        float max_val = *std::max_element(data.begin(), data.end());

        if (max_val == min_val) return std::string(width, '▄');

        std::string sparkline;
        const std::string levels = " ▁▂▃▄▅▆▇█";

        for (int i = 0; i < width && i < data.size(); ++i) {
            float normalized = (data[data.size() - width + i] - min_val) / (max_val - min_val);
            int level = static_cast<int>(normalized * 8);
            level = std::max(0, std::min(8, level));
            sparkline += levels[level];
        }

        // Pad if necessary
        while (sparkline.length() < width) {
            sparkline = " " + sparkline;
        }

        return sparkline;
    }
};

class NexusDashboard {
private:
    bool is_running;
    std::chrono::system_clock::time_point start_time;
    std::vector<float> cpu_history;
    std::vector<float> memory_history;
    std::vector<float> fps_history;
    mutable std::mutex dashboard_mutex;

    // System metrics
    struct SystemMetrics {
        float cpu_usage = 0.0f;
        size_t memory_usage = 0;
        size_t total_memory = 0;
        float fps = 0.0f;
        size_t active_entities = 0;
        size_t active_thoughts = 0;
        int quantum_mode = 0;
        std::string quantum_mode_name = "MIRROR";
        float quantum_intensity = 0.0f;
        size_t total_log_entries = 0;
        std::string active_preset = "Default";
    } current_metrics;

public:
    NexusDashboard() : is_running(false), start_time(std::chrono::system_clock::now()) {
        cpu_history.reserve(60);
        memory_history.reserve(60);
        fps_history.reserve(60);
    }

    void start() {
        std::lock_guard<std::mutex> lock(dashboard_mutex);
        is_running = true;
        start_time = std::chrono::system_clock::now();
        NEXUS_LOG_INFO("DASHBOARD", "System dashboard started");
    }

    void stop() {
        std::lock_guard<std::mutex> lock(dashboard_mutex);
        is_running = false;
        NEXUS_LOG_INFO("DASHBOARD", "System dashboard stopped");
    }

    void updateMetrics(const SystemMetrics& metrics) {
        std::lock_guard<std::mutex> lock(dashboard_mutex);
        current_metrics = metrics;

        // Update history
        cpu_history.push_back(metrics.cpu_usage);
        memory_history.push_back(static_cast<float>(metrics.memory_usage) / 1024.0f / 1024.0f); // MB
        fps_history.push_back(metrics.fps);

        // Keep only last 60 samples
        if (cpu_history.size() > 60) cpu_history.erase(cpu_history.begin());
        if (memory_history.size() > 60) memory_history.erase(memory_history.begin());
        if (fps_history.size() > 60) fps_history.erase(fps_history.begin());
    }

    void render(bool clear_screen = true) {
        std::lock_guard<std::mutex> lock(dashboard_mutex);

        if (clear_screen) {
            // Clear screen (works on most terminals)
            std::cout << "\033[2J\033[H";
        }

        // Header with logo
        std::cout << "\033[96m" << ASCIIArt::getNexusLogo() << "\033[0m\n";

        // System uptime
        auto uptime = std::chrono::system_clock::now() - start_time;
        auto uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(uptime).count();

        auto hours = uptime_seconds / 3600;
        auto minutes = (uptime_seconds % 3600) / 60;
        auto seconds = uptime_seconds % 60;

        std::cout << ASCIIArt::getStatusBorder("SYSTEM STATUS", 80);

        std::cout << "║ " << std::left << std::setw(20) << "System Uptime:"
                  << std::setw(55) << (std::to_string(hours) + "h " + std::to_string(minutes) + "m " + std::to_string(seconds) + "s")
                  << " ║\n";

        std::cout << "║ " << std::left << std::setw(20) << "Status:"
                  << std::setw(55) << (is_running ? "\033[32m● ONLINE\033[0m" : "\033[31m● OFFLINE\033[0m")
                  << " ║\n";

        // Performance metrics
        std::cout << "╠" << std::string(78, '═') << "╣\n";
        std::cout << "║ " << std::left << std::setw(76) << "\033[93mPERFORMANCE METRICS\033[0m" << " ║\n";
        std::cout << "╠" << std::string(78, '═') << "╣\n";

        // CPU Usage
        std::string cpu_bar = ASCIIArt::createProgressBar(current_metrics.cpu_usage, 30);
        std::string cpu_sparkline = ASCIIArt::createSparkline(cpu_history, 20);
        std::cout << "║ CPU: " << std::left << std::setw(35) << cpu_bar
                  << " " << std::setw(20) << cpu_sparkline << " ║\n";

        // Memory Usage
        float memory_percent = current_metrics.total_memory > 0 ?
                              (static_cast<float>(current_metrics.memory_usage) / current_metrics.total_memory * 100.0f) : 0.0f;
        std::string memory_bar = ASCIIArt::createProgressBar(memory_percent, 30);
        std::string memory_sparkline = ASCIIArt::createSparkline(memory_history, 20);
        std::cout << "║ MEM: " << std::left << std::setw(35) << memory_bar
                  << " " << std::setw(20) << memory_sparkline << " ║\n";

        // FPS
        std::string fps_bar = ASCIIArt::createProgressBar(std::min(current_metrics.fps * 100.0f / 120.0f, 100.0f), 30);
        std::string fps_sparkline = ASCIIArt::createSparkline(fps_history, 20);
        std::cout << "║ FPS: " << std::left << std::setw(35) << fps_bar
                  << " " << std::setw(20) << fps_sparkline << " ║\n";

        // Quantum Protocol Status
        std::cout << "╠" << std::string(78, '═') << "╣\n";
        std::cout << "║ " << std::left << std::setw(76) << "\033[95mQUANTUM PROTOCOL STATUS\033[0m" << " ║\n";
        std::cout << "╠" << std::string(78, '═') << "╣\n";

        std::cout << "║ " << std::left << std::setw(20) << "Processing Mode:"
                  << std::setw(55) << (std::to_string(current_metrics.quantum_mode) + " (" + current_metrics.quantum_mode_name + ")")
                  << " ║\n";

        std::string intensity_bar = ASCIIArt::createProgressBar(current_metrics.quantum_intensity * 100.0f, 40);
        std::cout << "║ Intensity: " << std::left << std::setw(64) << intensity_bar << " ║\n";

        // ASCII Quantum Portal
        std::cout << "║" << std::string(78, ' ') << "║\n";
        std::istringstream portal(ASCIIArt::getQuantumPortal());
        std::string portal_line;
        while (std::getline(portal, portal_line)) {
            if (!portal_line.empty()) {
                int padding = (78 - portal_line.length()) / 2;
                std::cout << "║" << std::string(padding, ' ') << "\033[94m" << portal_line << "\033[0m"
                          << std::string(78 - padding - portal_line.length(), ' ') << "║\n";
            }
        }

        // Cognitive Engine Status
        std::cout << "╠" << std::string(78, '═') << "╣\n";
        std::cout << "║ " << std::left << std::setw(76) << "\033[92mCOGNITIVE ENGINE STATUS\033[0m" << " ║\n";
        std::cout << "╠" << std::string(78, '═') << "╣\n";

        std::cout << "║ " << std::left << std::setw(20) << "Active Thoughts:"
                  << std::setw(55) << std::to_string(current_metrics.active_thoughts) << " ║\n";

        std::cout << "║ " << std::left << std::setw(20) << "Memory Status:"
                  << std::setw(55) << "\033[32m● PROCESSING\033[0m" << " ║\n";

        // ASCII Cognitive Web
        std::cout << "║" << std::string(78, ' ') << "║\n";
        std::istringstream web(ASCIIArt::getCognitiveWeb());
        std::string web_line;
        while (std::getline(web, web_line)) {
            if (!web_line.empty()) {
                int padding = (78 - web_line.length()) / 2;
                std::cout << "║" << std::string(padding, ' ') << "\033[92m" << web_line << "\033[0m"
                          << std::string(78 - padding - web_line.length(), ' ') << "║\n";
            }
        }

        // System Statistics
        std::cout << "╠" << std::string(78, '═') << "╣\n";
        std::cout << "║ " << std::left << std::setw(76) << "\033[91mSYSTEM STATISTICS\033[0m" << " ║\n";
        std::cout << "╠" << std::string(78, '═') << "╣\n";

        std::cout << "║ " << std::left << std::setw(20) << "Active Entities:"
                  << std::setw(55) << std::to_string(current_metrics.active_entities) << " ║\n";

        std::cout << "║ " << std::left << std::setw(20) << "Log Entries:"
                  << std::setw(55) << std::to_string(current_metrics.total_log_entries) << " ║\n";

        std::cout << "║ " << std::left << std::setw(20) << "Active Config:"
                  << std::setw(55) << current_metrics.active_preset << " ║\n";

        std::cout << ASCIIArt::getStatusFooter(80) << "\n\n";

        // Control instructions
        std::cout << "\033[90m";
        std::cout << "╭────────────────────────────────────────────────────────────────────────────╮\n";
        std::cout << "│ CONTROLS: [Q]uit | [R]efresh | [P]ause | [S]tats | [C]onfig | [L]ogs       │\n";
        std::cout << "╰────────────────────────────────────────────────────────────────────────────╯\n";
        std::cout << "\033[0m";

        std::cout.flush();
    }

    void renderCompact() {
        std::lock_guard<std::mutex> lock(dashboard_mutex);

        auto uptime = std::chrono::system_clock::now() - start_time;
        auto uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(uptime).count();

        std::cout << "\033[96m[NEXUS]\033[0m ";
        std::cout << "UP:" << uptime_seconds << "s ";
        std::cout << "CPU:" << static_cast<int>(current_metrics.cpu_usage) << "% ";
        std::cout << "MEM:" << current_metrics.memory_usage / 1024 / 1024 << "MB ";
        std::cout << "FPS:" << static_cast<int>(current_metrics.fps) << " ";
        std::cout << "QM:" << current_metrics.quantum_mode << " ";
        std::cout << "ENT:" << current_metrics.active_entities << " ";
        std::cout << "THT:" << current_metrics.active_thoughts;
        std::cout << std::endl;
    }

    void showSystemInfo() {
        std::cout << "\033[2J\033[H"; // Clear screen

        std::cout << "\033[96m" << ASCIIArt::getNexusLogo() << "\033[0m\n";
        std::cout << ASCIIArt::getStatusBorder("SYSTEM INFORMATION", 80);

        // Build info
        std::cout << "║ " << std::left << std::setw(20) << "Version:"
                  << std::setw(55) << "NEXUS Game Engine v1.0" << " ║\n";

        std::cout << "║ " << std::left << std::setw(20) << "Build Date:"
                  << std::setw(55) << __DATE__ << " ║\n";

        std::cout << "║ " << std::left << std::setw(20) << "Build Time:"
                  << std::setw(55) << __TIME__ << " ║\n";

        std::cout << "║ " << std::left << std::setw(20) << "Compiler:"
                  << std::setw(55) <<
#ifdef _MSC_VER
                     "Microsoft Visual C++"
#elif defined(__GNUC__)
                     "GCC"
#elif defined(__clang__)
                     "Clang"
#else
                     "Unknown"
#endif
                  << " ║\n";

        // Features
        std::cout << "╠" << std::string(78, '═') << "╣\n";
        std::cout << "║ " << std::left << std::setw(76) << "\033[93mFEATURES\033[0m" << " ║\n";
        std::cout << "╠" << std::string(78, '═') << "╣\n";

        std::cout << "║ ✓ Entity-Component System          ✓ Quantum Protocol Processing    ║\n";
        std::cout << "║ ✓ Audio-Reactive Visuals           ✓ Dynamic Color Palettes         ║\n";
        std::cout << "║ ✓ Trail Rendering System           ✓ Recursive Keeper Engine        ║\n";
        std::cout << "║ ✓ Cognitive Processing             ✓ Performance Profiling          ║\n";
        std::cout << "║ ✓ Advanced Logging System          ✓ Configuration Management       ║\n";
        std::cout << "║ ✓ Comprehensive Test Suite         ✓ Cross-Platform Support         ║\n";

        std::cout << ASCIIArt::getStatusFooter(80) << "\n\n";

        std::cout << "\033[90mPress any key to return to dashboard...\033[0m\n";
    }

    void startInteractiveMode() {
        start();

        while (is_running) {
            render();

            // Non-blocking input check (platform specific)
            // For now, we'll use a simple timed refresh
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            // Update with dummy data for demo
            updateDemoMetrics();
        }
    }

private:
    void updateDemoMetrics() {
        SystemMetrics demo_metrics;

        // Generate some demo data
        static float time = 0.0f;
        time += 0.1f;

        demo_metrics.cpu_usage = 30.0f + 20.0f * std::sin(time);
        demo_metrics.memory_usage = 150 * 1024 * 1024 + static_cast<size_t>(50 * 1024 * 1024 * std::sin(time * 0.5f));
        demo_metrics.total_memory = 8 * 1024 * 1024 * 1024; // 8GB
        demo_metrics.fps = 58.0f + 4.0f * std::sin(time * 2.0f);
        demo_metrics.active_entities = 100 + static_cast<size_t>(20 * std::sin(time * 0.3f));
        demo_metrics.active_thoughts = 15 + static_cast<size_t>(5 * std::sin(time * 0.7f));
        demo_metrics.quantum_mode = static_cast<int>(time) % 8;

        const std::vector<std::string> mode_names = {
            "MIRROR", "COSINE", "CHAOS", "ABSORB", "AMPLIFY", "PULSE", "FLOW", "FRAGMENT"
        };
        demo_metrics.quantum_mode_name = mode_names[demo_metrics.quantum_mode];
        demo_metrics.quantum_intensity = 0.5f + 0.3f * std::sin(time * 1.5f);
        demo_metrics.total_log_entries = static_cast<size_t>(1000 + time * 10);
        demo_metrics.active_preset = "Performance";

        updateMetrics(demo_metrics);
    }
};

// Global dashboard instance
class GlobalDashboard {
private:
    static std::unique_ptr<NexusDashboard> instance;
    static std::once_flag once_flag;

public:
    static NexusDashboard& getInstance() {
        std::call_once(once_flag, []() {
            instance = std::make_unique<NexusDashboard>();
        });
        return *instance;
    }
};

std::unique_ptr<NexusDashboard> GlobalDashboard::instance = nullptr;

// Convenience functions
inline void NEXUS_DASHBOARD_START() {
    GlobalDashboard::getInstance().start();
}

inline void NEXUS_DASHBOARD_STOP() {
    GlobalDashboard::getInstance().stop();
}

inline void NEXUS_DASHBOARD_RENDER() {
    GlobalDashboard::getInstance().render();
}

inline void NEXUS_DASHBOARD_INTERACTIVE() {
    GlobalDashboard::getInstance().startInteractiveMode();
}

} // namespace NEXUS
