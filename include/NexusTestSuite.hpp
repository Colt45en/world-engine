#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <memory>
#include <sstream>
#include <exception>
#include <thread>
#include <random>
#include <algorithm>
#include <cassert>

// Include all NEXUS headers
#include "NexusGameEngine.hpp"
#include "NexusProtocol.hpp"
#include "NexusVisuals.hpp"
#include "NexusTrailRenderer.hpp"
#include "NexusRecursiveKeeperEngine.hpp"
#include "NexusProfiler.hpp"
#include "NexusLogger.hpp"
#include "NexusConfig.hpp"

namespace NEXUS {

enum class TestResult {
    PASSED,
    FAILED,
    SKIPPED
};

struct TestCase {
    std::string name;
    std::string category;
    std::function<bool()> test_func;
    std::function<void()> setup_func;
    std::function<void()> teardown_func;

    TestCase(const std::string& n, const std::string& cat,
             std::function<bool()> func,
             std::function<void()> setup = nullptr,
             std::function<void()> teardown = nullptr)
        : name(n), category(cat), test_func(func), setup_func(setup), teardown_func(teardown) {}
};

struct TestResult_Detail {
    TestResult result;
    std::string name;
    std::string category;
    std::string error_message;
    std::chrono::milliseconds duration;

    TestResult_Detail(TestResult r, const std::string& n, const std::string& cat,
                     const std::string& error = "", std::chrono::milliseconds dur = std::chrono::milliseconds(0))
        : result(r), name(n), category(cat), error_message(error), duration(dur) {}
};

class NexusTestSuite {
private:
    std::vector<TestCase> test_cases;
    std::vector<TestResult_Detail> results;
    bool verbose_output;
    std::mt19937 random_engine;

public:
    NexusTestSuite(bool verbose = true)
        : verbose_output(verbose), random_engine(std::chrono::steady_clock::now().time_since_epoch().count()) {
        NEXUS_LOG_INFO("TEST", "NEXUS Test Suite initialized");
        setupAllTests();
    }

    void addTest(const std::string& name, const std::string& category,
                 std::function<bool()> test_func,
                 std::function<void()> setup = nullptr,
                 std::function<void()> teardown = nullptr) {
        test_cases.emplace_back(name, category, test_func, setup, teardown);
    }

    bool runAllTests() {
        NEXUS_LOG_INFO("TEST", "Starting comprehensive test suite...");
        NEXUS_LOG_INFO("TEST", "Total tests: " + std::to_string(test_cases.size()));

        results.clear();
        auto start_time = std::chrono::high_resolution_clock::now();

        size_t passed = 0, failed = 0, skipped = 0;

        for (const auto& test : test_cases) {
            auto result = runSingleTest(test);
            results.push_back(result);

            switch (result.result) {
                case TestResult::PASSED: passed++; break;
                case TestResult::FAILED: failed++; break;
                case TestResult::SKIPPED: skipped++; break;
            }

            if (verbose_output) {
                printTestResult(result);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        printSummary(passed, failed, skipped, total_duration);

        return failed == 0;
    }

    bool runCategory(const std::string& category) {
        NEXUS_LOG_INFO("TEST", "Running tests in category: " + category);

        size_t passed = 0, failed = 0, skipped = 0;

        for (const auto& test : test_cases) {
            if (test.category != category) continue;

            auto result = runSingleTest(test);
            results.push_back(result);

            switch (result.result) {
                case TestResult::PASSED: passed++; break;
                case TestResult::FAILED: failed++; break;
                case TestResult::SKIPPED: skipped++; break;
            }

            if (verbose_output) {
                printTestResult(result);
            }
        }

        NEXUS_LOG_INFO("TEST", "Category results - Passed: " + std::to_string(passed) +
                      ", Failed: " + std::to_string(failed) + ", Skipped: " + std::to_string(skipped));

        return failed == 0;
    }

    void generateReport(const std::string& filename = "nexus_test_report.html") {
        std::ofstream report(filename);
        if (!report.is_open()) {
            NEXUS_LOG_ERROR("TEST", "Failed to create test report: " + filename);
            return;
        }

        report << generateHTMLReport();
        report.close();

        NEXUS_LOG_INFO("TEST", "Test report generated: " + filename);
    }

private:
    void setupAllTests() {
        // Core Engine Tests
        addTest("GameEngine Initialization", "Core",
                [this]() { return testGameEngineInit(); });

        addTest("Entity Creation and Management", "Core",
                [this]() { return testEntityManagement(); });

        addTest("Component System", "Core",
                [this]() { return testComponentSystem(); });

        addTest("Resource Engine", "Core",
                [this]() { return testResourceEngine(); });

        // Quantum Protocol Tests
        addTest("Protocol Initialization", "Quantum",
                [this]() { return testQuantumProtocolInit(); });

        addTest("Processing Mode Switching", "Quantum",
                [this]() { return testProcessingModes(); });

        addTest("Callback System", "Quantum",
                [this]() { return testQuantumCallbacks(); });

        addTest("Audio Data Processing", "Quantum",
                [this]() { return testAudioProcessing(); });

        // Visual System Tests
        addTest("Palette System", "Visual",
                [this]() { return testPaletteSystem(); });

        addTest("Color Transitions", "Visual",
                [this]() { return testColorTransitions(); });

        addTest("Trail Renderer", "Visual",
                [this]() { return testTrailRenderer(); });

        addTest("Visual Effects", "Visual",
                [this]() { return testVisualEffects(); });

        // Cognitive Engine Tests
        addTest("Recursive Keeper Initialization", "Cognitive",
                [this]() { return testCognitiveInit(); });

        addTest("Thought Processing", "Cognitive",
                [this]() { return testThoughtProcessing(); });

        addTest("Memory Management", "Cognitive",
                [this]() { return testMemoryManagement(); });

        addTest("Philosophical Analysis", "Cognitive",
                [this]() { return testPhilosophicalAnalysis(); });

        // Performance Tests
        addTest("Profiler Functionality", "Performance",
                [this]() { return testProfiler(); });

        addTest("Benchmark System", "Performance",
                [this]() { return testBenchmarking(); });

        addTest("Memory Tracking", "Performance",
                [this]() { return testMemoryTracking(); });

        // Configuration Tests
        addTest("Config System", "Config",
                [this]() { return testConfigSystem(); });

        addTest("Preset Management", "Config",
                [this]() { return testPresets(); });

        addTest("File I/O", "Config",
                [this]() { return testConfigIO(); });

        // Logging Tests
        addTest("Logger Functionality", "Logging",
                [this]() { return testLogger(); });

        addTest("Multi-Sink Logging", "Logging",
                [this]() { return testMultiSinkLogging(); });

        addTest("Log Formatting", "Logging",
                [this]() { return testLogFormatting(); });

        // Integration Tests
        addTest("System Integration", "Integration",
                [this]() { return testSystemIntegration(); });

        addTest("Multi-threaded Operation", "Integration",
                [this]() { return testMultithreading(); });

        addTest("Stress Test", "Integration",
                [this]() { return testStressLoad(); });

        // Performance Benchmarks
        addTest("Entity Creation Performance", "Benchmark",
                [this]() { return benchmarkEntityCreation(); });

        addTest("Quantum Processing Performance", "Benchmark",
                [this]() { return benchmarkQuantumProcessing(); });

        addTest("Memory Allocation Performance", "Benchmark",
                [this]() { return benchmarkMemoryAllocation(); });
    }

    TestResult_Detail runSingleTest(const TestCase& test) {
        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            // Setup
            if (test.setup_func) {
                test.setup_func();
            }

            // Run test
            bool success = test.test_func();

            // Teardown
            if (test.teardown_func) {
                test.teardown_func();
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            return TestResult_Detail(
                success ? TestResult::PASSED : TestResult::FAILED,
                test.name, test.category, "", duration
            );

        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            return TestResult_Detail(
                TestResult::FAILED, test.name, test.category, e.what(), duration
            );
        }
    }

    void printTestResult(const TestResult_Detail& result) {
        std::string status_color;
        std::string status_text;

        switch (result.result) {
            case TestResult::PASSED:
                status_color = "\033[32m"; // Green
                status_text = "PASS";
                break;
            case TestResult::FAILED:
                status_color = "\033[31m"; // Red
                status_text = "FAIL";
                break;
            case TestResult::SKIPPED:
                status_color = "\033[33m"; // Yellow
                status_text = "SKIP";
                break;
        }

        std::cout << "[" << status_color << status_text << "\033[0m] "
                  << result.category << "::" << result.name
                  << " (" << result.duration.count() << "ms)";

        if (!result.error_message.empty()) {
            std::cout << " - " << result.error_message;
        }

        std::cout << std::endl;
    }

    void printSummary(size_t passed, size_t failed, size_t skipped, std::chrono::milliseconds total_time) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TEST SUITE SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::cout << "Total Tests: " << (passed + failed + skipped) << std::endl;
        std::cout << "\033[32mPassed: " << passed << "\033[0m" << std::endl;
        std::cout << "\033[31mFailed: " << failed << "\033[0m" << std::endl;
        std::cout << "\033[33mSkipped: " << skipped << "\033[0m" << std::endl;

        double success_rate = (passed + failed > 0) ?
                              (double)passed / (passed + failed) * 100.0 : 0.0;
        std::cout << "Success Rate: " << std::fixed << std::setprecision(1)
                  << success_rate << "%" << std::endl;

        std::cout << "Total Time: " << total_time.count() << "ms" << std::endl;

        if (failed > 0) {
            std::cout << "\n\033[31mFAILED TESTS:\033[0m" << std::endl;
            for (const auto& result : results) {
                if (result.result == TestResult::FAILED) {
                    std::cout << "  - " << result.category << "::" << result.name;
                    if (!result.error_message.empty()) {
                        std::cout << " (" << result.error_message << ")";
                    }
                    std::cout << std::endl;
                }
            }
        }

        std::cout << std::string(60, '=') << std::endl;
    }

    std::string generateHTMLReport() {
        std::ostringstream html;

        html << "<!DOCTYPE html>\n<html>\n<head>\n";
        html << "<title>NEXUS Test Report</title>\n";
        html << "<style>\n";
        html << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
        html << ".pass { color: green; }\n";
        html << ".fail { color: red; }\n";
        html << ".skip { color: orange; }\n";
        html << "table { border-collapse: collapse; width: 100%; }\n";
        html << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
        html << "th { background-color: #f2f2f2; }\n";
        html << "</style>\n</head>\n<body>\n";

        html << "<h1>NEXUS Test Suite Report</h1>\n";
        html << "<p>Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << "</p>\n";

        // Summary
        size_t passed = 0, failed = 0, skipped = 0;
        for (const auto& result : results) {
            switch (result.result) {
                case TestResult::PASSED: passed++; break;
                case TestResult::FAILED: failed++; break;
                case TestResult::SKIPPED: skipped++; break;
            }
        }

        html << "<h2>Summary</h2>\n";
        html << "<p>Total: " << results.size() << ", ";
        html << "<span class='pass'>Passed: " << passed << "</span>, ";
        html << "<span class='fail'>Failed: " << failed << "</span>, ";
        html << "<span class='skip'>Skipped: " << skipped << "</span></p>\n";

        // Detailed results
        html << "<h2>Detailed Results</h2>\n";
        html << "<table>\n";
        html << "<tr><th>Category</th><th>Test Name</th><th>Result</th><th>Duration</th><th>Error</th></tr>\n";

        for (const auto& result : results) {
            std::string result_class;
            std::string result_text;

            switch (result.result) {
                case TestResult::PASSED: result_class = "pass"; result_text = "PASS"; break;
                case TestResult::FAILED: result_class = "fail"; result_text = "FAIL"; break;
                case TestResult::SKIPPED: result_class = "skip"; result_text = "SKIP"; break;
            }

            html << "<tr>";
            html << "<td>" << result.category << "</td>";
            html << "<td>" << result.name << "</td>";
            html << "<td class='" << result_class << "'>" << result_text << "</td>";
            html << "<td>" << result.duration.count() << "ms</td>";
            html << "<td>" << result.error_message << "</td>";
            html << "</tr>\n";
        }

        html << "</table>\n</body>\n</html>";

        return html.str();
    }

    // Individual test implementations
    bool testGameEngineInit() {
        try {
            NexusGameEngine engine;
            return engine.getEntityCount() == 0;
        } catch (...) {
            return false;
        }
    }

    bool testEntityManagement() {
        NexusGameEngine engine;

        auto entity1 = engine.createEntity();
        auto entity2 = engine.createEntity();

        return engine.getEntityCount() == 2 &&
               engine.entityExists(entity1) &&
               engine.entityExists(entity2);
    }

    bool testComponentSystem() {
        NexusGameEngine engine;
        auto entity = engine.createEntity();

        auto* transform = engine.addComponent<TransformComponent>(entity);
        auto* audio = engine.addComponent<AudioSyncComponent>(entity);

        return transform != nullptr && audio != nullptr &&
               engine.hasComponent<TransformComponent>(entity) &&
               engine.hasComponent<AudioSyncComponent>(entity);
    }

    bool testResourceEngine() {
        try {
            NexusResourceEngine resource_engine;
            // Basic initialization test
            return true;
        } catch (...) {
            return false;
        }
    }

    bool testQuantumProtocolInit() {
        try {
            NexusProtocol protocol;
            return protocol.getCurrentMode() >= 0;
        } catch (...) {
            return false;
        }
    }

    bool testProcessingModes() {
        NexusProtocol protocol;

        for (int mode = 0; mode < 8; mode++) {
            protocol.setProcessingMode(mode);
            if (protocol.getCurrentMode() != mode) {
                return false;
            }
        }

        return true;
    }

    bool testQuantumCallbacks() {
        NexusProtocol protocol;
        bool callback_called = false;

        protocol.setModeChangeCallback([&callback_called](int old_mode, int new_mode) {
            callback_called = true;
        });

        protocol.setProcessingMode(1);
        return callback_called;
    }

    bool testAudioProcessing() {
        NexusProtocol protocol;

        // Test with dummy audio data
        std::vector<float> audio_data(1024, 0.5f);
        auto result = protocol.processAudioData(audio_data.data(), audio_data.size());

        return result.amplitude >= 0.0f && result.frequency >= 0.0f;
    }

    bool testPaletteSystem() {
        try {
            NexusVisuals visuals;
            auto palette = visuals.getCurrentPalette();
            return palette.size() > 0;
        } catch (...) {
            return false;
        }
    }

    bool testColorTransitions() {
        NexusVisuals visuals;

        // Test color interpolation
        Color start{1.0f, 0.0f, 0.0f, 1.0f};
        Color end{0.0f, 1.0f, 0.0f, 1.0f};

        Color mid = visuals.interpolateColor(start, end, 0.5f);

        return mid.r < 1.0f && mid.g > 0.0f;
    }

    bool testTrailRenderer() {
        try {
            NexusTrailRenderer renderer;
            renderer.addTrailPoint({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
            return renderer.getActiveTrails().size() > 0;
        } catch (...) {
            return false;
        }
    }

    bool testVisualEffects() {
        NexusTrailRenderer renderer;

        // Test glow effect
        renderer.setGlowIntensity(0.8f);

        // Test width modulation
        renderer.setAudioReactiveWidth(true);

        return true; // Basic functionality test
    }

    bool testCognitiveInit() {
        try {
            NexusRecursiveKeeperEngine cognitive;
            return true;
        } catch (...) {
            return false;
        }
    }

    bool testThoughtProcessing() {
        NexusRecursiveKeeperEngine cognitive;

        cognitive.processThought("consciousness", "What is the nature of consciousness?");

        return cognitive.getActiveThoughts().size() > 0;
    }

    bool testMemoryManagement() {
        NexusRecursiveKeeperEngine cognitive;

        cognitive.addMemory("test", "This is a test memory", 0.8f);

        return cognitive.getMemoryCount() > 0;
    }

    bool testPhilosophicalAnalysis() {
        NexusRecursiveKeeperEngine cognitive;

        auto analysis = cognitive.analyzePhilosophicalConcept("existence");

        return !analysis.concept.empty();
    }

    bool testProfiler() {
        try {
            auto& profiler = NexusProfiler::getInstance();
            profiler.startProfiling();

            {
                ScopedTimer timer("test_operation");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            profiler.stopProfiling();
            return true;
        } catch (...) {
            return false;
        }
    }

    bool testBenchmarking() {
        auto& profiler = NexusProfiler::getInstance();

        auto result = profiler.benchmark("simple_loop", []() {
            volatile int sum = 0;
            for (int i = 0; i < 1000; ++i) {
                sum += i;
            }
        }, 100);

        return result.iterations == 100;
    }

    bool testMemoryTracking() {
        auto& profiler = NexusProfiler::getInstance();

        auto initial_memory = profiler.getCurrentMemoryUsage();

        // Allocate some memory
        std::vector<int> large_vector(10000);

        auto final_memory = profiler.getCurrentMemoryUsage();

        return final_memory >= initial_memory;
    }

    bool testConfigSystem() {
        auto& config = NexusConfig::getInstance();

        config.set("test.value", 42, ConfigType::INTEGER);
        int value = config.get<int>("test.value", 0);

        return value == 42;
    }

    bool testPresets() {
        auto& config = NexusConfig::getInstance();

        config.set("preset.test", 100.0f, ConfigType::FLOAT);
        config.createPreset("test_preset", "Test preset");

        config.update("preset.test", 200.0f);
        config.loadPreset("test_preset");

        float value = config.get<float>("preset.test", 0.0f);
        return value == 100.0f; // Should revert to preset value
    }

    bool testConfigIO() {
        auto& config = NexusConfig::getInstance();

        config.set("io.test", std::string("hello"), ConfigType::STRING);

        bool save_success = config.saveToFile("test_config.ini");
        bool load_success = config.loadFromFile("test_config.ini");

        std::string value = config.get<std::string>("io.test", "");

        return save_success && load_success && value == "hello";
    }

    bool testLogger() {
        NEXUS_LOG_INFO("TEST", "Test log message");
        NEXUS_LOG_WARN("TEST", "Test warning message");
        NEXUS_LOG_ERROR("TEST", "Test error message");

        return true; // If we got here without crashing, logging works
    }

    bool testMultiSinkLogging() {
        auto& logger = NexusLogger::getInstance();

        // Add a memory sink for testing
        logger.addSink(std::make_unique<MemorySink>(100));

        NEXUS_LOG_DEBUG("TEST", "Multi-sink test message");

        return true;
    }

    bool testLogFormatting() {
        StandardFormatter formatter;
        LogEntry entry(LogLevel::INFO, "TEST", "Formatting test", __FILE__, __LINE__);

        std::string formatted = formatter.format(entry);

        return formatted.find("TEST") != std::string::npos &&
               formatted.find("Formatting test") != std::string::npos;
    }

    bool testSystemIntegration() {
        // Test integration between major systems
        NexusGameEngine engine;
        NexusProtocol protocol;
        NexusVisuals visuals;
        NexusRecursiveKeeperEngine cognitive;

        auto entity = engine.createEntity();
        auto* transform = engine.addComponent<TransformComponent>(entity);
        auto* audio = engine.addComponent<AudioSyncComponent>(entity);

        protocol.setProcessingMode(2);
        cognitive.processThought("integration", "Testing system integration");

        return transform != nullptr && audio != nullptr;
    }

    bool testMultithreading() {
        std::vector<std::thread> threads;
        std::atomic<int> counter(0);

        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&counter]() {
                for (int j = 0; j < 1000; ++j) {
                    counter++;
                    NEXUS_LOG_TRACE("THREAD", "Thread operation: " + std::to_string(j));
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        return counter.load() == 4000;
    }

    bool testStressLoad() {
        NexusGameEngine engine;

        // Create many entities
        std::vector<EntityID> entities;
        for (int i = 0; i < 1000; ++i) {
            auto entity = engine.createEntity();
            engine.addComponent<TransformComponent>(entity);
            entities.push_back(entity);
        }

        // Process many thoughts
        NexusRecursiveKeeperEngine cognitive;
        for (int i = 0; i < 100; ++i) {
            cognitive.processThought("stress", "Stress test thought " + std::to_string(i));
        }

        return engine.getEntityCount() == 1000;
    }

    bool benchmarkEntityCreation() {
        auto& profiler = NexusProfiler::getInstance();

        auto result = profiler.benchmark("entity_creation", []() {
            NexusGameEngine engine;
            for (int i = 0; i < 100; ++i) {
                auto entity = engine.createEntity();
                engine.addComponent<TransformComponent>(entity);
            }
        }, 10);

        NEXUS_LOG_INFO("BENCHMARK", "Entity creation: " + std::to_string(result.average_time_us) + "μs avg");

        return result.average_time_us > 0;
    }

    bool benchmarkQuantumProcessing() {
        auto& profiler = NexusProfiler::getInstance();

        auto result = profiler.benchmark("quantum_processing", []() {
            NexusProtocol protocol;
            std::vector<float> data(1024, 0.5f);

            for (int i = 0; i < 8; ++i) {
                protocol.setProcessingMode(i);
                protocol.processAudioData(data.data(), data.size());
            }
        }, 10);

        NEXUS_LOG_INFO("BENCHMARK", "Quantum processing: " + std::to_string(result.average_time_us) + "μs avg");

        return result.average_time_us > 0;
    }

    bool benchmarkMemoryAllocation() {
        auto& profiler = NexusProfiler::getInstance();

        auto result = profiler.benchmark("memory_allocation", []() {
            std::vector<std::vector<int>> vectors;
            for (int i = 0; i < 100; ++i) {
                vectors.emplace_back(1000, i);
            }
        }, 10);

        NEXUS_LOG_INFO("BENCHMARK", "Memory allocation: " + std::to_string(result.average_time_us) + "μs avg");

        return result.average_time_us > 0;
    }
};

} // namespace NEXUS
