// NEXUS Comprehensive System Demo
// This demo showcases all integrated systems working together

#include "../include/NexusGameEngine.hpp"
#include "../include/NexusProtocol.hpp"
#include "../include/NexusVisuals.hpp"
#include "../include/NexusTrailRenderer.hpp"
#include "../include/NexusRecursiveKeeperEngine.hpp"
#include "../include/NexusProfiler.hpp"
#include "../include/NexusLogger.hpp"
#include "../include/NexusConfig.hpp"
#include "../include/NexusTestSuite.hpp"
#include "../include/NexusDashboard.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <cmath>

using namespace NEXUS;

class NexusSystemDemo {
private:
    NexusGameEngine engine;
    NexusProtocol quantum_protocol;
    NexusVisuals visual_system;
    NexusTrailRenderer trail_renderer;
    NexusRecursiveKeeperEngine cognitive_engine;

    std::vector<EntityID> demo_entities;
    std::mt19937 random_engine;
    bool demo_running;

public:
    NexusSystemDemo() : random_engine(std::chrono::steady_clock::now().time_since_epoch().count()),
                        demo_running(false) {
        NEXUS_LOG_INFO("DEMO", "NEXUS System Demo initialized");
    }

    void initialize() {
        NEXUS_LOG_INFO("DEMO", "Initializing comprehensive system demo...");

        // Setup logging with multiple sinks
        auto& logger = NexusLogger::getInstance();
        logger.addSink(std::make_unique<FileSink>("nexus_demo.log", LogLevel::DEBUG));
        logger.addSink(std::make_unique<MemorySink>(500, LogLevel::TRACE));

        // Setup configuration
        auto& config = NexusConfig::getInstance();
        config.setConfigFile("nexus_demo.ini");
        config.loadFromFile();

        // Create performance preset
        config.createPreset("Performance", "High performance settings for demo");
        config.createPreset("Quality", "High quality visual settings");
        config.createPreset("Debug", "Debug and testing configuration");

        // Setup quantum protocol callbacks
        quantum_protocol.setModeChangeCallback([this](int old_mode, int new_mode) {
            NEXUS_LOG_INFO("QUANTUM", "Processing mode changed: " + std::to_string(old_mode) + " -> " + std::to_string(new_mode));

            // Update visual system based on mode
            switch (new_mode) {
                case 0: visual_system.setPalette("Mirror"); break;
                case 1: visual_system.setPalette("Cosine"); break;
                case 2: visual_system.setPalette("Chaos"); break;
                case 3: visual_system.setPalette("Absorb"); break;
                case 4: visual_system.setPalette("Amplify"); break;
                case 5: visual_system.setPalette("Pulse"); break;
                case 6: visual_system.setPalette("Flow"); break;
                case 7: visual_system.setPalette("Fragment"); break;
            }
        });

        // Setup audio processing callback
        quantum_protocol.setAudioProcessingCallback([this](const AudioFeatures& features) {
            // Update trail renderer with audio data
            trail_renderer.updateAudioFeatures(features.amplitude, features.frequency, features.spectral_centroid);

            // Feed audio characteristics to cognitive engine
            if (features.amplitude > 0.7f) {
                cognitive_engine.processThought("audio_intensity",
                    "Experiencing high amplitude audio: " + std::to_string(features.amplitude));
            }
        });

        // Create demo entities
        createDemoEntities();

        NEXUS_LOG_INFO("DEMO", "System initialization complete");
    }

    void runDemo() {
        demo_running = true;

        // Start profiler
        auto& profiler = NexusProfiler::getInstance();
        profiler.startProfiling();

        // Start dashboard
        NEXUS_DASHBOARD_START();

        NEXUS_LOG_INFO("DEMO", "Starting comprehensive demo simulation...");

        std::cout << "\n\033[96m";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                          NEXUS SYSTEM DEMO STARTED                          ║\n";
        std::cout << "║                                                                              ║\n";
        std::cout << "║  • Entity-Component System with " << demo_entities.size() << " active entities                             ║\n";
        std::cout << "║  • Quantum Protocol processing with 8 different modes                       ║\n";
        std::cout << "║  • Audio-reactive visual systems and trail rendering                        ║\n";
        std::cout << "║  • Recursive Keeper Engine for cognitive processing                         ║\n";
        std::cout << "║  • Real-time performance profiling and system monitoring                    ║\n";
        std::cout << "║                                                                              ║\n";
        std::cout << "║  Dashboard will update every second. Press Ctrl+C to exit.                  ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\033[0m\n";

        // Wait a moment for user to read
        std::this_thread::sleep_for(std::chrono::seconds(3));

        auto last_update = std::chrono::steady_clock::now();
        auto last_cognitive_update = std::chrono::steady_clock::now();
        auto last_quantum_mode_change = std::chrono::steady_clock::now();

        int frame_count = 0;

        while (demo_running) {
            auto current_time = std::chrono::steady_clock::now();
            auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_update).count() / 1000.0f;

            {
                ScopedTimer frame_timer("demo_frame");

                // Update all systems
                updateEntitySystems(delta_time);
                updateQuantumProtocol(current_time, last_quantum_mode_change);
                updateVisualSystems(delta_time);
                updateCognitiveEngine(current_time, last_cognitive_update);

                // Update dashboard metrics
                updateDashboardMetrics();

                // Render dashboard
                NEXUS_DASHBOARD_RENDER();

                frame_count++;
                last_update = current_time;
            }

            // Control demo speed
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            // Safety exit after 60 seconds for demo purposes
            if (frame_count > 60) {
                NEXUS_LOG_INFO("DEMO", "Demo time limit reached, stopping...");
                break;
            }
        }

        shutdown();
    }

    void runTestSuite() {
        NEXUS_LOG_INFO("DEMO", "Running comprehensive test suite...");

        NexusTestSuite test_suite(true);
        bool all_passed = test_suite.runAllTests();

        if (all_passed) {
            NEXUS_LOG_INFO("TEST", "All tests passed! System is ready for production.");
            std::cout << "\n\033[32m✓ ALL TESTS PASSED - SYSTEM READY\033[0m\n\n";
        } else {
            NEXUS_LOG_ERROR("TEST", "Some tests failed. Please review the results.");
            std::cout << "\n\033[31m✗ SOME TESTS FAILED - PLEASE REVIEW\033[0m\n\n";
        }

        // Generate test report
        test_suite.generateReport();
        NEXUS_LOG_INFO("TEST", "Test report generated: nexus_test_report.html");
    }

    void showSystemInfo() {
        auto& dashboard = GlobalDashboard::getInstance();
        dashboard.showSystemInfo();

        std::cout << "\nPress Enter to continue...";
        std::cin.get();
    }

    void benchmarkSystems() {
        NEXUS_LOG_INFO("DEMO", "Running system benchmarks...");

        auto& profiler = NexusProfiler::getInstance();

        // Benchmark entity creation
        auto entity_benchmark = profiler.benchmark("entity_creation", [this]() {
            for (int i = 0; i < 100; ++i) {
                auto entity = engine.createEntity();
                engine.addComponent<TransformComponent>(entity);
                engine.addComponent<AudioSyncComponent>(entity);
            }
        }, 10);

        // Benchmark quantum processing
        auto quantum_benchmark = profiler.benchmark("quantum_processing", [this]() {
            std::vector<float> audio_data(1024);
            std::iota(audio_data.begin(), audio_data.end(), 0.0f);

            for (int mode = 0; mode < 8; ++mode) {
                quantum_protocol.setProcessingMode(mode);
                quantum_protocol.processAudioData(audio_data.data(), audio_data.size());
            }
        }, 10);

        // Benchmark cognitive processing
        auto cognitive_benchmark = profiler.benchmark("cognitive_processing", [this]() {
            for (int i = 0; i < 10; ++i) {
                cognitive_engine.processThought("benchmark", "Benchmark thought " + std::to_string(i));
            }
        }, 10);

        // Display results
        std::cout << "\n\033[93m" << std::string(80, '=') << "\033[0m\n";
        std::cout << "\033[93mBENCHMARK RESULTS\033[0m\n";
        std::cout << "\033[93m" << std::string(80, '=') << "\033[0m\n\n";

        std::cout << "Entity Creation:\n";
        std::cout << "  Average: " << entity_benchmark.average_time_us << " μs\n";
        std::cout << "  Min/Max: " << entity_benchmark.min_time_us << " / " << entity_benchmark.max_time_us << " μs\n\n";

        std::cout << "Quantum Processing:\n";
        std::cout << "  Average: " << quantum_benchmark.average_time_us << " μs\n";
        std::cout << "  Min/Max: " << quantum_benchmark.min_time_us << " / " << quantum_benchmark.max_time_us << " μs\n\n";

        std::cout << "Cognitive Processing:\n";
        std::cout << "  Average: " << cognitive_benchmark.average_time_us << " μs\n";
        std::cout << "  Min/Max: " << cognitive_benchmark.min_time_us << " / " << cognitive_benchmark.max_time_us << " μs\n\n";

        std::cout << "\033[93m" << std::string(80, '=') << "\033[0m\n";

        NEXUS_LOG_INFO("BENCHMARK", "Benchmark complete");
    }

    void stop() {
        demo_running = false;
        NEXUS_LOG_INFO("DEMO", "Demo stop requested");
    }

private:
    void createDemoEntities() {
        NEXUS_LOG_INFO("DEMO", "Creating demo entities...");

        // Create various types of entities
        for (int i = 0; i < 50; ++i) {
            auto entity = engine.createEntity();

            // Add transform component
            auto* transform = engine.addComponent<TransformComponent>(entity);
            std::uniform_real_distribution<float> pos_dist(-100.0f, 100.0f);
            transform->position = {pos_dist(random_engine), pos_dist(random_engine), pos_dist(random_engine)};

            // Add audio sync component to some entities
            if (i % 2 == 0) {
                auto* audio = engine.addComponent<AudioSyncComponent>(entity);
                audio->amplitude_sensitivity = 1.0f + static_cast<float>(i) * 0.1f;
            }

            // Add art sync component to others
            if (i % 3 == 0) {
                auto* art = engine.addComponent<ArtSyncComponent>(entity);
                art->color_sensitivity = 0.5f + static_cast<float>(i) * 0.05f;
            }

            // Add physics to some entities
            if (i % 5 == 0) {
                auto* physics = engine.addComponent<PhysicsComponent>(entity);
                std::uniform_real_distribution<float> vel_dist(-10.0f, 10.0f);
                physics->velocity = {vel_dist(random_engine), vel_dist(random_engine), vel_dist(random_engine)};
            }

            demo_entities.push_back(entity);
        }

        NEXUS_LOG_INFO("DEMO", "Created " + std::to_string(demo_entities.size()) + " demo entities");
    }

    void updateEntitySystems(float delta_time) {
        ScopedTimer timer("entity_update");

        // Update transforms and physics
        for (auto entity : demo_entities) {
            auto* transform = engine.getComponent<TransformComponent>(entity);
            auto* physics = engine.getComponent<PhysicsComponent>(entity);

            if (transform && physics) {
                // Simple physics integration
                transform->position.x += physics->velocity.x * delta_time;
                transform->position.y += physics->velocity.y * delta_time;
                transform->position.z += physics->velocity.z * delta_time;

                // Boundary bouncing
                const float boundary = 100.0f;
                if (std::abs(transform->position.x) > boundary) physics->velocity.x *= -0.8f;
                if (std::abs(transform->position.y) > boundary) physics->velocity.y *= -0.8f;
                if (std::abs(transform->position.z) > boundary) physics->velocity.z *= -0.8f;

                // Add some rotation
                transform->rotation.y += delta_time * 45.0f;
            }
        }
    }

    void updateQuantumProtocol(std::chrono::steady_clock::time_point current_time,
                              std::chrono::steady_clock::time_point& last_mode_change) {
        ScopedTimer timer("quantum_update");

        // Change quantum mode every 5 seconds
        if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_mode_change).count() >= 5) {
            int new_mode = (quantum_protocol.getCurrentMode() + 1) % 8;
            quantum_protocol.setProcessingMode(new_mode);
            last_mode_change = current_time;
        }

        // Generate synthetic audio data
        static float audio_time = 0.0f;
        audio_time += 0.1f;

        std::vector<float> audio_data(512);
        for (size_t i = 0; i < audio_data.size(); ++i) {
            float freq = 440.0f + 200.0f * std::sin(audio_time * 0.5f);
            audio_data[i] = 0.3f * std::sin(2.0f * M_PI * freq * i / 44100.0f) *
                           (0.5f + 0.5f * std::sin(audio_time));
        }

        // Process audio data
        quantum_protocol.processAudioData(audio_data.data(), audio_data.size());
    }

    void updateVisualSystems(float delta_time) {
        ScopedTimer timer("visual_update");

        // Update visual system
        visual_system.update(delta_time);

        // Add trail points
        for (size_t i = 0; i < 5; ++i) {
            float angle = static_cast<float>(i) * 2.0f * M_PI / 5.0f + delta_time;
            Vec3 position = {
                50.0f * std::cos(angle),
                20.0f * std::sin(angle * 2.0f),
                30.0f * std::sin(angle * 0.5f)
            };

            auto palette = visual_system.getCurrentPalette();
            Color color = palette.empty() ? Color{1.0f, 1.0f, 1.0f, 1.0f} : palette[i % palette.size()];

            trail_renderer.addTrailPoint(position, color);
        }

        trail_renderer.update(delta_time);
    }

    void updateCognitiveEngine(std::chrono::steady_clock::time_point current_time,
                              std::chrono::steady_clock::time_point& last_cognitive_update) {
        ScopedTimer timer("cognitive_update");

        // Add new thoughts periodically
        if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_cognitive_update).count() >= 3) {
            std::vector<std::string> thought_topics = {
                "consciousness", "reality", "existence", "time", "space", "identity", "knowledge",
                "perception", "truth", "beauty", "meaning", "purpose", "infinity", "paradox"
            };

            std::uniform_int_distribution<size_t> topic_dist(0, thought_topics.size() - 1);
            std::string topic = thought_topics[topic_dist(random_engine)];

            std::string thought = "Demo contemplation on " + topic +
                                " - frame " + std::to_string(current_time.time_since_epoch().count());

            cognitive_engine.processThought(topic, thought);
            last_cognitive_update = current_time;
        }

        cognitive_engine.update(0.1f);
    }

    void updateDashboardMetrics() {
        auto& profiler = NexusProfiler::getInstance();

        NexusDashboard::SystemMetrics metrics;
        metrics.cpu_usage = 45.0f; // Would normally get from system
        metrics.memory_usage = profiler.getCurrentMemoryUsage();
        metrics.total_memory = 8ULL * 1024 * 1024 * 1024; // 8GB
        metrics.fps = 60.0f;
        metrics.active_entities = demo_entities.size();
        metrics.active_thoughts = cognitive_engine.getActiveThoughts().size();
        metrics.quantum_mode = quantum_protocol.getCurrentMode();

        const std::vector<std::string> mode_names = {
            "MIRROR", "COSINE", "CHAOS", "ABSORB", "AMPLIFY", "PULSE", "FLOW", "FRAGMENT"
        };
        metrics.quantum_mode_name = mode_names[metrics.quantum_mode];

        auto audio_features = quantum_protocol.getLastAudioFeatures();
        metrics.quantum_intensity = audio_features.amplitude;

        metrics.total_log_entries = 1000; // Would get from logger

        auto& config = NexusConfig::getInstance();
        metrics.active_preset = config.getActivePreset();

        GlobalDashboard::getInstance().updateMetrics(metrics);
    }

    void shutdown() {
        NEXUS_LOG_INFO("DEMO", "Shutting down demo systems...");

        // Stop profiler
        auto& profiler = NexusProfiler::getInstance();
        profiler.stopProfiling();

        // Save configuration
        auto& config = NexusConfig::getInstance();
        config.saveToFile();

        // Generate final profiling report
        profiler.generateDetailedReport("nexus_demo_profile.txt");

        // Stop dashboard
        NEXUS_DASHBOARD_STOP();

        NEXUS_LOG_INFO("DEMO", "Demo shutdown complete");
        NexusLogger::getInstance().flush();
    }
};

int main() {
    try {
        std::cout << "\033[2J\033[H"; // Clear screen

        NexusSystemDemo demo;
        demo.initialize();

        // Show menu
        while (true) {
            std::cout << "\n\033[96m";
            std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                           NEXUS SYSTEM DEMO MENU                            ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║  1. Run Interactive Demo (60 seconds with live dashboard)                   ║\n";
            std::cout << "║  2. Run Complete Test Suite                                                  ║\n";
            std::cout << "║  3. Show System Information                                                  ║\n";
            std::cout << "║  4. Run Performance Benchmarks                                              ║\n";
            std::cout << "║  5. Exit                                                                     ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
            std::cout << "\033[0m";
            std::cout << "\nEnter your choice (1-5): ";

            int choice;
            if (!(std::cin >> choice)) {
                std::cin.clear();
                std::cin.ignore(10000, '\n');
                std::cout << "\033[31mInvalid input. Please enter a number.\033[0m\n";
                continue;
            }
            std::cin.ignore(); // Consume newline

            switch (choice) {
                case 1:
                    demo.runDemo();
                    break;
                case 2:
                    demo.runTestSuite();
                    std::cout << "\nPress Enter to continue...";
                    std::cin.get();
                    break;
                case 3:
                    demo.showSystemInfo();
                    break;
                case 4:
                    demo.benchmarkSystems();
                    std::cout << "\nPress Enter to continue...";
                    std::cin.get();
                    break;
                case 5:
                    std::cout << "\n\033[96mThank you for using NEXUS! Goodbye.\033[0m\n\n";
                    return 0;
                default:
                    std::cout << "\033[31mInvalid choice. Please select 1-5.\033[0m\n";
                    break;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\033[31mError: " << e.what() << "\033[0m" << std::endl;
        return 1;
    }

    return 0;
}
