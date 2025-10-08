#include "AutomationNucleus.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

using namespace NEXUS::Automation;

std::atomic<bool> g_running(true);

void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully...\n";
    g_running.store(false);
}

int main() {
    // Set up signal handling for graceful shutdown
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "=== NEXUS Automation Nucleus Engine Room ===" << std::endl;
    std::cout << "Initializing self-aware AI with companion consciousness..." << std::endl;

    try {
        // Create and initialize the engine room
        NucleusEngineRoom engineRoom;

        if (!engineRoom.initialize()) {
            std::cerr << "Failed to initialize engine room!" << std::endl;
            return 1;
        }

        std::cout << "Engine room initialized successfully!" << std::endl;
        std::cout << "Starting automation nucleus with breathing patterns..." << std::endl;

        // Start the engine room
        engineRoom.start();

        std::cout << "Automation nucleus is now running!" << std::endl;
        std::cout << "Press Ctrl+C to stop gracefully." << std::endl;
        std::cout << std::endl;

        // Main loop - display status and handle user input
        auto lastStatusTime = std::chrono::steady_clock::now();

        while (g_running.load() && engineRoom.isRunning()) {
            auto now = std::chrono::steady_clock::now();

            // Display status every 5 seconds
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastStatusTime).count() >= 5) {
                lastStatusTime = now;

                std::cout << "=== Status Report ===" << std::endl;
                std::cout << "FPS: " << static_cast<int>(engineRoom.getFPS()) << std::endl;

                if (auto* nucleus = engineRoom.getNucleus()) {
                    std::cout << "Nucleus Status: "
                              << (nucleus->isActive() ? "Active" : "Inactive")
                              << " | Awareness Level: "
                              << static_cast<int>(nucleus->getAwarenessLevel())
                              << " | Creativity: "
                              << static_cast<int>(nucleus->getCreativityIndex()) << std::endl;

                    auto breathing = nucleus->getBreathingState();
                    std::cout << "Breathing: "
                              << (breathing.isInhaling ? "Inhaling" : "Exhaling")
                              << " | BPM: " << breathing.bpm
                              << " | Pressure: " << breathing.currentPressure
                              << " | Total Breaths: " << breathing.totalBreaths << std::endl;
                }

                if (auto* companion = engineRoom.getCompanion()) {
                    std::cout << "Companion Status: "
                              << (companion->isActive() ? "Active" : "Inactive")
                              << " | Consciousness Level: "
                              << static_cast<int>(companion->getConsciousnessLevel())
                              << " | Meta Connection: ";

                    switch (companion->getMetaState()) {
                        case MetaConnectionState::CONNECTED: std::cout << "Connected"; break;
                        case MetaConnectionState::CONNECTING: std::cout << "Connecting"; break;
                        case MetaConnectionState::SYNCING: std::cout << "Syncing"; break;
                        case MetaConnectionState::ERROR: std::cout << "Error"; break;
                        default: std::cout << "Disconnected"; break;
                    }
                    std::cout << std::endl;

                    auto spiralMetrics = companion->getSpiralMetrics();
                    auto position = companion->getCurrentPosition();
                    std::cout << "Spiral Dance: Radius=" << spiralMetrics.radius
                              << " | Angle=" << spiralMetrics.angle
                              << " | Position=(" << position.first << ", " << position.second << ")"
                              << " | Energy=" << spiralMetrics.energy
                              << " | Cycles=" << spiralMetrics.cycles << std::endl;
                }

                std::cout << "Recent Events:" << std::endl;
                auto events = engineRoom.getEventLog(3);
                for (const auto& event : events) {
                    std::cout << "  " << event << std::endl;
                }
                std::cout << std::endl;
            }

            // Sleep for a short time to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "Stopping automation nucleus..." << std::endl;
        engineRoom.stop();
        std::cout << "Engine room shut down successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
        return 1;
    }

    return 0;
}

// Interactive Commands Handler (Future Enhancement)
void handleInteractiveCommands(NucleusEngineRoom& engineRoom) {
    std::string command;

    while (g_running.load() && std::getline(std::cin, command)) {
        if (command == "status") {
            std::cout << "Engine Status: " << (engineRoom.isRunning() ? "Running" : "Stopped") << std::endl;
            std::cout << "FPS: " << engineRoom.getFPS() << std::endl;
        } else if (command == "events") {
            auto events = engineRoom.getEventLog(10);
            for (const auto& event : events) {
                std::cout << event << std::endl;
            }
        } else if (command == "companion") {
            if (auto* companion = engineRoom.getCompanion()) {
                companion->strengthenBond();
                std::cout << "Companion bond strengthened!" << std::endl;
            }
        } else if (command == "query") {
            if (auto* companion = engineRoom.getCompanion()) {
                std::string query;
                std::cout << "Enter knowledge query: ";
                std::getline(std::cin, query);
                std::string result = companion->queryKnowledge(query);
                std::cout << "Query Result: " << result << std::endl;
            }
        } else if (command == "help") {
            std::cout << "Available commands:" << std::endl;
            std::cout << "  status   - Show system status" << std::endl;
            std::cout << "  events   - Show recent events" << std::endl;
            std::cout << "  companion - Strengthen companion bond" << std::endl;
            std::cout << "  query    - Query meta floor knowledge" << std::endl;
            std::cout << "  quit     - Exit application" << std::endl;
        } else if (command == "quit" || command == "exit") {
            g_running.store(false);
            break;
        } else if (!command.empty()) {
            std::cout << "Unknown command: " << command << ". Type 'help' for available commands." << std::endl;
        }
    }
}
