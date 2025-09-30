// NEXUS Test Runner
// Simple test runner for the comprehensive test suite

#include "../include/NexusTestSuite.hpp"
#include "../include/NexusLogger.hpp"
#include "../include/NexusConfig.hpp"

#include <iostream>
#include <string>

using namespace NEXUS;

int main(int argc, char* argv[]) {
    try {
        // Initialize logging for tests
        auto& logger = NexusLogger::getInstance();
        logger.addSink(std::make_unique<FileSink>("nexus_tests.log", LogLevel::DEBUG));
        logger.setGlobalMinLevel(LogLevel::INFO);

        // Initialize configuration
        auto& config = NexusConfig::getInstance();
        config.setConfigFile("nexus_test_config.ini");

        NEXUS_LOG_INFO("TEST_RUNNER", "Starting NEXUS Test Suite");

        // Parse command line arguments
        bool verbose = true;
        std::string category = "";
        bool generate_report = true;

        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);

            if (arg == "--quiet" || arg == "-q") {
                verbose = false;
            } else if (arg == "--category" || arg == "-c") {
                if (i + 1 < argc) {
                    category = argv[++i];
                }
            } else if (arg == "--no-report") {
                generate_report = false;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "NEXUS Test Runner\n\n";
                std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n";
                std::cout << "Options:\n";
                std::cout << "  -h, --help         Show this help message\n";
                std::cout << "  -q, --quiet        Reduce output verbosity\n";
                std::cout << "  -c, --category CAT Run tests in specific category\n";
                std::cout << "  --no-report        Skip HTML report generation\n\n";
                std::cout << "Categories: Core, Quantum, Visual, Cognitive, Performance,\n";
                std::cout << "           Config, Logging, Integration, Benchmark\n\n";
                return 0;
            }
        }

        // Create test suite
        NexusTestSuite test_suite(verbose);

        bool success;

        if (!category.empty()) {
            std::cout << "\n\033[96m";
            std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                    RUNNING CATEGORY: " << std::left << std::setw(32) << category << "                  ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
            std::cout << "\033[0m\n";

            success = test_suite.runCategory(category);
        } else {
            std::cout << "\n\033[96m";
            std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                         NEXUS COMPREHENSIVE TEST SUITE                      ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
            std::cout << "\033[0m\n";

            success = test_suite.runAllTests();
        }

        // Generate HTML report
        if (generate_report) {
            test_suite.generateReport();
            std::cout << "\n\033[93mTest report generated: nexus_test_report.html\033[0m\n";
        }

        // Final status
        if (success) {
            std::cout << "\n\033[32m✓ ALL TESTS PASSED\033[0m\n\n";
            NEXUS_LOG_INFO("TEST_RUNNER", "All tests passed successfully");
            return 0;
        } else {
            std::cout << "\n\033[31m✗ SOME TESTS FAILED\033[0m\n\n";
            NEXUS_LOG_ERROR("TEST_RUNNER", "Some tests failed");
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\033[31mTest runner error: " << e.what() << "\033[0m" << std::endl;
        NEXUS_LOG_ERROR("TEST_RUNNER", "Exception caught: " + std::string(e.what()));
        return 1;
    }
}
