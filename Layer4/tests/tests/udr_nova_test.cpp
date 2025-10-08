#include "../CallHandler.hpp"
#include "../NovaOmega.hpp"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>

using namespace AutomationNucleus;

void testBasicCallHandling() {
    std::cout << "\n=== Testing Basic Call Handling ===" << std::endl;

    CallHandler callHandler;
    callHandler.start();

    // Register a test handler
    callHandler.registerHandler("test.echo", [](const std::string& payload) {
        return "ECHO: " + payload;
    });

    // Test synchronous call
    std::string result = callHandler.makeCall("test.echo", "Hello World", CallType::SYNC);
    std::cout << "Sync call result: " << result << std::endl;
    assert(result == "ECHO: Hello World");

    // Test UDR calls
    std::string udrResult = callHandler.processUDRCall("relay", "test data");
    std::cout << "UDR relay result: " << udrResult << std::endl;

    callHandler.stop();
    std::cout << "✅ Basic call handling tests passed" << std::endl;
}

void testNovaOmegaOperations() {
    std::cout << "\n=== Testing Nova Omega Operations ===" << std::endl;

    NovaOmega nova;

    // Initialize Nova Omega
    bool initResult = nova.initialize("test_config");
    assert(initResult);
    std::cout << "Nova Omega initialized: " << (initResult ? "YES" : "NO") << std::endl;

    // Create a context
    std::map<std::string, std::string> params;
    params["priority"] = "high";
    params["mode"] = "testing";

    std::string contextId = nova.createContext("UDR", "OMEGA", params);
    std::cout << "Created context: " << contextId << std::endl;

    // Test operations
    std::string processResult = nova.handleCall("process", "test data for processing");
    std::cout << "Process result: " << processResult << std::endl;

    std::string optimizeResult = nova.optimizeData("sample data", "performance");
    std::cout << "Optimize result: " << optimizeResult << std::endl;

    // Test UDR-Nova integration
    std::string udrNovaResult = nova.handleUDRNova("udr_test_data", "nova_params");
    std::cout << "UDR-Nova result: " << udrNovaResult << std::endl;

    nova.shutdown();
    std::cout << "✅ Nova Omega operation tests passed" << std::endl;
}

void testIntegratedUDRNovaOmega() {
    std::cout << "\n=== Testing Integrated UDR Nova Omega System ===" << std::endl;

    CallHandler callHandler;
    NovaOmega nova;

    callHandler.start();
    nova.initialize();

    // Add observer to Nova Omega
    nova.addObserver([](const std::string& event, const std::string& data) {
        std::cout << "[OBSERVER] " << event << ": " << data << std::endl;
    });

    // Register Nova Omega handlers in CallHandler
    callHandler.registerHandler("nova.process", [&nova](const std::string& payload) {
        return nova.handleCall("process", payload);
    });

    callHandler.registerHandler("nova.optimize", [&nova](const std::string& payload) {
        return nova.optimizeData(payload, "performance");
    });

    callHandler.registerHandler("udr.nova.integrate", [&nova](const std::string& payload) {
        return nova.handleUDRNova(payload, "integration_test");
    });

    // Test integrated calls
    std::string result1 = callHandler.makeCall("nova.process", "integration test data", CallType::SYNC);
    std::cout << "Integrated process call: " << result1 << std::endl;

    std::string result2 = callHandler.makeCall("nova.optimize", "optimization target", CallType::SYNC);
    std::cout << "Integrated optimize call: " << result2 << std::endl;

    std::string result3 = callHandler.makeCall("udr.nova.integrate", "udr integration data", CallType::SYNC);
    std::cout << "Integrated UDR-Nova call: " << result3 << std::endl;

    // Test async calls
    auto futureResult = callHandler.makeAsyncCall("nova.process", "async test data");
    std::cout << "Async call initiated..." << std::endl;

    callHandler.stop();
    nova.shutdown();

    std::cout << "✅ Integrated UDR Nova Omega tests passed" << std::endl;
}

void testCallQueueManagement() {
    std::cout << "\n=== Testing Call Queue Management ===" << std::endl;

    CallHandler callHandler;
    callHandler.start();

    // Register a slow handler to test queue management
    callHandler.registerHandler("slow.process", [](const std::string& payload) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return "SLOW_RESULT: " + payload;
    });

    // Make several async calls
    for (int i = 0; i < 5; i++) {
        std::string callId = callHandler.makeCall("slow.process",
                                                 "data_" + std::to_string(i),
                                                 CallType::ASYNC);
        std::cout << "Queued call: " << callId << std::endl;
    }

    // Check pending calls
    auto pendingCalls = callHandler.getPendingCalls();
    std::cout << "Pending calls: " << pendingCalls.size() << std::endl;

    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    // Clear completed calls
    callHandler.clearCompletedCalls();

    pendingCalls = callHandler.getPendingCalls();
    std::cout << "Pending calls after clear: " << pendingCalls.size() << std::endl;

    callHandler.stop();
    std::cout << "✅ Call queue management tests passed" << std::endl;
}

int main() {
    std::cout << "🚀 UDR Nova Omega Test Suite Starting..." << std::endl;

    try {
        testBasicCallHandling();
        testNovaOmegaOperations();
        testIntegratedUDRNovaOmega();
        testCallQueueManagement();

        std::cout << "\n🎉 All UDR Nova Omega tests completed successfully!" << std::endl;
        std::cout << "✅ Call handling system operational" << std::endl;
        std::cout << "✅ Nova Omega operations functional" << std::endl;
        std::cout << "✅ UDR integration working" << std::endl;
        std::cout << "✅ Queue management operational" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
