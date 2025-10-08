// NEXUS WebSocket Bridge for Dashboard Integration
// Real-time communication between web dashboards and C++ NEXUS engine
// Handles data exchange, control commands, and streaming for both utilities and testing

#pragma once
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include <queue>

namespace NEXUS {

// ============ MESSAGE TYPES ============

enum class MessageType {
    UTILITY_DATA,           // Data from/to engine utilities
    TEST_COMMAND,          // Test commands for nucleus learning
    TEST_RESULT,           // Results from dashboard tests
    ENGINE_STATUS,         // Engine status updates
    DASHBOARD_REGISTER,    // Dashboard registration
    METRICS_REQUEST,       // Request for metrics data
    CONTROL_COMMAND        // Control commands (start/stop/configure)
};

struct WebSocketMessage {
    MessageType type;
    std::string dashboard_id;
    std::string payload;
    std::chrono::steady_clock::time_point timestamp;

    WebSocketMessage(MessageType t, const std::string& id, const std::string& data)
        : type(t), dashboard_id(id), payload(data), timestamp(std::chrono::steady_clock::now()) {}
};

// ============ CONNECTION MANAGEMENT ============

struct DashboardConnection {
    websocketpp::connection_hdl handle;
    std::string dashboard_id;
    std::string dashboard_type;
    bool is_active{true};
    std::chrono::steady_clock::time_point last_ping;
    size_t messages_sent{0};
    size_t messages_received{0};

    DashboardConnection(websocketpp::connection_hdl hdl, const std::string& id, const std::string& type)
        : handle(hdl), dashboard_id(id), dashboard_type(type), last_ping(std::chrono::steady_clock::now()) {}
};

// ============ WEBSOCKET BRIDGE ============

class WebSocketBridge {
private:
    using WebSocketServer = websocketpp::server<websocketpp::config::asio>;
    using json = nlohmann::json;

    WebSocketServer server;
    std::thread server_thread;
    std::atomic<bool> running{false};
    uint16_t port{8765};

    // Connection management
    std::unordered_map<std::string, std::unique_ptr<DashboardConnection>> connections;
    std::mutex connections_mutex;

    // Message queues
    std::queue<WebSocketMessage> incoming_messages;
    std::queue<WebSocketMessage> outgoing_messages;
    std::mutex message_mutex;

    // Message handlers
    std::unordered_map<MessageType, std::function<void(const WebSocketMessage&)>> message_handlers;

    // Dashboard Integration Manager reference
    class DashboardIntegrationManager* integration_manager{nullptr};

public:
    WebSocketBridge(uint16_t server_port = 8765) : port(server_port) {
        setupServer();
        registerDefaultHandlers();
    }

    ~WebSocketBridge() {
        stop();
    }

    void setIntegrationManager(DashboardIntegrationManager* manager) {
        integration_manager = manager;
    }

    void start() {
        if (running.load()) return;

        running.store(true);
        server_thread = std::thread([this]() {
            serverLoop();
        });
    }

    void stop() {
        if (!running.load()) return;

        running.store(false);
        server.stop();

        if (server_thread.joinable()) {
            server_thread.join();
        }
    }

    // Send message to specific dashboard
    bool sendMessage(const std::string& dashboard_id, MessageType type, const json& data) {
        std::lock_guard<std::mutex> lock(connections_mutex);

        auto it = connections.find(dashboard_id);
        if (it == connections.end() || !it->second->is_active) {
            return false;
        }

        json message;
        message["type"] = static_cast<int>(type);
        message["dashboard_id"] = dashboard_id;
        message["data"] = data;
        message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        std::error_code ec;
        server.send(it->second->handle, message.dump(), websocketpp::frame::opcode::text, ec);

        if (!ec) {
            it->second->messages_sent++;
            return true;
        }

        return false;
    }

    // Broadcast message to all dashboards of specific type
    void broadcastMessage(const std::string& dashboard_type, MessageType type, const json& data) {
        std::lock_guard<std::mutex> lock(connections_mutex);

        for (const auto& pair : connections) {
            if (pair.second->dashboard_type == dashboard_type && pair.second->is_active) {
                sendMessage(pair.first, type, data);
            }
        }
    }

    // Register custom message handler
    void registerMessageHandler(MessageType type, std::function<void(const WebSocketMessage&)> handler) {
        message_handlers[type] = handler;
    }

    // Process incoming messages
    void processMessages() {
        std::lock_guard<std::mutex> lock(message_mutex);

        while (!incoming_messages.empty()) {
            auto message = incoming_messages.front();
            incoming_messages.pop();

            auto it = message_handlers.find(message.type);
            if (it != message_handlers.end()) {
                it->second(message);
            }
        }
    }

    // Connection status
    std::vector<std::string> getConnectedDashboards() const {
        std::lock_guard<std::mutex> lock(connections_mutex);

        std::vector<std::string> dashboards;
        for (const auto& pair : connections) {
            if (pair.second->is_active) {
                dashboards.push_back(pair.first);
            }
        }
        return dashboards;
    }

    size_t getConnectionCount() const {
        std::lock_guard<std::mutex> lock(connections_mutex);

        size_t active_count = 0;
        for (const auto& pair : connections) {
            if (pair.second->is_active) {
                active_count++;
            }
        }
        return active_count;
    }

    // Statistics
    json getConnectionStats() const {
        std::lock_guard<std::mutex> lock(connections_mutex);

        json stats;
        stats["total_connections"] = connections.size();
        stats["active_connections"] = 0;
        stats["total_messages_sent"] = 0;
        stats["total_messages_received"] = 0;

        for (const auto& pair : connections) {
            if (pair.second->is_active) {
                stats["active_connections"] = stats["active_connections"].get<int>() + 1;
            }
            stats["total_messages_sent"] = stats["total_messages_sent"].get<size_t>() + pair.second->messages_sent;
            stats["total_messages_received"] = stats["total_messages_received"].get<size_t>() + pair.second->messages_received;
        }

        return stats;
    }

private:
    void setupServer() {
        server.set_access_channels(websocketpp::log::alevel::all);
        server.clear_access_channels(websocketpp::log::alevel::frame_payload);

        server.init_asio();
        server.set_reuse_addr(true);

        // Bind handlers
        server.set_message_handler([this](websocketpp::connection_hdl hdl, WebSocketServer::message_ptr msg) {
            onMessage(hdl, msg);
        });

        server.set_open_handler([this](websocketpp::connection_hdl hdl) {
            onOpen(hdl);
        });

        server.set_close_handler([this](websocketpp::connection_hdl hdl) {
            onClose(hdl);
        });

        server.listen(port);
        server.start_accept();
    }

    void serverLoop() {
        try {
            server.run();
        } catch (const std::exception& e) {
            // Log error
        }
    }

    void onOpen(websocketpp::connection_hdl hdl) {
        // Connection opened - will register dashboard on first message
    }

    void onClose(websocketpp::connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connections_mutex);

        // Find and mark connection as inactive
        for (auto& pair : connections) {
            if (pair.second->handle.lock() == hdl.lock()) {
                pair.second->is_active = false;
                break;
            }
        }
    }

    void onMessage(websocketpp::connection_hdl hdl, WebSocketServer::message_ptr msg) {
        try {
            json message = json::parse(msg->get_payload());

            MessageType type = static_cast<MessageType>(message["type"].get<int>());
            std::string dashboard_id = message["dashboard_id"].get<std::string>();

            // Register connection if not exists
            if (type == MessageType::DASHBOARD_REGISTER) {
                registerDashboardConnection(hdl, dashboard_id, message["dashboard_type"].get<std::string>());
            }

            // Update connection stats
            {
                std::lock_guard<std::mutex> lock(connections_mutex);
                auto it = connections.find(dashboard_id);
                if (it != connections.end()) {
                    it->second->messages_received++;
                    it->second->last_ping = std::chrono::steady_clock::now();
                }
            }

            // Queue message for processing
            {
                std::lock_guard<std::mutex> lock(message_mutex);
                incoming_messages.emplace(type, dashboard_id, message["data"].dump());
            }

        } catch (const std::exception& e) {
            // Log parsing error
        }
    }

    void registerDashboardConnection(websocketpp::connection_hdl hdl,
                                   const std::string& dashboard_id,
                                   const std::string& dashboard_type) {
        std::lock_guard<std::mutex> lock(connections_mutex);

        auto connection = std::make_unique<DashboardConnection>(hdl, dashboard_id, dashboard_type);
        connections[dashboard_id] = std::move(connection);

        // Send welcome message
        json welcome;
        welcome["status"] = "registered";
        welcome["server_time"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        sendMessage(dashboard_id, MessageType::ENGINE_STATUS, welcome);
    }

    void registerDefaultHandlers() {
        // Utility data handler
        registerMessageHandler(MessageType::UTILITY_DATA, [this](const WebSocketMessage& msg) {
            handleUtilityData(msg);
        });

        // Test command handler
        registerMessageHandler(MessageType::TEST_COMMAND, [this](const WebSocketMessage& msg) {
            handleTestCommand(msg);
        });

        // Test result handler
        registerMessageHandler(MessageType::TEST_RESULT, [this](const WebSocketMessage& msg) {
            handleTestResult(msg);
        });

        // Metrics request handler
        registerMessageHandler(MessageType::METRICS_REQUEST, [this](const WebSocketMessage& msg) {
            handleMetricsRequest(msg);
        });

        // Control command handler
        registerMessageHandler(MessageType::CONTROL_COMMAND, [this](const WebSocketMessage& msg) {
            handleControlCommand(msg);
        });
    }

    void handleUtilityData(const WebSocketMessage& msg) {
        if (!integration_manager) return;

        try {
            json data = json::parse(msg.payload);
            std::string utility_name = data["utility"].get<std::string>();
            std::string utility_data = data["data"].dump();

            // Process through utility bridge
            integration_manager->getUtilityBridge()->processUtilityData(utility_name, utility_data);

            // Get output and send back
            std::string output = integration_manager->getUtilityBridge()->getUtilityOutput(utility_name);
            if (!output.empty()) {
                json response;
                response["utility"] = utility_name;
                response["output"] = json::parse(output);

                sendMessage(msg.dashboard_id, MessageType::UTILITY_DATA, response);
            }

        } catch (const std::exception& e) {
            // Log error
        }
    }

    void handleTestCommand(const WebSocketMessage& msg) {
        if (!integration_manager) return;

        try {
            json data = json::parse(msg.payload);
            std::string command = data["command"].get<std::string>();

            if (command == "run_test") {
                // Trigger nucleus test
                // This would integrate with the learning system
            } else if (command == "get_test_patterns") {
                // Return available test patterns for this dashboard
                json response;
                response["patterns"] = {"pattern1", "pattern2", "pattern3"};
                sendMessage(msg.dashboard_id, MessageType::TEST_RESULT, response);
            }

        } catch (const std::exception& e) {
            // Log error
        }
    }

    void handleTestResult(const WebSocketMessage& msg) {
        if (!integration_manager) return;

        // Process test result through learning system
        // This would feed into the nucleus self-learning
    }

    void handleMetricsRequest(const WebSocketMessage& msg) {
        if (!integration_manager) return;

        try {
            json data = json::parse(msg.payload);
            std::string metric_type = data["type"].get<std::string>();

            json response;

            if (metric_type == "system_status") {
                auto status = integration_manager->getSystemStatus();
                for (const auto& pair : status) {
                    response[pair.first] = pair.second;
                }
            } else if (metric_type == "dashboard_performance") {
                auto performance = integration_manager->getLearningSystem()->getDashboardPerformance();
                for (const auto& pair : performance) {
                    response[pair.first] = pair.second;
                }
            } else if (metric_type == "connection_stats") {
                response = getConnectionStats();
            }

            sendMessage(msg.dashboard_id, MessageType::METRICS_REQUEST, response);

        } catch (const std::exception& e) {
            // Log error
        }
    }

    void handleControlCommand(const WebSocketMessage& msg) {
        try {
            json data = json::parse(msg.payload);
            std::string command = data["command"].get<std::string>();

            json response;
            response["command"] = command;

            if (command == "ping") {
                response["status"] = "pong";
            } else if (command == "shutdown") {
                response["status"] = "shutting_down";
                // Initiate graceful shutdown
            } else if (command == "restart_learning") {
                if (integration_manager) {
                    integration_manager->getLearningSystem()->stopSelfLearning();
                    integration_manager->getLearningSystem()->startSelfLearning();
                    response["status"] = "learning_restarted";
                }
            }

            sendMessage(msg.dashboard_id, MessageType::CONTROL_COMMAND, response);

        } catch (const std::exception& e) {
            // Log error
        }
    }
};

// ============ JAVASCRIPT CLIENT TEMPLATE ============

const char* WEBSOCKET_CLIENT_JS = R"(
class NexusWebSocketClient {
    constructor(dashboardId, dashboardType, serverUrl = 'ws://localhost:8765') {
        this.dashboardId = dashboardId;
        this.dashboardType = dashboardType;
        this.serverUrl = serverUrl;
        this.socket = null;
        this.isConnected = false;
        this.messageHandlers = new Map();

        this.MessageType = {
            UTILITY_DATA: 0,
            TEST_COMMAND: 1,
            TEST_RESULT: 2,
            ENGINE_STATUS: 3,
            DASHBOARD_REGISTER: 4,
            METRICS_REQUEST: 5,
            CONTROL_COMMAND: 6
        };
    }

    connect() {
        try {
            this.socket = new WebSocket(this.serverUrl);

            this.socket.onopen = () => {
                this.isConnected = true;
                this.register();
                console.log('Connected to NEXUS WebSocket Bridge');
            };

            this.socket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            };

            this.socket.onclose = () => {
                this.isConnected = false;
                console.log('Disconnected from NEXUS WebSocket Bridge');
                // Auto-reconnect after 5 seconds
                setTimeout(() => this.connect(), 5000);
            };

            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

        } catch (e) {
            console.error('Failed to connect:', e);
        }
    }

    register() {
        this.sendMessage(this.MessageType.DASHBOARD_REGISTER, {
            dashboard_type: this.dashboardType,
            capabilities: this.getCapabilities()
        });
    }

    sendMessage(type, data) {
        if (!this.isConnected || !this.socket) return false;

        const message = {
            type: type,
            dashboard_id: this.dashboardId,
            data: data,
            timestamp: Date.now()
        };

        this.socket.send(JSON.stringify(message));
        return true;
    }

    handleMessage(message) {
        const handler = this.messageHandlers.get(message.type);
        if (handler) {
            handler(message.data);
        }
    }

    onMessage(type, handler) {
        this.messageHandlers.set(type, handler);
    }

    // Utility methods for different dashboard types
    sendUtilityData(utilityName, data) {
        return this.sendMessage(this.MessageType.UTILITY_DATA, {
            utility: utilityName,
            data: data
        });
    }

    runTest(testData) {
        return this.sendMessage(this.MessageType.TEST_COMMAND, {
            command: 'run_test',
            test_data: testData
        });
    }

    requestMetrics(metricType) {
        return this.sendMessage(this.MessageType.METRICS_REQUEST, {
            type: metricType
        });
    }

    getCapabilities() {
        // Override in dashboard implementations
        return [];
    }
}

// Auto-initialize for dashboards
window.NexusWebSocketClient = NexusWebSocketClient;
)";

} // namespace NEXUS
